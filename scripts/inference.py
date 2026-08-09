import os
import cv2
import copy
import torch

# --- PyTorch 2.6+ Compatibility Monkey Patch ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ---------------------------------------------

import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.mask_utils import adjust_face_box
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.audio_utils import get_active_audio_frame_range, parse_bool
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, release_landmark_models

def smooth_bbox_sequence(coord_list, window_size):
    """Smooth valid face bounding boxes while stabilizing crop scale over time."""
    if window_size <= 1:
        return coord_list

    half_window = window_size // 2
    smoothed_coords = []
    valid_coords = [
        None if bbox == coord_placeholder else np.asarray(bbox, dtype=np.float32)
        for bbox in coord_list
    ]

    for index, current_bbox in enumerate(valid_coords):
        if current_bbox is None:
            smoothed_coords.append(coord_placeholder)
            continue

        start = max(0, index - half_window)
        end = min(len(valid_coords), index + half_window + 1)
        window_coords = [bbox for bbox in valid_coords[start:end] if bbox is not None]
        if not window_coords:
            smoothed_coords.append(coord_list[index])
            continue

        window_array = np.asarray(window_coords, dtype=np.float32)
        centers_x = (window_array[:, 0] + window_array[:, 2]) / 2.0
        centers_y = (window_array[:, 1] + window_array[:, 3]) / 2.0
        widths = window_array[:, 2] - window_array[:, 0]
        heights = window_array[:, 3] - window_array[:, 1]

        center_x = float(np.mean(centers_x))
        center_y = float(np.mean(centers_y))
        width = float(np.median(widths))
        height = float(np.median(heights))

        x1 = int(round(center_x - width / 2.0))
        y1 = int(round(center_y - height / 2.0))
        x2 = int(round(center_x + width / 2.0))
        y2 = int(round(center_y + height / 2.0))
        smoothed_coords.append([x1, y1, x2, y2])

    return smoothed_coords

def fast_check_ffmpeg():
    """Return whether ffmpeg is available in the current PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    
    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    
    # Process each task
    has_error = False
    for task_id in inference_config:
        try:
            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            if "result_name" in inference_config[task_id]:
                args.output_vid_name = inference_config[task_id]["result_name"]
            
            # Pre-validate files
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Input face video/image/directory path does not exist: {video_path}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Input audio path does not exist: {audio_path}")
            if not os.path.isdir(video_path) and os.path.getsize(video_path) == 0:
                raise ValueError(f"Input face video/image file is empty (0 bytes): {video_path}")
            if os.path.getsize(audio_path) == 0:
                raise ValueError(f"Input audio file is empty (0 bytes): {audio_path}")

            # Set bbox_shift based on version
            bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
            
            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"
            
            # Create temporary directories
            temp_dir = os.path.join(args.result_dir, f"{args.version}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Set result save paths
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            
            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            output_vid_name_concat = os.path.join(temp_dir, output_basename + "_concat.mp4")
            
            # Extract frames from source video
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                ret_code = os.system(cmd)
                if ret_code != 0:
                    raise RuntimeError(f"ffmpeg frame extraction failed with status code {ret_code}.")
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                if not input_img_list:
                    raise ValueError(f"No frames extracted from {video_path}. The video may be corrupted (e.g., moov atom missing).")
                fps = get_video_fps(video_path)
                if fps <= 0:
                    raise ValueError(f"Invalid fps detected for video {video_path}: {fps}")
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
                if not input_img_list:
                    raise ValueError(f"No images found in directory: {video_path}")
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

            # Preprocess input images
            try:
                if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                    print("Using saved coordinates")
                    with open(crop_coord_save_path, 'rb') as f:
                        coord_list = pickle.load(f)
                    frame_list = read_imgs(input_img_list)
                else:
                    print("Extracting landmarks... time-consuming operation")
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                    with open(crop_coord_save_path, 'wb') as f:
                        pickle.dump(coord_list, f)
            finally:
                release_landmark_models()

            # Load generation models after landmark detection to avoid GPU memory overlap.
            vae, unet, pe = load_all_model(
                unet_model_path=args.unet_model_path,
                vae_type=args.vae_type,
                unet_config=args.unet_config,
                device=device
            )
            timesteps = torch.tensor([0], device=device)

            if args.use_float16:
                pe = pe.half()
                vae.vae = vae.vae.half()
                unet.model = unet.model.half()

            pe = pe.to(device)
            vae.vae = vae.vae.to(device)
            unet.model = unet.model.to(device)
            weight_dtype = unet.model.dtype

            audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
            whisper = WhisperModel.from_pretrained(args.whisper_dir)
            whisper = whisper.to(device=device, dtype=weight_dtype).eval()
            whisper.requires_grad_(False)

            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            if whisper_input_features is None:
                raise ValueError(f"Failed to process audio or extract features from {audio_path}")
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )
            active_audio_start_frame, active_audio_end_frame = get_active_audio_frame_range(
                audio_path,
                fps
            )
            active_audio_start_frame = min(active_audio_start_frame, len(whisper_chunks))
            active_audio_end_frame = min(active_audio_end_frame, len(whisper_chunks))
            print(
                f"Active audio spans frames {active_audio_start_frame}:"
                f"{active_audio_end_frame}/{len(whisper_chunks)}; leading and trailing "
                "silence will use the neutral mouth"
            )
            del whisper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.bbox_smooth_window > 1:
                print(f"Smoothing bbox coordinates with window size: {args.bbox_smooth_window}")
                coord_list = smooth_bbox_sequence(coord_list, args.bbox_smooth_window)
            
            print(f"Number of frames: {len(frame_list)}")         
            
            # Process each frame (ensuring strict length alignment with frame_list to prevent cumulative index drift)
            input_latent_list = []
            last_valid_bbox = None
            last_valid_latents = None
            
            # Find the first valid bbox and latents as initial fallback
            first_valid_bbox = None
            first_valid_latents = None
            for bbox, frame in zip(coord_list, frame_list):
                if bbox != coord_placeholder and (bbox[2]-bbox[0]) > 0 and (bbox[3]-bbox[1]) > 0:
                    x1, y1, x2, y2 = bbox
                    if args.version == "v15":
                        x1, y1, x2, y2 = adjust_face_box(
                            bbox,
                            frame.shape,
                            bbox_left_ratio=args.bbox_left_ratio,
                            bbox_right_ratio=args.bbox_right_ratio,
                            bbox_top_ratio=args.bbox_top_ratio,
                            bbox_bottom_ratio=args.bbox_bottom_ratio
                        )
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    try:
                        crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                        first_valid_latents = vae.get_latents_for_unet(crop_frame)
                        first_valid_bbox = bbox
                        break
                    except:
                        continue
            
            # Absolute fallback if no valid face is found at all
            if first_valid_bbox is None:
                first_valid_bbox = [0, 0, 100, 100]
                dummy = np.zeros((256, 256, 3), dtype=np.uint8)
                first_valid_latents = vae.get_latents_for_unet(dummy)

            last_valid_bbox = first_valid_bbox
            last_valid_latents = first_valid_latents

            for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder or (bbox[2]-bbox[0]) <= 0 or (bbox[3]-bbox[1]) <= 0:
                    # Face detection failed, fallback to previous valid frame data without skipping
                    coord_list[idx] = last_valid_bbox
                    input_latent_list.append(last_valid_latents)
                    continue
                
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    x1, y1, x2, y2 = adjust_face_box(
                        bbox,
                        frame.shape,
                        bbox_left_ratio=args.bbox_left_ratio,
                        bbox_right_ratio=args.bbox_right_ratio,
                        bbox_top_ratio=args.bbox_top_ratio,
                        bbox_bottom_ratio=args.bbox_bottom_ratio
                    )
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                
                crop_frame = frame[y1:y2, x1:x2]
                try:
                    crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                    latents = vae.get_latents_for_unet(crop_frame)
                    last_valid_bbox = bbox
                    last_valid_latents = latents
                    input_latent_list.append(latents)
                except Exception as e:
                    # Fallback on resize/extraction failures
                    coord_list[idx] = last_valid_bbox
                    input_latent_list.append(last_valid_latents)
        
            # Ping-Pong mirror cycle to eliminate seam at loop boundary
            # Original: [0, 1, ..., N-1] → Extended: [0, 1, ..., N-1, N-2, ..., 1]
            frame_list_cycle = frame_list + frame_list[-2::-1] if len(frame_list) > 1 else frame_list
            coord_list_cycle = coord_list + coord_list[-2::-1] if len(coord_list) > 1 else coord_list
            input_latent_list_cycle = input_latent_list + input_latent_list[-2::-1] if len(input_latent_list) > 1 else input_latent_list

            print(f"Frame cycle length: {len(frame_list)} → {len(frame_list_cycle)} (ping-pong)")
            
            # Batch inference
            print("Starting inference")
            video_num = len(whisper_chunks)
            batch_size = args.batch_size
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))
            
            # Execute inference
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            if args.version == "v15":
                fp = FaceParsing(
                    left_cheek_width=args.left_cheek_width,
                    right_cheek_width=args.right_cheek_width
                )
            else:
                fp = FaceParsing()

            # Pad generated images to original video size
            print("Padding generated images to original video size")
            neutral_frame = copy.deepcopy(frame_list[0])
            neutral_bbox = coord_list[0]
            neutral_face = None
            if (
                neutral_bbox != coord_placeholder
                and (neutral_bbox[2] - neutral_bbox[0]) > 0
                and (neutral_bbox[3] - neutral_bbox[1]) > 0
            ):
                neutral_x1, neutral_y1, neutral_x2, neutral_y2 = neutral_bbox
                if args.version == "v15":
                    neutral_x1, neutral_y1, neutral_x2, neutral_y2 = adjust_face_box(
                        neutral_bbox,
                        neutral_frame.shape,
                        bbox_left_ratio=args.bbox_left_ratio,
                        bbox_right_ratio=args.bbox_right_ratio,
                        bbox_top_ratio=args.bbox_top_ratio,
                        bbox_bottom_ratio=args.bbox_bottom_ratio
                    )
                    neutral_y2 = min(neutral_y2 + args.extra_margin, neutral_frame.shape[0])
                neutral_face = neutral_frame[neutral_y1:neutral_y2, neutral_x1:neutral_x2]

            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                
                # If no face is detected or coordinate is invalid, fallback to original frame to avoid sequence gap
                if bbox == coord_placeholder or (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue
                
                if args.version == "v15":
                    x1, y1, x2, y2 = adjust_face_box(
                        bbox,
                        ori_frame.shape,
                        bbox_left_ratio=args.bbox_left_ratio,
                        bbox_right_ratio=args.bbox_right_ratio,
                        bbox_top_ratio=args.bbox_top_ratio,
                        bbox_bottom_ratio=args.bbox_bottom_ratio
                    )
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, ori_frame.shape[0])

                should_close_mouth = (
                    (i < active_audio_start_frame and args.close_mouth_start)
                    or (i >= active_audio_end_frame and args.close_mouth_end)
                )
                if should_close_mouth:
                    try:
                        if neutral_face is None or neutral_face.size == 0:
                            raise ValueError("Neutral reference face is unavailable")
                        closed_face = cv2.resize(
                            neutral_face,
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                        if args.version == "v15":
                            combine_frame = get_image(
                                ori_frame,
                                closed_face,
                                [x1, y1, x2, y2],
                                mode=args.parsing_mode,
                                fp=fp,
                                side_protect_ratio=args.side_protect_ratio
                            )
                        else:
                            combine_frame = get_image(
                                ori_frame,
                                closed_face,
                                [x1, y1, x2, y2],
                                fp=fp
                            )
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                    except Exception as e:
                        print(f"Warning: failed to blend neutral mouth at frame {i}: {e}")
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue

                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except Exception as e:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue
                
                # Merge results with version-specific parameters
                try:
                    if args.version == "v15":
                        combine_frame = get_image(
                            ori_frame,
                            res_frame,
                            [x1, y1, x2, y2],
                            mode=args.parsing_mode,
                            fp=fp,
                            side_protect_ratio=args.side_protect_ratio
                        )
                    else:
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                except Exception as e:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)

            # Save prediction results
            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            print("Video generation command:", cmd_img2video)
            os.system(cmd_img2video)   
            
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
            print("Audio combination command:", cmd_combine_audio) 
            os.system(cmd_combine_audio)
            
            # Clean up temporary files
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)
            
            shutil.rmtree(save_dir_full)
            if not args.saved_coord:
                os.remove(crop_coord_save_path)
                    
            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)
            has_error = True

    if has_error:
        print("\n❌ Error: One or more tasks failed during processing.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=float, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--side_protect_ratio", type=float, default=0.08, help="Side-edge blend protection ratio")
    parser.add_argument("--bbox_left_ratio", type=float, default=0.0, help="Left bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_right_ratio", type=float, default=0.0, help="Right bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_top_ratio", type=float, default=0.0, help="Top bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_bottom_ratio", type=float, default=0.0, help="Bottom bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_smooth_window", type=int, default=1, help="Centered moving-average window for bbox smoothing; 1 disables smoothing")
    parser.add_argument("--close_mouth_start", type=parse_bool, default=True, help="Close mouth during leading silence (true/false)")
    parser.add_argument("--close_mouth_end", type=parse_bool, default=False, help="Close mouth during trailing silence (true/false)")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    args = parser.parse_args()
    main(args)
