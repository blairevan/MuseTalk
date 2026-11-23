import os
import tempfile
import cv2
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
import pickle
import argparse
import numpy as np
import subprocess
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.mask_utils import adjust_face_box
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.audio_utils import (
    build_audio_mux_command,
    get_active_audio_frame_intervals,
    get_active_audio_frame_range,
    parse_bool,
    pause_close_weights,
)
from musetalk.utils.clarity import (
    ENHANCE_MODES,
    PASTE_INTERPOLATION_MODES,
    SUPPORTED_RESOLUTIONS,
    blend_faces,
    correct_teeth_whiteness,
    resize_generated_face,
    sharpen_lip_region,
    validate_resolution,
)
from musetalk.utils.frame_io import FrameSequence
from musetalk.utils.resource_utils import release_torch_resources
from musetalk.utils.sequence_utils import build_ping_pong_cycle
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_frame_bboxes, coord_placeholder, release_landmark_models

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

def enhance_generated_face(face, args, fp):
    """Apply the opt-in teeth correction and lip sharpening to a resized face.

    Args:
        face (np.ndarray): Generated BGR face resized to the destination box.
        args (argparse.Namespace): Parsed CLI options.
        fp (FaceParsing | None): Face parser used to find the mouth interior.

    Returns:
        np.ndarray: Enhanced face, or the input when every option is off.
    """
    if args.fix_teeth_whiteness and fp is not None:
        mouth_mask = None
        try:
            parsed = fp(Image.fromarray(face[:, :, ::-1]), mode="mouth")
            parsed = parsed.resize((face.shape[1], face.shape[0]), Image.NEAREST)
            mouth_mask = np.array(parsed)
        except Exception as e:
            print(f"Warning: mouth parsing failed, skipping teeth correction: {e}")
        face = correct_teeth_whiteness(face, mouth_mask)
    if args.enhance_mode == "usm":
        generated_size = validate_resolution(getattr(args, "resolution", 256))
        face = sharpen_lip_region(
            face,
            amount_override=args.usm_amount,
            generated_size=generated_size,
        )
    return face

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
    
    # Validate generation resolution before loading any models
    resolution = validate_resolution(getattr(args, "resolution", 256))

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    
    # Process each task
    has_error = False
    for task_id in inference_config:
        vae = None
        unet = None
        pe = None
        whisper = None
        fp = None
        audio_processor = None
        timesteps = None
        whisper_input_features = None
        whisper_chunks = None
        input_latent_list = None
        input_latent_list_cycle = None
        gen = None
        res_frame_list = None
        whisper_batch = None
        latent_batch = None
        audio_feature_batch = None
        pred_latents = None
        recon = None
        task_directory = None
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
            
            # Keep temporary frames separate across jobs and retries.
            task_directory = tempfile.TemporaryDirectory(prefix="inference-", dir=temp_dir)
            result_img_save_path = os.path.join(task_directory.name, "rendered")
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            
            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            
            # Extract frames from source video
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(task_directory.name, "source")
                os.makedirs(save_dir_full, exist_ok=True)
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", video_path,
                     "-start_number", "0", os.path.join(save_dir_full, "%08d.png")],
                    check=True,
                )
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
            frames = FrameSequence(input_img_list)
            try:
                coord_list = None
                if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                    with open(crop_coord_save_path, 'rb') as f:
                        saved_coords = pickle.load(f)
                    if len(saved_coords) == len(frames):
                        print("Using saved coordinates")
                        coord_list = saved_coords
                    else:
                        print("Ignoring saved coordinates with a different frame count")
                if coord_list is None:
                    print("Extracting landmarks... time-consuming operation")
                    coord_list = get_frame_bboxes(frames, bbox_shift)
                    if len(coord_list) != len(frames):
                        raise ValueError("Expected one face box per source frame")
                    # Only persist coordinates on request: the file is keyed by the face
                    # file name and shared by every task with the same template
                    if args.saved_coord:
                        with open(crop_coord_save_path, 'wb') as f:
                            pickle.dump(coord_list, f)
            finally:
                release_landmark_models()

            # Load generation models after landmark detection to avoid GPU memory overlap.
            vae, unet, pe = load_all_model(
                unet_model_path=args.unet_model_path,
                vae_type=args.vae_type,
                unet_config=args.unet_config,
                device=device,
                vae_latent_mode=args.vae_latent_mode,
                resolution=resolution,
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
            pause_weights = np.zeros(len(whisper_chunks), dtype=np.float32)
            if args.close_mouth_pauses:
                voiced_intervals = get_active_audio_frame_intervals(
                    audio_path,
                    fps,
                    top_db=args.pause_top_db,
                )
                pause_weights = pause_close_weights(
                    voiced_intervals,
                    len(whisper_chunks),
                    min_pause_frames=args.pause_min_frames,
                    fade_frames=args.pause_fade_frames,
                )
                print(
                    f"Mid-sentence pauses: {int(np.count_nonzero(pause_weights))} frames "
                    "blend toward the neutral mouth"
                )
            whisper_cleanup_errors = release_torch_resources(whisper)
            for cleanup_error in whisper_cleanup_errors:
                print(f"Warning: failed to release Whisper model: {cleanup_error}")
            whisper = None

            if args.bbox_smooth_window > 1:
                print(f"Smoothing bbox coordinates with window size: {args.bbox_smooth_window}")
                coord_list = smooth_bbox_sequence(coord_list, args.bbox_smooth_window)
            
            print(f"Number of frames: {len(frames)}")
            
            # Keep box and latent indices aligned with the source frames.
            input_latent_list = []
            last_valid_bbox = None
            last_valid_latents = None
            
            # Find the first valid bbox and latents as initial fallback
            first_valid_bbox = None
            first_valid_latents = None
            for index, bbox in enumerate(coord_list):
                if bbox != coord_placeholder and (bbox[2]-bbox[0]) > 0 and (bbox[3]-bbox[1]) > 0:
                    frame = frames[index]
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
                        crop_frame = cv2.resize(
                            crop_frame,
                            (resolution, resolution),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        first_valid_latents = vae.get_latents_for_unet(crop_frame)
                        first_valid_bbox = bbox
                        break
                    except:
                        continue
            
            # Absolute fallback if no valid face is found at all
            if first_valid_bbox is None:
                first_valid_bbox = [0, 0, 100, 100]
                dummy = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                first_valid_latents = vae.get_latents_for_unet(dummy)

            last_valid_bbox = first_valid_bbox
            last_valid_latents = first_valid_latents

            for idx, bbox in enumerate(coord_list):
                if bbox == coord_placeholder or (bbox[2]-bbox[0]) <= 0 or (bbox[3]-bbox[1]) <= 0:
                    # Face detection failed, fallback to previous valid frame data without skipping
                    coord_list[idx] = last_valid_bbox
                    input_latent_list.append(last_valid_latents)
                    continue
                
                frame = frames[idx]
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
                    crop_frame = cv2.resize(
                        crop_frame,
                        (resolution, resolution),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
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
            frame_index_cycle = build_ping_pong_cycle(list(range(len(frames))))
            input_latent_list_cycle = [input_latent_list[index] for index in frame_index_cycle]

            print(f"Frame cycle length: {len(frames)} → {len(frame_index_cycle)} (ping-pong)")
            
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
                audio_feature_batch = pe(whisper_batch) if args.use_audio_pe else whisper_batch
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            res_frame_list = res_frame_list[:video_num]

            if args.version == "v15":
                fp = FaceParsing(
                    left_cheek_width=args.left_cheek_width,
                    right_cheek_width=args.right_cheek_width
                )
            else:
                fp = FaceParsing()

            # Pad generated images to original video size
            print("Padding generated images to original video size")
            neutral_bbox = coord_list[0]
            neutral_face = None
            if (
                neutral_bbox != coord_placeholder
                and (neutral_bbox[2] - neutral_bbox[0]) > 0
                and (neutral_bbox[3] - neutral_bbox[1]) > 0
            ):
                neutral_frame = frames[0]
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
                neutral_face = neutral_frame[neutral_y1:neutral_y2, neutral_x1:neutral_x2].copy()
                del neutral_frame

            for i, res_frame in enumerate(tqdm(res_frame_list)):
                source_index = frame_index_cycle[i % len(frame_index_cycle)]
                bbox = coord_list[source_index]
                ori_frame = frames[source_index]
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

                pause_weight = float(pause_weights[i]) if i < len(pause_weights) else 0.0
                has_neutral_face = neutral_face is not None and neutral_face.size > 0
                should_close_mouth = (
                    (i < active_audio_start_frame and args.close_mouth_start)
                    or (i >= active_audio_end_frame and args.close_mouth_end)
                    or (pause_weight >= 1.0 and has_neutral_face)
                )
                if should_close_mouth:
                    try:
                        if not has_neutral_face:
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
                                side_protect_ratio=args.side_protect_ratio,
                                protect_upper_face=args.protect_upper_face,
                            )
                        else:
                            combine_frame = get_image(
                                ori_frame,
                                closed_face,
                                [x1, y1, x2, y2],
                                fp=fp,
                                protect_upper_face=args.protect_upper_face,
                            )
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                    except Exception as e:
                        print(f"Warning: failed to blend neutral mouth at frame {i}: {e}")
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue

                try:
                    res_frame = resize_generated_face(
                        res_frame, x2 - x1, y2 - y1, mode=args.paste_interpolation
                    )
                except Exception as e:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue

                # Opt-in teeth correction and lip sharpening of the generated face
                res_frame = enhance_generated_face(res_frame, args, fp)

                # Fade toward the neutral mouth at the edges of mid-sentence pauses
                if pause_weight > 0.0 and has_neutral_face:
                    closed_face = cv2.resize(
                        neutral_face,
                        (x2 - x1, y2 - y1),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    res_frame = blend_faces(res_frame, closed_face, pause_weight)
                
                # Merge results with version-specific parameters
                try:
                    if args.version == "v15":
                        combine_frame = get_image(
                            ori_frame,
                            res_frame,
                            [x1, y1, x2, y2],
                            mode=args.parsing_mode,
                            fp=fp,
                            side_protect_ratio=args.side_protect_ratio,
                            protect_upper_face=args.protect_upper_face,
                        )
                    else:
                        combine_frame = get_image(
                            ori_frame,
                            res_frame,
                            [x1, y1, x2, y2],
                            fp=fp,
                            protect_upper_face=args.protect_upper_face,
                        )
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                except Exception as e:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)

            # Save prediction results
            temp_vid_path = os.path.join(task_directory.name, "rendered.mp4")
            cmd_img2video = [
                "ffmpeg", "-nostdin", "-y", "-v", "warning", "-r", str(fps), "-f", "image2",
                "-i", os.path.join(result_img_save_path, "%08d.png"), "-vcodec", "libx264",
                "-vf", "format=yuv420p", "-crf", "18", temp_vid_path,
            ]
            print("Video generation command:", cmd_img2video)
            subprocess.run(cmd_img2video, check=True)
            
            cmd_combine_audio = build_audio_mux_command(audio_path, temp_vid_path, output_vid_name)
            print("Audio combination command:", cmd_combine_audio) 
            subprocess.run(cmd_combine_audio, check=True)
            
            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)
            has_error = True
        finally:
            if task_directory is not None:
                task_directory.cleanup()
            task_resources = (
                getattr(vae, "vae", None),
                getattr(unet, "model", None),
                pe,
                whisper,
                getattr(fp, "net", None),
            )
            vae = None
            unet = None
            pe = None
            whisper = None
            fp = None
            audio_processor = None
            timesteps = None
            whisper_input_features = None
            whisper_chunks = None
            input_latent_list = None
            input_latent_list_cycle = None
            gen = None
            res_frame_list = None
            whisper_batch = None
            latent_batch = None
            audio_feature_batch = None
            pred_latents = None
            recon = None
            cleanup_errors = release_torch_resources(*task_resources)
            task_resources = None
            for cleanup_error in cleanup_errors:
                print(f"Warning: failed to release task model: {cleanup_error}")

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
    parser.add_argument("--resolution", type=int, default=256, choices=list(SUPPORTED_RESOLUTIONS),
                        help="UNet/VAE generation resolution in pixels (256 legacy or 512 high-resolution)")
    # Clarity options. The first four defaults were adopted after 4K A/B runs on
    # an RTX 5070; pass the legacy values to reproduce the previous output.
    parser.add_argument("--vae_latent_mode", type=str, default="mode", choices=["sample", "mode"],
                        help="VAE encoding: 'mode' (deterministic, matches training) or 'sample' (legacy)")
    parser.add_argument("--paste_interpolation", type=str, default="lanczos", choices=list(PASTE_INTERPOLATION_MODES),
                        help="Resize when pasting the generated face: 'lanczos' (area down, Lanczos up) or 'linear' (legacy)")
    parser.add_argument("--enhance_mode", type=str, default="usm", choices=list(ENHANCE_MODES),
                        help="Lip clarity enhancement: 'usm' (luma unsharp mask on the lip area) or 'none' (legacy)")
    parser.add_argument("--usm_amount", type=float, default=-1.0,
                        help="Unsharp-mask amount; negative selects it from the face enlargement factor")
    parser.add_argument("--fix_teeth_whiteness", type=parse_bool, default=False,
                        help="Tone down over-bright teeth inside the parsed mouth interior (true/false)")
    parser.add_argument("--protect_upper_face", type=parse_bool, default=True,
                        help="Keep exact source pixels above the blend boundary (true/false; legacy false)")
    parser.add_argument("--close_mouth_pauses", type=parse_bool, default=False,
                        help="Blend to the neutral mouth during mid-sentence pauses (true/false)")
    parser.add_argument("--pause_min_frames", type=int, default=8, help="Shortest pause, in frames, that closes the mouth")
    parser.add_argument("--pause_fade_frames", type=int, default=4, help="Frames used to fade into and out of a pause")
    parser.add_argument("--pause_top_db", type=float, default=38.0, help="Silence threshold below the audio peak, in dB")
    parser.add_argument("--use_audio_pe", type=parse_bool, default=True,
                        help="Add positional encoding to Whisper features (legacy true; for A/B against training)")
    args = parser.parse_args()
    main(args)
