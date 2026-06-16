import os
import time
import pdb
import re

import gradio as gr
import numpy as np
import sys
import subprocess

from huggingface_hub import snapshot_download
import requests

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *
from transformers import WhisperModel

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

def draw_dashed_rectangle(image, point1, point2, color=(0, 0, 255), thickness=2, dash_length=12):
    """Draw a dashed rectangle on an image and return the same image."""
    image = np.ascontiguousarray(image)
    x1, y1 = point1
    x2, y2 = point2

    for x in range(x1, x2, dash_length * 2):
        cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

    for y in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    return image

def draw_debug_parameter_overlay(
        image,
        face_box,
        original_bottom,
        bbox_shift,
        extra_margin,
        parsing_mode,
        left_cheek_width,
        right_cheek_width,
        bbox_left_ratio,
        bbox_right_ratio,
        bbox_top_ratio,
        bbox_bottom_ratio,
        side_protect_ratio):
    """Draw visual guides for the main inpainting debug parameters."""
    image = draw_dashed_rectangle(image, face_box[:2], face_box[2:])
    x1, y1, x2, y2 = face_box
    width = max(x2 - x1, 1)
    center_x = x1 + width // 2
    label_x = max(x1, 8)
    label_y = max(y1 - 12, 24)

    cv2.putText(
        image,
        f"mode={parsing_mode} shift={bbox_shift}",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    if extra_margin > 0 and original_bottom < y2:
        cv2.line(image, (x1, original_bottom), (x2, original_bottom), (0, 255, 255), 2)
        cv2.arrowedLine(image, (x2 + 12, original_bottom), (x2 + 12, y2), (0, 255, 255), 2, tipLength=0.25)
        cv2.putText(
            image,
            f"extra_margin={extra_margin}",
            (min(x2 + 18, image.shape[1] - 180), min(y2, image.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    shift_end_y = int(np.clip(y1 + bbox_shift, 0, image.shape[0] - 1))
    cv2.line(image, (center_x, y1), (center_x, y2), (255, 255, 255), 1)
    if bbox_shift != 0:
        cv2.arrowedLine(image, (center_x, y1), (center_x, shift_end_y), (255, 0, 255), 2, tipLength=0.25)

    left_boundary = int(np.clip(center_x - left_cheek_width * width / 512, x1, x2))
    right_boundary = int(np.clip(center_x + right_cheek_width * width / 512, x1, x2))
    side_protect_width = int(round(width * side_protect_ratio))
    side_protect_width = int(np.clip(side_protect_width, 0, width // 2))
    cv2.line(image, (left_boundary, y1), (left_boundary, y2), (255, 128, 0), 2)
    cv2.line(image, (right_boundary, y1), (right_boundary, y2), (255, 128, 0), 2)
    if side_protect_width > 0:
        cv2.line(image, (x1 + side_protect_width, y1), (x1 + side_protect_width, y2), (0, 255, 0), 2)
        cv2.line(image, (x2 - side_protect_width, y1), (x2 - side_protect_width, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"L={left_cheek_width} R={right_cheek_width} box={bbox_left_ratio:.2f}/{bbox_right_ratio:.2f}/{bbox_top_ratio:.2f}/{bbox_bottom_ratio:.2f} side={side_protect_ratio:.2f}",
        (label_x, min(y2 + 24, image.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 128, 0),
        1,
        cv2.LINE_AA
    )

    return image

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

@torch.no_grad()
def debug_inpainting(video_path, version, bbox_shift, extra_margin=8, parsing_mode="jaw",
                    left_cheek_width=120, right_cheek_width=120, bbox_smooth_window=7,
                    show_face_coordinates=False, side_protect_ratio=0.08,
                    bbox_left_ratio=0.0, bbox_right_ratio=0.0,
                    bbox_top_ratio=0.0, bbox_bottom_ratio=0.0):
    """Debug inpainting parameters, only process the first frame"""
    # Set default parameters
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 25, 
        "batch_size": 1, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15" if version == "v1.5" else version,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width,
        "side_protect_ratio": side_protect_ratio,
        "bbox_left_ratio": bbox_left_ratio,
        "bbox_right_ratio": bbox_right_ratio,
        "bbox_top_ratio": bbox_top_ratio,
        "bbox_bottom_ratio": bbox_bottom_ratio
    }
    args = Namespace(**args_dict)

    # Create debug directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Read first frame
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Save first frame
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Get face coordinates
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # Process first frame
    x1, y1, x2, y2 = adjust_face_box(
        bbox,
        frame.shape,
        bbox_left_ratio=args.bbox_left_ratio,
        bbox_right_ratio=args.bbox_right_ratio,
        bbox_top_ratio=args.bbox_top_ratio,
        bbox_bottom_ratio=args.bbox_bottom_ratio
    )
    original_y2 = y2
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    # Generate random audio features
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    
    # Get latents
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    
    # Generate prediction results
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    combine_frame = get_image(
        frame,
        res_frame,
        [x1, y1, x2, y2],
        mode=args.parsing_mode,
        fp=fp,
        side_protect_ratio=args.side_protect_ratio
    )

    if show_face_coordinates:
        combine_frame = draw_debug_parameter_overlay(
            combine_frame,
            (x1, y1, x2, y2),
            original_y2,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            bbox_left_ratio,
            bbox_right_ratio,
            bbox_top_ratio,
            bbox_bottom_ratio,
            side_protect_ratio
        )
    
    # Save results (no need to convert color space again since get_image already returns RGB format)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    # Create information text
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"version: {version}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"bbox_left_ratio: {bbox_left_ratio}\n" + \
                f"bbox_right_ratio: {bbox_right_ratio}\n" + \
                f"bbox_top_ratio: {bbox_top_ratio}\n" + \
                f"bbox_bottom_ratio: {bbox_bottom_ratio}\n" + \
                f"side_protect_ratio: {side_protect_ratio}\n" + \
                f"bbox_smooth_window: {bbox_smooth_window}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")




download_model()  # for huggingface deployment.

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.mask_utils import adjust_face_box
from musetalk.utils.audio_processor import AudioProcessor
import torch

# --- PyTorch 2.6+ Compatibility Monkey Patch ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ---------------------------------------------

from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(audio_path, video_path, version, bbox_shift, extra_margin=8, parsing_mode="jaw",
              left_cheek_width=120, right_cheek_width=120, bbox_smooth_window=7,
              side_protect_ratio=0.08, bbox_left_ratio=0.0, bbox_right_ratio=0.0,
              bbox_top_ratio=0.0, bbox_bottom_ratio=0.0,
              progress=gr.Progress(track_tqdm=True)):
    # Set default parameters, aligned with inference.py
    args_dict = {
        "result_dir": './results/output', 
        "fps": 25, 
        "batch_size": 8, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15" if version == "v1.5" else version,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width,
        "bbox_smooth_window": bbox_smooth_window,
        "side_protect_ratio": side_protect_ratio,
        "bbox_left_ratio": bbox_left_ratio,
        "bbox_right_ratio": bbox_right_ratio,
        "bbox_top_ratio": bbox_top_ratio,
        "bbox_bottom_ratio": bbox_bottom_ratio
    }
    args = Namespace(**args_dict)

    # Check ffmpeg
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # Create temporary directory
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set result save path
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        # Read video
        reader = imageio.get_reader(video_path)

        # Save images
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    ############################################## extract audio feature ##############################################
    # Extract audio features
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
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
        
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    if args.bbox_smooth_window > 1:
        print(f"Smoothing bbox coordinates with window size: {args.bbox_smooth_window}")
        coord_list = smooth_bbox_sequence(coord_list, args.bbox_smooth_window)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
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
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    ############################################## inference batch by batch ##############################################
    print("start inference")
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
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        # Ensure latent_batch is consistent with model weight type
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
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
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        
        # Use v15 version blending
        combine_frame = get_image(
            ori_frame,
            res_frame,
            [x1, y1, x2, y2],
            mode=args.parsing_mode,
            fp=fp,
            side_protect_ratio=args.side_protect_ratio
        )
            
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # Frame rate
    fps = 25
    # Output video path
    output_video = 'temp.mp4'

    # Read images
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # Save video
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read video
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # Get original video frame rate
    reader.close() # Otherwise, error on win11: PermissionError: [WinError 32] Another program is using this file, process cannot access. : 'temp.mp4'
    # Store frames in list
    frames = images
    
    print(len(frames))

    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text



# load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth", 
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help="Path to ffmpeg executable")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
parser.add_argument("--share", action="store_true", help="Create a public link")
parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
args = parser.parse_args()

# Set data type
if args.use_float16:
    # Convert models to half precision for better performance
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

# Move models to specified device
pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)

timesteps = torch.tensor([0], device=device)

# Initialize audio processor and Whisper model
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)


def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # save video
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video


def set_generate_running():
    """Disable the generate button and show generation status."""
    return (
        gr.update(value="Generating...", interactive=False),
        gr.update(value="正在生成视频，请稍候。页面会显示进度条，生成完成后按钮会自动恢复。", visible=True)
    )


def set_generate_idle():
    """Restore the generate button and hide generation status."""
    return (
        gr.update(value="2. Generate", interactive=True),
        gr.update(value="", visible=False)
    )



css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        version = gr.Radio(label="Version", choices=["v1.5"], value="v1.5")
        bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
        bbox_left_ratio = gr.Slider(label="BBox Left Adjust Ratio", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
        bbox_right_ratio = gr.Slider(label="BBox Right Adjust Ratio", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
        bbox_top_ratio = gr.Slider(label="BBox Top Adjust Ratio", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
        bbox_bottom_ratio = gr.Slider(label="BBox Bottom Adjust Ratio", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
        extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=8, step=1)
        parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
        left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=120, step=5)
        right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=120, step=5)
        bbox_smooth_window = gr.Slider(label="BBox Smooth Window", minimum=1, maximum=15, value=7, step=2)
        side_protect_ratio = gr.Slider(label="Side Protect Ratio", minimum=0.0, maximum=0.2, value=0.08, step=0.01)
        show_face_coordinates = gr.Checkbox(label="Show Detected Face Coordinates", value=True)

    bbox_shift_scale = gr.Markdown(visible=False)

    with gr.Row():
        debug_btn = gr.Button("1. Test Inpainting ")
        btn = gr.Button("2. Generate")
    generate_status = gr.Markdown(visible=False)

    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Reference Video",sources=['upload'])
            audio = gr.Audio(label="Drving Audio",type="filepath")
        with gr.Column():
            debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
            debug_info = gr.Textbox(label="Parameter Information", lines=5)
            out1 = gr.Video()

    gr.Markdown(
        """
### 参数说明

- **BBox_shift value, px**：控制检测到的人脸框整体在垂直方向上的偏移。正值通常让编辑区域向下移动，嘴部张开效果可能更明显；负值通常让编辑区域向上移动，嘴部变化会更收敛。建议先点击 `1. Test Inpainting` 查看可调范围，再在范围内微调。
- **BBox Left/Right/Top/Bottom Adjust Ratio**：分别控制红色检测框四条边。正数表示向外扩大，负数表示向内缩小；例如左耳贴近红框时把 `BBox Left Adjust Ratio` 调到 `-0.04` 或 `-0.06`，下巴区域不足时把 `BBox Bottom Adjust Ratio` 调到 `0.04`。
- **Version**：当前页面加载的是 MuseTalk `v1.5` 权重，因此页面版本固定为 `v1.5`。
- **Extra Margin**：在人脸框底部额外向下扩展的像素范围，主要影响下巴和 jaw 区域的融合空间。数值过小可能导致下巴附近融合不足，数值过大可能影响到脖子或衣领区域。
- **Parsing Mode**：控制融合 mask 的构建方式。`jaw` 会更关注下颌和脸颊边界，通常适合真人视频；`raw` 更接近原始解析区域，适合在 `jaw` 出现边缘异常时对比排查。
- **Left Cheek Width**：在 `jaw` 模式下控制左脸颊侧的编辑保护范围。数值越大，左侧脸颊参与编辑的范围越收敛；数值越小，左侧脸颊更容易被融合区域影响。
- **Right Cheek Width**：在 `jaw` 模式下控制右脸颊侧的编辑保护范围。数值越大，右侧脸颊参与编辑的范围越收敛；数值越小，右侧脸颊更容易被融合区域影响。
- **BBox Smooth Window**：对完整视频生成时的人脸框序列做居中滑动平滑，`1` 表示关闭。数值越大，框抖动越少，但过大可能让快速头部运动跟随变慢；常用值为 `5` 或 `7`。
- **Side Protect Ratio**：降低人脸框左右边缘的融合强度，用于保护耳朵和头发边界。耳朵有拼接线时建议从 `0.08` 调到 `0.10` 或 `0.12`；如果嘴角或脸颊变化被压得太小，再降到 `0.04` 到 `0.06`。
- **Show Detected Face Coordinates**：开启后，`1. Test Inpainting` 的预览图会叠加调试标注。红色虚线框表示最终检测框，黄色标注表示 `Extra Margin`，紫色箭头表示 `BBox_shift`，橙色竖线表示左右脸颊宽度的近似影响边界，绿色竖线表示 `Side Protect Ratio` 的左右保护边界。

建议调参顺序：先调 `BBox_shift value, px`，耳朵贴近红框时调对应侧的 `BBox Adjust Ratio` 为负数，再调 `Extra Margin`，然后只在脸颊边缘异常时调整 `Left Cheek Width` 和 `Right Cheek Width`，耳朵或头发边缘仍有拼接线时调整 `Side Protect Ratio`，最后根据视频抖动情况调整 `BBox Smooth Window`。

### 推荐参数

- **通用起始值**：`Version = v1.5`，`BBox_shift value, px = 0`，四个 `BBox Adjust Ratio = 0`，`Extra Margin = 8`，`Parsing Mode = jaw`，`Left Cheek Width = 120`，`Right Cheek Width = 120`，`BBox Smooth Window = 7`，`Side Protect Ratio = 0.08`。
- **嘴型张开不够**：优先把 `BBox_shift value, px` 往正数方向微调，例如 `3` 到 `8`；如果下巴融合空间不足，再把 `Extra Margin` 调到 `12` 到 `18`。
- **嘴型变化过大或下巴变形**：优先把 `BBox_shift value, px` 往负数方向微调，例如 `-3` 到 `-8`；必要时降低 `Extra Margin`。
- **脸颊边缘被明显改动**：在 `jaw` 模式下增大对应侧的 cheek width，例如把 `Left Cheek Width` 或 `Right Cheek Width` 从 `90` 调到 `110` 到 `130`。
- **融合边缘不自然**：先保持 `Parsing Mode = jaw`，微调 `Extra Margin`；如果 jaw 模式边界异常明显，再切换到 `raw` 对比效果。
- **耳朵或头发边缘有拼接线**：保持 `Parsing Mode = jaw`，优先把对应侧的 `BBox Adjust Ratio` 调到 `-0.04` 或 `-0.06`；如果仍有线，再把 `Side Protect Ratio` 调到 `0.10` 或 `0.12`。

### 常见问题：两边嘴角有黑点

嘴角黑点通常是融合 mask 边缘或嘴角区域被过度编辑导致的。建议先勾选 `Show Detected Face Coordinates`，点击 `1. Test Inpainting` 查看红色检测框和橙色脸颊边界。

- 优先增大左右脸颊保护范围：把 `Left Cheek Width` 和 `Right Cheek Width` 从 `90` 调到 `110` 或 `120`。
- 如果黑点出现在下嘴角或下巴附近：把 `Extra Margin` 从 `10` 降到 `6` 或 `8`。
- 如果黑点仍然存在：把 `BBox_shift value, px` 往负数方向微调，例如从 `5` 依次试 `2`、`0`、`-3`。
- 如果 `jaw` 模式持续出现边缘异常：切换到 `raw` 对比一次。

推荐尝试组合：`BBox_shift value, px = 0`，`Extra Margin = 8`，`Parsing Mode = jaw`，`Left Cheek Width = 120`，`Right Cheek Width = 120`。如果黑点消失但嘴型不够明显，再只把 `BBox_shift value, px` 从 `0` 微调到 `2` 到 `5`。
"""
    )
    
    video.change(
        fn=check_video, inputs=[video], outputs=[video]
    )
    generate_event = btn.click(
        fn=set_generate_running,
        inputs=[],
        outputs=[btn, generate_status]
    )
    generate_event.then(
        fn=inference,
        inputs=[
            audio,
            video,
            version,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            bbox_smooth_window,
            side_protect_ratio,
            bbox_left_ratio,
            bbox_right_ratio,
            bbox_top_ratio,
            bbox_bottom_ratio
        ],
        outputs=[out1,bbox_shift_scale]
    ).then(
        fn=set_generate_idle,
        inputs=[],
        outputs=[btn, generate_status]
    )
    debug_btn.click(
        fn=debug_inpainting,
        inputs=[
            video,
            version,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            bbox_smooth_window,
            show_face_coordinates,
            side_protect_ratio,
            bbox_left_ratio,
            bbox_right_ratio,
            bbox_top_ratio,
            bbox_bottom_ratio
        ],
        outputs=[debug_image, debug_info]
    )

# Check ffmpeg and add to PATH
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    # According to operating system, choose path separator
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# Solve asynchronous IO issues on Windows
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Start Gradio application
demo.queue().launch(
    share=args.share, 
    debug=True, 
    server_name=args.ip, 
    server_port=args.port
)
