import argparse
import os
import sys
import tempfile
import yaml
import subprocess

from musetalk.utils.audio_utils import parse_bool
from musetalk.utils.clarity import (
    REF_FRAME_MODES,
    SUPPORTED_RESOLUTIONS,
    validate_ref_frame_mode,
    validate_resolution,
)


DEFAULT_512_UNET_PATH = "./models/musetalkV15_512/unet.pth"
DEFAULT_512_CONFIG_PATH = "./models/musetalkV15_512/musetalk.json"


def resolve_model_settings(
    model_version,
    resolution=None,
    unet_model_path=None,
    unet_config=None,
    use_audio_pe=None,
    ref_frame_mode=None,
    batch_size=None,
    use_float16=None,
    path_exists=None,
):
    """Resolve model checkpoint paths, resolution, precision, and inference defaults.

    When ``resolution`` is ``None`` and ``model_version == "v1.5"`` with no
    explicit UNet overrides, automatically selects ``512`` if both
    :data:`DEFAULT_512_UNET_PATH` and :data:`DEFAULT_512_CONFIG_PATH` exist on
    disk; otherwise falls back cleanly to ``256``.

    Args:
        model_version (str): CLI model version (``"v1.0"`` or ``"v1.5"``).
        resolution (int | None): Explicit generation resolution (``256`` or
            ``512``), or ``None`` to auto-detect 512 weights.
        unet_model_path (str | None): Explicit UNet weights path override.
        unet_config (str | None): Explicit UNet config JSON path override.
        use_audio_pe (bool | None): Explicit audio PE flag, or ``None`` to pick
            the resolution default (``True`` for 256, ``False`` for 512).
        ref_frame_mode (str | None): Explicit UNet reference-frame mode, or
            ``None`` to pick the resolution default (``"self"`` for 256,
            ``"auto"`` for 512).
        batch_size (int | None): Explicit batch size, or ``None`` to pick the
            resolution default (``8`` for 256, ``2`` for 512).
        use_float16 (bool | None): Explicit FP16 toggle, or ``None`` to pick
            the resolution default (``False`` for 256, ``True`` for 512).
        path_exists (Callable[[str], bool] | None): Optional file existence
            predicate; defaults to ``os.path.isfile`` at call time.

    Returns:
        dict: Mapping with ``unet_model_path``, ``unet_config``, ``version_arg``,
            ``resolution``, ``use_audio_pe``, ``ref_frame_mode``, ``batch_size``,
            and ``use_float16``.

    Raises:
        ValueError: If ``resolution``, ``ref_frame_mode``, or ``batch_size`` is
            invalid, or ``v1.0`` is paired with ``resolution=512`` without
            explicit UNet path overrides.
    """
    exists_fn = os.path.isfile if path_exists is None else path_exists
    if resolution is None:
        has_512_weights = (
            model_version == "v1.5"
            and not unet_model_path
            and not unet_config
            and bool(exists_fn(DEFAULT_512_UNET_PATH))
            and bool(exists_fn(DEFAULT_512_CONFIG_PATH))
        )
        resolved_resolution = 512 if has_512_weights else 256
    else:
        resolved_resolution = validate_resolution(resolution)

    if model_version == "v1.0":
        if resolved_resolution != 256 and not (unet_model_path and unet_config):
            raise ValueError(
                "model_version 'v1.0' only ships with 256px weights; "
                "use --model_version v1.5 or pass explicit --unet_model_path and --unet_config."
            )
        default_unet = "./models/musetalk/pytorch_model.bin"
        default_cfg = "./models/musetalk/musetalk.json"
        version_arg = "v1"
    elif resolved_resolution == 512:
        default_unet = DEFAULT_512_UNET_PATH
        default_cfg = DEFAULT_512_CONFIG_PATH
        version_arg = "v15"
    else:
        default_unet = "./models/musetalkV15/unet.pth"
        default_cfg = "./models/musetalkV15/musetalk.json"
        version_arg = "v15"

    resolved_audio_pe = (resolved_resolution == 256) if use_audio_pe is None else bool(use_audio_pe)
    resolved_ref_mode = (
        ("auto" if resolved_resolution == 512 else "self")
        if ref_frame_mode is None
        else validate_ref_frame_mode(ref_frame_mode)
    )
    if batch_size is None:
        resolved_batch_size = 2 if resolved_resolution == 512 else 8
    else:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        resolved_batch_size = batch_size
    resolved_float16 = (resolved_resolution == 512) if use_float16 is None else bool(use_float16)

    return {
        "unet_model_path": unet_model_path or default_unet,
        "unet_config": unet_config or default_cfg,
        "version_arg": version_arg,
        "resolution": resolved_resolution,
        "use_audio_pe": resolved_audio_pe,
        "ref_frame_mode": resolved_ref_mode,
        "batch_size": resolved_batch_size,
        "use_float16": resolved_float16,
    }

def validate_inputs(face_path, audio_path):
    # Check if face_path exists and is non-empty
    if not os.path.exists(face_path):
        print(f"❌ 错误: 输入的 Face 文件/目录不存在: {face_path}")
        return False
    if not os.path.isdir(face_path) and os.path.getsize(face_path) == 0:
        print(f"❌ 错误: 输入的 Face 文件为空 (0 字节): {face_path}")
        return False
        
    # Check if audio_path exists and is non-empty
    if not os.path.exists(audio_path):
        print(f"❌ 错误: 输入的 Audio 文件不存在: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        print(f"❌ 错误: 输入的 Audio 文件为空 (0 字节): {audio_path}")
        return False

    # Check face path validity (video or image)
    _, ext = os.path.splitext(face_path)
    ext_lower = ext.lower()
    if ext_lower in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        # Try to open with OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(face_path)
            if not cap.isOpened():
                print(f"❌ 错误: 无法打开视频文件 {face_path}，文件可能损坏或写入不完整（如 moov atom 缺失）。")
                cap.release()
                return False
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps <= 0:
                print(f"❌ 错误: 视频文件 {face_path} 的帧率无效（{fps}），文件可能损坏或写入不完整。")
                return False
        except ImportError:
            # If cv2 is not installed (though it should be), we fall back to a basic check
            pass
    elif ext_lower in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        try:
            import cv2
            img = cv2.imread(face_path)
            if img is None or img.size == 0:
                print(f"❌ 错误: 无法读取图像文件 {face_path}，文件可能损坏或不合法。")
                return False
        except ImportError:
            pass
    elif os.path.isdir(face_path):
        # Check if directory contains images
        import glob
        img_list = glob.glob(os.path.join(face_path, '*.[jpJP][pnPN]*[gG]'))
        if not img_list:
            print(f"❌ 错误: 输入目录中未找到任何图片: {face_path}")
            return False
    else:
        print(f"❌ 错误: 不支持的 Face 输入类型或后缀 {face_path}。")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="MuseTalk CLI Wrapper")
    parser.add_argument("--face", type=str, required=True, help="Input face video path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path")
    parser.add_argument("--outfile", type=str, required=True, help="Output video path (must be a specific .mp4 file)")
    parser.add_argument("--model_version", type=str, default="v1.5", choices=["v1.0", "v1.5"])
    parser.add_argument("--resolution", type=int, default=None, choices=list(SUPPORTED_RESOLUTIONS),
                        help="UNet/VAE generation resolution in pixels (default: auto 512 when models/musetalkV15_512 exists, else 256)")
    parser.add_argument("--unet_model_path", type=str, default=None,
                        help="Optional override path for UNet weights")
    parser.add_argument("--unet_config", type=str, default=None,
                        help="Optional override path for UNet config JSON")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box vertical shift in pixels")
    parser.add_argument("--extra_margin", type=int, default=8, help="Extra margin for face cropping")
    parser.add_argument("--bbox_left_ratio", type=float, default=0.0, help="Left bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_right_ratio", type=float, default=0.0, help="Right bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_top_ratio", type=float, default=0.0, help="Top bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--bbox_bottom_ratio", type=float, default=0.0, help="Bottom bbox adjust ratio; positive expands, negative shrinks")
    parser.add_argument("--parsing_mode", type=str, default="jaw", choices=["jaw", "raw"], help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=120, help="Width of left cheek editing region")
    parser.add_argument("--right_cheek_width", type=int, default=120, help="Width of right cheek editing region")
    parser.add_argument("--side_protect_ratio", type=float, default=0.08, help="Side-edge blend protection ratio")
    parser.add_argument("--bbox_smooth_window", type=int, default=7, help="Centered moving-average window for bbox smoothing; 1 disables smoothing")
    parser.add_argument("--close_mouth_start", type=parse_bool, default=True, help="Close mouth during leading silence (true/false)")
    parser.add_argument("--close_mouth_end", type=parse_bool, default=False, help="Close mouth during trailing silence (true/false)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for inference (default: 8 for 256, 2 for 512)")
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--use_float16", action="store_true", dest="use_float16", default=None,
                            help="Use float16 for lower GPU memory usage (default: enabled for 512, disabled for 256)")
    fp16_group.add_argument("--no_float16", action="store_false", dest="use_float16",
                            help="Force float32 inference even when resolution is 512")
    # Clarity options. The first four defaults were adopted after 4K A/B runs on
    # an RTX 5070; pass the legacy values to reproduce the previous output.
    parser.add_argument("--vae_latent_mode", type=str, default="mode", choices=["sample", "mode"],
                        help="VAE encoding: 'mode' (deterministic, matches training) or 'sample' (legacy)")
    parser.add_argument("--paste_interpolation", type=str, default="lanczos", choices=["linear", "lanczos"],
                        help="Resize when pasting the generated face back: 'lanczos' or 'linear' (legacy)")
    parser.add_argument("--enhance_mode", type=str, default="usm", choices=["none", "usm"],
                        help="Lip clarity enhancement: 'usm' or 'none' (legacy)")
    parser.add_argument("--usm_amount", type=float, default=-1.0,
                        help="Unsharp-mask amount; negative selects it automatically")
    parser.add_argument("--fix_teeth_whiteness", type=parse_bool, default=False,
                        help="Tone down over-bright teeth inside the mouth (true/false)")
    parser.add_argument("--protect_upper_face", type=parse_bool, default=True,
                        help="Keep exact source pixels above the blend boundary (true/false; legacy false)")
    parser.add_argument("--close_mouth_pauses", type=parse_bool, default=False,
                        help="Blend to the neutral mouth during mid-sentence pauses (true/false)")
    parser.add_argument("--pause_min_frames", type=int, default=8, help="Shortest pause, in frames, that closes the mouth")
    parser.add_argument("--pause_fade_frames", type=int, default=4, help="Frames used to fade into and out of a pause")
    parser.add_argument("--pause_top_db", type=float, default=38.0, help="Silence threshold below the audio peak, in dB")
    parser.add_argument("--use_audio_pe", type=parse_bool, default=None,
                        help="Add positional encoding to Whisper features (default: true for 256, false for 512)")
    parser.add_argument("--ref_frame_mode", type=str, default=None, choices=list(REF_FRAME_MODES),
                        help="UNet reference latent source: 'self', 'first', 'offset', or 'auto' (default: self for 256, auto for 512)")
    
    args = parser.parse_args()

    # 将输入的路径转化为绝对路径，确保 MuseTalk 底层能够正确识别
    face_path = os.path.abspath(args.face)
    audio_path = os.path.abspath(args.audio)
    outfile_path = os.path.abspath(args.outfile)

    # 验证输入文件的有效性
    if not validate_inputs(face_path, audio_path):
        sys.exit(1)

    # MuseTalk 默认是通过 yaml 配置文件来执行批量任务的，为了支持单次 CLI 调用，这里动态生成一个临时的 yaml
    config_dict = {
        "task_0": {
            "video_path": face_path,
            "audio_path": audio_path
        }
    }
    
    # 每次调用使用唯一的临时 yaml，避免同一目录下并发任务互相覆盖输入路径
    temp_yaml_dir = os.path.abspath("configs/inference")
    os.makedirs(temp_yaml_dir, exist_ok=True)
    yaml_fd, temp_yaml = tempfile.mkstemp(prefix="temp_cli_task_", suffix=".yaml", dir=temp_yaml_dir)
    with os.fdopen(yaml_fd, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
        
    # 提取输出目录
    out_dir = os.path.dirname(outfile_path)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve UNet weights, config, resolution, and audio PE settings
    try:
        model_settings = resolve_model_settings(
            args.model_version,
            args.resolution,
            unet_model_path=args.unet_model_path,
            unet_config=args.unet_config,
            use_audio_pe=args.use_audio_pe,
            ref_frame_mode=args.ref_frame_mode,
            batch_size=args.batch_size,
            use_float16=args.use_float16,
        )
    except ValueError as err:
        print(f"❌ 错误: {err}")
        sys.exit(1)

    # 组装最终的底层调用命令
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", temp_yaml,
        "--result_dir", out_dir,
        "--output_vid_name", outfile_path,  # 传入绝对路径，底层代码 os.path.join 遇到绝对路径会自动使用绝对路径
        "--unet_model_path", model_settings["unet_model_path"],
        "--unet_config", model_settings["unet_config"],
        "--version", model_settings["version_arg"],
        "--resolution", str(model_settings["resolution"]),
        "--bbox_shift", str(args.bbox_shift),
        "--extra_margin", str(args.extra_margin),
        "--bbox_left_ratio", str(args.bbox_left_ratio),
        "--bbox_right_ratio", str(args.bbox_right_ratio),
        "--bbox_top_ratio", str(args.bbox_top_ratio),
        "--bbox_bottom_ratio", str(args.bbox_bottom_ratio),
        "--parsing_mode", args.parsing_mode,
        "--left_cheek_width", str(args.left_cheek_width),
        "--right_cheek_width", str(args.right_cheek_width),
        "--side_protect_ratio", str(args.side_protect_ratio),
        "--bbox_smooth_window", str(args.bbox_smooth_window),
        "--close_mouth_start", str(args.close_mouth_start).lower(),
        "--close_mouth_end", str(args.close_mouth_end).lower(),
        "--batch_size", str(model_settings["batch_size"]),
        "--vae_latent_mode", args.vae_latent_mode,
        "--paste_interpolation", args.paste_interpolation,
        "--enhance_mode", args.enhance_mode,
        "--usm_amount", str(args.usm_amount),
        "--fix_teeth_whiteness", str(args.fix_teeth_whiteness).lower(),
        "--protect_upper_face", str(args.protect_upper_face).lower(),
        "--close_mouth_pauses", str(args.close_mouth_pauses).lower(),
        "--pause_min_frames", str(args.pause_min_frames),
        "--pause_fade_frames", str(args.pause_fade_frames),
        "--pause_top_db", str(args.pause_top_db),
        "--use_audio_pe", str(model_settings["use_audio_pe"]).lower(),
        "--ref_frame_mode", model_settings["ref_frame_mode"],
    ]
    if model_settings["use_float16"]:
        cmd.append("--use_float16")
    
    print(f"🚀 开始执行数字人合成任务...\n底层调用命令: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 任务圆满完成！\n视频已保存至: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 推理过程中发生错误。底层推理命令执行失败，请检查上方日志。")
        sys.exit(1)
    finally:
        # 清理临时的 yaml 配置文件
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

if __name__ == "__main__":
    main()
