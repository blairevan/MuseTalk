import argparse
import hashlib
import json
import os
import stat
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
DEFAULT_MAX_INPUT_LONG_EDGE = 1920
_VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mov", ".flv", ".mkv"})


def _normalize_even_long_edge(max_long_edge):
    """Validate and floor ``max_long_edge >= 2`` down to an even pixel cap."""
    if isinstance(max_long_edge, bool) or int(max_long_edge) < 2:
        raise ValueError(
            f"max_input_long_edge must be 0 (disabled) or >= 2, got {max_long_edge!r}"
        )
    return (int(max_long_edge) // 2) * 2


def compute_max_short_edge(max_long_edge):
    """Compute even 16:9 short-edge cap corresponding to ``max_long_edge``."""
    even_long = _normalize_even_long_edge(max_long_edge)
    return max(2, int(round(even_long * 9.0 / 16.0 / 2.0)) * 2)


def needs_input_video_resize(width, height, max_long_edge=DEFAULT_MAX_INPUT_LONG_EDGE):
    """Return whether a video frame size exceeds the cap or has odd dimensions."""
    if max_long_edge is None:
        return False
    if isinstance(max_long_edge, bool) or int(max_long_edge) < 0 or int(max_long_edge) == 1:
        raise ValueError(
            f"max_input_long_edge must be 0 (disabled) or >= 2, got {max_long_edge!r}"
        )
    if int(max_long_edge) == 0:
        return False
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return False
    even_long = _normalize_even_long_edge(max_long_edge)
    long_edge = max(w, h)
    short_edge = min(w, h)
    max_short = compute_max_short_edge(even_long)
    return (
        long_edge > even_long
        or short_edge > max_short
        or (w % 2 != 0)
        or (h % 2 != 0)
    )


def build_normalized_cache_path(face_path, max_long_edge=DEFAULT_MAX_INPUT_LONG_EDGE):
    """Build a collision-free sibling cache path guaranteed never to equal ``face_path``.

    Standard lowercase ``.mp4`` inputs whose stem does not end in ``_{tag}`` or
    contain ``.musetalk_`` map to ``{stem}_{tag}.mp4`` (e.g.
    ``digital_human_video_4136480240_1080p.mp4``). All other persistent inputs
    (non-MP4 extensions such as ``clip.mov``, uppercase extensions, or stems
    already ending in ``_{tag}``) map to ``{base_name}.musetalk_{hash8}_{tag}.mp4``
    so two distinct filenames in the same directory can never collide. Hidden
    transient inputs (e.g. ``.video-timeline-*.mp4``) return ``None``.
    """
    parent_dir = os.path.dirname(face_path)
    base_name = os.path.basename(face_path)
    if base_name.startswith("."):
        return None
    stem, ext = os.path.splitext(base_name)
    even_long = _normalize_even_long_edge(max_long_edge)
    tag = "1080p" if even_long == DEFAULT_MAX_INPUT_LONG_EDGE else f"max{even_long}"
    suffix = f"_{tag}"
    if ext == ".mp4" and not stem.endswith(suffix) and ".musetalk_" not in stem:
        candidate = os.path.join(parent_dir, f"{stem}{suffix}.mp4")
    else:
        digest = hashlib.sha256(base_name.encode("utf-8")).hexdigest()[:8]
        candidate = os.path.join(parent_dir, f"{base_name}.musetalk_{digest}{suffix}.mp4")
    if os.path.abspath(candidate) == os.path.abspath(face_path):
        return None
    return candidate


def _cache_sidecar_path(cache_path):
    """Return hidden JSON metadata sidecar path paired with ``cache_path``."""
    return os.path.join(
        os.path.dirname(cache_path),
        f".{os.path.basename(cache_path)}.meta.json",
    )


def _cache_lock_path(cache_path):
    """Return exclusive lockfile path paired with ``cache_path``."""
    return os.path.join(
        os.path.dirname(cache_path),
        f".{os.path.basename(cache_path)}.lock",
    )


def _file_sha256_nofollow(path):
    """Compute SHA-256 of a regular non-symlink file, or ``None`` if unsafe."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        st_link = os.lstat(path)
        if not stat.S_ISREG(st_link.st_mode):
            return None
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st_fd = os.fstat(fd)
        if not stat.S_ISREG(st_fd.st_mode) or st_fd.st_ino != st_link.st_ino:
            return None
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            hasher.update(chunk)
        return hasher.hexdigest(), int(st_fd.st_size)
    except OSError:
        return None
    finally:
        os.close(fd)


def _source_identity_record(face_path, even_long):
    """Build deterministic source identity metadata for cache validation."""
    st = os.stat(face_path)
    return {
        "source_realpath": os.path.realpath(face_path),
        "source_size": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
        "max_long_edge": int(even_long),
    }


def _read_cache_sidecar_nofollow(cache_path):
    """Load cache sidecar JSON without following symlinks; otherwise ``None``."""
    meta_path = _cache_sidecar_path(cache_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        st_link = os.lstat(meta_path)
        if not stat.S_ISREG(st_link.st_mode):
            return None
        fd = os.open(meta_path, flags)
    except OSError:
        return None
    try:
        raw = os.read(fd, 65536)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    finally:
        os.close(fd)


def _verify_owned_valid_cache(cache_path, expected_source_meta, even_long):
    """Verify ``cache_path`` is owned by ``expected_source_meta`` and intact."""
    if (
        os.path.lexists(_cache_lock_path(cache_path))
        or os.path.islink(cache_path)
        or os.path.islink(_cache_sidecar_path(cache_path))
    ):
        return False
    meta = _read_cache_sidecar_nofollow(cache_path)
    if meta is None:
        return False
    for key, val in expected_source_meta.items():
        if meta.get(key) != val:
            return False
    digest_info = _file_sha256_nofollow(cache_path)
    if digest_info is None:
        return False
    sha256_hex, byte_size = digest_info
    if byte_size <= 0 or byte_size != meta.get("cache_size") or sha256_hex != meta.get("cache_sha256"):
        return False
    cached_probe = _probe_video_size(cache_path)
    if cached_probe is None or needs_input_video_resize(cached_probe[0], cached_probe[1], even_long):
        return False
    return True


def _probe_video_size(video_path):
    """Return ``(width, height, fps)`` of ``video_path`` via OpenCV, or ``None``."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        return None
    return width, height, fps


def _write_all(fd, data):
    """Write every byte of ``data`` to ``fd``, handling partial ``os.write`` calls."""
    view = memoryview(data)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:
            raise OSError("os.write returned 0 bytes")
        total += written


def _copy_regular_file_nofollow(src_path, dst_dir, prefix="norm_snap_"):
    """Copy regular non-symlink ``src_path`` into a private new inode in ``dst_dir``.

    Using an independent inode via ``O_NOFOLLOW`` and full-write loops guarantees
    complete isolation across mount points, concurrent ``os.link``/``os.replace``
    operations, partial writes, and external in-place truncation of ``src_path``.
    """
    target_dir = dst_dir or "."
    os.makedirs(target_dir, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        st_src = os.lstat(src_path)
        if not stat.S_ISREG(st_src.st_mode):
            return None
        in_fd = os.open(src_path, flags)
    except OSError:
        return None

    snap_path = None
    try:
        st_in = os.fstat(in_fd)
        if not stat.S_ISREG(st_in.st_mode) or st_in.st_ino != st_src.st_ino:
            return None
        out_fd, snap_path = tempfile.mkstemp(prefix=prefix, suffix=".mp4", dir=target_dir)
        try:
            while True:
                chunk = os.read(in_fd, 65536)
                if not chunk:
                    break
                _write_all(out_fd, chunk)
        finally:
            os.close(out_fd)
        return snap_path
    except OSError:
        if snap_path and os.path.lexists(snap_path):
            try:
                os.unlink(snap_path)
            except OSError:
                pass
        return None
    finally:
        os.close(in_fd)


def _publish_cache_write_once(src_path, cache_path, expected_source_meta):
    """Publish ``src_path`` to ``cache_path`` strictly once via non-overwriting ``os.link``.

    Never calls ``os.replace`` or ``os.unlink`` on ``cache_path`` or ``meta_path``,
    so an external process creating or modifying files at any instant can never
    have its file overwritten or deleted.
    """
    if os.path.lexists(cache_path) or os.path.lexists(_cache_sidecar_path(cache_path)):
        return False

    cache_dir = os.path.dirname(cache_path) or "."
    meta_path = _cache_sidecar_path(cache_path)
    lock_path = _cache_lock_path(cache_path)
    lock_fd = None
    stage_vid = None
    meta_tmp = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return False

    try:
        if os.path.lexists(cache_path) or os.path.lexists(meta_path):
            return False
        src_digest = _file_sha256_nofollow(src_path)
        if src_digest is None or src_digest[1] <= 0:
            return False
        sha256_hex, byte_size = src_digest

        stage_vid = _copy_regular_file_nofollow(src_path, cache_dir, prefix=".norm_vid_stage_")
        if stage_vid is None or _file_sha256_nofollow(stage_vid) != src_digest:
            return False

        full_meta = dict(expected_source_meta)
        full_meta["cache_sha256"] = sha256_hex
        full_meta["cache_size"] = byte_size
        meta_bytes = json.dumps(full_meta).encode("utf-8")

        meta_fd, meta_tmp = tempfile.mkstemp(prefix=".norm_meta_tmp_", suffix=".json", dir=cache_dir)
        try:
            _write_all(meta_fd, meta_bytes)
        finally:
            os.close(meta_fd)

        os.link(stage_vid, cache_path, follow_symlinks=False)
        os.link(meta_tmp, meta_path, follow_symlinks=False)
        return True
    except OSError:
        return False
    finally:
        for temp_item in (stage_vid, meta_tmp):
            if temp_item and os.path.lexists(temp_item):
                try:
                    os.unlink(temp_item)
                except OSError:
                    pass
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def _pin_verified_cache_snapshot(cache_path, out_dir, expected_source_meta, even_long):
    """Pin a private snapshot of ``cache_path`` and verify its digest AFTER pinning."""
    if (
        os.path.lexists(_cache_lock_path(cache_path))
        or os.path.islink(cache_path)
        or os.path.islink(_cache_sidecar_path(cache_path))
    ):
        return None
    meta = _read_cache_sidecar_nofollow(cache_path)
    if meta is None:
        return None
    for key, val in expected_source_meta.items():
        if meta.get(key) != val:
            return None

    snap_path = _copy_regular_file_nofollow(cache_path, out_dir)
    if snap_path is None:
        return None
    digest_info = _file_sha256_nofollow(snap_path)
    cached_probe = _probe_video_size(snap_path) if digest_info is not None else None
    if (
        digest_info is None
        or digest_info[1] <= 0
        or digest_info != (meta.get("cache_sha256"), meta.get("cache_size"))
        or cached_probe is None
        or needs_input_video_resize(cached_probe[0], cached_probe[1], even_long)
    ):
        try:
            os.unlink(snap_path)
        except OSError:
            pass
        return None
    return snap_path


def prepare_normalized_face_input(
    face_path,
    out_dir,
    max_long_edge=DEFAULT_MAX_INPUT_LONG_EDGE,
    pin_run_snapshot=False,
):
    """Downscale oversized video inputs to <=1080p while preserving source FPS.

    Uses write-once ``os.link`` cache publication (never overwriting or deleting
    existing files at ``cache_path``/``meta_path``) and SHA-256 verification.
    When ``pin_run_snapshot=True``, unconditionally returns a private verified
    inode in ``out_dir`` so external modifications can never alter in-flight input.

    Args:
        face_path (str): Absolute path to the validated input video/image/dir.
        out_dir (str): Output directory used when a transient temp file is needed.
        max_long_edge (int): Maximum long-edge pixels (``0`` disables scaling;
            otherwise must be ``>= 2``).
        pin_run_snapshot (bool): When ``True``, always return a per-run private
            verified file path in ``out_dir`` for normalized videos.

    Returns:
        tuple[str, str | None]: ``(effective_face_path, temp_cleanup_path)``.
    """
    if isinstance(max_long_edge, bool) or int(max_long_edge) < 0 or int(max_long_edge) == 1:
        raise ValueError(
            f"max_input_long_edge must be 0 (disabled) or >= 2, got {max_long_edge!r}"
        )
    raw_max_long = int(max_long_edge)
    if raw_max_long == 0 or os.path.isdir(face_path):
        return face_path, None

    _, ext = os.path.splitext(face_path)
    if ext.lower() not in _VIDEO_EXTENSIONS:
        return face_path, None

    even_long = _normalize_even_long_edge(raw_max_long)
    probed = _probe_video_size(face_path)
    if probed is None:
        return face_path, None
    width, height, _ = probed
    if not needs_input_video_resize(width, height, even_long):
        return face_path, None

    eff_out_dir = out_dir or "."
    expected_meta = _source_identity_record(face_path, even_long)
    cache_path = build_normalized_cache_path(face_path, even_long)
    if cache_path is not None:
        meta_path = _cache_sidecar_path(cache_path)
        if os.path.lexists(cache_path):
            if pin_run_snapshot:
                pinned = _pin_verified_cache_snapshot(cache_path, eff_out_dir, expected_meta, even_long)
                if pinned is not None:
                    print(f"♻️ 复用已规范化的 1080p 缓存视频: {cache_path}")
                    return pinned, pinned
            elif _verify_owned_valid_cache(cache_path, expected_meta, even_long):
                print(f"♻️ 复用已规范化的 1080p 缓存视频: {cache_path}")
                return cache_path, None
            cache_path = None
        elif os.path.lexists(meta_path):
            cache_path = None

    max_short = compute_max_short_edge(even_long)
    vf = (
        f"scale='if(gt(iw,ih),min({even_long},iw),min({max_short},iw))':"
        f"'if(gt(iw,ih),min({max_short},ih),min({even_long},ih))':"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        "scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)':flags=lanczos"
    )

    os.makedirs(eff_out_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="norm1080p_tmp_", suffix=".mp4", dir=eff_out_dir)
    os.close(tmp_fd)

    ffmpeg_cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "warning",
        "-i", face_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        tmp_path,
    ]
    print(f"📐 检测到超清输入视频 ({width}x{height})，正在规范化至 <= {even_long}x{max_short}...")
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        tmp_digest = _file_sha256_nofollow(tmp_path)
        if tmp_digest is None or tmp_digest[1] <= 0:
            raise RuntimeError(f"Normalized video output is empty or invalid: {tmp_path}")

        published = False
        if cache_path is not None:
            published = _publish_cache_write_once(tmp_path, cache_path, expected_meta)
            if published:
                print(f"✅ 已生成规范化缓存视频: {cache_path}")

        if pin_run_snapshot:
            if _file_sha256_nofollow(tmp_path) != tmp_digest:
                raise RuntimeError(f"Transient normalized snapshot changed unexpectedly: {tmp_path}")
            return tmp_path, tmp_path

        if published and _verify_owned_valid_cache(cache_path, expected_meta, even_long):
            os.remove(tmp_path)
            return cache_path, None
        return tmp_path, tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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

    When ``resolution`` is ``None``, defaults to the production-hardened ``256``
    official checkpoint; pass ``resolution=512`` explicitly to route to
    :data:`DEFAULT_512_UNET_PATH` and :data:`DEFAULT_512_CONFIG_PATH`.

    Args:
        model_version (str): CLI model version (``"v1.0"`` or ``"v1.5"``).
        resolution (int | None): Explicit generation resolution (``256`` or
            ``512``), or ``None`` for the default ``256`` pipeline.
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
        path_exists (Callable[[str], bool] | None): Unused legacy hook retained
            for caller compatibility.

    Returns:
        dict: Mapping with ``unet_model_path``, ``unet_config``, ``version_arg``,
            ``resolution``, ``use_audio_pe``, ``ref_frame_mode``, ``batch_size``,
            and ``use_float16``.

    Raises:
        ValueError: If ``resolution``, ``ref_frame_mode``, or ``batch_size`` is
            invalid, or ``v1.0`` is paired with ``resolution=512`` without
            explicit UNet path overrides.
    """
    _ = path_exists
    resolved_resolution = 256 if resolution is None else validate_resolution(resolution)

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
                        help="UNet/VAE generation resolution in pixels (default: 256; pass 512 for musetalkV15_512)")
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
    parser.add_argument("--max_input_long_edge", type=int, default=DEFAULT_MAX_INPUT_LONG_EDGE,
                        help="Downscale input videos exceeding this long-edge size before inference; 0 disables (default: 1920)")
    
    args = parser.parse_args()

    # 将输入的路径转化为绝对路径，确保 MuseTalk 底层能够正确识别
    face_path = os.path.abspath(args.face)
    audio_path = os.path.abspath(args.audio)
    outfile_path = os.path.abspath(args.outfile)

    # 验证输入文件的有效性
    if not validate_inputs(face_path, audio_path):
        sys.exit(1)

    # 先完成模型与超参数校验，避免无效参数在视频规范化之后才退出遗留临时文件
    try:
        if args.max_input_long_edge < 0 or args.max_input_long_edge == 1:
            raise ValueError(
                f"max_input_long_edge must be 0 (disabled) or >= 2, got {args.max_input_long_edge!r}"
            )
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

    # 提取输出目录
    out_dir = os.path.dirname(outfile_path)
    os.makedirs(out_dir, exist_ok=True)

    norm_cleanup_path = None
    temp_yaml = None
    try:
        # 超清视频自动规范化至 <= 1080p（持久底板生成同目录缓存，并为当次运行固定不可变快照防并发刷新串流）
        try:
            effective_face_path, norm_cleanup_path = prepare_normalized_face_input(
                face_path,
                out_dir,
                max_long_edge=args.max_input_long_edge,
                pin_run_snapshot=True,
            )
        except (ValueError, subprocess.CalledProcessError) as err:
            print(f"❌ 错误: 输入视频分辨率规范化失败: {err}")
            sys.exit(1)

        # MuseTalk 默认是通过 yaml 配置文件来执行批量任务的，为了支持单次 CLI 调用，这里动态生成一个临时的 yaml
        config_dict = {
            "task_0": {
                "video_path": effective_face_path,
                "audio_path": audio_path
            }
        }

        # 每次调用使用唯一的临时 yaml，避免同一目录下并发任务互相覆盖输入路径
        temp_yaml_dir = os.path.abspath("configs/inference")
        os.makedirs(temp_yaml_dir, exist_ok=True)
        yaml_fd, temp_yaml = tempfile.mkstemp(prefix="temp_cli_task_", suffix=".yaml", dir=temp_yaml_dir)
        with os.fdopen(yaml_fd, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

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
        except subprocess.CalledProcessError:
            print(f"\n❌ 推理过程中发生错误。底层推理命令执行失败，请检查上方日志。")
            sys.exit(1)
    finally:
        # 清理临时的 yaml 配置文件与临时规范化视频（持久缓存文件保留）
        if temp_yaml and os.path.exists(temp_yaml):
            os.remove(temp_yaml)
        if norm_cleanup_path and os.path.exists(norm_cleanup_path):
            os.remove(norm_cleanup_path)

if __name__ == "__main__":
    main()
