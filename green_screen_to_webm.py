#!/usr/bin/env python3
"""
Green screen to transparent WebM converter.

Converts a green-screen MP4 video to a transparent-background WebM video
using FFmpeg's colorkey filter and VP9 codec with alpha channel support.

Usage:
    python green_screen_to_webm.py --input input.mp4 --output output.webm
    python green_screen_to_webm.py --input input.mp4  # outputs input.webm
"""

import argparse
import os
import shutil
import subprocess
import sys


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def get_video_info(video_path: str) -> dict:
    """Extract basic video info using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,duration",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(",")
        if len(parts) >= 4:
            return {
                "width": int(parts[0]),
                "height": int(parts[1]),
                "fps": parts[2],
                "duration": float(parts[3]),
            }
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return {}


def convert(
    input_path: str,
    output_path: str,
    key_color: str = "0x00FF00",
    similarity: float = 0.15,
    blend: float = 0.1,
    bitrate: str = "3M",
    use_colorkey: bool = True,
    no_audio: bool = False,
    despill: bool = True,
    despill_factor: float = 0.5,
    crop_pixels: int = 0,
    alpha_erode: int = 0,
) -> bool:
    """
    Convert green screen MP4 to transparent WebM.

    Args:
        input_path: Input MP4 file path.
        output_path: Output WebM file path.
        key_color: Chroma key color in hex (default: 0x00FF00 pure green).
        similarity: Similarity threshold (0~1). Lower = stricter keying.
        blend: Blend softness (0~1). Higher = smoother edges.
        bitrate: Video bitrate (e.g. "2M", "5M").
        use_colorkey: Use colorkey filter (better edges) vs chromakey.
        no_audio: Strip audio from output.
        despill: Apply despill filter to remove green spill.
        despill_factor: Despill strength (0~1).
        crop_pixels: Crop N pixels from all edges before keying.
        alpha_erode: Shrink alpha channel edges by N pixels to remove white borders.

    Returns:
        True on success, False on failure.
    """
    filter_chain = []

    # Crop edges first to remove compression artifacts
    if crop_pixels > 0:
        filter_chain.append(f"crop=iw-{crop_pixels*2}:ih-{crop_pixels*2}:{crop_pixels}:{crop_pixels}")

    # Apply chroma key
    filter_name = "colorkey" if use_colorkey else "chromakey"
    filter_chain.append(f"{filter_name}={key_color}:{similarity}:{blend}")

    # Apply despill to remove green spill artifacts
    if despill:
        filter_chain.append(f"despill=type=green:mix={despill_factor}")

    if alpha_erode > 0:
        # Need -filter_complex: split into RGB + alpha paths, shrink alpha edges, merge back.
        # Strategy: extract alpha → scale down → pad back to original size with black fill.
        # This shrinks the opaque region by exactly `alpha_erode` pixels from all edges.
        pre_filters = ",".join(filter_chain)
        shrink = alpha_erode * 2
        # scale: (iw-2N, ih-2N), then pad back: (iw+2N, ih+2N) with offset (N, N)
        shrink_filter = (
            f"scale=iw-{shrink}:ih-{shrink},"
            f"pad=iw+{shrink}:ih+{shrink}:{alpha_erode}:{alpha_erode}:black"
        )
        filter_complex = (
            f"[0:v]{pre_filters}[v];"
            f"[v]split[rgb][a];"
            f"[a]alphaextract,{shrink_filter}[ea];"
            f"[rgb][ea]alphamerge,format=rgba[out]"
        )
        map_label = "[out]"
    else:
        # Simple linear chain is fine
        filter_chain.append("format=rgba")
        filter_complex = None
        map_label = None

    cmd = ["ffmpeg", "-y", "-v", "warning", "-i", input_path]

    if filter_complex is not None:
        cmd += ["-filter_complex", filter_complex, "-map", map_label]
    else:
        vf = ",".join(filter_chain)
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libvpx-vp9",
        "-b:v", bitrate,
        "-pix_fmt", "yuva420p",
        "-r", "30",
    ]

    if no_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "libopus"]

    cmd.append(output_path)

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def verify_output(output_path: str) -> bool:
    """Verify output has alpha channel."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=pix_fmt,codec_name",
        "-of", "csv=p=0",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            codec, pix_fmt = parts[0], parts[1]
            has_alpha = "yuva" in pix_fmt
            print(f"Codec: {codec}, Pixel format: {pix_fmt}, Alpha: {'YES' if has_alpha else 'NO'}")
            return has_alpha
    except subprocess.CalledProcessError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert green screen MP4 to transparent WebM video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python green_screen_to_webm.py -i input.mp4
  python green_screen_to_webm.py -i input.mp4 -o result.webm
  python green_screen_to_webm.py -i input.mp4 --similarity 0.08 --blend 0.2
  python green_screen_to_webm.py -i input.mp4 --crop 4  # remove edge artifacts
  python green_screen_to_webm.py -i input.mp4 --alpha-erode 3  # shrink white borders
  python green_screen_to_webm.py -i input.mp4 --crop 4 --alpha-erode 3 --despill-factor 0.8
  python green_screen_to_webm.py -i input.mp4 --chromakey --bitrate 5M
""",
    )
    parser.add_argument("-i", "--input", required=True, help="Input MP4 file path")
    parser.add_argument("-o", "--output", default=None, help="Output WebM file path (default: same name with .webm extension)")
    parser.add_argument("--key-color", default="0x00FF00", help="Chroma key color hex (default: 0x00FF00)")
    parser.add_argument("--similarity", type=float, default=0.15, help="Similarity threshold 0~1 (default: 0.15, lower = stricter)")
    parser.add_argument("--blend", type=float, default=0.1, help="Blend softness 0~1 (default: 0.1, higher = smoother edges)")
    parser.add_argument("--bitrate", default="3M", help="Video bitrate (default: 3M)")
    parser.add_argument("--chromakey", action="store_true", help="Use chromakey filter instead of colorkey")
    parser.add_argument("--no-audio", action="store_true", help="Remove audio from output")
    parser.add_argument("--no-despill", action="store_true", help="Disable despill filter")
    parser.add_argument("--despill-factor", type=float, default=0.5, help="Despill strength 0~1 (default: 0.5)")
    parser.add_argument("--crop", type=int, default=0, help="Crop N pixels from all edges to remove compression artifacts (default: 0)")
    parser.add_argument("--alpha-erode", type=int, default=0, help="Shrink alpha channel edges by N pixels to remove white borders (default: 0)")

    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not check_ffmpeg():
        print("Error: FFmpeg not found in PATH. Please install FFmpeg first.", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + ".webm"

    # Show input info
    info = get_video_info(args.input)
    if info:
        print(f"Input: {args.input}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}, Duration: {info['duration']:.2f}s")
        print()

    print(f"Output: {args.output}")
    print(f"Key color: {args.key_color}, Similarity: {args.similarity}, Blend: {args.blend}")
    if args.crop > 0:
        print(f"Crop: {args.crop}px from all edges")
    if args.alpha_erode > 0:
        print(f"Alpha erode: {args.alpha_erode}px (shrink transparent edges)")
    if not args.no_despill:
        print(f"Despill: enabled (factor={args.despill_factor})")
    print()

    # Convert
    print("Converting...")
    success = convert(
        input_path=args.input,
        output_path=args.output,
        key_color=args.key_color,
        similarity=args.similarity,
        blend=args.blend,
        bitrate=args.bitrate,
        use_colorkey=not args.chromakey,
        no_audio=args.no_audio,
        despill=not args.no_despill,
        despill_factor=args.despill_factor,
        crop_pixels=args.crop,
        alpha_erode=args.alpha_erode,
    )

    if not success:
        print("\nConversion FAILED.", file=sys.stderr)
        sys.exit(1)

    # Verify
    print("\nVerifying output...")
    if verify_output(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nDone! Output saved to: {args.output} ({size_mb:.2f} MB)")
    else:
        print("\nWarning: Output may not have transparent channel. Check the file manually.", file=sys.stderr)


if __name__ == "__main__":
    main()
