import contextlib
import math
import os
import subprocess
import tempfile
import wave

import numpy as np


def parse_bool(value):
    """Parse a human-readable boolean value for command-line arguments."""
    normalized = str(value).strip().lower()
    true_values = {"1", "true", "yes", "on"}
    false_values = {"0", "false", "no", "off"}
    if normalized in true_values:
        return True
    if normalized in false_values:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def is_pcm16_mono_16k(path):
    """Return whether ``path`` is an uncompressed 16-bit mono 16 kHz WAV file."""
    try:
        with wave.open(path, "rb") as wav_file:
            return (
                wav_file.getframerate() == 16000
                and wav_file.getnchannels() == 1
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
            )
    except (wave.Error, EOFError, OSError):
        return False


def _run_ffmpeg_to_pcm16_mono_16k(input_path, output_path):
    """Convert ``input_path`` to 16 kHz mono PCM WAV at ``output_path``."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "-f", "wav", output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextlib.contextmanager
def converted_wav(input_path):
    """Yield a 16 kHz mono PCM WAV path for ``input_path`` and clean up afterwards.

    A conforming input is yielded unchanged. Anything else is converted into a
    private temporary file that only this call can see, which is deleted when
    the context exits. Nothing is written next to the caller's audio, so
    concurrent tasks and sources that share a base name cannot affect each other.

    Args:
        input_path (str): Audio file readable by ffmpeg.

    Yields:
        str: Path to read with librosa. A missing input is yielded as is so the
        caller reports the original error.

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not isinstance(input_path, str) or not os.path.exists(input_path):
        yield input_path
        return
    if is_pcm16_mono_16k(input_path):
        yield input_path
        return

    fd, temp_path = tempfile.mkstemp(prefix="musetalk_audio_", suffix=".wav")
    os.close(fd)
    try:
        _run_ffmpeg_to_pcm16_mono_16k(input_path, temp_path)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def ensure_wav(input_path: str, target_path: str | None = None) -> str:
    """
    Convert any audio (mp3/ogg/m4a/wav/…) to 16kHz mono PCM WAV via ffmpeg.
    Returns path to the converted .wav (original if already correct).

    An input that is already 16 kHz mono PCM is returned unchanged when no
    explicit target is requested. Otherwise the default target name keeps the
    source extension (``foo.mp3`` -> ``foo_mp3_16k.wav``) so sources that share
    a base name never overwrite each other. ffmpeg writes a unique temporary
    file that is atomically renamed to the target, so concurrent readers never
    see a partially written file. The conversion always runs; no cached target
    is reused.

    Raises:
        ValueError: If the target path would overwrite the input file.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not isinstance(input_path, str) or not os.path.exists(input_path):
        return input_path
    if target_path is None and is_pcm16_mono_16k(input_path):
        return input_path

    base, ext = os.path.splitext(input_path)
    if target_path is None:
        ext_tag = ext.lower().lstrip(".")
        target_path = f"{base}_{ext_tag}_16k.wav" if ext_tag else f"{base}_16k.wav"
    if os.path.abspath(target_path) == os.path.abspath(input_path):
        raise ValueError(f"Refusing to overwrite the input audio in place: {input_path}")

    target_dir = os.path.dirname(os.path.abspath(target_path))
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(target_path) + ".",
        suffix=".tmp.wav",
        dir=target_dir,
    )
    os.close(fd)
    try:
        _run_ffmpeg_to_pcm16_mono_16k(input_path, temp_path)
        os.replace(temp_path, target_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return target_path


def audio_samples_to_frame_range(start_sample, end_sample, sample_rate, fps):
    """Map an active audio sample range to an inclusive video-frame boundary."""
    active_start_frame = int(math.floor(start_sample / sample_rate * float(fps)))
    active_end_frame = int(math.ceil(end_sample / sample_rate * float(fps)))
    return max(0, active_start_frame), max(active_start_frame, active_end_frame)


def calculate_audio_target_frames(audio_samples, sample_rate, fps):
    """Calculate strictly aligned total video frames for the given audio duration."""
    if audio_samples <= 0 or sample_rate <= 0 or fps <= 0:
        return 0
    return int(round((audio_samples / float(sample_rate)) * float(fps)))


def build_audio_mux_command(audio_path, video_path, output_path):
    """Build the ffmpeg command that adds the driving audio to a rendered video.

    The video stream is copied instead of re-encoded, so the rendered frames
    keep the quality of the first (CRF 18) encode. Only the audio is encoded to
    AAC, which is what ffmpeg already chose by default for MP4 output.

    Args:
        audio_path (str): Driving audio file.
        video_path (str): Rendered, silent H.264 video.
        output_path (str): Destination file; must be a container that accepts
            H.264 (e.g. ``.mp4``).

    Returns:
        list[str]: Argument list for ``subprocess.run``.
    """
    return [
        "ffmpeg", "-nostdin", "-y", "-v", "warning",
        "-i", audio_path, "-i", video_path,
        "-map", "1:v:0", "-map", "0:a:0",
        "-c:v", "copy", "-c:a", "aac",
        output_path,
    ]


def get_active_audio_frame_range(audio_path, fps, top_db=40.0, silence_threshold=1e-4):
    """Return the video-frame range containing detected active audio."""
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    if audio.size == 0:
        return 0, 0
    if float(abs(audio).max()) <= silence_threshold:
        return 0, 0

    _, trim_indices = librosa.effects.trim(audio, top_db=top_db)
    active_start_sample = int(trim_indices[0])
    active_end_sample = int(trim_indices[1])
    return audio_samples_to_frame_range(
        active_start_sample,
        active_end_sample,
        sample_rate,
        fps,
    )


def get_active_audio_frame_intervals(audio_path, fps, top_db=38.0, silence_threshold=1e-4):
    """Return every voiced span of the audio as video-frame intervals.

    Args:
        audio_path (str): Audio file readable by librosa.
        fps (float): Video frame rate.
        top_db (float): Threshold below the peak that counts as silence.
        silence_threshold (float): Absolute peak below which audio is silent.

    Returns:
        list[tuple[int, int]]: Half-open ``(start_frame, end_frame)`` intervals,
        with starts rounded down and ends rounded up to preserve speech.
    """
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    if audio.size == 0 or float(abs(audio).max()) <= silence_threshold:
        return []

    sample_intervals = librosa.effects.split(
        audio, top_db=top_db, frame_length=1024, hop_length=256
    )
    return [
        audio_samples_to_frame_range(int(start), int(end), sample_rate, fps)
        for start, end in sample_intervals
    ]


def pause_close_weights(active_intervals, num_frames, min_pause_frames=8, fade_frames=4):
    """Build per-frame closed-mouth weights for pauses between voiced spans.

    Only gaps strictly between two voiced intervals are considered; leading and
    trailing silence are left to the existing start/end closing options. Inside
    a qualifying gap the weight ramps up with a cosine over ``fade_frames``,
    stays at 1, and ramps down again, so voiced frames are never altered.

    Args:
        active_intervals (list[tuple[int, int]]): Half-open voiced frame spans.
        num_frames (int): Total number of output frames.
        min_pause_frames (int): Shortest gap, in frames, that is closed.
        fade_frames (int): Frames used for each fade in and fade out.

    Returns:
        np.ndarray: ``float32`` weights in ``[0, 1]`` of length ``num_frames``.
    """
    num_frames = max(int(num_frames), 0)
    weights = np.zeros(num_frames, dtype=np.float32)
    spans = sorted(
        (max(int(start), 0), min(int(end), num_frames))
        for start, end in active_intervals
    )
    merged = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    min_pause_frames = max(int(min_pause_frames), 1)
    for (_, gap_start), (gap_end, _) in zip(merged, merged[1:]):
        gap_length = gap_end - gap_start
        if gap_length < min_pause_frames:
            continue
        # Leave at least one fully closed frame between the two fades
        fade = min(max(int(fade_frames), 0), (gap_length - 1) // 2)
        gap_weights = np.ones(gap_length, dtype=np.float32)
        if fade > 0:
            steps = np.arange(1, fade + 1, dtype=np.float32) / (fade + 1)
            ramp = 0.5 * (1.0 - np.cos(np.pi * steps))
            gap_weights[:fade] = ramp
            gap_weights[gap_length - fade:] = ramp[::-1]
        weights[gap_start:gap_end] = np.maximum(weights[gap_start:gap_end], gap_weights)
    return weights
