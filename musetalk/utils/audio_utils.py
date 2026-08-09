import math
import os
import subprocess


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


def ensure_wav(input_path: str, target_path: str | None = None) -> str:
    """
    Convert any audio (mp3/ogg/m4a/wav/…) to 16kHz mono PCM WAV via ffmpeg.
    Returns path to the converted .wav (original if already correct).
    """
    if not isinstance(input_path, str) or not os.path.exists(input_path):
        return input_path
    base, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    if target_path is None:
        target_path = base + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", target_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target_path


def audio_samples_to_frame_range(start_sample, end_sample, sample_rate, fps):
    """Map an active audio sample range to an inclusive video-frame boundary."""
    active_start_frame = int(math.floor(start_sample / sample_rate * float(fps)))
    active_end_frame = int(math.ceil(end_sample / sample_rate * float(fps)))
    return max(0, active_start_frame), max(active_start_frame, active_end_frame)


def get_active_audio_frame_range(audio_path, fps, top_db=40.0):
    """Return the video-frame range containing detected active audio."""
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    if audio.size == 0:
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
