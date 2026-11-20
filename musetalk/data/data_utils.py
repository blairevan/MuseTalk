"""Pure helpers for the training dataset that do not need torch or decord."""

import contextlib
import random

import numpy as np


def read_meta_list(list_path):
    """Read a header-less clip list written by ``scripts/preprocess.py``.

    Every non-empty line names one meta JSON file in its first column. Lines
    starting with ``#`` are comments. No line is skipped as a header.

    Args:
        list_path (str): Path to ``train.txt`` / ``val.txt``.

    Returns:
        list[str]: Meta file names in file order.
    """
    names = []
    with open(list_path, "r", encoding="utf-8") as list_file:
        for line in list_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.append(stripped.split()[0])
    return names


def check_face_size_config(image_size, min_face_size, enabled):
    """Reject a minimum face size that no resized crop can ever satisfy.

    Crops are resized to ``image_size`` before the per-frame size check, so a
    larger ``min_face_size`` would make every sample fail.

    Raises:
        ValueError: If the check is enabled and ``min_face_size > image_size``.
    """
    if enabled and int(min_face_size) > int(image_size):
        raise ValueError(
            f"min_face_size ({min_face_size}) exceeds image_size ({image_size}); "
            "every sample would be rejected after resizing."
        )


def mouth_mask_bounds(lip_landmarks, padding_pixel_mouth, crop_margin, image_size):
    """Compute the mouth mask rectangle in resized-crop coordinates.

    Args:
        lip_landmarks (np.ndarray): Lip landmarks already scaled to the crop,
            shape ``(N, 2)``.
        padding_pixel_mouth (float): Horizontal padding around the lips.
        crop_margin (float): Vertical shift caused by the dynamic jaw margin.
        image_size (int): Side length of the resized square crop.

    Returns:
        tuple[float, float, float, float]: ``(min_x, min_y, max_x, max_y)``
        clamped to ``[0, image_size]``.
    """
    points = np.asarray(lip_landmarks, dtype=np.float64)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    min_x -= padding_pixel_mouth
    max_x += padding_pixel_mouth

    width = max_x - min_x
    center_y = (max_y + min_y) / 2
    min_y = center_y - width / 4 - crop_margin
    max_y = center_y + width / 4 - crop_margin

    limit = float(image_size)
    return (
        float(np.clip(min_x, 0, limit)),
        float(np.clip(min_y, 0, limit)),
        float(np.clip(max_x, 0, limit)),
        float(np.clip(max_y, 0, limit)),
    )


def whisper_window(start_index, fps=25, window_seconds=30, context_frames=50, shift_seconds=4):
    """Locate the 30 s Whisper input window for a clip that starts at ``start_index``.

    The window containing the start frame is used; when fewer than
    ``context_frames`` frames remain in it, the window is moved later by
    ``shift_seconds`` so the audio context still fits. For starts inside the
    first 30 s this matches the original behaviour; later starts now select the
    window they actually fall in instead of falling back to the clip start.

    Args:
        start_index (int): Absolute start frame of the sampled clip.
        fps (int): Video frame rate used by the dataset.
        window_seconds (int): Whisper input length in seconds.
        context_frames (int): Frames of audio context required after the start.
        shift_seconds (int): How far the window moves when context is short.

    Returns:
        tuple[int, int]: ``(window_start_seconds, frame_index_inside_window)``.
    """
    frames_per_window = int(window_seconds * fps)
    segment = int(start_index) // frames_per_window
    local_index = int(start_index) - segment * frames_per_window
    window_start = segment * window_seconds
    if local_index + context_frames >= frames_per_window:
        local_index -= shift_seconds * fps
        window_start += shift_seconds
    return window_start, local_index


@contextlib.contextmanager
def seeded_random_state(seed):
    """Temporarily seed Python and NumPy RNGs, restoring both afterwards.

    Used to make validation samples reproducible without disturbing the
    training process's random streams.

    Args:
        seed (int): Seed applied inside the context.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
