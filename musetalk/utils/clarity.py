"""Opt-in clarity helpers for pasting generated mouths back into source frames.

Every helper keeps the legacy behaviour unless it is explicitly enabled, so the
production output does not change until an A/B comparison approves a switch.
Pixel maths is implemented with NumPy only, which keeps it testable without
OpenCV; only :func:`resize_generated_face` imports ``cv2`` lazily.
"""

import math

import numpy as np

PASTE_INTERPOLATION_MODES = ("linear", "lanczos")
ENHANCE_MODES = ("none", "usm")
SUPPORTED_RESOLUTIONS = (256, 512)
REF_FRAME_MODES = ("self", "first", "offset", "auto")

# Legacy generated face size before pasting back.
_GENERATED_FACE_SIZE = 256.0


def validate_ref_frame_mode(mode):
    """Validate an inference reference-frame selection mode.

    Args:
        mode (str): ``"self"`` (legacy: pair each frame with its own full latent),
            ``"first"`` (use frame 0 as a fixed reference latent for every frame),
            ``"offset"`` (use the half-cycle offset frame as reference), or
            ``"auto"`` (automatically pick the best closed-mouth frontal frame).

    Returns:
        str: Validated mode string.

    Raises:
        ValueError: If ``mode`` is not in :data:`REF_FRAME_MODES`.
    """
    if mode not in REF_FRAME_MODES:
        raise ValueError(
            f"Unsupported ref_frame_mode {mode!r}; expected one of {REF_FRAME_MODES}"
        )
    return mode


def resolve_ref_frame_index(frame_index, num_frames, mode="self", auto_ref_index=0):
    """Choose the source frame index supplying the UNet reference latent.

    Args:
        frame_index (int): Index of the target frame being rendered.
        num_frames (int): Total number of unique source frames before ping-pong cycling.
        mode (str): One of :data:`REF_FRAME_MODES`.
        auto_ref_index (int): Pre-selected neutral frame index used when
            ``mode == "auto"``.

    Returns:
        int: Reference frame index in ``[0, num_frames)``.

    Raises:
        ValueError: If ``mode`` is unknown, ``num_frames <= 0``, or an index
            is out of bounds.
    """
    validated_mode = validate_ref_frame_mode(mode)
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(f"frame_index {frame_index} out of range for num_frames={num_frames}")
    if validated_mode == "self" or num_frames == 1:
        return frame_index
    if validated_mode == "first":
        return 0
    if validated_mode == "auto":
        if auto_ref_index < 0 or auto_ref_index >= num_frames:
            raise ValueError(
                f"auto_ref_index {auto_ref_index} out of range for num_frames={num_frames}"
            )
        return int(auto_ref_index)
    return (frame_index + num_frames // 2) % num_frames


def score_landmark_neutral_ref(landmarks_68):
    """Compute mouth-openness ratio and frontal yaw asymmetry from 68 landmarks.

    Expects standard 68-point facial landmarks (DWPose ``keypoints[0][23:91]``):
    jaw contour ``0..16``, nose tip ``30``, outer mouth corners ``48`` and ``54``,
    and inner lip pairs ``(61, 67)``, ``(62, 66)``, ``(63, 65)``.

    Args:
        landmarks_68 (np.ndarray): Array-like of shape ``(68, 2)``.

    Returns:
        tuple[float, float] | None: ``(mouth_aspect_ratio, yaw_asymmetry)``, or
            ``None`` if landmarks are malformed, non-finite, or degenerate.
    """
    if landmarks_68 is None:
        return None
    pts = np.asarray(landmarks_68, dtype=np.float32)
    if pts.shape != (68, 2) or not np.all(np.isfinite(pts)):
        return None
    mouth_w = float(np.linalg.norm(pts[54] - pts[48]))
    jaw_w = float(np.linalg.norm(pts[16] - pts[0]))
    if mouth_w < 1e-6 or jaw_w < 1e-6:
        return None
    inner_gap = float(
        np.mean([
            np.linalg.norm(pts[61] - pts[67]),
            np.linalg.norm(pts[62] - pts[66]),
            np.linalg.norm(pts[63] - pts[65]),
        ])
    )
    mar = inner_gap / mouth_w
    left_dist = float(np.linalg.norm(pts[30] - pts[0]))
    right_dist = float(np.linalg.norm(pts[16] - pts[30]))
    yaw_asym = abs(left_dist - right_dist) / jaw_w
    return mar, yaw_asym


def estimate_mouth_cavity_darkness(face_bgr):
    """Estimate mouth openness via dark-cavity pixel ratio inside the mouth ROI.

    Used as a zero-model CPU fallback for ``ref_frame_mode="auto"`` when cached
    bounding boxes are reused without saved 68-point landmarks.

    Args:
        face_bgr (np.ndarray): Cropped face image, ``(H, W, 3)`` in BGR order.

    Returns:
        float: Fraction in ``[0, 1]`` of mouth-ROI pixels darker than 55% of the
            80th-percentile skin luma; lower implies a more closed mouth.
    """
    if face_bgr is None or getattr(face_bgr, "ndim", 0) != 3 or face_bgr.shape[2] != 3:
        return 1.0
    h, w = face_bgr.shape[:2]
    y1, y2 = int(h * 0.56), int(h * 0.86)
    x1, x2 = int(w * 0.22), int(w * 0.78)
    if y2 <= y1 or x2 <= x1:
        return 1.0
    roi = face_bgr[y1:y2, x1:x2].astype(np.float32)
    luma = roi[:, :, 0] * 0.114 + roi[:, :, 1] * 0.587 + roi[:, :, 2] * 0.299
    skin_ref = float(np.percentile(luma, 80))
    if skin_ref < 1e-5:
        return 1.0
    return float(np.mean(luma < 0.55 * skin_ref))


# Natural closed-mouth MAR plateau: any inner-lip ratio below this threshold is
# treated as equally closed so auto-selection picks the most frontal, median-box
# natural frame instead of a tightly pursed outlier frame.
_CLOSED_MOUTH_MAR_FLOOR = 0.035


def select_auto_ref_frame_index(
    coord_list,
    landmarks_list=None,
    mouth_darkness_scores=None,
    placeholder=(0.0, 0.0, 0.0, 0.0),
):
    """Pick the best closed-mouth, front-facing, median-pose reference frame.

    Args:
        coord_list (list): Per-frame face bounding boxes ``[x1, y1, x2, y2]``.
        landmarks_list (list | None): Optional per-frame ``(68, 2)`` landmarks
            from DWPose. When valid entries exist, selection uses
            ``3.0 * max(mar, 0.035) + 1.0 * yaw_asym + 1.0 * pose_dev``.
        mouth_darkness_scores (list | None): Optional per-frame dark-cavity ratios
            from :func:`estimate_mouth_cavity_darkness` used when landmarks are
            unavailable (e.g. ``--use_saved_coord``).
        placeholder (tuple): Sentinel value marking missing face detections.

    Returns:
        int: Selected frame index in ``[0, len(coord_list))`` (or ``0`` if empty).
    """
    valid_indices = []
    boxes = []
    for idx, bbox in enumerate(coord_list):
        if bbox is None or tuple(bbox) == tuple(placeholder):
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            valid_indices.append(idx)
            boxes.append((float(x1), float(y1), float(x2), float(y2)))

    if not valid_indices:
        return 0
    if len(valid_indices) == 1:
        return valid_indices[0]

    arr = np.asarray(boxes, dtype=np.float32)
    cxs = 0.5 * (arr[:, 0] + arr[:, 2])
    cys = 0.5 * (arr[:, 1] + arr[:, 3])
    ws = np.maximum(arr[:, 2] - arr[:, 0], 1.0)
    hs = np.maximum(arr[:, 3] - arr[:, 1], 1.0)
    med_cx, med_cy = float(np.median(cxs)), float(np.median(cys))
    med_w, med_h = float(np.median(ws)), float(np.median(hs))

    pose_dev = {}
    for pos, idx in enumerate(valid_indices):
        pose_dev[idx] = (
            abs(float(cxs[pos]) - med_cx) / med_w
            + abs(float(cys[pos]) - med_cy) / med_h
            + abs(float(ws[pos]) - med_w) / med_w
            + abs(float(hs[pos]) - med_h) / med_h
        )

    # Primary path: DWPose 68-point landmarks (mouth closedness + frontal yaw + median pose)
    if landmarks_list is not None and len(landmarks_list) == len(coord_list):
        landmark_candidates = []
        for idx in valid_indices:
            scored = score_landmark_neutral_ref(landmarks_list[idx])
            if scored is not None:
                mar, yaw_asym = scored
                effective_mar = max(float(mar), _CLOSED_MOUTH_MAR_FLOOR)
                total = 3.0 * effective_mar + 1.0 * yaw_asym + 1.0 * pose_dev[idx]
                landmark_candidates.append((total, idx))
        if landmark_candidates:
            landmark_candidates.sort(key=lambda item: (item[0], item[1]))
            return landmark_candidates[0][1]

    # Fallback path: CPU image darkness ratio (e.g., --use_saved_coord legacy .pkl)
    if mouth_darkness_scores is not None and len(mouth_darkness_scores) == len(coord_list):
        dark_candidates = []
        for idx in valid_indices:
            dark = mouth_darkness_scores[idx]
            if dark is not None and np.isfinite(dark):
                total = 3.0 * float(dark) + 0.5 * pose_dev[idx]
                dark_candidates.append((total, idx))
        if dark_candidates:
            dark_candidates.sort(key=lambda item: (item[0], item[1]))
            return dark_candidates[0][1]

    return min(valid_indices, key=lambda idx: (pose_dev[idx], idx))


def rebind_ref_latents(input_latent_list, ref_frame_mode="self", ref_frame_index=None):
    """Replace the reference-half channels of concatenated UNet latents.

    ``VAE.get_latents_for_unet`` concatenates ``[masked_latents, ref_latents]``
    along channel dimension 1. Keeping ``ref_frame_mode="self"`` returns
    ``input_latent_list`` unchanged; ``"first"``, ``"offset"``, and ``"auto"``
    pair each frame's masked latent with another frame's reference latent so a
    speaking template video does not leak its original mouth state into
    cross-driven generation.

    Args:
        input_latent_list (list): Sequence of 4-D ``(B, 2*C, H, W)`` tensors or
            NumPy arrays, one per unique source frame.
        ref_frame_mode (str): One of :data:`REF_FRAME_MODES`.
        ref_frame_index (int | None): Pre-selected frame index used when
            ``ref_frame_mode == "auto"`` (defaults to ``0`` if ``None``).

    Returns:
        list: Original list when ``ref_frame_mode == "self"`` or fewer than two
            frames are present; otherwise a new list with rebound reference halves.

    Raises:
        ValueError: If ``ref_frame_mode`` is invalid, ``ref_frame_index`` is out
            of bounds, or any latent lacks an even channel count on axis 1.
    """
    validated_mode = validate_ref_frame_mode(ref_frame_mode)
    num_frames = len(input_latent_list)
    if validated_mode == "self" or num_frames <= 1:
        return input_latent_list

    sample = input_latent_list[0]
    if getattr(sample, "ndim", None) != 4 or sample.shape[1] < 2 or sample.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected 4-D latents with even channel count >= 2, got shape {getattr(sample, 'shape', None)}"
        )
    half_channels = sample.shape[1] // 2
    is_numpy = isinstance(sample, np.ndarray)
    cat_fn = None if is_numpy else __import__("torch").cat
    auto_idx = 0 if ref_frame_index is None else int(ref_frame_index)

    rebound = []
    for idx in range(num_frames):
        ref_idx = resolve_ref_frame_index(
            idx,
            num_frames,
            mode=validated_mode,
            auto_ref_index=auto_idx,
        )
        masked_half = input_latent_list[idx][:, :half_channels]
        ref_half = input_latent_list[ref_idx][:, half_channels:]
        if is_numpy:
            rebound.append(np.concatenate([masked_half, ref_half], axis=1))
        else:
            rebound.append(cat_fn([masked_half, ref_half], dim=1))
    return rebound


def validate_resolution(resolution):
    """Validate an inference generation resolution.

    Args:
        resolution (int): Target square generation size in pixels.

    Returns:
        int: Validated resolution (``256`` or ``512``).

    Raises:
        ValueError: If ``resolution`` is a boolean or not in
            :data:`SUPPORTED_RESOLUTIONS`.
    """
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution {resolution!r}; expected one of {SUPPORTED_RESOLUTIONS}"
        )
    return resolution


def validate_cached_avatar_resolution(avatar_id, cached_info, current_resolution):
    """Ensure a cached realtime avatar's latents match the active resolution.

    Args:
        avatar_id (str): Identifier of the cached avatar.
        cached_info (dict): Metadata loaded from ``avator_info.json``; older
            caches without a ``resolution`` field default to ``256``.
        current_resolution (int): Active inference resolution (``256`` or ``512``).

    Raises:
        ValueError: If the cached resolution differs from ``current_resolution``.
    """
    expected = validate_resolution(current_resolution)
    cached_res = cached_info.get("resolution", 256)
    if cached_res != expected:
        raise ValueError(
            f"Cached avatar '{avatar_id}' was prepared at resolution={cached_res}, "
            f"which is incompatible with --resolution {expected}; "
            "re-create the avatar with preparation=True."
        )

# BT.601 luma weights in BGR channel order.
_LUMA_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def resize_generated_face(face, target_w, target_h, mode="linear"):
    """Resize a generated face crop to the destination face box.

    Args:
        face (np.ndarray): Generated BGR face, usually ``256x256x3``.
        target_w (int): Destination width in pixels.
        target_h (int): Destination height in pixels.
        mode (str): ``"linear"`` reproduces the legacy ``cv2.resize`` default;
            ``"lanczos"`` shrinks with ``INTER_AREA`` and enlarges with
            ``INTER_LANCZOS4``, handling each axis independently.

    Returns:
        np.ndarray: ``uint8`` face of shape ``(target_h, target_w, 3)``.

    Raises:
        ValueError: If the target size is not positive or the mode is unknown.
    """
    import cv2

    target_w = int(target_w)
    target_h = int(target_h)
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target face size: {target_w}x{target_h}")
    if mode not in PASTE_INTERPOLATION_MODES:
        raise ValueError(f"Unknown paste interpolation mode: {mode}")

    face = face.astype(np.uint8, copy=False)
    if mode == "linear":
        return cv2.resize(face, (target_w, target_h))

    src_h, src_w = face.shape[:2]
    reduced = (min(src_w, target_w), min(src_h, target_h))
    if reduced != (src_w, src_h):
        face = cv2.resize(face, reduced, interpolation=cv2.INTER_AREA)
    if reduced != (target_w, target_h):
        face = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return face


def blend_faces(generated, closed, closed_weight):
    """Linearly blend a generated face toward a closed-mouth reference.

    Args:
        generated (np.ndarray): Generated face, ``uint8`` HxWx3.
        closed (np.ndarray): Closed-mouth reference with the same shape.
        closed_weight (float): Weight of ``closed`` in ``[0, 1]``.

    Returns:
        np.ndarray: Blended ``uint8`` face.

    Raises:
        ValueError: If the two faces have different shapes.
    """
    if generated.shape != closed.shape:
        raise ValueError(f"Face shapes differ: {generated.shape} vs {closed.shape}")
    weight = float(np.clip(closed_weight, 0.0, 1.0))
    if weight <= 0.0:
        return generated
    if weight >= 1.0:
        return closed
    mixed = generated.astype(np.float32) * (1.0 - weight) + closed.astype(np.float32) * weight
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def gaussian_kernel_1d(sigma):
    """Build a normalized 1-D Gaussian kernel with radius ``ceil(3 * sigma)``.

    Args:
        sigma (float): Standard deviation in pixels; must be positive.

    Returns:
        np.ndarray: ``float32`` kernel summing to one.

    Raises:
        ValueError: If ``sigma`` is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def _blur_axis(image, kernel, axis):
    """Convolve a 2-D float array with a 1-D kernel along one axis."""
    radius = len(kernel) // 2
    length = image.shape[axis]
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    pad_mode = "reflect" if length > radius else "edge"
    padded = np.pad(image, pad_width, mode=pad_mode)

    blurred = np.zeros(image.shape, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        window = [slice(None)] * image.ndim
        window[axis] = slice(offset, offset + length)
        blurred += weight * padded[tuple(window)]
    return blurred


def gaussian_blur_2d(image, sigma):
    """Apply a separable Gaussian blur to a single-channel image.

    Args:
        image (np.ndarray): 2-D array.
        sigma (float): Standard deviation in pixels.

    Returns:
        np.ndarray: Blurred ``float32`` array with the input shape.
    """
    kernel = gaussian_kernel_1d(sigma)
    image = image.astype(np.float32, copy=False)
    return _blur_axis(_blur_axis(image, kernel, axis=1), kernel, axis=0)


def compute_usm_params(face_w, face_h, amount_override=-1.0, generated_size=_GENERATED_FACE_SIZE):
    """Choose unsharp-mask strength from how much the generated face is enlarged.

    Args:
        face_w (int): Destination face-box width.
        face_h (int): Destination face-box height.
        amount_override (float): Fixed amount when ``>= 0``; negative means auto.
        generated_size (float): Native square generation resolution (e.g. 256 or 512).

    Returns:
        tuple[float, float] | None: ``(amount, sigma)``, or ``None`` when the
        face is not enlarged enough for sharpening to help.

    Raises:
        ValueError: If ``generated_size`` is not positive.
    """
    if generated_size <= 0:
        raise ValueError(f"generated_size must be positive, got {generated_size}")
    scale = max(int(face_w), int(face_h)) / float(generated_size)
    if scale <= 0.95:
        return None
    if amount_override is not None and amount_override >= 0:
        amount = float(amount_override)
    else:
        amount = float(np.clip(0.35 + 0.25 * (scale - 1.0), 0.30, 0.85))
    sigma = float(np.clip(scale, 1.0, 2.2))
    return amount, sigma


def mouth_ellipse_weight(height, width, inner_ratio=0.6):
    """Build a feathered ellipse weight centred on the lips of a face box.

    The ellipse is centred at ``(0.50 W, 0.68 H)`` with semi-axes
    ``(0.36 W, 0.22 H)``. Weight is 1 inside ``inner_ratio`` of the radius and
    falls to 0 at the ellipse edge with a cosine ramp.

    Args:
        height (int): Face-box height.
        width (int): Face-box width.
        inner_ratio (float): Normalized radius where the fade starts.

    Returns:
        np.ndarray: ``float32`` weight of shape ``(height, width)``.
    """
    ys = (np.arange(height, dtype=np.float32) + 0.5)[:, np.newaxis]
    xs = (np.arange(width, dtype=np.float32) + 0.5)[np.newaxis, :]
    center_x, center_y = 0.50 * width, 0.68 * height
    axis_x, axis_y = max(0.36 * width, 1e-6), max(0.22 * height, 1e-6)
    distance = np.sqrt(((xs - center_x) / axis_x) ** 2 + ((ys - center_y) / axis_y) ** 2)
    fade = np.clip((distance - inner_ratio) / (1.0 - inner_ratio), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * fade))).astype(np.float32)


def sharpen_lip_region(face_bgr, amount_override=-1.0, threshold=3.0, generated_size=_GENERATED_FACE_SIZE):
    """Sharpen the lip and teeth area of a pasted face on its luma channel.

    Sharpening only luma and adding the same delta to all channels avoids the
    colour fringes an RGB unsharp mask produces. A soft threshold leaves flat
    skin noise untouched, and an ellipse weight keeps cheeks and nose unchanged.

    Args:
        face_bgr (np.ndarray): ``uint8`` BGR face already resized to the face box.
        amount_override (float): Fixed amount when ``>= 0``; negative means auto.
        threshold (float): Luma detail below this magnitude is ignored.
        generated_size (float): Native square generation resolution (e.g. 256 or 512).

    Returns:
        np.ndarray: Sharpened ``uint8`` face, or the input when skipped.
    """
    height, width = face_bgr.shape[:2]
    params = compute_usm_params(width, height, amount_override, generated_size=generated_size)
    if params is None:
        return face_bgr
    amount, sigma = params
    if amount <= 0:
        return face_bgr

    pixels = face_bgr.astype(np.float32)
    luma = pixels @ _LUMA_WEIGHTS_BGR
    detail = luma - gaussian_blur_2d(luma, sigma)
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    delta = amount * detail * mouth_ellipse_weight(height, width)
    sharpened = pixels + delta[:, :, np.newaxis]
    return np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)


def correct_teeth_whiteness(
    face_bgr,
    mouth_mask,
    brightness_reduce=0.10,
    warm_shift=6.0,
    sat_boost=0.15,
    min_mouth_fraction=0.001,
):
    """Tone down over-bright, colourless pixels inside the mouth interior.

    Adapted from upstream PR #423, but restricted to a mouth-interior mask so
    that highlights on skin or lips are never touched.

    Args:
        face_bgr (np.ndarray): ``uint8`` BGR face.
        mouth_mask (np.ndarray | None): HxW mask of the mouth interior; non-zero
            marks the mouth. ``None`` or an empty mask skips the correction.
        brightness_reduce (float): Fractional brightness reduction at full weight.
        warm_shift (float): Blue-channel reduction at full weight.
        sat_boost (float): Fractional saturation increase at full weight.
        min_mouth_fraction (float): Skip when the mouth covers less of the face.

    Returns:
        np.ndarray: Corrected ``uint8`` face, or the input when skipped.
    """
    if mouth_mask is None:
        return face_bgr
    mouth = np.asarray(mouth_mask) > 0
    if mouth.shape != face_bgr.shape[:2]:
        raise ValueError(f"Mouth mask shape {mouth.shape} does not match face {face_bgr.shape[:2]}")
    if mouth.mean() < min_mouth_fraction:
        return face_bgr

    pixels = face_bgr.astype(np.float32)
    value = pixels.max(axis=2)
    chroma = value - pixels.min(axis=2)
    saturation = np.where(value > 0, chroma / np.maximum(value, 1e-6) * 255.0, 0.0)
    teeth = mouth & (value > 170) & (saturation < 60)
    if not teeth.any():
        return face_bgr

    weight = np.clip(gaussian_blur_2d(teeth.astype(np.float32), 1.0), 0.0, 1.0)
    weight = np.where(mouth, weight, 0.0)[:, :, np.newaxis]

    luma = (pixels @ _LUMA_WEIGHTS_BGR)[:, :, np.newaxis]
    corrected = luma + (pixels - luma) * (1.0 + sat_boost * weight)
    corrected = corrected * (1.0 - brightness_reduce * weight)
    corrected[:, :, 0] -= warm_shift * weight[:, :, 0]
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
