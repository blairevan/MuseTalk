"""Pose-aligned reference-frame helpers for cross-driven MuseTalk inference.

Pairs each target frame's upper-face masked latent with a 2D similarity-aligned
reference crop so head tilt, translation, and detector bbox-height jitter do not
misalign the lower jawline or bleed background pixels onto the neck.

Geometry math uses NumPy only so it stays deterministic and testable without
OpenCV; only :func:`align_reference_face_crop` imports ``cv2`` lazily.
"""

import math

import numpy as np

from musetalk.utils.clarity import (
    score_landmark_neutral_ref,
    select_auto_ref_frame_index,
    validate_ref_frame_mode,
)

# Standard 68-point facial landmarks invariant to mouth opening:
# upper jaw/ears (0..2, 14..16), eyebrows (17..26), nose (27..35), eyes (36..47).
RIGID_UPPER_FACE_INDICES = (
    0, 1, 2, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
)
_LEFT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
_RIGHT_EYE_INDICES = (42, 43, 44, 45, 46, 47)

# Default guardrails for rejecting anomalous or out-of-plane pose warps.
DEFAULT_MIN_SCALE = 0.80
DEFAULT_MAX_SCALE = 1.25
DEFAULT_MAX_ANGLE_DEG = 25.0
DEFAULT_MAX_NORM_RESIDUAL = 0.12
# Note: yaw_asym = |d_left - d_right| / jaw_w equals ~2x nose horizontal shift
# ratio, so a mild ~12deg talking-head turn yields ~0.25 while a true >25deg
# out-of-plane 3/4 profile turn exceeds ~0.38.
DEFAULT_MAX_YAW_DELTA = 0.38
DEFAULT_MIN_IN_BOUNDS_RATIO = 0.85

# Default thresholds for detecting an already closed-mouth template video.
DEFAULT_CLOSED_TEMPLATE_P95_MAR = 0.06
DEFAULT_CLOSED_TEMPLATE_MAX_MAR = 0.10


def is_closed_mouth_template(
    landmarks_list,
    p95_mar_threshold=DEFAULT_CLOSED_TEMPLATE_P95_MAR,
    max_mar_threshold=DEFAULT_CLOSED_TEMPLATE_MAX_MAR,
):
    """Return whether a template video already keeps its mouth closed throughout.

    When every frame of the template has a closed mouth (e.g. an idle nodding
    avatar clip), per-frame ``self`` reference has zero geometric mismatch and
    no source-mouth leakage, making cross-frame reference binding unnecessary.

    Args:
        landmarks_list (list | None): Per-frame ``(68, 2)`` landmark arrays.
        p95_mar_threshold (float): Maximum 95th-percentile MAR still considered
            closed.
        max_mar_threshold (float): Maximum single-frame MAR still considered
            closed.

    Returns:
        tuple[bool, float | None]: ``(is_closed, p95_mar)``. Returns
            ``(False, None)`` when fewer than two valid frames exist.
    """
    if not landmarks_list:
        return False, None

    mars = []
    for lm in landmarks_list:
        scored = score_landmark_neutral_ref(lm)
        if scored is not None:
            mars.append(float(scored[0]))

    if len(mars) < 2:
        return False, None

    p95_mar = float(np.percentile(mars, 95))
    max_mar = float(max(mars))
    is_closed = (p95_mar <= float(p95_mar_threshold)) and (max_mar <= float(max_mar_threshold))
    return is_closed, p95_mar


def _fit_similarity_2d(src_pts, dst_pts):
    """Solve closed-form 4-DoF 2D similarity (scale, rotation, translation).

    Finds ``M`` of shape ``(2, 3)`` minimizing ``sum ||M @ [x, y, 1]^T - dst||^2``
    without reflection (Umeyama 2D).

    Args:
        src_pts (np.ndarray): Source coordinates, shape ``(N, 2)``.
        dst_pts (np.ndarray): Target coordinates, shape ``(N, 2)``.

    Returns:
        np.ndarray | None: ``(2, 3)`` float64 affine matrix, or ``None`` if the
            point set is degenerate.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2 or src.shape[0] < 3:
        return None

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_var = float(np.sum(src_c * src_c))
    if src_var < 1e-8:
        return None

    # Closed-form 2D similarity without reflection:
    # R = [[a, -b], [b, a]] with a = s*cos(theta), b = s*sin(theta).
    a = float(np.sum(src_c[:, 0] * dst_c[:, 0] + src_c[:, 1] * dst_c[:, 1])) / src_var
    b = float(np.sum(src_c[:, 0] * dst_c[:, 1] - src_c[:, 1] * dst_c[:, 0])) / src_var
    if not (math.isfinite(a) and math.isfinite(b)):
        return None

    tx = float(dst_mean[0] - (a * src_mean[0] - b * src_mean[1]))
    ty = float(dst_mean[1] - (b * src_mean[0] + a * src_mean[1]))
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def _inter_ocular_distance(landmarks_68):
    """Compute distance between left and right eye centroids."""
    left_eye = landmarks_68[list(_LEFT_EYE_INDICES)].mean(axis=0)
    right_eye = landmarks_68[list(_RIGHT_EYE_INDICES)].mean(axis=0)
    return float(np.linalg.norm(right_eye - left_eye))


def _signed_yaw_asymmetry(landmarks_68):
    """Compute signed nose-to-jaw horizontal yaw ratio (left vs right turn)."""
    jaw_w = float(np.linalg.norm(landmarks_68[16] - landmarks_68[0]))
    if jaw_w < 1e-6:
        return 0.0
    left_dist = float(np.linalg.norm(landmarks_68[30] - landmarks_68[0]))
    right_dist = float(np.linalg.norm(landmarks_68[16] - landmarks_68[30]))
    return (left_dist - right_dist) / jaw_w


def estimate_rigid_face_similarity(
    ref_landmarks,
    target_landmarks,
    min_scale=DEFAULT_MIN_SCALE,
    max_scale=DEFAULT_MAX_SCALE,
    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
    max_norm_residual=DEFAULT_MAX_NORM_RESIDUAL,
    max_yaw_delta=DEFAULT_MAX_YAW_DELTA,
):
    """Estimate a validated 2D similarity mapping reference face to target face.

    Uses a 1-step trimmed least-squares fit on upper-face rigid landmarks so a
    few jittery points do not skew the transform, then enforces scale, rotation,
    normalized residual, and 3D yaw-delta guardrails.

    Args:
        ref_landmarks (np.ndarray | None): Reference ``(68, 2)`` landmarks.
        target_landmarks (np.ndarray | None): Target ``(68, 2)`` landmarks.
        min_scale (float): Minimum accepted similarity scale.
        max_scale (float): Maximum accepted similarity scale.
        max_angle_deg (float): Maximum accepted absolute in-plane roll angle.
        max_norm_residual (float): Maximum median landmark residual divided by
            the target inter-ocular distance.
        max_yaw_delta (float): Maximum allowed difference in yaw asymmetry
            between target and reference before rejecting as out-of-plane turn.

    Returns:
        tuple[np.ndarray | None, str]: ``(matrix_2x3, status)`` where
            ``status`` is ``"aligned"`` when valid, ``"missing_landmarks"`` when
            either landmark array is absent/malformed, or ``"fallback_self"``
            when rejected by a safety guardrail.
    """
    ref_scored = score_landmark_neutral_ref(ref_landmarks)
    tgt_scored = score_landmark_neutral_ref(target_landmarks)
    if ref_scored is None or tgt_scored is None:
        return None, "missing_landmarks"

    ref_pts = np.asarray(ref_landmarks, dtype=np.float64)
    tgt_pts = np.asarray(target_landmarks, dtype=np.float64)
    if abs(_signed_yaw_asymmetry(tgt_pts) - _signed_yaw_asymmetry(ref_pts)) > float(max_yaw_delta):
        return None, "fallback_self"

    iod = _inter_ocular_distance(tgt_pts)
    if not math.isfinite(iod) or iod < 4.0:
        return None, "fallback_self"

    idx = list(RIGID_UPPER_FACE_INDICES)
    src = ref_pts[idx]
    dst = tgt_pts[idx]

    matrix = _fit_similarity_2d(src, dst)
    if matrix is None:
        return None, "fallback_self"

    # Trim the worst 20% residuals and refit once for deterministic robustness.
    pred = src @ matrix[:, :2].T + matrix[:, 2]
    errs = np.linalg.norm(pred - dst, axis=1)
    keep_count = max(6, int(round(len(errs) * 0.8)))
    inliers = np.argsort(errs)[:keep_count]
    refined = _fit_similarity_2d(src[inliers], dst[inliers])
    if refined is not None:
        matrix = refined

    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    scale = math.hypot(a, b)
    angle_deg = abs(math.degrees(math.atan2(b, a)))
    if not (min_scale <= scale <= max_scale) or angle_deg > float(max_angle_deg):
        return None, "fallback_self"

    pred_final = src @ matrix[:, :2].T + matrix[:, 2]
    median_err = float(np.median(np.linalg.norm(pred_final - dst, axis=1)))
    if (median_err / iod) > float(max_norm_residual):
        return None, "fallback_self"

    return matrix.astype(np.float32), "aligned"


def _normalize_output_size(output_size):
    """Normalize an integer square size or ``(width, height)`` pair."""
    if isinstance(output_size, (tuple, list)):
        if len(output_size) != 2:
            raise ValueError(f"Expected 2-element output_size, got {output_size!r}")
        out_w, out_h = int(output_size[0]), int(output_size[1])
    else:
        if isinstance(output_size, bool):
            raise ValueError(f"Invalid boolean output_size={output_size!r}")
        out_w = out_h = int(output_size)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"output_size must be positive, got {output_size!r}")
    return out_w, out_h


def build_upper_body_pose_box(face_box, image_shape, full_frame_ratio=0.85):
    """Expand a detected face box into an upper-body ROI for top-down DWPose.

    Full-body 4K portrait videos shrink the mouth to a few pixels if the entire
    frame is resized to ``288x384``. Cropping a head-and-shoulders ``ubody`` box
    around the S3FD face preserves 2-3x finer facial keypoint resolution while
    matching ``rtmpose-l_coco-ubody`` framing. Close-up / half-body frames where
    the expanded box already spans ``>= full_frame_ratio`` of both axes return
    ``None`` so legacy full-frame inference stays untouched.

    Args:
        face_box (Sequence[float] | None): S3FD detection ``(x1, y1, x2, y2, ...)``.
        image_shape (Sequence[int]): Frame shape ``(H, W, ...)``.
        full_frame_ratio (float): Fraction of frame width/height above which
            cropping is skipped.

    Returns:
        np.ndarray | None: ``(1, 4)`` float32 ``[[x1, y1, x2, y2]]`` box for
            ``inference_topdown(..., bboxes=...)``, or ``None`` for full-frame.
    """
    if face_box is None or len(face_box) < 4 or len(image_shape) < 2:
        return None
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    if img_h <= 0 or img_w <= 0:
        return None

    fx1, fy1, fx2, fy2 = [float(v) for v in face_box[:4]]
    fw = fx2 - fx1
    fh = fy2 - fy1
    if fw <= 2.0 or fh <= 2.0:
        return None

    fcx = 0.5 * (fx1 + fx2)
    ux1 = max(0.0, fcx - 2.2 * fw)
    uy1 = max(0.0, fy1 - 0.8 * fh)
    ux2 = min(float(img_w), fcx + 2.2 * fw)
    uy2 = min(float(img_h), fy2 + 3.2 * fh)

    if (ux2 - ux1) >= full_frame_ratio * img_w and (uy2 - uy1) >= full_frame_ratio * img_h:
        return None
    if (ux2 - ux1) <= 4.0 or (uy2 - uy1) <= 4.0:
        return None
    return np.array([[ux1, uy1, ux2, uy2]], dtype=np.float32)


def compose_aligned_crop_matrix(similarity_matrix, target_box, output_size):
    """Compose a full-frame similarity warp and box resize into one 2x3 matrix.

    Matches OpenCV ``cv2.resize`` pixel-center alignment so an identity
    similarity produces zero half-pixel drift relative to direct crop+resize.

    Args:
        similarity_matrix (np.ndarray): ``(2, 3)`` matrix mapping reference
            frame coordinates into target frame coordinates.
        target_box (tuple[int, int, int, int]): Target face crop box
            ``(x1, y1, x2, y2)``.
        output_size (int | tuple[int, int]): Square resolution ``int`` or
            ``(out_width, out_height)`` pair.

    Returns:
        np.ndarray: ``(2, 3)`` float32 affine matrix mapping reference frame
            coordinates directly into ``output_size``.

    Raises:
        ValueError: If ``target_box`` or ``output_size`` has non-positive size.
    """
    x1, y1, x2, y2 = [float(v) for v in target_box]
    width = x2 - x1
    height = y2 - y1
    out_w, out_h = _normalize_output_size(output_size)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Invalid target_box={target_box!r} or output_size={output_size!r}"
        )

    m = np.asarray(similarity_matrix, dtype=np.float64)
    if m.shape != (2, 3) or not np.all(np.isfinite(m)):
        raise ValueError(f"Expected finite (2, 3) matrix, got shape {m.shape}")

    sx = float(out_w) / width
    sy = float(out_h) / height
    tx_c = -sx * x1 + 0.5 * (sx - 1.0)
    ty_c = -sy * y1 + 0.5 * (sy - 1.0)

    composed = np.array(
        [
            [sx * m[0, 0], sx * m[0, 1], sx * m[0, 2] + tx_c],
            [sy * m[1, 0], sy * m[1, 1], sy * m[1, 2] + ty_c],
        ],
        dtype=np.float32,
    )
    return composed


def compute_crop_in_bounds_ratio(crop_matrix, output_size, ref_shape):
    """Estimate fraction of the output crop that samples inside ``ref_frame``.

    Samples a regular grid in output space, inverse-maps points back into
    reference-frame coordinates, and returns the fraction inside image bounds.

    Args:
        crop_matrix (np.ndarray): ``(2, 3)`` matrix mapping ref -> crop space.
        output_size (int | tuple[int, int]): Square size or ``(out_w, out_h)``.
        ref_shape (tuple[int, ...]): Shape of ``ref_frame`` ``(H, W, ...)``.

    Returns:
        float: Ratio in ``[0.0, 1.0]`` of sampled points inside ``ref_frame``.
    """
    ref_h, ref_w = int(ref_shape[0]), int(ref_shape[1])
    try:
        out_w, out_h = _normalize_output_size(output_size)
    except ValueError:
        return 0.0
    if ref_h <= 0 or ref_w <= 0:
        return 0.0

    m = np.asarray(crop_matrix, dtype=np.float64)
    lin = m[:, :2]
    det = float(lin[0, 0] * lin[1, 1] - lin[0, 1] * lin[1, 0])
    if abs(det) < 1e-8:
        return 0.0
    inv_lin = np.linalg.inv(lin)
    t = m[:, 2]

    xs = np.linspace(0.0, float(out_w - 1), 9, dtype=np.float64)
    ys = np.linspace(0.0, float(out_h - 1), 9, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    src_pts = (pts - t) @ inv_lin.T

    inside = (
        (src_pts[:, 0] >= 0.0)
        & (src_pts[:, 0] <= float(ref_w - 1))
        & (src_pts[:, 1] >= 0.0)
        & (src_pts[:, 1] <= float(ref_h - 1))
    )
    return float(np.mean(inside))


def align_reference_face_crop(
    ref_frame,
    ref_landmarks,
    target_landmarks,
    target_box,
    output_size,
    min_in_bounds_ratio=DEFAULT_MIN_IN_BOUNDS_RATIO,
):
    """Warp ``ref_frame`` into ``target_box`` via one Lanczos4 affine pass.

    Args:
        ref_frame (np.ndarray): Full BGR reference frame.
        ref_landmarks (np.ndarray | None): ``(68, 2)`` landmarks of ``ref_frame``.
        target_landmarks (np.ndarray | None): ``(68, 2)`` landmarks of target frame.
        target_box (tuple[int, int, int, int]): Target crop box ``(x1, y1, x2, y2)``.
        output_size (int | tuple[int, int]): Square resolution (``256``/``512``)
            or rectangular ``(width, height)`` destination size.
        min_in_bounds_ratio (float): Minimum in-bounds sampling fraction before
            rejecting to prevent edge-replicate streaks.

    Returns:
        tuple[np.ndarray | None, str]: ``(aligned_crop_bgr, status)`` where
            ``status`` is ``"aligned"``, ``"missing_landmarks"``, or
            ``"fallback_self"``.
    """
    if ref_frame is None or getattr(ref_frame, "ndim", 0) != 3:
        return None, "fallback_self"

    try:
        out_w, out_h = _normalize_output_size(output_size)
    except ValueError:
        return None, "fallback_self"

    sim_matrix, status = estimate_rigid_face_similarity(ref_landmarks, target_landmarks)
    if sim_matrix is None:
        return None, status

    try:
        crop_matrix = compose_aligned_crop_matrix(sim_matrix, target_box, (out_w, out_h))
    except ValueError:
        return None, "fallback_self"

    if compute_crop_in_bounds_ratio(crop_matrix, (out_w, out_h), ref_frame.shape) < float(min_in_bounds_ratio):
        return None, "fallback_self"

    import cv2

    aligned = cv2.warpAffine(
        ref_frame,
        crop_matrix,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, "aligned"


def plan_reference_frame_mode(
    ref_frame_mode,
    coord_list,
    landmarks_list=None,
    placeholder=(0.0, 0.0, 0.0, 0.0),
):
    """Decide how reference latents should be built for a source video.

    When 68-point landmarks are available:
    - In ``"auto"`` mode, templates whose mouth is already closed throughout
      bypass cross-frame binding and keep per-frame ``"self"`` references.
    - Speaking templates select a natural closed-mouth neutral reference frame
      and enable per-frame 2D rigid pose alignment.
    - Explicit ``"first"`` / ``"offset"`` modes also enable 2D rigid alignment.

    When landmarks are unavailable (e.g. ``--use_saved_coord``), falls back to
    post-encoding ``rebind_ref_latents`` without landmark alignment.

    Args:
        ref_frame_mode (str): Requested mode from :data:`REF_FRAME_MODES`.
        coord_list (list): Per-frame face bounding boxes.
        landmarks_list (list | None): Optional per-frame ``(68, 2)`` landmarks.
        placeholder (tuple): Sentinel box for missing face detections.

    Returns:
        dict: Plan dictionary with keys ``effective_mode``, ``auto_ref_idx``,
            ``use_landmark_align``, ``closed_template``, and ``p95_mar``.
    """
    validated_mode = validate_ref_frame_mode(ref_frame_mode)
    if validated_mode == "self" or len(coord_list) <= 1:
        return {
            "effective_mode": "self",
            "auto_ref_idx": None,
            "use_landmark_align": False,
            "closed_template": False,
            "p95_mar": None,
        }

    if landmarks_list is None or len(landmarks_list) != len(coord_list):
        return {
            "effective_mode": validated_mode,
            "auto_ref_idx": None,
            "use_landmark_align": False,
            "closed_template": False,
            "p95_mar": None,
        }

    valid_lm_count = sum(
        1 for lm in landmarks_list if score_landmark_neutral_ref(lm) is not None
    )
    if valid_lm_count < 2:
        return {
            "effective_mode": validated_mode,
            "auto_ref_idx": None,
            "use_landmark_align": False,
            "closed_template": False,
            "p95_mar": None,
        }

    if validated_mode in ("auto", "dynamic"):
        closed_template, p95_mar = is_closed_mouth_template(landmarks_list)
        if closed_template:
            return {
                "effective_mode": "self",
                "auto_ref_idx": None,
                "use_landmark_align": False,
                "closed_template": True,
                "p95_mar": p95_mar,
            }
        auto_ref_idx = select_auto_ref_frame_index(
            coord_list,
            landmarks_list=landmarks_list,
            placeholder=placeholder,
        )
        return {
            "effective_mode": validated_mode,
            "auto_ref_idx": auto_ref_idx,
            "use_landmark_align": True,
            "closed_template": False,
            "p95_mar": p95_mar,
        }

    return {
        "effective_mode": validated_mode,
        "auto_ref_idx": 0 if validated_mode == "first" else None,
        "use_landmark_align": True,
        "closed_template": False,
        "p95_mar": None,
    }
