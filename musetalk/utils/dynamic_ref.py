"""Dynamic pose-and-time nearest open/closed reference selection and neck healing.

In cross-driven inference on speaking template videos, a single static closed
reference frame loses real teeth identity and fails under larger 3D head turns,
while ``self`` reference leaks source mouth openings during silent pauses.

``ref_frame_mode="dynamic"`` solves both problems plus open-jaw double-chin:
1. Splits valid source frames into ``open_pool`` (clear teeth/articulation,
   preferring frames with visible inner-mouth teeth when source frames are
   provided) and ``closed_pool`` (calm natural closures, rejecting 1-frame
   tight consonant outliers when calm spans exist).
2. Pairs voiced frames & short comma breaths with the pose-and-time nearest
   ``open_pool`` frame (with dwell hysteresis and monotone convex crossfading
   even across interrupted switches), and smoothly routes long pauses
   (``>= min_long_pause_sec``) and leading/trailing silence to ``closed_pool``.
3. Evaluates ``[masked_latent, ref_latent]`` lazily per frame via
   :class:`DynamicLatentSequence` so GPU memory stays ``O(batch_size)``
   regardless of output audio length.
4. Pre-heals open-jaw source frames' lower-chin / upper-neck region with the
   pose-aligned nearest ``closed_pool`` frame inside a semantic skin/neck
   intersection mask to eliminate double-chin and Adam's-apple artifacts.
"""

from collections import OrderedDict
import math

import numpy as np

from musetalk.utils.clarity import gaussian_blur_2d, score_landmark_neutral_ref
from musetalk.utils.ref_align import (
    DEFAULT_MAX_YAW_DELTA,
    DEFAULT_MIN_IN_BOUNDS_RATIO,
    RIGID_UPPER_FACE_INDICES,
    align_reference_face_crop,
    compose_aligned_crop_matrix,
    compute_crop_in_bounds_ratio,
    estimate_rigid_face_similarity,
)

# Minimum inner-mouth bright neutral-pixel fraction indicating visible teeth.
_VISIBLE_TEETH_MIN_SCORE = 0.04


def extract_rigid_pose_descriptor(landmarks_68):
    """Extract mouth openness and signed 5-D rigid head-pose features.

    Uses only rigid upper-face landmarks for pose features so jaw drop during
    speech never perturbs head-pose matching.

    Args:
        landmarks_68 (np.ndarray | None): ``(68, 2)`` DWPose facial landmarks.

    Returns:
        dict | None: Feature mapping with ``mar``, ``signed_yaw``,
            ``signed_pitch``, ``roll_rad``, ``cx``, ``cy``, ``jaw_w``, and
            ``mouth_w_ratio``, or ``None`` if landmarks are invalid.
    """
    if landmarks_68 is None:
        return None
    pts = np.asarray(landmarks_68, dtype=np.float32)
    if pts.shape != (68, 2) or not np.all(np.isfinite(pts)):
        return None

    mouth_w = float(np.linalg.norm(pts[54] - pts[48]))
    jaw_w = float(np.linalg.norm(pts[16] - pts[0]))
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    iod = float(np.linalg.norm(right_eye - left_eye))
    if mouth_w < 1e-6 or jaw_w < 1e-6 or iod < 1e-4:
        return None

    inner_gap = float(
        np.mean([
            np.linalg.norm(pts[61] - pts[67]),
            np.linalg.norm(pts[62] - pts[66]),
            np.linalg.norm(pts[63] - pts[65]),
        ])
    )
    mar = inner_gap / mouth_w
    d_left = float(np.linalg.norm(pts[30] - pts[0]))
    d_right = float(np.linalg.norm(pts[16] - pts[30]))
    signed_yaw = (d_left - d_right) / jaw_w
    roll_rad = float(
        math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0]))
    )
    eye_mid_y = 0.5 * float(left_eye[1] + right_eye[1])
    signed_pitch = (float(pts[30, 1]) - eye_mid_y) / iod
    rigid_center = pts[list(RIGID_UPPER_FACE_INDICES)].mean(axis=0)
    return {
        "mar": mar,
        "signed_yaw": signed_yaw,
        "signed_pitch": signed_pitch,
        "roll_rad": roll_rad,
        "cx": float(rigid_center[0]),
        "cy": float(rigid_center[1]),
        "jaw_w": jaw_w,
        "mouth_w_ratio": mouth_w / jaw_w,
        "teeth_score": None,
    }


def estimate_inner_mouth_teeth_score(frame_bgr, landmarks_68):
    """Estimate visible teeth fraction inside the 68-landmark inner-lip aperture.

    Measures bright, low-saturation pixels inside the inner-lip ROI relative to
    surrounding perioral skin so open frames showing teeth rank above equally
    open frames showing only a dark oral cavity or red lip tissue.

    Args:
        frame_bgr (np.ndarray | None): Source BGR image ``(H, W, 3)``.
        landmarks_68 (np.ndarray | None): ``(68, 2)`` facial landmarks.

    Returns:
        float | None: Fraction in ``[0.0, 1.0]`` of inner-mouth pixels matching
            visible teeth, or ``None`` if image/landmarks cannot be sampled.
    """
    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3 or frame_bgr.shape[2] != 3:
        return None
    if landmarks_68 is None:
        return None
    pts = np.asarray(landmarks_68, dtype=np.float32)
    if pts.shape != (68, 2) or not np.all(np.isfinite(pts)):
        return None

    img_h, img_w = frame_bgr.shape[:2]
    inner = pts[60:68]
    ix1 = max(0, int(math.floor(float(inner[:, 0].min()))))
    ix2 = min(img_w, int(math.ceil(float(inner[:, 0].max()))) + 1)
    iy1 = max(0, int(math.floor(float(inner[:, 1].min()))))
    iy2 = min(img_h, int(math.ceil(float(inner[:, 1].max()))) + 1)
    if (ix2 - ix1) < 3 or (iy2 - iy1) < 2:
        return 0.0

    # Sample cheek/chin skin reference around outer mouth to adapt to studio exposure.
    outer = pts[48:60]
    mw = max(float(outer[:, 0].max() - outer[:, 0].min()), 4.0)
    sx1 = max(0, int(round(float(outer[:, 0].min()) - 0.35 * mw)))
    sx2 = min(img_w, int(round(float(outer[:, 0].max()) + 0.35 * mw)))
    sy1 = max(0, int(round(float(pts[33, 1]))))
    sy2 = min(img_h, int(round(float(outer[:, 1].max()) + 0.25 * mw)))
    if sx2 > sx1 and sy2 > sy1:
        skin_patch = frame_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
        skin_luma = skin_patch[:, :, 0] * 0.114 + skin_patch[:, :, 1] * 0.587 + skin_patch[:, :, 2] * 0.299
        skin_ref = float(np.percentile(skin_luma, 75))
    else:
        skin_ref = 150.0

    roi = frame_bgr[iy1:iy2, ix1:ix2].astype(np.float32)
    luma = roi[:, :, 0] * 0.114 + roi[:, :, 1] * 0.587 + roi[:, :, 2] * 0.299
    val = roi.max(axis=2)
    chroma = val - roi.min(axis=2)
    sat = np.where(val > 1e-5, chroma / np.maximum(val, 1e-5) * 255.0, 0.0)
    red_excess = roi[:, :, 2] - 0.5 * (roi[:, :, 1] + roi[:, :, 0])

    min_tooth_luma = max(85.0, min(195.0, 0.72 * skin_ref))
    teeth_mask = (luma >= min_tooth_luma) & (sat <= 95.0) & (red_excess <= 38.0)
    return float(np.mean(teeth_mask))


def build_mouth_state_pools(
    coord_list,
    landmarks_list,
    placeholder=(0.0, 0.0, 0.0, 0.0),
    frames=None,
):
    """Build ``open_pool`` (teeth/speaking) and ``closed_pool`` (calm neutral).

    Args:
        coord_list (Sequence): Per-frame face bounding boxes ``[x1, y1, x2, y2]``.
        landmarks_list (Sequence): Per-frame ``(68, 2)`` landmark arrays.
        placeholder (tuple): Sentinel value marking missing face detections.
        frames (Sequence | None): Optional indexable BGR frames used to score
            inner-mouth teeth visibility on open-mouth candidates.

    Returns:
        dict: Dictionary containing ``open_pool``, ``closed_pool``,
            ``descriptors``, and ``median_jaw_w``.
    """
    n = len(coord_list)
    descriptors = [None] * n
    valid = []
    for idx in range(n):
        bbox = coord_list[idx]
        if bbox is None or tuple(bbox) == tuple(placeholder):
            continue
        if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
            continue
        lm = landmarks_list[idx] if (landmarks_list is not None and idx < len(landmarks_list)) else None
        desc = extract_rigid_pose_descriptor(lm)
        if desc is not None:
            descriptors[idx] = desc
            valid.append(idx)

    if not valid:
        fallback = [0] if n > 0 else []
        return {
            "open_pool": fallback,
            "closed_pool": fallback,
            "descriptors": descriptors,
            "median_jaw_w": 100.0,
        }

    med_jaw_w = float(np.median([descriptors[i]["jaw_w"] for i in valid]))
    med_mw_ratio = float(np.median([descriptors[i]["mouth_w_ratio"] for i in valid]))

    # Measure contiguous calm closed-mouth run length per frame (mar <= 0.058).
    calm_run_len = [0] * n
    run_start = None
    for idx in range(n + 1):
        is_calm = (
            idx < n
            and descriptors[idx] is not None
            and descriptors[idx]["mar"] <= 0.058
        )
        if is_calm and run_start is None:
            run_start = idx
        elif not is_calm and run_start is not None:
            span = idx - run_start
            for k in range(run_start, idx):
                calm_run_len[k] = span
            run_start = None

    # Closed pool: prefer >=4-frame calm spans over isolated 1-frame /m/ squeezes.
    closed_t1 = [
        i for i in valid
        if descriptors[i]["mar"] <= 0.048 and calm_run_len[i] >= 4
    ]
    relaxed_t1 = [
        i for i in closed_t1
        if abs(descriptors[i]["mouth_w_ratio"] - med_mw_ratio) <= 0.055
    ]
    if len(relaxed_t1) >= 3:
        closed_t1 = relaxed_t1

    if closed_t1:
        closed_pool = closed_t1
    else:
        closed_t2 = [i for i in valid if descriptors[i]["mar"] <= 0.055]
        if closed_t2:
            closed_pool = closed_t2
        else:
            k = max(1, len(valid) // 10)
            closed_pool = sorted(sorted(valid, key=lambda i: descriptors[i]["mar"])[:k])

    # Open pool: clear speaking mouth opening, preferring visible teeth when frames exist.
    open_t1 = [i for i in valid if 0.10 <= descriptors[i]["mar"] <= 0.30]
    if len(open_t1) >= min(3, len(valid)):
        open_pool = open_t1
    else:
        open_t2 = [i for i in valid if 0.07 <= descriptors[i]["mar"] <= 0.36]
        if open_t2:
            open_pool = open_t2
        else:
            k = max(1, len(valid) // 10)
            open_pool = sorted(sorted(valid, key=lambda i: -descriptors[i]["mar"])[:k])

    if frames is not None and open_pool:
        for idx in open_pool:
            try:
                descriptors[idx]["teeth_score"] = estimate_inner_mouth_teeth_score(
                    frames[idx], landmarks_list[idx]
                )
            except Exception:
                descriptors[idx]["teeth_score"] = None
        teeth_visible = [
            idx for idx in open_pool
            if (descriptors[idx]["teeth_score"] or 0.0) >= _VISIBLE_TEETH_MIN_SCORE
        ]
        if len(teeth_visible) >= min(3, len(open_pool)):
            open_pool = teeth_visible

    return {
        "open_pool": closed_pool if not open_pool else open_pool,
        "closed_pool": closed_pool,
        "descriptors": descriptors,
        "median_jaw_w": max(med_jaw_w, 1.0),
    }


def _pose_only_cost(d_src, d_cand, med_jaw_w):
    """Compute dimensionless 5-D rigid pose distance between two frames."""
    dyaw = abs(d_src["signed_yaw"] - d_cand["signed_yaw"])
    dpitch = abs(d_src["signed_pitch"] - d_cand["signed_pitch"])
    droll = abs(d_src["roll_rad"] - d_cand["roll_rad"])
    dpos = math.hypot(d_src["cx"] - d_cand["cx"], d_src["cy"] - d_cand["cy"]) / max(med_jaw_w, 1.0)
    yaw_guard = 10.0 if dyaw > DEFAULT_MAX_YAW_DELTA else 0.0
    return yaw_guard + 2.5 * dyaw + 1.6 * dpitch + 1.5 * droll + 1.2 * dpos


def _candidate_total_cost(src_idx, cand_idx, descriptors, med_jaw_w, fps, role):
    """Combine rigid pose distance, saturated time distance, and mouth/teeth quality."""
    d_src = descriptors[src_idx]
    d_cand = descriptors[cand_idx]
    dt_sec = abs(int(src_idx) - int(cand_idx)) / max(float(fps), 1.0)
    time_cost = 0.08 * min(dt_sec, 4.0) + 0.01 * max(0.0, dt_sec - 4.0)
    if d_src is None or d_cand is None:
        return time_cost + 5.0
    pose_cost = _pose_only_cost(d_src, d_cand, med_jaw_w)
    if role == "open":
        quality_cost = 0.20 * abs(d_cand["mar"] - 0.17)
        t_score = d_cand.get("teeth_score")
        if t_score is not None:
            quality_cost += 0.15 * (1.0 - min(float(t_score) / 0.15, 1.0))
    else:
        quality_cost = 0.60 * max(0.0, d_cand["mar"] - 0.032)
    return pose_cost + time_cost + quality_cost


def select_pose_time_nearest_per_source(num_frames, pools_info, fps=30.0):
    """Select best open and closed reference indices for each source frame.

    Args:
        num_frames (int): Total unique source frames.
        pools_info (dict): Output of :func:`build_mouth_state_pools`.
        fps (float): Source video frame rate.

    Returns:
        tuple[list[int], list[int]]: ``(nearest_open_by_src, nearest_closed_by_src)``.
    """
    open_pool = pools_info["open_pool"]
    closed_pool = pools_info["closed_pool"]
    descriptors = pools_info["descriptors"]
    med_jaw_w = pools_info["median_jaw_w"]

    nearest_open = [0] * num_frames
    nearest_closed = [0] * num_frames
    last_open = open_pool[0] if open_pool else 0
    last_closed = closed_pool[0] if closed_pool else 0

    for s in range(num_frames):
        if descriptors[s] is None:
            nearest_open[s] = last_open
            nearest_closed[s] = last_closed
            continue
        best_o = min(
            open_pool,
            key=lambda j: (_candidate_total_cost(s, j, descriptors, med_jaw_w, fps, "open"), abs(s - j)),
        )
        best_c = min(
            closed_pool,
            key=lambda j: (_candidate_total_cost(s, j, descriptors, med_jaw_w, fps, "closed"), abs(s - j)),
        )
        nearest_open[s] = int(best_o)
        nearest_closed[s] = int(best_c)
        last_open = int(best_o)
        last_closed = int(best_c)
    return nearest_open, nearest_closed


def stabilize_ref_sequence(
    source_index_seq,
    raw_ref_by_source,
    pools_info,
    fps=30.0,
    role="open",
    min_dwell_frames=10,
    hysteresis_margin=0.05,
    max_pose_drift=0.22,
):
    """Suppress frame-to-frame reference chatter while tracking real head turns.

    Args:
        source_index_seq (Sequence[int]): Source frame index per output frame.
        raw_ref_by_source (Sequence[int]): Best candidate per unique source frame.
        pools_info (dict): Output of :func:`build_mouth_state_pools`.
        fps (float): Video frame rate.
        role (str): ``"open"`` or ``"closed"``.
        min_dwell_frames (int): Minimum dwell before non-urgent candidate switch.
        hysteresis_margin (float): Cost improvement required to switch after dwell.
        max_pose_drift (float): Pose-cost ceiling that forces immediate tracking.

    Returns:
        list[int]: Stabilized reference frame index for each output frame.
    """
    if not source_index_seq:
        return []
    descriptors = pools_info["descriptors"]
    med_jaw_w = pools_info["median_jaw_w"]
    cur_ref = int(raw_ref_by_source[source_index_seq[0]])
    dwell = 0
    stabilized = []

    for s in source_index_seq:
        cand_ref = int(raw_ref_by_source[s])
        if cand_ref == cur_ref:
            dwell += 1
            stabilized.append(cur_ref)
            continue

        d_src = descriptors[s]
        d_cur = descriptors[cur_ref]
        urgent_pose = (
            d_src is not None
            and d_cur is not None
            and _pose_only_cost(d_src, d_cur, med_jaw_w) > float(max_pose_drift)
        )
        cur_cost = _candidate_total_cost(s, cur_ref, descriptors, med_jaw_w, fps, role)
        cand_cost = _candidate_total_cost(s, cand_ref, descriptors, med_jaw_w, fps, role)
        if urgent_pose or (dwell >= int(min_dwell_frames) and cand_cost + float(hysteresis_margin) < cur_cost):
            cur_ref = cand_ref
            dwell = 1
        else:
            dwell += 1
        stabilized.append(cur_ref)
    return stabilized


def build_dynamic_closed_weights(
    voiced_intervals,
    total_frames,
    fps=30.0,
    min_long_pause_sec=0.48,
    fade_sec=0.13,
    close_start=True,
    close_end=True,
):
    """Compute per-output-frame ``closed_pool`` reference weight in ``[0, 1]``.

    Voiced segments and short comma breaths (strictly ``< min_long_pause_sec``
    at any frame rate) stay at ``0.0`` so UNet retains teeth identity and
    breathes naturally, while pauses ``>= min_long_pause_sec`` and leading or
    trailing silence cosine-ramp to ``1.0``.

    Args:
        voiced_intervals (Sequence[tuple[int, int]]): Active audio intervals.
        total_frames (int): Output frame count.
        fps (float): Video frame rate.
        min_long_pause_sec (float): Exact minimum silent pause duration in
            seconds that routes UNet's reference latent to ``closed_pool``.
        fade_sec (float): Cosine transition duration (seconds) inside silent edges.
        close_start (bool): Route leading silence to ``closed_pool``.
        close_end (bool): Route trailing silence to ``closed_pool``.

    Returns:
        np.ndarray: ``float32`` weights of shape ``(total_frames,)``.
    """
    n = int(total_frames)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    cleaned = []
    for start, end in voiced_intervals or []:
        s = max(0, min(n, int(start)))
        e = max(0, min(n, int(end)))
        if e > s:
            cleaned.append((s, e))
    if not cleaned:
        return np.ones(n, dtype=np.float32)

    cleaned.sort()
    merged = [list(cleaned[0])]
    for s, e in cleaned[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    eff_fps = max(float(fps), 1.0)
    min_pause_frames = max(1, int(math.ceil(float(min_long_pause_sec) * eff_fps - 1e-9)))
    fade_frames = max(2, int(round(float(fade_sec) * eff_fps)))
    weights = np.zeros(n, dtype=np.float32)

    def _fill_span(g0, g1, fade_left, fade_right):
        length = g1 - g0
        if length <= 0:
            return
        span_w = np.ones(length, dtype=np.float32)
        fl = min(fade_frames, length // 2 if fade_right else length) if fade_left else 0
        fr = min(fade_frames, length // 2 if fade_left else length) if fade_right else 0
        if fl > 0:
            t = (np.arange(fl, dtype=np.float32) + 1.0) / float(fl + 1)
            span_w[:fl] = 0.5 * (1.0 - np.cos(np.pi * t))
        if fr > 0:
            t = (np.arange(fr, dtype=np.float32) + 1.0) / float(fr + 1)
            span_w[-fr:] = np.minimum(span_w[-fr:], 0.5 * (1.0 - np.cos(np.pi * t))[::-1])
        weights[g0:g1] = np.maximum(weights[g0:g1], span_w)

    if close_start and merged[0][0] > 0:
        _fill_span(0, merged[0][0], fade_left=False, fade_right=True)
    for idx in range(len(merged) - 1):
        g0, g1 = merged[idx][1], merged[idx + 1][0]
        if (g1 - g0) >= min_pause_frames:
            _fill_span(g0, g1, fade_left=True, fade_right=True)
    if close_end and merged[-1][1] < n:
        _fill_span(merged[-1][1], n, fade_left=True, fade_right=False)
    return weights


def build_dynamic_reference_plan(
    coord_list,
    landmarks_list,
    frame_index_cycle,
    total_output_frames,
    voiced_intervals,
    fps=30.0,
    placeholder=(0.0, 0.0, 0.0, 0.0),
    frames=None,
):
    """Build complete dynamic open/closed reference schedule for ``inference.py``.

    Args:
        coord_list (Sequence): Per-source-frame bounding boxes.
        landmarks_list (Sequence): Per-source-frame ``(68, 2)`` landmarks.
        frame_index_cycle (Sequence[int]): Ping-pong source index cycle.
        total_output_frames (int): Output video frame count (``len(whisper_chunks)``).
        voiced_intervals (Sequence[tuple[int, int]]): Active speech intervals.
        fps (float): Video frame rate.
        placeholder (tuple): Missing-detection sentinel bbox.
        frames (Sequence | None): Optional source BGR frames for teeth scoring.

    Returns:
        dict: Schedule containing pools, per-source nearest maps, stabilized
            per-output-frame ``open_ref_seq`` / ``closed_ref_seq``, and
            ``closed_weights``.
    """
    pools_info = build_mouth_state_pools(
        coord_list, landmarks_list, placeholder=placeholder, frames=frames
    )
    nearest_open, nearest_closed = select_pose_time_nearest_per_source(
        len(coord_list), pools_info, fps=fps
    )
    cycle_len = len(frame_index_cycle)
    source_seq = [int(frame_index_cycle[i % cycle_len]) for i in range(int(total_output_frames))]
    open_ref_seq = stabilize_ref_sequence(
        source_seq, nearest_open, pools_info, fps=fps, role="open"
    )
    closed_ref_seq = stabilize_ref_sequence(
        source_seq, nearest_closed, pools_info, fps=fps, role="closed"
    )
    closed_weights = build_dynamic_closed_weights(
        voiced_intervals, total_output_frames, fps=fps
    )
    return {
        "pools_info": pools_info,
        "nearest_open_by_source": nearest_open,
        "nearest_closed_by_source": nearest_closed,
        "source_seq": source_seq,
        "open_ref_seq": open_ref_seq,
        "closed_ref_seq": closed_ref_seq,
        "closed_weights": closed_weights,
    }


def _track_branch_distributions(seq, crossfade_frames):
    """Build per-frame convex ``{ref_idx: weight}`` dicts across arbitrary switches.

    When a switch ``A -> B`` is interrupted mid-fade by ``-> C``, snapshots the
    exact in-progress blend state ``start_dist`` so ``A`` and ``B`` decay
    monotonically while ``C`` ramps in with zero step discontinuity.
    """
    if not seq:
        return []
    k = max(0, int(crossfade_frames))
    cur_target = int(seq[0])
    cur_dist = {cur_target: 1.0}
    start_dist = None
    step = 0
    out = []

    for raw_r in seq:
        r = int(raw_r)
        if r != cur_target:
            if k > 0:
                start_dist = dict(cur_dist)
                cur_target = r
                step = 1
            else:
                cur_target = r
                cur_dist = {r: 1.0}
                start_dist = None
                step = 0

        if start_dist is not None and 1 <= step <= k:
            alpha = float(step) / float(k + 1)
            blended = {}
            for idx_k, w_k in start_dist.items():
                val = float(w_k) * (1.0 - alpha)
                if val > 1e-5:
                    blended[idx_k] = val
            blended[cur_target] = blended.get(cur_target, 0.0) + alpha
            total = sum(blended.values())
            cur_dist = {idx_k: v / total for idx_k, v in blended.items()}
            step += 1
            if step > k:
                start_dist = None
                step = 0
        else:
            cur_dist = {cur_target: 1.0}

        out.append(cur_dist)
    return out


class DynamicLatentSequence:
    """Lazy sequence of ``[masked_latent, ref_latent]`` pairs keeping GPU VRAM O(1).

    Precomputes only lightweight scalar blend weights ``((ref_idx, weight), ...)``
    per output frame and evaluates tensors on demand when indexed by ``datagen``.
    """

    def __init__(self, masked_latents_by_source, dynamic_plan, encode_ref_fn, crossfade_frames=4):
        """Initialize lazy dynamic latent sequence."""
        self._masked = masked_latents_by_source
        self._source_seq = list(dynamic_plan["source_seq"])
        self._encode_ref_fn = encode_ref_fn
        self._frame_mixes = []

        if not self._source_seq:
            self._is_numpy = True
            self._cat_fn = None
            return

        sample = self._masked[0]
        self._is_numpy = isinstance(sample, np.ndarray)
        self._cat_fn = None if self._is_numpy else __import__("torch").cat

        open_dists = _track_branch_distributions(dynamic_plan["open_ref_seq"], crossfade_frames)
        closed_dists = _track_branch_distributions(dynamic_plan["closed_ref_seq"], crossfade_frames)
        closed_weights = dynamic_plan["closed_weights"]

        for i in range(len(self._source_seq)):
            w = float(closed_weights[i])
            if w <= 0.0:
                merged = open_dists[i]
            elif w >= 1.0:
                merged = closed_dists[i]
            else:
                merged = {}
                for r_idx, rw in open_dists[i].items():
                    if rw * (1.0 - w) > 1e-5:
                        merged[r_idx] = merged.get(r_idx, 0.0) + rw * (1.0 - w)
                for r_idx, rw in closed_dists[i].items():
                    if rw * w > 1e-5:
                        merged[r_idx] = merged.get(r_idx, 0.0) + rw * w
            total = sum(merged.values()) or 1.0
            self._frame_mixes.append(
                tuple((int(r_idx), float(rw / total)) for r_idx, rw in sorted(merged.items()))
            )

    def __len__(self):
        """Return number of output video frames."""
        return len(self._source_seq)

    def __getitem__(self, index):
        """Assemble ``[masked_latent, ref_latent]`` on demand for frame ``index``."""
        if index < 0:
            index += len(self._source_seq)
        if index < 0 or index >= len(self._source_seq):
            raise IndexError(f"DynamicLatentSequence index {index} out of range")

        s_idx = self._source_seq[index]
        masked_lat = self._masked[s_idx]
        mix = self._frame_mixes[index]

        ref_lat = None
        for r_idx, weight in mix:
            part = self._encode_ref_fn(s_idx, r_idx)
            scaled = part if len(mix) == 1 else (part * weight)
            ref_lat = scaled if ref_lat is None else (ref_lat + scaled)

        if self._is_numpy:
            return np.concatenate([masked_lat, ref_lat.astype(masked_lat.dtype, copy=False)], axis=1)
        ref_lat = ref_lat.to(device=masked_lat.device, dtype=masked_lat.dtype)
        return self._cat_fn([masked_lat, ref_lat], dim=1)


def make_resilient_ref_encoder(
    frames,
    landmarks_list,
    adjusted_boxes_by_source,
    self_ref_crops_by_source,
    resolution,
    encode_crop_fn,
    initial_fallback_latent,
    align_counts=None,
    align_fn=align_reference_face_crop,
    max_cache_size=256,
):
    """Build a fault-tolerant, CPU-LRU-cached ``(src_idx, ref_idx)`` latent encoder.

    Catches per-frame alignment or VAE encoding exceptions and falls back first
    to the target frame's own crop and then to ``last_valid_ref_latent`` so a
    single corrupted crop never aborts full-video rendering.
    """
    cache = OrderedDict()
    cap = max(1, int(max_cache_size))
    last_valid = [initial_fallback_latent]

    def _encode(src_idx, ref_idx):
        key = (int(src_idx), int(ref_idx))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        target_box = adjusted_boxes_by_source[src_idx]
        self_crop = self_ref_crops_by_source[src_idx]
        ref_crop = None
        used_self = False

        if src_idx == ref_idx:
            ref_crop = self_crop
            used_self = True
        elif target_box is not None:
            try:
                aligned_crop, status = align_fn(
                    frames[ref_idx],
                    landmarks_list[ref_idx],
                    landmarks_list[src_idx],
                    target_box,
                    resolution,
                )
            except Exception:
                aligned_crop, status = None, "fallback_self"
            if status == "aligned" and aligned_crop is not None:
                ref_crop = aligned_crop
            else:
                ref_crop = self_crop

        primary_tag = (
            "self"
            if used_self
            else ("aligned" if (ref_crop is not None and ref_crop is not self_crop) else "fallback_self")
        )
        candidates = [(ref_crop, primary_tag)]
        if ref_crop is not self_crop and self_crop is not None:
            candidates.append((self_crop, "fallback_self"))

        lat = None
        chosen_tag = "fallback_self"
        for crop_img, tag in candidates:
            if crop_img is None:
                continue
            try:
                lat = encode_crop_fn(crop_img)
                chosen_tag = tag
                last_valid[0] = lat
                break
            except Exception:
                lat = None

        if lat is None:
            lat = last_valid[0]
            chosen_tag = "fallback_self"

        if align_counts is not None:
            align_counts[chosen_tag] = align_counts.get(chosen_tag, 0) + 1

        cache[key] = lat
        if len(cache) > cap:
            cache.popitem(last=False)
        return lat

    return _encode


def build_dynamic_unet_latents(
    masked_latents_by_source,
    dynamic_plan,
    encode_ref_fn,
    crossfade_frames=4,
):
    """Return a lazy :class:`DynamicLatentSequence` for ``datagen`` consumption."""
    return DynamicLatentSequence(
        masked_latents_by_source,
        dynamic_plan,
        encode_ref_fn,
        crossfade_frames=crossfade_frames,
    )


def compute_healed_neck_roi(
    ori_frame,
    ori_landmarks,
    face_box,
    ref_closed_frame,
    ref_closed_landmarks,
    fp=None,
):
    """Pre-heal an open-jaw source frame's lower-chin and upper-neck ROI.

    Replaces the dropped open-jaw chin shadow on ``ori_frame``'s neck with the
    pose-aligned clean neck from ``ref_closed_frame``, restricted to the semantic
    skin+neck intersection so hair, clothing, and green-screen background are
    never touched.

    Args:
        ori_frame (np.ndarray): Target source BGR frame ``(H, W, 3)``.
        ori_landmarks (np.ndarray | None): ``(68, 2)`` landmarks of ``ori_frame``.
        face_box (Sequence[int]): Adjusted crop box ``(x1, y1, x2, y2)``.
        ref_closed_frame (np.ndarray): Closed-mouth BGR frame ``(H, W, 3)``.
        ref_closed_landmarks (np.ndarray | None): ``(68, 2)`` landmarks of ``ref_closed_frame``.
        fp (Callable | None): Optional ``FaceParsing`` callable supporting ``mode="neck"``.

    Returns:
        tuple[tuple[int, int, int, int], np.ndarray] | None: ``((rx1, ry1, rx2, ry2), healed_roi)``
            or ``None`` when ``ori_frame``'s jaw is already closed or unaligned.
    """
    ori_scored = score_landmark_neutral_ref(ori_landmarks)
    ref_scored = score_landmark_neutral_ref(ref_closed_landmarks)
    if ori_scored is None or ref_scored is None:
        return None

    mar_ori, mar_ref = float(ori_scored[0]), float(ref_scored[0])
    strength = float(np.clip((mar_ori - 0.040) / 0.030, 0.0, 1.0))
    if strength <= 1e-3 or mar_ori <= mar_ref + 0.012:
        return None

    sim_matrix, status = estimate_rigid_face_similarity(ref_closed_landmarks, ori_landmarks)
    if status != "aligned" or sim_matrix is None:
        return None

    img_h, img_w = ori_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_box]
    fw, fh = float(x2 - x1), float(y2 - y1)
    if fw <= 4.0 or fh <= 4.0:
        return None

    cx = 0.5 * (x1 + x2)
    rx1 = max(0, int(round(cx - 0.85 * fw)))
    rx2 = min(img_w, int(round(cx + 0.85 * fw)))
    ry1 = max(0, int(round(y1 + 0.35 * fh)))
    ry2 = min(img_h, int(round(y2 + 0.35 * fh)))
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 4 or rh <= 4:
        return None

    crop_matrix = compose_aligned_crop_matrix(sim_matrix, (rx1, ry1, rx2, ry2), (rw, rh))
    if compute_crop_in_bounds_ratio(crop_matrix, (rw, rh), ref_closed_frame.shape) < DEFAULT_MIN_IN_BOUNDS_RATIO:
        return None

    import cv2
    from PIL import Image

    warped_ref = cv2.warpAffine(
        ref_closed_frame,
        crop_matrix,
        (rw, rh),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    roi_tgt = ori_frame[ry1:ry2, rx1:rx2]

    ys = np.arange(ry1, ry2, dtype=np.float32)
    xs = np.arange(rx1, rx2, dtype=np.float32)
    t_top = np.clip((ys - (y1 + 0.50 * fh)) / max(0.12 * fh, 1.0), 0.0, 1.0)
    t_bot = np.clip(((y2 + 0.28 * fh) - ys) / max(0.16 * fh, 1.0), 0.0, 1.0)
    wy = (0.5 * (1.0 - np.cos(np.pi * t_top))) * (0.5 * (1.0 - np.cos(np.pi * t_bot)))

    dx = np.abs(xs - cx)
    tx = np.clip((0.78 * fw - dx) / max(0.23 * fw, 1.0), 0.0, 1.0)
    wx = 0.5 * (1.0 - np.cos(np.pi * tx))

    valid_skin_neck = np.ones((rh, rw), dtype=bool)
    soft_sem = np.ones((rh, rw), dtype=np.float32)
    if fp is not None:
        m_tgt = np.asarray(fp(Image.fromarray(roi_tgt[:, :, ::-1]), mode="neck").resize((rw, rh), Image.BILINEAR))
        m_ref = np.asarray(fp(Image.fromarray(warped_ref[:, :, ::-1]), mode="neck").resize((rw, rh), Image.BILINEAR))
        valid_skin_neck = (m_tgt > 127) & (m_ref > 127)
        if not np.any(valid_skin_neck):
            return None
        blurred = gaussian_blur_2d(valid_skin_neck.astype(np.float32), 2.0)
        soft_sem = np.where(valid_skin_neck, np.clip((blurred - 0.5) * 2.0, 0.0, 1.0), 0.0).astype(np.float32)

    alpha = (strength * wy[:, np.newaxis] * wx[np.newaxis, :] * soft_sem).astype(np.float32)
    if float(alpha.max()) <= 1e-3:
        return None

    tgt_f = roi_tgt.astype(np.float32)
    ref_f = warped_ref.astype(np.float32)
    tone_ring = valid_skin_neck & ((ys >= (y2 + 0.08 * fh))[:, np.newaxis]) & ((ys <= (y2 + 0.26 * fh))[:, np.newaxis])
    if int(np.count_nonzero(tone_ring)) >= 25:
        delta_bgr = np.clip(np.median(tgt_f[tone_ring] - ref_f[tone_ring], axis=0), -10.0, 10.0)
        ref_f = np.clip(ref_f + delta_bgr, 0.0, 255.0)

    blended = tgt_f * (1.0 - alpha[:, :, np.newaxis]) + ref_f * alpha[:, :, np.newaxis]
    return (rx1, ry1, rx2, ry2), np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def apply_healed_neck_roi(ori_frame, roi_patch):
    """Return ``ori_frame`` with ``roi_patch`` pasted, or ``ori_frame`` if ``None``."""
    if roi_patch is None:
        return ori_frame
    (rx1, ry1, rx2, ry2), patch = roi_patch
    out = ori_frame.copy()
    out[ry1:ry2, rx1:rx2] = patch
    return out
