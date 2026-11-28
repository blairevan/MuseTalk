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
4. Pre-heals open-jaw source frames' dropped-chin bulge and sub-mental crease
   using same-frame upper-throat skin continuation inside a semantic skin/neck
   mask (with zero cross-frame neck warping), eliminating double-chin and
   Adam's-apple artifacts while keeping the neck 100% stationary.
"""

from collections import OrderedDict
import math

import numpy as np

from musetalk.utils.clarity import (
    gaussian_blur_2d,
    gaussian_kernel_1d,
    score_landmark_neutral_ref,
)
from musetalk.utils.ref_align import (
    DEFAULT_MAX_YAW_DELTA,
    RIGID_UPPER_FACE_INDICES,
    align_reference_face_crop,
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
    neutral_chin_ratio, neutral_mar = compute_neutral_chin_stats(
        landmarks_list, pools_info["closed_pool"]
    )
    chin_heal_strengths = build_smoothed_chin_heal_strengths(
        landmarks_list,
        neutral_chin_ratio=neutral_chin_ratio,
        neutral_mar=neutral_mar,
    )
    return {
        "pools_info": pools_info,
        "nearest_open_by_source": nearest_open,
        "nearest_closed_by_source": nearest_closed,
        "source_seq": source_seq,
        "open_ref_seq": open_ref_seq,
        "closed_ref_seq": closed_ref_seq,
        "closed_weights": closed_weights,
        "neutral_chin_ratio": neutral_chin_ratio,
        "neutral_mar": neutral_mar,
        "chin_heal_strengths": chin_heal_strengths,
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


def _extract_chin_pose_metrics(landmarks_68):
    """Extract roll-invariant eye-to-chin projection, IOD, eye midpoint, and MAR."""
    if landmarks_68 is None:
        return None
    pts = np.asarray(landmarks_68, dtype=np.float32)
    if pts.shape != (68, 2) or not np.all(np.isfinite(pts)):
        return None
    scored = score_landmark_neutral_ref(pts)
    if scored is None:
        return None
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    eye_vec = right_eye - left_eye
    iod = float(np.linalg.norm(eye_vec))
    if iod < 1e-3:
        return None
    u_axis = eye_vec / iod
    down_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float32)
    eye_mid = 0.5 * (left_eye + right_eye)
    chin_proj = float(np.dot(pts[8] - eye_mid, down_axis))
    return {
        "pts": pts,
        "mar": float(scored[0]),
        "iod": iod,
        "eye_mid": eye_mid,
        "down_axis": down_axis,
        "chin_proj": chin_proj,
        "chin_ratio": chin_proj / iod,
    }


def compute_neutral_chin_stats(landmarks_list, closed_pool=None):
    """Estimate calm closed-mouth ``(neutral_chin_ratio, neutral_mar)`` from source landmarks.

    Args:
        landmarks_list (Sequence | None): Per-source-frame ``(68, 2)`` landmarks.
        closed_pool (Sequence[int] | None): Indices of calm closed-mouth frames.

    Returns:
        tuple[float, float]: ``(neutral_chin_ratio, neutral_mar)``.
    """
    if not landmarks_list:
        return 1.75, 0.025
    indices = list(closed_pool) if closed_pool else list(range(len(landmarks_list)))
    ratios, mars = [], []
    for idx in indices:
        if 0 <= int(idx) < len(landmarks_list):
            m = _extract_chin_pose_metrics(landmarks_list[int(idx)])
            if m is not None:
                ratios.append(m["chin_ratio"])
                mars.append(m["mar"])
    if not ratios:
        for lm in landmarks_list:
            m = _extract_chin_pose_metrics(lm)
            if m is not None:
                ratios.append(m["chin_ratio"])
                mars.append(m["mar"])
    if not ratios:
        return 1.75, 0.025
    return float(np.median(ratios)), float(np.median(mars))


def _raw_chin_heal_strength(metrics, neutral_chin_ratio, neutral_mar):
    """Compute instantaneous open-jaw sub-chin heal strength in ``[0, 1]``."""
    mar_gate = float(np.clip((metrics["mar"] - (float(neutral_mar) + 0.010)) / 0.030, 0.0, 1.0))
    if mar_gate <= 1e-4:
        return 0.0
    chin_drop_norm = max(0.0, metrics["chin_ratio"] - float(neutral_chin_ratio))
    mar_drop_norm = max(0.0, metrics["mar"] - float(neutral_mar)) * 0.55
    eff_drop_norm = max(chin_drop_norm, mar_drop_norm)
    drop_ramp = float(np.clip((eff_drop_norm - 0.015) / 0.075, 0.0, 1.0))
    return mar_gate * drop_ramp


def build_smoothed_chin_heal_strengths(
    landmarks_list,
    neutral_chin_ratio=1.75,
    neutral_mar=0.025,
):
    """Compute 5-frame temporally smoothed sub-chin heal strengths per source frame.

    Carries the last valid strength across missing-landmark frames and pads
    edges symmetrically so ping-pong cycle turnarounds remain continuous.

    Args:
        landmarks_list (Sequence | None): Per-source-frame ``(68, 2)`` landmarks.
        neutral_chin_ratio (float): Closed-mouth ``chin_proj / iod`` baseline.
        neutral_mar (float): Closed-mouth inner-lip MAR baseline.

    Returns:
        np.ndarray: ``float32`` array of shape ``(len(landmarks_list),)``.
    """
    n = len(landmarks_list) if landmarks_list is not None else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    raw = np.zeros(n, dtype=np.float32)
    last_val = 0.0
    for idx in range(n):
        m = _extract_chin_pose_metrics(landmarks_list[idx])
        if m is not None:
            last_val = _raw_chin_heal_strength(m, neutral_chin_ratio, neutral_mar)
        raw[idx] = last_val

    if n < 3:
        return raw
    kernel = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=np.float32)
    padded = np.pad(raw, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _smooth_horizontal_profile(profile_bgr, sigma=3.0):
    """Smooth a ``(W, 3)`` BGR profile along X with a 1-D Gaussian in pure NumPy."""
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(profile_bgr.astype(np.float32, copy=False), ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(profile_bgr, dtype=np.float32)
    width = profile_bgr.shape[0]
    for offset, weight in enumerate(kernel):
        out += float(weight) * padded[offset : offset + width]
    return out


def compute_healed_neck_roi(
    ori_frame,
    ori_landmarks,
    face_box,
    ref_closed_frame=None,
    ref_closed_landmarks=None,
    fp=None,
    neutral_chin_ratio=None,
    neutral_mar=None,
    strength_override=None,
):
    """Pre-heal an open-jaw source frame's dropped-chin bulge and sub-mental crease.

    Projects ``ori_frame``'s own clean upper-throat skin (immediately below the
    dropped-chin rim in the same frame) upward across the narrow sub-chin band
    ``[top_y, open_chin_y + 0.035 * fh]`` within ``chin_x +/- 0.34 * fw``.
    Performs zero cross-frame warping and modifies zero pixels on the neck below
    ``open_chin_y + 0.035 * fh`` or on the left/right neck borders, eliminating
    double-chin / Adam's-apple artifacts with zero neck jitter.

    Args:
        ori_frame (np.ndarray): Target source BGR frame ``(H, W, 3)``.
        ori_landmarks (np.ndarray | None): ``(68, 2)`` landmarks of ``ori_frame``.
        face_box (Sequence[int]): Adjusted crop box ``(x1, y1, x2, y2)``.
        ref_closed_frame (np.ndarray | None): Unused legacy parameter kept for
            API compatibility (never warped).
        ref_closed_landmarks (np.ndarray | None): Optional closed-mouth landmarks
            used to derive ``neutral_chin_ratio`` / ``neutral_mar`` when omitted.
        fp (Callable | None): Optional ``FaceParsing`` callable supporting ``mode="neck"``.
        neutral_chin_ratio (float | None): Closed-mouth ``chin_proj / iod`` baseline.
        neutral_mar (float | None): Closed-mouth inner-lip MAR baseline.
        strength_override (float | None): Optional temporally smoothed strength in ``[0, 1]``.

    Returns:
        tuple[tuple[int, int, int, int], np.ndarray] | None: ``((rx1, ry1, rx2, ry2), healed_roi)``
            or ``None`` when ``ori_frame``'s jaw is closed, invalid, or occluded by clothing.
    """
    del ref_closed_frame
    if ori_frame is None or getattr(ori_frame, "ndim", 0) != 3 or ori_frame.shape[2] != 3:
        return None
    ori_m = _extract_chin_pose_metrics(ori_landmarks)
    if ori_m is None:
        return None

    if neutral_chin_ratio is None or neutral_mar is None:
        ref_m = _extract_chin_pose_metrics(ref_closed_landmarks)
        if neutral_chin_ratio is None:
            neutral_chin_ratio = ref_m["chin_ratio"] if ref_m is not None else 1.75
        if neutral_mar is None:
            neutral_mar = ref_m["mar"] if ref_m is not None else 0.025

    if strength_override is not None:
        strength = float(np.clip(strength_override, 0.0, 1.0))
    else:
        strength = _raw_chin_heal_strength(ori_m, neutral_chin_ratio, neutral_mar)
    if strength <= 1e-3:
        return None

    img_h, img_w = ori_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_box]
    fw, fh = float(x2 - x1), float(y2 - y1)
    if fw <= 8.0 or fh <= 8.0:
        return None

    pts = ori_m["pts"]
    chin_x = float(pts[8, 0])
    open_chin_y = float(pts[8, 1])
    neutral_chin_y = float(
        ori_m["eye_mid"][1] + float(neutral_chin_ratio) * ori_m["iod"] * float(ori_m["down_axis"][1])
    )

    top_y = min(neutral_chin_y - 0.04 * fh, open_chin_y - 0.14 * fh)
    bot_y = open_chin_y + 0.035 * fh
    rx1 = max(0, int(round(chin_x - 0.34 * fw)))
    rx2 = min(img_w, int(round(chin_x + 0.34 * fw)))
    ry1 = max(0, int(round(top_y)))
    ry2 = min(img_h, int(round(bot_y)))
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 6 or rh <= 4:
        return None

    sy1 = max(0, min(img_h - 2, int(round(open_chin_y + 0.045 * fh))))
    sy2 = max(sy1 + 2, min(img_h, int(round(open_chin_y + 0.115 * fh))))
    if sy2 - sy1 < 2:
        return None

    soft_sem = np.ones((rh, rw), dtype=np.float32)
    m_throat = None
    if fp is not None:
        from PIL import Image

        pad = int(round(0.25 * fw))
        px1 = max(0, min(x1, rx1) - pad)
        py1 = max(0, min(y1, ry1))
        px2 = min(img_w, max(x2, rx2) + pad)
        py2 = min(img_h, max(ry2, sy2) + pad)
        pw, ph = px2 - px1, py2 - py1
        if pw <= 4 or ph <= 4:
            return None
        m_full = np.asarray(
            fp(Image.fromarray(ori_frame[py1:py2, px1:px2, ::-1]), mode="neck").resize(
                (pw, ph), Image.BILINEAR
            )
        )
        m_roi = m_full[ry1 - py1 : ry2 - py1, rx1 - px1 : rx2 - px1] > 127
        m_throat = m_full[sy1 - py1 : sy2 - py1, rx1 - px1 : rx2 - px1] > 127
        if not np.any(m_roi) or int(np.count_nonzero(m_throat)) < max(8, int(0.20 * m_throat.size)):
            return None
        blurred = gaussian_blur_2d(m_roi.astype(np.float32), 2.0)
        soft_sem = np.where(m_roi, np.clip((blurred - 0.4) * 1.8, 0.0, 1.0), 0.0).astype(np.float32)

    throat_strip = ori_frame[sy1:sy2, rx1:rx2].astype(np.float32)
    if m_throat is not None:
        col_med = np.zeros((rw, 3), dtype=np.float32)
        valid_cols = []
        for col in range(rw):
            col_mask = m_throat[:, col]
            if np.any(col_mask):
                col_med[col] = np.median(throat_strip[col_mask, col], axis=0)
                valid_cols.append(col)
        if not valid_cols:
            return None
        if len(valid_cols) < rw:
            vx = np.asarray(valid_cols, dtype=np.float32)
            all_x = np.arange(rw, dtype=np.float32)
            for ch in range(3):
                col_med[:, ch] = np.interp(all_x, vx, col_med[valid_cols, ch])
    else:
        col_med = np.median(throat_strip, axis=0)

    col_smooth = _smooth_horizontal_profile(col_med, sigma=3.0)
    ys = np.arange(ry1, ry2, dtype=np.float32)
    xs = np.arange(rx1, rx2, dtype=np.float32)
    crease_anchor_y = min(neutral_chin_y, open_chin_y - 0.06 * fh) + 0.02 * fh
    shade = 1.0 - 0.035 * np.exp(-0.5 * ((ys - crease_anchor_y) / max(0.03 * fh, 2.0)) ** 2)
    fill_bgr = col_smooth[np.newaxis, :, :] * shade[:, np.newaxis, np.newaxis]

    t_top = np.clip((ys - top_y) / max(0.05 * fh, 2.0), 0.0, 1.0)
    t_bot = np.clip((bot_y - ys) / max(0.035 * fh, 2.0), 0.0, 1.0)
    wy = (0.5 * (1.0 - np.cos(np.pi * t_top))) * (0.5 * (1.0 - np.cos(np.pi * t_bot)))

    dx = np.abs(xs - chin_x)
    tx = np.clip((0.33 * fw - dx) / max(0.11 * fw, 2.0), 0.0, 1.0)
    wx = 0.5 * (1.0 - np.cos(np.pi * tx))

    alpha = np.clip(strength * wy[:, np.newaxis] * wx[np.newaxis, :] * soft_sem, 0.0, 1.0).astype(np.float32)
    if float(alpha.max()) <= 1e-3:
        return None

    roi_tgt = ori_frame[ry1:ry2, rx1:rx2].astype(np.float32)
    blended = roi_tgt * (1.0 - alpha[:, :, np.newaxis]) + fill_bgr * alpha[:, :, np.newaxis]
    return (rx1, ry1, rx2, ry2), np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def apply_healed_neck_roi(ori_frame, roi_patch):
    """Return ``ori_frame`` with ``roi_patch`` pasted, or ``ori_frame`` if ``None``."""
    if roi_patch is None:
        return ori_frame
    (rx1, ry1, rx2, ry2), patch = roi_patch
    out = ori_frame.copy()
    out[ry1:ry2, rx1:rx2] = patch
    return out
