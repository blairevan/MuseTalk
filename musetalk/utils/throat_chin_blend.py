"""Jaw-tracked lower-face detail preservation, elastic throat lift, and chin blending.

Addresses three coupled cross-driven digital-human rendering artifacts without
requiring OpenCV (all pixel math uses pure NumPy and PIL):
1. ``256x256`` SD-VAE (`32x32` latents) erases small lower-face identity marks
   such as moles below the mouth and skin pores. :func:`restore_jaw_skin_details`
   registers the source frame's lower-jaw skin vertically to the generated chin
   height and transfers high-frequency skin details via normalized skin-only
   convolution while strictly excluding dilated lips and oral cavity.
2. When a speaking template's source jaw is more open than the generated jaw,
   the chin's cast shadow on the throat sits too low and bobs to the old speech
   rhythm. :func:`lift_throat_shadow_same_frame` elastically lifts the same
   frame's real sub-chin shadow and upper throat vertically (``dx == 0``) with
   non-skin safety clamping so hair, clothing, and neck borders never shift.
3. :func:`composite_dynamic_chin_clean` composites along the generated face's
   true jawline (``classes 1, 11, 12, 13`` with mouth-centered horizontal
   cheek-silhouette protection), blocking synthetic ``class 14`` neck spillover
   onto the throat.
"""

from collections import OrderedDict
import math

import numpy as np
from PIL import Image

from musetalk.utils.clarity import gaussian_blur_2d
from musetalk.utils.mask_utils import apply_side_protect_mask, confine_mask_below

_FACE_AND_LIP_CLASSES = (1, 11, 12, 13)
_LIP_AND_MOUTH_CLASSES = (11, 12, 13)
_SKIN_AND_NECK_CLASSES = (1, 11, 12, 13, 14)
_TEMPORAL_5TAP = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=np.float32)

MAX_SAFE_CHEEK_WIDTH = 90
MIN_SAFE_CHEEK_WIDTH = 20
MIN_SAFE_SIDE_PROTECT_RATIO = 0.15
MAX_SAFE_SIDE_PROTECT_RATIO = 0.45


def clamp_blend_safety_params(
    left_cheek_width,
    right_cheek_width,
    side_protect_ratio,
    force_min_side_protect=True,
):
    """Clamp CLI cheek-width and side-protect values into seam-safe bounds.

    Prevents legacy external callers (e.g. passing ``--left_cheek_width 120``
    and ``--side_protect_ratio 0.08``) from pushing the blend mask onto outer
    cheek-to-hair silhouettes.

    Args:
        left_cheek_width (int | float): Requested left cheek width in 512-space.
        right_cheek_width (int | float): Requested right cheek width in 512-space.
        side_protect_ratio (float): Requested side-edge fade ratio.
        force_min_side_protect (bool): Enforce ``>= MIN_SAFE_SIDE_PROTECT_RATIO``
            even when ``side_protect_ratio <= 0``.

    Returns:
        tuple[int, int, float]: ``(safe_left_w, safe_right_w, safe_side_ratio)``.
    """
    safe_l = int(round(float(np.clip(left_cheek_width, MIN_SAFE_CHEEK_WIDTH, MAX_SAFE_CHEEK_WIDTH))))
    safe_r = int(round(float(np.clip(right_cheek_width, MIN_SAFE_CHEEK_WIDTH, MAX_SAFE_CHEEK_WIDTH))))
    raw_side = float(side_protect_ratio)
    if force_min_side_protect or raw_side > 0.0:
        safe_side = float(np.clip(raw_side, MIN_SAFE_SIDE_PROTECT_RATIO, MAX_SAFE_SIDE_PROTECT_RATIO))
    else:
        safe_side = 0.0
    return safe_l, safe_r, safe_side


_DISK_RADIUS2_OFFSETS = tuple(
    (dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dx * dx + dy * dy) <= 4
)


def _erode_disk_radius2(mask_bool):
    """Exact 2-pixel Euclidean disk erosion on a 2-D boolean array in pure NumPy."""
    h, w = mask_bool.shape
    padded = np.pad(mask_bool, ((2, 2), (2, 2)), mode="edge")
    out = np.ones((h, w), dtype=bool)
    for dy, dx in _DISK_RADIUS2_OFFSETS:
        out &= padded[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
    return out


def _erode_horizontal_1d(mask_bool, radius_px):
    """1-D horizontal erosion keeping pixels >= ``radius_px`` inside left/right false borders."""
    r = max(0, int(round(float(radius_px))))
    if r <= 0:
        return mask_bool.copy()
    h, w = mask_bool.shape
    # Pad horizontal borders with False so face edges touching crop boundaries also erode inward
    padded = np.pad(mask_bool, ((0, 0), (r, r)), mode="constant", constant_values=False)
    csum = np.cumsum(padded.astype(np.int32), axis=1)
    span = 2 * r + 1
    win_sum = csum[:, span - 1 : span - 1 + w] - np.concatenate(
        [np.zeros((h, 1), dtype=np.int32), csum[:, : w - 1]], axis=1
    )
    return win_sum == span


def _dilate_isotropic(mask_bool, radius_px):
    """Smooth isotropic Euclidean dilation of ``mask_bool`` by ``radius_px``."""
    sigma = max(1.2, 0.45 * float(radius_px))
    return gaussian_blur_2d(mask_bool.astype(np.float32), sigma) > 0.035


def parse_semantic_class_map(fp, bgr_img):
    """Return 2-D ``uint8`` BiSeNet semantic class map matching ``bgr_img`` shape.

    Uses ``fp.parse_classes`` when available, falling back to ``mode`` queries
    for lightweight mock callables in unit tests.

    Args:
        fp (Callable): ``FaceParsing`` instance or compatible callable.
        bgr_img (np.ndarray): ``(H, W, 3)`` BGR ``uint8`` image.

    Returns:
        np.ndarray: ``(H, W)`` ``uint8`` class map.
    """
    h, w = bgr_img.shape[:2]
    rgb_pil = Image.fromarray(bgr_img[:, :, ::-1])
    if hasattr(fp, "parse_classes"):
        cls_pil = fp.parse_classes(rgb_pil)
        if cls_pil.size != (w, h):
            cls_pil = cls_pil.resize((w, h), Image.NEAREST)
        return np.asarray(cls_pil, dtype=np.uint8)

    raw_m = np.asarray(fp(rgb_pil, mode="raw").resize((w, h), Image.NEAREST)) > 127
    neck_m = np.asarray(fp(rgb_pil, mode="neck").resize((w, h), Image.NEAREST)) > 127
    mouth_m = np.asarray(fp(rgb_pil, mode="mouth").resize((w, h), Image.NEAREST)) > 127
    cls_map = np.zeros((h, w), dtype=np.uint8)
    cls_map[neck_m & (~raw_m)] = 14
    cls_map[raw_m] = 1
    cls_map[mouth_m] = 11
    return cls_map


def measure_chin_bottom_in_crop(cls_map, rel_chin_x=None):
    """Measure median bottom row of face classes inside the central chin strip.

    Args:
        cls_map (np.ndarray): ``(H, W)`` BiSeNet class map of a face crop.
        rel_chin_x (float | None): Horizontal chin center inside the crop.

    Returns:
        float: Estimated chin bottom row in ``[0, H - 1]``.
    """
    h, w = cls_map.shape
    if h <= 1 or w <= 1:
        return 0.0
    half_band = max(3, int(round(0.11 * w)))
    cx_raw = (0.5 * w) if rel_chin_x is None else float(rel_chin_x)
    cx = int(np.clip(round(cx_raw), half_band, max(half_band, w - 1 - half_band)))
    x0 = max(0, cx - half_band)
    x1 = min(w, cx + half_band + 1)
    face_band = np.isin(cls_map[:, x0:x1], _FACE_AND_LIP_CLASSES)

    bottoms = []
    for col in range(face_band.shape[1]):
        ys = np.where(face_band[:, col])[0]
        if ys.size > 0:
            bottoms.append(float(ys[-1]))
    if not bottoms:
        ys_all = np.where(np.isin(cls_map, _FACE_AND_LIP_CLASSES))[0]
        return float(ys_all.max()) if ys_all.size > 0 else float(h - 1)
    return float(np.median(bottoms))


def _masked_skin_blur(img_f32, skin_mask_bool, sigma=3.2):
    """Normalized Gaussian blur over ``skin_mask_bool`` to prevent boundary halos."""
    w_mask = skin_mask_bool.astype(np.float32)
    denom = np.maximum(gaussian_blur_2d(w_mask, sigma), 1e-4)
    out = np.empty_like(img_f32, dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = gaussian_blur_2d(img_f32[:, :, ch] * w_mask, sigma) / denom
    return out


def restore_jaw_skin_details(
    res_crop_bgr,
    ori_crop_bgr,
    res_cls,
    ori_cls,
    delta_chin_px,
    sigma=3.2,
    strength=1.0,
):
    """Restore high-frequency chin/cheek skin features (moles, pores) onto ``res_crop_bgr``.

    Args:
        res_crop_bgr (np.ndarray): Generated BGR face crop ``(H, W, 3)``.
        ori_crop_bgr (np.ndarray): Source BGR face crop ``(H, W, 3)``.
        res_cls (np.ndarray): ``(H, W)`` class map of ``res_crop_bgr``.
        ori_cls (np.ndarray): ``(H, W)`` class map of ``ori_crop_bgr``.
        delta_chin_px (float): ``src_chin_y - gen_chin_y`` vertical mandible offset.
        sigma (float): High-pass cutoff standard deviation in pixels.
        strength (float): Detail blend weight in ``[0, 1]``.

    Returns:
        np.ndarray: Detail-restored ``uint8`` BGR face crop.
    """
    eff_strength = float(np.clip(strength, 0.0, 1.0))
    if eff_strength <= 1e-3 or res_crop_bgr.shape != ori_crop_bgr.shape:
        return res_crop_bgr

    h, w = res_crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return res_crop_bgr

    ys = np.arange(h, dtype=np.float32)
    ramp = np.clip((ys - 0.50 * h) / max(0.18 * h, 1.0), 0.0, 1.0)
    dy_row = float(delta_chin_px) * (0.5 * (1.0 - np.cos(np.pi * ramp)))

    src_y = np.clip(ys + dy_row, 0.0, float(h - 1))
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    wy = (src_y - y0)[:, np.newaxis, np.newaxis]
    ori_f = ori_crop_bgr.astype(np.float32)
    warped_ori_f = ori_f[y0] * (1.0 - wy) + ori_f[y1] * wy
    warped_ori_cls = ori_cls[np.clip(np.rint(src_y).astype(np.int32), 0, h - 1), :]

    ori_skin = warped_ori_cls == 1
    res_skin = res_cls == 1
    both_skin_inner = _erode_disk_radius2(ori_skin & res_skin)

    lip_margin = max(5.0, 0.065 * float(w))
    ori_lips_dil = _dilate_isotropic(np.isin(warped_ori_cls, _LIP_AND_MOUTH_CLASSES), lip_margin)
    res_lips_dil = _dilate_isotropic(np.isin(res_cls, _LIP_AND_MOUTH_CLASSES), lip_margin)

    valid_detail = both_skin_inner & (~ori_lips_dil) & (~res_lips_dil)
    valid_detail[: int(0.48 * h), :] = False
    if not np.any(valid_detail):
        return res_crop_bgr

    inner_valid = _erode_disk_radius2(valid_detail)
    soft_source = inner_valid if np.any(inner_valid) else valid_detail
    feathered = np.clip(gaussian_blur_2d(soft_source.astype(np.float32), 2.2) * 1.25, 0.0, 1.0)
    alpha = np.where(valid_detail, np.clip(eff_strength * feathered, 0.0, 1.0), 0.0).astype(np.float32)
    if float(alpha.max()) <= 1e-3:
        return res_crop_bgr

    ori_low = _masked_skin_blur(warped_ori_f, ori_skin, sigma=sigma)
    ori_high = np.clip(warped_ori_f - ori_low, -90.0, 45.0)
    res_f = res_crop_bgr.astype(np.float32)
    res_low = _masked_skin_blur(res_f, res_skin, sigma=sigma)
    detailed = res_low + ori_high

    blended = res_f * (1.0 - alpha[:, :, np.newaxis]) + detailed * alpha[:, :, np.newaxis]
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def lift_throat_shadow_same_frame(
    ori_frame,
    face_box,
    chin_x,
    src_chin_y,
    gen_chin_y,
    ctx_cls,
    ctx_bounds,
):
    """Elastically lift same-frame sub-chin shadow when ``src_chin_y > gen_chin_y``.

    Uses strictly vertical remapping (``dx == 0``) clamped against non-skin/neck
    pixels in ``ctx_cls`` so hair, clothing, and neck borders never move.

    Args:
        ori_frame (np.ndarray): Full source BGR frame ``(H, W, 3)``.
        face_box (Sequence[int]): Adjusted face box ``(x1, y1, x2, y2)``.
        chin_x (float): Horizontal chin center in full-frame coordinates.
        src_chin_y (float): Smoothed source chin Y in full-frame coordinates.
        gen_chin_y (float): Smoothed generated chin Y in full-frame coordinates.
        ctx_cls (np.ndarray): Full-context BiSeNet class map covering ``ctx_bounds``.
        ctx_bounds (tuple[int, int, int, int]): ``(px1, py1, px2, py2)`` of ``ctx_cls``.

    Returns:
        tuple[np.ndarray, bool]: ``(lifted_frame, did_lift)``.
    """
    lift_px = float(src_chin_y) - float(gen_chin_y)
    if lift_px <= 0.4:
        return ori_frame, False

    img_h, img_w = ori_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_box]
    fw, fh = float(x2 - x1), float(y2 - y1)
    if fw <= 8.0 or fh <= 8.0:
        return ori_frame, False

    px1, py1, px2, py2 = [int(v) for v in ctx_bounds]
    rx1 = max(px1, int(round(float(chin_x) - 0.42 * fw)))
    rx2 = min(px2, int(round(float(chin_x) + 0.42 * fw)))
    ry1 = max(py1, int(round(float(gen_chin_y) - 0.10 * fh)))
    ry2 = min(py2, int(round(float(src_chin_y) + 0.22 * fh)))
    sample_y2 = min(py2, ry2 + int(math.ceil(lift_px)) + 2)
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 8 or rh <= 8 or sample_y2 <= ry1 + 2:
        return ori_frame, False

    valid_ctx = np.isin(
        ctx_cls[ry1 - py1 : sample_y2 - py1, rx1 - px1 : rx2 - px1],
        _SKIN_AND_NECK_CLASSES,
    )
    skin_neck = valid_ctx[:rh, :]
    if not np.any(skin_neck):
        return ori_frame, False

    skin_neck_inner = _erode_disk_radius2(skin_neck)
    sem_weight = np.where(skin_neck, gaussian_blur_2d(skin_neck_inner.astype(np.float32), 2.5), 0.0)

    ys = np.arange(ry1, ry2, dtype=np.float32)
    xs = np.arange(rx1, rx2, dtype=np.float32)
    t_up = np.clip((ys - float(ry1)) / max(0.06 * fh, 2.0), 0.0, 1.0)
    w_up = 0.5 * (1.0 - np.cos(np.pi * t_up))
    plateau_end = float(gen_chin_y) + 0.03 * fh
    t_down = np.clip((ys - plateau_end) / max(float(ry2) - plateau_end, 4.0), 0.0, 1.0)
    w_down = 0.5 * (1.0 + np.cos(np.pi * t_down))
    wy = w_up * w_down

    dx = np.abs(xs - float(chin_x))
    tx = np.clip((dx - 0.20 * fw) / max(0.22 * fw, 2.0), 0.0, 1.0)
    wx = 0.5 * (1.0 + np.cos(np.pi * tx))

    dy_map = float(lift_px) * wy[:, np.newaxis] * wx[np.newaxis, :] * sem_weight

    sh = valid_ctx.shape[0]
    dist_to_invalid = np.zeros((rh, rw), dtype=np.float32)
    max_step = int(math.ceil(lift_px)) + 2
    cur_valid = skin_neck.copy()
    base_rows = np.arange(rh, dtype=np.int32)[:, np.newaxis]
    col_indices = np.arange(rw, dtype=np.int32)[np.newaxis, :]
    for step in range(1, max_step + 1):
        in_bounds = (base_rows + step) < sh
        check_rows = np.minimum(base_rows + step, sh - 1)
        cur_valid &= in_bounds & valid_ctx[check_rows, col_indices]
        dist_to_invalid[cur_valid] = float(step)
    hard_max_dy = np.where(skin_neck, np.maximum(0.0, dist_to_invalid - 1.0), 0.0)
    safe_max_dy = np.minimum(hard_max_dy, gaussian_blur_2d(hard_max_dy, 2.0))
    dy_map = np.minimum(dy_map, safe_max_dy)
    if float(dy_map.max()) <= 0.25:
        return ori_frame, False

    src_strip = ori_frame[ry1:sample_y2, rx1:rx2].astype(np.float32)
    local_y = np.clip((ys[:, np.newaxis] - float(ry1)) + dy_map, 0.0, float(sh - 1))
    y0 = np.floor(local_y).astype(np.int32)
    y1_idx = np.where(dy_map > 0.0, np.minimum(y0 + 1, sh - 1), y0)
    frac = (local_y - y0)[:, :, np.newaxis]

    remapped = src_strip[y0, col_indices] * (1.0 - frac) + src_strip[y1_idx, col_indices] * frac
    out = ori_frame.copy()
    out[ry1:ry2, rx1:rx2] = np.clip(np.rint(remapped), 0, 255).astype(np.uint8)
    return out, True


def _build_lower_alpha_pure_numpy(
    mask_pil,
    ori_shape,
    face_rect,
    upper_boundary_ratio,
    blur_ratio,
    side_protect_ratio,
    protect_upper_face,
):
    """Pure-NumPy equivalent of ``blending._build_lower_face_alpha`` without OpenCV."""
    width, height = mask_pil.size
    top_boundary = int(height * upper_boundary_ratio)
    blur_k = int(blur_ratio * ori_shape[0] // 2 * 2) + 1
    paste_top = min(top_boundary + (blur_k // 2 if protect_upper_face else 0), height)

    modified = Image.new("L", ori_shape, 0)
    modified.paste(mask_pil.crop((0, paste_top, width, height)), (0, paste_top))
    sigma = max(0.3 * ((blur_k - 1) * 0.5 - 1.0) + 0.8, 0.5)
    blurred = gaussian_blur_2d(np.asarray(modified, dtype=np.float32), sigma)
    mask_arr = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    mask_arr = apply_side_protect_mask(mask_arr, face_rect, side_protect_ratio)
    if protect_upper_face:
        mask_arr = confine_mask_below(mask_arr, top_boundary)
    return mask_arr


def composite_dynamic_chin_clean(
    body_bgr,
    res_face_bgr,
    face_box,
    res_cls,
    ori_cls=None,
    rel_chin_x=None,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
    upper_boundary_ratio=0.5,
    expand=1.5,
    blur_ratio=0.055,
    side_protect_ratio=0.15,
    protect_upper_face=True,
):
    """Composite ``res_face_bgr`` within mouth/chin column while protecting cheek silhouettes.

    Enforces ``left_cheek_width`` / ``right_cheek_width`` horizontal inward cheek
    erosion, inward-feathered jawline clamping (strictly zero alpha on synthetic
    ``class 14`` neck and non-skin source classes), and ``parsing_mode``
    (``"jaw"`` vs ``"raw"``) semantics.
    """
    if parsing_mode not in ("jaw", "raw"):
        raise ValueError(f"Unsupported parsing_mode: {parsing_mode!r}; expected 'jaw' or 'raw'")

    body = Image.fromarray(body_bgr[:, :, ::-1])
    face = Image.fromarray(res_face_bgr[:, :, ::-1])

    x, y, x1, y1 = [int(v) for v in face_box]
    fw, fh = x1 - x, y1 - y
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    s = int(max(fw, fh) // 2 * float(expand))
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    x_s, y_s, x_e, y_e = crop_box
    ori_shape = (x_e - x_s, y_e - y_s)
    w_crop = float(max(ori_shape[0], 1))
    face_rect = (x - x_s, y - y_s, x1 - x_s, y1 - y_s)

    safe_left_w, safe_right_w, safe_side_ratio = clamp_blend_safety_params(
        left_cheek_width,
        right_cheek_width,
        side_protect_ratio,
        force_min_side_protect=True,
    )

    res_face = np.isin(res_cls, _FACE_AND_LIP_CLASSES)
    lip_mask = np.isin(res_cls, _LIP_AND_MOUTH_CLASSES)
    mouth_xs = np.where(lip_mask)[1]
    if mouth_xs.size > 0:
        cx_local = float(mouth_xs.mean())
    elif rel_chin_x is not None:
        cx_local = float(np.clip(rel_chin_x, 0.2 * fw, 0.8 * fw))
    else:
        cx_local = 0.5 * float(fw)

    # Convert clamped 512-space cheek widths into crop pixels around mouth centroid
    left_inner = (float(safe_left_w) / 512.0) * w_crop
    right_inner = (float(safe_right_w) / 512.0) * w_crop

    guard_px = max(3, int(round(0.055 * w_crop)))
    erode_px = max(guard_px, int(round(0.065 * w_crop)))
    res_face_guarded = _erode_horizontal_1d(res_face, guard_px)
    if ori_cls is None:
        silhouette_guarded = res_face_guarded
        cheek_eroded = _erode_horizontal_1d(res_face, erode_px)
        central_safe = silhouette_guarded.copy()
    else:
        ori_face = np.isin(ori_cls, _FACE_AND_LIP_CLASSES)
        ori_skin_or_neck = np.isin(ori_cls, _SKIN_AND_NECK_CLASSES)
        ori_face_guarded = _erode_horizontal_1d(ori_face, guard_px)
        silhouette_guarded = res_face_guarded & ori_face_guarded
        cheek_eroded = _erode_horizontal_1d(res_face & ori_face, erode_px)

        if parsing_mode == "jaw":
            chin_tip_row = int(0.78 * fh)
            inner_face_cols = np.any(ori_face_guarded, axis=0)[np.newaxis, :]
            ori_neck_only = ori_cls == 14
            ori_jaw_guarded = _erode_horizontal_1d(ori_skin_or_neck, guard_px)
            neck_tip_ext = res_face_guarded & ori_neck_only & ori_jaw_guarded & inner_face_cols
            central_safe = silhouette_guarded.copy()
            central_safe[chin_tip_row:, :] |= neck_tip_ext[chin_tip_row:, :]
        else:
            central_safe = silhouette_guarded.copy()

    xs = np.arange(fw, dtype=np.float32)
    taper_span = max(0.11 * w_crop, 4.0)
    left_t = np.clip(((cx_local - xs) - left_inner) / taper_span, 0.0, 1.0)
    right_t = np.clip(((xs - cx_local) - right_inner) / taper_span, 0.0, 1.0)
    horiz_taper = (0.5 * (1.0 + np.cos(np.pi * np.maximum(left_t, right_t)))).astype(np.float32)

    in_central_col = (xs >= (cx_local - left_inner)) & (xs <= (cx_local + right_inner))
    hard_allow = np.where(in_central_col[np.newaxis, :], central_safe, cheek_eroded) & (
        horiz_taper[np.newaxis, :] > 1e-3
    )

    # Erode inward before Gaussian feathering so alpha tapers smoothly inside hard_allow
    # and strictly equals 0 on synthetic class-14 neck, cheek silhouette guards, and non-skin
    eroded_once = _erode_disk_radius2(hard_allow)
    eroded_twice = _erode_disk_radius2(eroded_once)
    soft_core = eroded_twice if np.any(eroded_twice) else (eroded_once if np.any(eroded_once) else hard_allow)
    if np.any(lip_mask):
        soft_core = soft_core | (lip_mask & eroded_once)

    weighted_small = np.clip(
        np.rint(soft_core.astype(np.float32) * horiz_taper[np.newaxis, :] * 255.0),
        0,
        255,
    ).astype(np.uint8)

    mask_pil = Image.new("L", ori_shape, 0)
    mask_pil.paste(Image.fromarray(weighted_small), face_rect)

    mask_arr = _build_lower_alpha_pure_numpy(
        mask_pil,
        ori_shape,
        face_rect,
        upper_boundary_ratio=upper_boundary_ratio,
        blur_ratio=blur_ratio,
        side_protect_ratio=safe_side_ratio,
        protect_upper_face=protect_upper_face,
    )
    hard_large = Image.new("L", ori_shape, 0)
    hard_large.paste(Image.fromarray(hard_allow.astype(np.uint8) * 255), face_rect)
    hard_large_bool = np.asarray(hard_large, dtype=np.uint8) > 127
    mask_arr = np.where(hard_large_bool, mask_arr, 0).astype(np.uint8)

    face_large = body.crop(crop_box)
    face_large.paste(face, face_rect)
    body.paste(face_large, crop_box[:2], Image.fromarray(mask_arr))
    return np.asarray(body)[:, :, ::-1].copy()


class DynamicChinThroatCoordinator:
    """Bounded-memory 5-frame lookahead coordinator for dynamic throat & chin blending."""

    def __init__(
        self,
        total_frames,
        frames,
        landmarks_list,
        fp,
        prepare_frame_fn,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        side_protect_ratio=0.15,
        protect_upper_face=True,
        max_source_cache=256,
    ):
        """Initialize bounded lookahead coordinator."""
        if parsing_mode not in ("jaw", "raw"):
            raise ValueError(f"Unsupported parsing_mode: {parsing_mode!r}; expected 'jaw' or 'raw'")
        self._total = int(total_frames)
        self._frames = frames
        self._landmarks = landmarks_list
        self._fp = fp
        self._prepare_fn = prepare_frame_fn
        self._parsing_mode = parsing_mode
        self._left_cheek_w, self._right_cheek_w, self._side_protect = clamp_blend_safety_params(
            left_cheek_width,
            right_cheek_width,
            side_protect_ratio,
            force_min_side_protect=True,
        )
        self._protect_upper = bool(protect_upper_face)
        self._max_src_cache = max(1, int(max_source_cache))
        self._src_cache = OrderedDict()
        self._slots = {}
        self.lifted_source_frames = set()
        self.evaluated_source_frames = set()

    def _get_source_context(self, source_index, face_box):
        """Fetch or compute full-context semantic map and chin anchor for ``source_index``."""
        src_idx = int(source_index)
        x1, y1, x2, y2 = [int(v) for v in face_box]
        key = (src_idx, (x1, y1, x2, y2))
        if key in self._src_cache:
            self._src_cache.move_to_end(key)
            return self._src_cache[key]

        ori_frame = self._frames[src_idx]
        img_h, img_w = ori_frame.shape[:2]
        fw, fh = float(x2 - x1), float(y2 - y1)
        lm = self._landmarks[src_idx] if (self._landmarks is not None and src_idx < len(self._landmarks)) else None
        if lm is not None:
            pts = np.asarray(lm, dtype=np.float32)
            chin_x = float(pts[8, 0]) if (pts.shape == (68, 2) and np.all(np.isfinite(pts))) else 0.5 * (x1 + x2)
        else:
            chin_x = 0.5 * (x1 + x2)

        pad_x = int(round(0.45 * fw))
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1)
        px2 = min(img_w, x2 + pad_x)
        py2 = min(img_h, y2 + int(round(0.38 * fh)))
        ctx_cls = parse_semantic_class_map(self._fp, ori_frame[py1:py2, px1:px2])
        ori_cls = ctx_cls[y1 - py1 : y2 - py1, x1 - px1 : x2 - px1]
        raw_src_chin_y = float(y1) + measure_chin_bottom_in_crop(ori_cls, chin_x - float(x1))

        entry = {
            "ctx_bounds": (px1, py1, px2, py2),
            "ctx_cls": ctx_cls,
            "chin_x": chin_x,
            "raw_src_chin_y": raw_src_chin_y,
        }
        self._src_cache[key] = entry
        if len(self._src_cache) > self._max_src_cache:
            self._src_cache.popitem(last=False)
        return entry

    def _ensure_slot(self, idx):
        """Populate lookahead slot ``idx`` if not yet evaluated, isolating per-frame errors."""
        if idx in self._slots:
            return
        src_idx = None
        try:
            src_idx_raw, box, face_bgr, detail_strength = self._prepare_fn(idx)
            src_idx = int(src_idx_raw)
            if box is None or face_bgr is None or not (0 <= src_idx < len(self._frames)):
                self._slots[idx] = {"src_idx": src_idx, "valid": False}
                return
            img_h, img_w = self._frames[src_idx].shape[:2]
            bx1, by1, bx2, by2 = [int(v) for v in box]
            if (
                bx1 < 0
                or by1 < 0
                or bx2 > img_w
                or by2 > img_h
                or bx2 <= bx1
                or by2 <= by1
                or face_bgr.shape[:2] != (by2 - by1, bx2 - bx1)
            ):
                self._slots[idx] = {"src_idx": src_idx, "valid": False}
                return
            norm_box = (bx1, by1, bx2, by2)
            src_ctx = self._get_source_context(src_idx, norm_box)
            res_cls = parse_semantic_class_map(self._fp, face_bgr)
            raw_gen_chin_y = float(by1) + measure_chin_bottom_in_crop(
                res_cls, src_ctx["chin_x"] - float(bx1)
            )
            self._slots[idx] = {
                "src_idx": src_idx,
                "valid": True,
                "box": norm_box,
                "face_bgr": face_bgr,
                "res_cls": res_cls,
                "detail_strength": float(detail_strength),
                "raw_src_chin_y": src_ctx["raw_src_chin_y"],
                "raw_gen_chin_y": raw_gen_chin_y,
            }
        except Exception as slot_err:
            print(f"Warning: dynamic slot {idx} preparation failed, isolating fallback: {slot_err}")
            self._slots[idx] = {"src_idx": src_idx, "valid": False}

    def render_frame(self, index):
        """Render composited BGR frame ``index`` and release expired lookahead buffers."""
        i = int(index)
        for stale in [k for k in self._slots if k < i - 2]:
            del self._slots[stale]
        for k in range(max(0, i - 2), min(self._total, i + 3)):
            self._ensure_slot(k)

        cur = self._slots[i]
        src_idx = cur["src_idx"]
        if not cur["valid"]:
            if src_idx is not None and 0 <= src_idx < len(self._frames):
                return self._frames[src_idx]
            raise RuntimeError(f"Dynamic slot {i} invalid without valid source index")
        ori_frame = self._frames[src_idx]

        src_vals, gen_vals = [], []
        for offset in (-2, -1, 0, 1, 2):
            nk = max(0, min(self._total - 1, i + offset))
            nslot = self._slots[nk]
            if nslot["valid"]:
                src_vals.append(nslot["raw_src_chin_y"])
                gen_vals.append(nslot["raw_gen_chin_y"])
            else:
                src_vals.append(cur["raw_src_chin_y"])
                gen_vals.append(cur["raw_gen_chin_y"])

        smooth_src_y = float(np.dot(_TEMPORAL_5TAP, np.asarray(src_vals, dtype=np.float32)))
        smooth_gen_y = float(np.dot(_TEMPORAL_5TAP, np.asarray(gen_vals, dtype=np.float32)))
        delta_chin = smooth_src_y - smooth_gen_y
        cur_raw_lift = float(cur["raw_src_chin_y"]) - float(cur["raw_gen_chin_y"])

        x1, y1, x2, y2 = cur["box"]
        src_ctx = self._get_source_context(src_idx, cur["box"])
        px1, py1, px2, py2 = src_ctx["ctx_bounds"]
        ctx_cls = src_ctx["ctx_cls"]
        ori_cls = ctx_cls[y1 - py1 : y2 - py1, x1 - px1 : x2 - px1]
        ori_crop = ori_frame[y1:y2, x1:x2]
        res_cls = cur["res_cls"]

        rf_detailed = restore_jaw_skin_details(
            cur["face_bgr"],
            ori_crop,
            res_cls,
            ori_cls,
            delta_chin_px=delta_chin,
            strength=cur["detail_strength"],
        )
        if cur_raw_lift <= 0.4:
            lifted_ori, did_lift = ori_frame, False
        else:
            eff_lift = min(max(0.0, delta_chin), cur_raw_lift)
            lifted_ori, did_lift = lift_throat_shadow_same_frame(
                ori_frame,
                cur["box"],
                src_ctx["chin_x"],
                smooth_gen_y + eff_lift,
                smooth_gen_y,
                ctx_cls,
                src_ctx["ctx_bounds"],
            )
        self.evaluated_source_frames.add(src_idx)
        if did_lift:
            self.lifted_source_frames.add(src_idx)

        combined = composite_dynamic_chin_clean(
            lifted_ori,
            rf_detailed,
            cur["box"],
            res_cls,
            ori_cls=ori_cls,
            rel_chin_x=src_ctx["chin_x"] - float(x1),
            parsing_mode=self._parsing_mode,
            left_cheek_width=self._left_cheek_w,
            right_cheek_width=self._right_cheek_w,
            side_protect_ratio=self._side_protect,
            protect_upper_face=self._protect_upper,
        )

        # Free image buffers immediately while keeping scalar chin heights for i+1, i+2
        cur["face_bgr"] = None
        cur["res_cls"] = None
        return combined
