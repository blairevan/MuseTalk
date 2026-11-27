import numpy as np


def adjust_face_box(
    face_box,
    image_shape,
    bbox_left_ratio=0.0,
    bbox_right_ratio=0.0,
    bbox_top_ratio=0.0,
    bbox_bottom_ratio=0.0,
):
    """Adjust each side of a face bbox independently.

    Args:
        face_box (tuple[int, int, int, int]): Face box as ``(x1, y1, x2, y2)``.
        image_shape (tuple[int, ...]): Source image shape used to clamp output.
        bbox_left_ratio (float): Positive expands the left edge outward,
            negative shrinks it inward.
        bbox_right_ratio (float): Positive expands the right edge outward,
            negative shrinks it inward.
        bbox_top_ratio (float): Positive expands the top edge outward,
            negative shrinks it inward.
        bbox_bottom_ratio (float): Positive expands the bottom edge outward,
            negative shrinks it inward.

    Returns:
        list[int]: Clamped adjusted bbox.
    """
    x1, y1, x2, y2 = [int(value) for value in face_box]
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    x1 = int(np.clip(x1, 0, image_width))
    x2 = int(np.clip(x2, x1, image_width))
    y1 = int(np.clip(y1, 0, image_height))
    y2 = int(np.clip(y2, y1, image_height))

    width = x2 - x1
    height = y2 - y1
    if width <= 2 or height <= 2:
        return [x1, y1, x2, y2]

    new_x1 = x1 - int(round(width * float(bbox_left_ratio)))
    new_x2 = x2 + int(round(width * float(bbox_right_ratio)))
    new_y1 = y1 - int(round(height * float(bbox_top_ratio)))
    new_y2 = y2 + int(round(height * float(bbox_bottom_ratio)))

    new_x1 = int(np.clip(new_x1, 0, image_width))
    new_x2 = int(np.clip(new_x2, 0, image_width))
    new_y1 = int(np.clip(new_y1, 0, image_height))
    new_y2 = int(np.clip(new_y2, 0, image_height))

    if new_x2 - new_x1 < 2:
        center_x = int(round((x1 + x2) / 2.0))
        new_x1 = int(np.clip(center_x - 1, 0, max(image_width - 2, 0)))
        new_x2 = min(new_x1 + 2, image_width)

    if new_y2 - new_y1 < 2:
        center_y = int(round((y1 + y2) / 2.0))
        new_y1 = int(np.clip(center_y - 1, 0, max(image_height - 2, 0)))
        new_y2 = min(new_y1 + 2, image_height)

    return [new_x1, new_y1, new_x2, new_y2]


def shrink_face_box(face_box, image_shape, bbox_shrink_ratio=0.0):
    """Shrink the left and right edges of a face bbox around its center."""
    return adjust_face_box(
        face_box,
        image_shape,
        bbox_left_ratio=-bbox_shrink_ratio,
        bbox_right_ratio=-bbox_shrink_ratio,
    )


def apply_side_protect_mask(mask_array, face_rect, side_protect_ratio=0.0):
    """Fade blend alpha near the left and right edges of a face box.

    Args:
        mask_array (np.ndarray): Single-channel alpha mask.
        face_rect (tuple[int, int, int, int]): Face box inside ``mask_array`` as
            ``(left, top, right, bottom)``.
        side_protect_ratio (float): Fraction of face-box width to protect on
            each side. ``0`` disables protection.

    Returns:
        np.ndarray: Mask with reduced side-edge alpha.
    """
    if side_protect_ratio <= 0:
        return mask_array

    height, width = mask_array.shape[:2]
    left, top, right, bottom = [int(value) for value in face_rect]
    left = int(np.clip(left, 0, width))
    right = int(np.clip(right, left, width))
    top = int(np.clip(top, 0, height))
    bottom = int(np.clip(bottom, top, height))

    face_width = right - left
    face_height = bottom - top
    if face_width <= 1 or face_height <= 0:
        return mask_array

    protect_width = int(round(face_width * float(side_protect_ratio)))
    protect_width = int(np.clip(protect_width, 0, face_width // 2))
    if protect_width <= 0:
        return mask_array

    protected_mask = mask_array.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, protect_width, endpoint=False, dtype=np.float32)

    protected_mask[top:bottom, left:left + protect_width] *= ramp[np.newaxis, :]
    protected_mask[top:bottom, right - protect_width:right] *= ramp[::-1][np.newaxis, :]

    if np.issubdtype(mask_array.dtype, np.integer):
        return np.clip(
            protected_mask,
            0,
            np.iinfo(mask_array.dtype).max,
        ).astype(mask_array.dtype)

    return protected_mask.astype(mask_array.dtype, copy=False)


def confine_mask_below(mask_array, top_row):
    """Zero every alpha value above ``top_row`` so those pixels keep the source frame.

    Args:
        mask_array (np.ndarray): Single-channel alpha mask.
        top_row (int): First row that may keep non-zero alpha.

    Returns:
        np.ndarray: Copy of the mask with rows ``[0, top_row)`` set to zero.
    """
    confined = np.array(mask_array, copy=True)
    top_row = int(np.clip(top_row, 0, confined.shape[0]))
    confined[:top_row] = 0
    return confined


def _cosine_horizontal_taper(x_coords, center_x, inner_half, outer_half):
    """Return a smooth horizontal weight in ``[0, 1]`` centered at ``center_x``."""
    dist = np.abs(x_coords - float(center_x))
    if outer_half <= inner_half + 1e-6:
        return (dist <= inner_half).astype(np.float32)
    t = np.clip((dist - float(inner_half)) / float(outer_half - inner_half), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


def guard_jaw_dilation_mask(
    parsing,
    jaw_mask,
    chin_inner_ratio=0.16,
    chin_outer_ratio=0.25,
    bg_inner_ratio=0.12,
    bg_outer_ratio=0.18,
):
    """Taper jaw-mode dilation spillover outside the mouth-anchored chin column.

    Preserves 100% of original face-skin (class 1) and mouth/lip (classes 11-13)
    pixels so wide faces and large ``cheek_width`` values continue to cover the
    full cheek interior. Only the dilated spillover into non-face regions (neck
    class 14, background class 0, cloth, hair) is smoothly attenuated outside
    the chin column anchored to the mouth centroid, preventing generated
    background corners from painting notches onto the outer edges of the neck.

    Args:
        parsing (np.ndarray): 2-D integer BiSeNet class map.
        jaw_mask (np.ndarray): 2-D ``uint8`` mask produced by jaw dilation.
        chin_inner_ratio (float): Half-width (fraction of face skin width) around
            the mouth center that keeps full downward dilation onto non-background
            pixels (e.g. central chin drop onto upper neck).
        chin_outer_ratio (float): Half-width where non-face spillover fades to 0.
        bg_inner_ratio (float): Tighter inner half-width for pure background
            pixels (class 0).
        bg_outer_ratio (float): Tighter outer half-width where background
            spillover fades to 0.

    Returns:
        np.ndarray: ``uint8`` guarded mask of the same shape as ``jaw_mask``.
    """
    mask_arr = np.asarray(jaw_mask, dtype=np.uint8)
    parse_arr = np.asarray(parsing)
    if parse_arr.shape != mask_arr.shape or parse_arr.ndim != 2:
        raise ValueError(
            f"parsing shape {parse_arr.shape} must match 2-D jaw_mask shape {mask_arr.shape}"
        )

    skin_ys, skin_xs = np.where(parse_arr == 1)
    if skin_xs.size == 0:
        return mask_arr.copy()

    face_width = float(skin_xs.max() - skin_xs.min())
    if face_width < 2.0:
        return mask_arr.copy()

    # Anchor to the mouth centroid so 3/4-profile heads track the shifted chin.
    mouth_mask = np.isin(parse_arr, (11, 12, 13))
    mouth_xs = np.where(mouth_mask)[1]
    if mouth_xs.size > 0:
        chin_cx = float(mouth_xs.mean())
    else:
        y_cutoff = float(skin_ys.min()) + 0.70 * float(skin_ys.max() - skin_ys.min())
        lower_xs = skin_xs[skin_ys >= y_cutoff]
        chin_cx = float(lower_xs.mean()) if lower_xs.size > 0 else float(skin_xs.mean())

    x_coords = np.arange(mask_arr.shape[1], dtype=np.float32)
    neck_taper = _cosine_horizontal_taper(
        x_coords, chin_cx, chin_inner_ratio * face_width, chin_outer_ratio * face_width
    )
    bg_taper = _cosine_horizontal_taper(
        x_coords, chin_cx, bg_inner_ratio * face_width, bg_outer_ratio * face_width
    )

    face_interior = (parse_arr == 1) | mouth_mask
    spillover = (mask_arr > 0) & (~face_interior)
    if not np.any(spillover):
        return mask_arr.copy()

    out = mask_arr.astype(np.float32, copy=True)
    bg_spill = spillover & (parse_arr == 0)
    non_bg_spill = spillover & (parse_arr != 0)

    if np.any(non_bg_spill):
        weights = np.broadcast_to(neck_taper[np.newaxis, :], mask_arr.shape)
        out[non_bg_spill] *= weights[non_bg_spill]
    if np.any(bg_spill):
        weights = np.broadcast_to(bg_taper[np.newaxis, :], mask_arr.shape)
        out[bg_spill] *= weights[bg_spill]

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
