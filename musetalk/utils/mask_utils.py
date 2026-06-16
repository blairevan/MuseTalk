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
