import unittest

import numpy as np

from musetalk.utils.mask_utils import (
    adjust_face_box,
    apply_side_protect_mask,
    shrink_face_box,
)


class ApplySideProtectMaskTest(unittest.TestCase):
    """Tests for side-edge blend protection around the face box."""

    def test_fades_face_box_edges_without_touching_center(self):
        """Side protection should fade bbox edges while preserving center alpha."""
        mask = np.full((6, 20), 100, dtype=np.uint8)

        protected = apply_side_protect_mask(
            mask,
            face_rect=(5, 1, 15, 5),
            side_protect_ratio=0.2,
        )

        self.assertEqual(protected[2, 5], 0)
        self.assertEqual(protected[2, 6], 50)
        self.assertEqual(protected[2, 7], 100)
        self.assertEqual(protected[2, 12], 100)
        self.assertEqual(protected[2, 13], 50)
        self.assertEqual(protected[2, 14], 0)

    def test_zero_ratio_leaves_mask_unchanged(self):
        """A disabled side protection ratio should keep the mask unchanged."""
        mask = np.full((4, 8), 123, dtype=np.uint8)

        protected = apply_side_protect_mask(
            mask,
            face_rect=(1, 0, 7, 4),
            side_protect_ratio=0.0,
        )

        np.testing.assert_array_equal(protected, mask)


class ShrinkFaceBoxTest(unittest.TestCase):
    """Tests for shrinking detected face boxes before inpainting."""

    def test_shrinks_box_width_around_center(self):
        """Horizontal shrink should move both side edges inward."""
        box = shrink_face_box(
            face_box=(10, 20, 110, 120),
            image_shape=(200, 200, 3),
            bbox_shrink_ratio=0.1,
        )

        self.assertEqual(box, [20, 20, 100, 120])

    def test_zero_ratio_leaves_box_unchanged(self):
        """A disabled shrink ratio should keep the bbox unchanged."""
        box = shrink_face_box(
            face_box=(10, 20, 110, 120),
            image_shape=(200, 200, 3),
            bbox_shrink_ratio=0.0,
        )

        self.assertEqual(box, [10, 20, 110, 120])


class AdjustFaceBoxTest(unittest.TestCase):
    """Tests for independent four-side bbox adjustment."""

    def test_expands_and_shrinks_each_side_independently(self):
        """Positive ratios expand edges and negative ratios shrink edges."""
        box = adjust_face_box(
            face_box=(20, 30, 120, 130),
            image_shape=(200, 200, 3),
            bbox_left_ratio=0.1,
            bbox_right_ratio=-0.2,
            bbox_top_ratio=-0.1,
            bbox_bottom_ratio=0.2,
        )

        self.assertEqual(box, [10, 40, 100, 150])

    def test_clamps_expanded_box_to_image_bounds(self):
        """Expanded edges should not leave the source image."""
        box = adjust_face_box(
            face_box=(10, 20, 90, 120),
            image_shape=(130, 100, 3),
            bbox_left_ratio=0.5,
            bbox_right_ratio=0.5,
            bbox_top_ratio=0.5,
            bbox_bottom_ratio=0.5,
        )

        self.assertEqual(box, [0, 0, 100, 130])


if __name__ == "__main__":
    unittest.main()
