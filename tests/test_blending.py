import unittest

import numpy as np

from musetalk.utils.mask_utils import (
    adjust_face_box,
    apply_side_protect_mask,
    guard_jaw_dilation_mask,
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


class GuardJawDilationMaskTest(unittest.TestCase):
    """Tests for mouth-anchored neck-flank protection in jaw parsing mode."""

    def test_preserves_face_skin_and_chin_center_while_blocking_neck_flanks(self):
        """Face skin stays 255, central chin dilation stays 255, neck/bg flanks drop to 0."""
        parsing = np.zeros((200, 200), dtype=np.uint8)
        parsing[40:120, 50:150] = 1      # face skin (width=100, center=100)
        parsing[95:108, 85:115] = 12     # mouth/lips centered at x=100
        parsing[120:170, 70:130] = 14    # neck under face

        # Simulate wide jaw dilation (e.g. left/right_cheek_width=120) spilling into neck & bg
        jaw_mask = np.zeros((200, 200), dtype=np.uint8)
        jaw_mask[40:140, 45:155] = 255

        guarded = guard_jaw_dilation_mask(parsing, jaw_mask)

        # Full face skin (even far cheek edge) and lips remain 100% intact
        self.assertEqual(int(guarded[80, 52]), 255)
        self.assertEqual(int(guarded[80, 148]), 255)
        self.assertEqual(int(guarded[100, 100]), 255)
        # Central chin drop onto upper neck remains 255
        self.assertEqual(int(guarded[130, 100]), 255)
        # Outer neck flanks & pure background beside neck are zeroed
        self.assertEqual(int(guarded[130, 72]), 0)
        self.assertEqual(int(guarded[130, 128]), 0)
        self.assertEqual(int(guarded[130, 50]), 0)
        self.assertEqual(int(guarded[130, 150]), 0)

    def test_anchors_chin_column_to_shifted_mouth_for_profile_faces(self):
        """On a 3/4-profile face, chin protection tracks the shifted mouth center."""
        parsing = np.zeros((200, 200), dtype=np.uint8)
        parsing[40:120, 50:150] = 1      # face width=100
        parsing[95:108, 115:140] = 12    # shifted mouth centered around x=127
        parsing[120:170, 60:145] = 14    # neck

        jaw_mask = np.zeros((200, 200), dtype=np.uint8)
        jaw_mask[40:140, 50:150] = 255

        guarded = guard_jaw_dilation_mask(parsing, jaw_mask)
        # Under shifted mouth (x=127) keeps full dilation; far opposite neck flank (x=65) is 0
        self.assertEqual(int(guarded[130, 127]), 255)
        self.assertEqual(int(guarded[130, 65]), 0)

    def test_handles_empty_skin_and_rejects_shape_mismatch(self):
        """Missing face skin returns mask copy; mismatched shapes raise ValueError."""
        parsing = np.zeros((32, 32), dtype=np.uint8)
        jaw_mask = np.full((32, 32), 255, dtype=np.uint8)
        np.testing.assert_array_equal(guard_jaw_dilation_mask(parsing, jaw_mask), jaw_mask)
        with self.assertRaises(ValueError):
            guard_jaw_dilation_mask(parsing, np.zeros((16, 16), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
