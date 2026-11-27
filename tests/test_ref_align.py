import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from musetalk.utils import ref_align


def _make_full_landmarks(
    cx=500.0,
    cy=400.0,
    scale=1.0,
    angle_deg=0.0,
    mouth_gap=0.0,
    yaw_shift=0.0,
):
    """Build synthetic 68-point facial landmarks with realistic rigid structure."""
    pts = np.zeros((68, 2), dtype=np.float32)

    # Jaw contour 0..16 around face width 200, height 220
    angles = np.linspace(math.pi, 0.0, 17)
    for i, a in enumerate(angles):
        pts[i] = [100.0 * math.cos(a), 110.0 * math.sin(a)]

    # Eyebrows 17..26
    for i, x in enumerate(np.linspace(-65.0, -15.0, 5)):
        pts[17 + i] = [x, -55.0 - 4.0 * math.sin(i / 4.0 * math.pi)]
    for i, x in enumerate(np.linspace(15.0, 65.0, 5)):
        pts[22 + i] = [x, -55.0 - 4.0 * math.sin(i / 4.0 * math.pi)]

    # Nose bridge 27..30 & nostrils 31..35
    for i, y in enumerate(np.linspace(-35.0, 10.0, 4)):
        pts[27 + i] = [yaw_shift * (i / 3.0), y]
    for i, x in enumerate(np.linspace(-18.0, 18.0, 5)):
        pts[31 + i] = [x + yaw_shift, 20.0]

    # Left eye 36..41, right eye 42..47 (inter-ocular distance = 80)
    eye_offsets = [(-12, 0), (-5, -5), (5, -5), (12, 0), (5, 5), (-5, 5)]
    for i, (dx, dy) in enumerate(eye_offsets):
        pts[36 + i] = [-40.0 + dx, -30.0 + dy]
        pts[42 + i] = [40.0 + dx, -30.0 + dy]

    # Outer mouth 48..59 & inner lips 60..67
    pts[48] = [-35.0, 55.0]
    pts[54] = [35.0, 55.0]
    for i in range(49, 54):
        pts[i] = [-35.0 + 70.0 * (i - 48) / 6.0, 50.0 - mouth_gap * 0.5]
    for i in range(55, 60):
        pts[i] = [35.0 - 70.0 * (i - 54) / 6.0, 60.0 + mouth_gap * 0.5]
    pts[60] = [-28.0, 55.0]
    pts[64] = [28.0, 55.0]
    for top_i, bot_i, x in ((61, 67, -12.0), (62, 66, 0.0), (63, 65, 12.0)):
        pts[top_i] = [x, 55.0 - mouth_gap * 0.5]
        pts[bot_i] = [x, 55.0 + mouth_gap * 0.5]

    rad = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]],
        dtype=np.float32,
    )
    return (pts * float(scale)) @ rot.T + np.array([cx, cy], dtype=np.float32)


class ClosedMouthTemplateTest(unittest.TestCase):
    """Tests for detecting already closed-mouth templates."""

    def test_closed_template_detected_as_closed(self):
        """Templates with consistently low MAR bypass cross-frame binding."""
        lms = [_make_full_landmarks(mouth_gap=1.0 + 0.2 * i) for i in range(6)]
        is_closed, p95_mar = ref_align.is_closed_mouth_template(lms)
        self.assertTrue(is_closed)
        self.assertIsNotNone(p95_mar)
        self.assertLess(p95_mar, 0.06)

    def test_speaking_template_detected_as_open(self):
        """Even a single clearly open mouth frame enables auto cross-frame ref."""
        lms = [_make_full_landmarks(mouth_gap=1.0) for _ in range(19)]
        lms.append(_make_full_landmarks(mouth_gap=15.0))
        is_closed, p95_mar = ref_align.is_closed_mouth_template(lms)
        self.assertFalse(is_closed)
        self.assertIsNotNone(p95_mar)

    def test_insufficient_landmarks_returns_false(self):
        """Empty or single-frame landmark lists never trigger closed bypass."""
        self.assertEqual(ref_align.is_closed_mouth_template(None), (False, None))
        self.assertEqual(
            ref_align.is_closed_mouth_template([_make_full_landmarks(mouth_gap=0.0)]),
            (False, None),
        )


class RigidSimilarityTest(unittest.TestCase):
    """Tests for 4-DoF 2D rigid alignment and safety guardrails."""

    def test_recovers_known_similarity_independent_of_mouth_opening(self):
        """Mouth opening in target frame does not skew upper-face rigid fit."""
        ref_lm = _make_full_landmarks(cx=500.0, cy=400.0, scale=1.0, angle_deg=0.0, mouth_gap=0.5)
        tgt_lm = _make_full_landmarks(cx=518.0, cy=388.0, scale=1.10, angle_deg=-11.5, mouth_gap=26.0)

        matrix, status = ref_align.estimate_rigid_face_similarity(ref_lm, tgt_lm)
        self.assertEqual(status, "aligned")
        self.assertIsNotNone(matrix)

        idx = list(ref_align.RIGID_UPPER_FACE_INDICES)
        warped = ref_lm[idx] @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(warped, tgt_lm[idx], atol=1e-3)

    def test_trimmed_fit_tolerates_single_landmark_glitch(self):
        """One wildly jittery upper-face landmark should not break alignment."""
        ref_lm = _make_full_landmarks(cx=500.0, cy=400.0, scale=1.0, angle_deg=0.0)
        tgt_lm = _make_full_landmarks(cx=508.0, cy=405.0, scale=1.03, angle_deg=6.0)
        tgt_lm[17] += np.array([45.0, -40.0], dtype=np.float32)

        matrix, status = ref_align.estimate_rigid_face_similarity(ref_lm, tgt_lm)
        self.assertEqual(status, "aligned")
        # Check clean eyes align accurately despite corrupted eyebrow corner
        eye_pred = ref_lm[36:48] @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(eye_pred, tgt_lm[36:48], atol=0.5)

    def test_guardrails_reject_extreme_scale_roll_yaw_and_corruption(self):
        """Out-of-range scale, roll, 3D yaw turn, or noisy landmarks fall back safely."""
        ref_lm = _make_full_landmarks()

        _, status_scale = ref_align.estimate_rigid_face_similarity(
            ref_lm, _make_full_landmarks(scale=1.40)
        )
        self.assertEqual(status_scale, "fallback_self")

        _, status_roll = ref_align.estimate_rigid_face_similarity(
            ref_lm, _make_full_landmarks(angle_deg=32.0)
        )
        self.assertEqual(status_roll, "fallback_self")

        _, status_yaw = ref_align.estimate_rigid_face_similarity(
            ref_lm, _make_full_landmarks(yaw_shift=50.0)
        )
        self.assertEqual(status_yaw, "fallback_self")

        # Opposite left/right turns of equal magnitude must also exceed signed yaw delta
        _, status_opp_yaw = ref_align.estimate_rigid_face_similarity(
            _make_full_landmarks(yaw_shift=-25.0),
            _make_full_landmarks(yaw_shift=25.0),
        )
        self.assertEqual(status_yaw, "fallback_self")
        self.assertEqual(status_opp_yaw, "fallback_self")

        noisy = _make_full_landmarks()
        rng = np.random.default_rng(7)
        noisy[list(ref_align.RIGID_UPPER_FACE_INDICES)] += rng.normal(0.0, 22.0, (37, 2)).astype(np.float32)
        _, status_noise = ref_align.estimate_rigid_face_similarity(ref_lm, noisy)
        self.assertEqual(status_noise, "fallback_self")

        _, status_missing = ref_align.estimate_rigid_face_similarity(ref_lm, None)
        self.assertEqual(status_missing, "missing_landmarks")


class ComposedWarpAndPlanTest(unittest.TestCase):
    """Tests for single-step crop composition, bounds check, and mode planning."""

    def test_identity_composition_matches_pixel_center_resize(self):
        """Identity similarity maps crop borders with zero half-pixel drift."""
        ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        crop_m = ref_align.compose_aligned_crop_matrix(ident, (100, 200, 200, 400), 512)
        # Pixel center at output 0 should invert to x = 100 + 0.5*(100/512) - 0.5
        sx = 512.0 / 100.0
        sy = 512.0 / 200.0
        self.assertAlmostEqual(float(crop_m[0, 0]), sx, places=4)
        self.assertAlmostEqual(float(crop_m[1, 1]), sy, places=4)
        src_x0 = (0.0 - crop_m[0, 2]) / crop_m[0, 0]
        self.assertAlmostEqual(src_x0, 100.0 + 0.5 * (100.0 / 512.0) - 0.5, places=4)
        ratio = ref_align.compute_crop_in_bounds_ratio(crop_m, 512, (1000, 1000, 3))
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_align_reference_face_crop_uses_single_lanczos_warp_and_checks_bounds(self):
        """Valid warp calls cv2.warpAffine once; out-of-frame target falls back safely."""
        calls = []

        def fake_warp(img, mat, dsize, flags=None, borderMode=None):
            calls.append((dsize, flags, borderMode))
            return np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

        fake_cv2 = SimpleNamespace(
            warpAffine=fake_warp,
            INTER_LANCZOS4="lanczos4",
            BORDER_REPLICATE="replicate",
        )
        ref_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        ref_lm = _make_full_landmarks(cx=500.0, cy=400.0)
        tgt_lm = _make_full_landmarks(cx=510.0, cy=405.0, angle_deg=5.0)

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            out, status = ref_align.align_reference_face_crop(
                ref_frame, ref_lm, tgt_lm, (400, 300, 620, 520), 512
            )
            self.assertEqual(status, "aligned")
            self.assertEqual(out.shape, (512, 512, 3))
            self.assertEqual(calls, [((512, 512), "lanczos4", "replicate")])

            # Shifted far outside ref_frame should trigger bounds guardrail
            far_lm = _make_full_landmarks(cx=50.0, cy=50.0)
            out_bad, status_bad = ref_align.align_reference_face_crop(
                ref_frame,
                _make_full_landmarks(cx=10.0, cy=10.0),
                far_lm,
                (0, 0, 200, 200),
                512,
            )
            self.assertIsNone(out_bad)
            self.assertEqual(status_bad, "fallback_self")

    def test_plan_reference_frame_mode_covers_all_modes(self):
        """Auto bypasses closed templates, aligns speaking templates, and preserves fallback."""
        coords = [[400, 300, 600, 500]] * 4
        closed_lms = [_make_full_landmarks(mouth_gap=1.0) for _ in range(4)]
        speaking_lms = [
            _make_full_landmarks(mouth_gap=20.0),
            _make_full_landmarks(mouth_gap=1.5),
            _make_full_landmarks(mouth_gap=18.0),
            _make_full_landmarks(mouth_gap=1.0),
        ]

        plan_closed = ref_align.plan_reference_frame_mode("auto", coords, closed_lms)
        self.assertEqual(plan_closed["effective_mode"], "self")
        self.assertTrue(plan_closed["closed_template"])
        self.assertFalse(plan_closed["use_landmark_align"])

        plan_speak = ref_align.plan_reference_frame_mode("auto", coords, speaking_lms)
        self.assertEqual(plan_speak["effective_mode"], "auto")
        self.assertTrue(plan_speak["use_landmark_align"])
        self.assertIn(plan_speak["auto_ref_idx"], (1, 3))

        plan_no_lm = ref_align.plan_reference_frame_mode("auto", coords, None)
        self.assertEqual(plan_no_lm["effective_mode"], "auto")
        self.assertFalse(plan_no_lm["use_landmark_align"])

        # Same-length list of unusable None landmarks must also degrade to non-landmark fallback
        plan_all_none_lm = ref_align.plan_reference_frame_mode("dynamic", coords, [None] * 4)
        self.assertEqual(plan_all_none_lm["effective_mode"], "dynamic")
        self.assertFalse(plan_all_none_lm["use_landmark_align"])

        plan_first = ref_align.plan_reference_frame_mode("first", coords, speaking_lms)
        self.assertEqual(plan_first["effective_mode"], "first")
        self.assertEqual(plan_first["auto_ref_idx"], 0)
        self.assertTrue(plan_first["use_landmark_align"])

    def test_rectangular_output_size_and_upper_body_pose_box(self):
        """Rectangular output_size warps to (w, h) for close_mouth; ubody box crops full-body only."""
        calls = []

        def fake_warp(img, mat, dsize, flags=None, borderMode=None):
            calls.append(dsize)
            return np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

        fake_cv2 = SimpleNamespace(
            warpAffine=fake_warp,
            INTER_LANCZOS4="lanczos4",
            BORDER_REPLICATE="replicate",
        )
        ref_frame = np.zeros((1200, 1200, 3), dtype=np.uint8)
        ref_lm = _make_full_landmarks(cx=600.0, cy=500.0)
        tgt_lm = _make_full_landmarks(cx=608.0, cy=504.0, angle_deg=4.0)
        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            out, status = ref_align.align_reference_face_crop(
                ref_frame, ref_lm, tgt_lm, (460, 320, 740, 720), (280, 400)
            )
        self.assertEqual(status, "aligned")
        self.assertEqual(out.shape, (400, 280, 3))
        self.assertEqual(calls, [(280, 400)])

        # Full-body 4K frame (2160x3840) with 280x360 face produces compact ubody box
        ubox = ref_align.build_upper_body_pose_box((980, 400, 1260, 760), (3840, 2160, 3))
        self.assertIsNotNone(ubox)
        self.assertEqual(ubox.shape, (1, 4))
        self.assertLess(float(ubox[0, 3] - ubox[0, 1]), 2200.0)

        # Close-up frame where ubody spans >85% returns None (keeps full-frame behavior)
        self.assertIsNone(ref_align.build_upper_body_pose_box((150, 100, 450, 500), (800, 600, 3)))
        self.assertIsNone(ref_align.build_upper_body_pose_box(None, (1080, 1920, 3)))


if __name__ == "__main__":
    unittest.main()
