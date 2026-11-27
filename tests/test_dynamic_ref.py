"""Unit tests for dynamic pose-and-time reference pools, audio routing, and neck healing."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from musetalk.utils import clarity, dynamic_ref, ref_align


def _make_landmarks(
    cx=500.0,
    cy=400.0,
    scale=1.0,
    angle_deg=0.0,
    mouth_gap=2.0,
    yaw_shift=0.0,
    mouth_half_w=35.0,
):
    """Build synthetic 68-point facial landmarks with tunable pose and mouth state."""
    pts = np.zeros((68, 2), dtype=np.float32)
    angles = np.linspace(math.pi, 0.0, 17)
    for i, a in enumerate(angles):
        pts[i] = [100.0 * math.cos(a), 110.0 * math.sin(a)]
    for i, x in enumerate(np.linspace(-65.0, -15.0, 5)):
        pts[17 + i] = [x, -55.0]
    for i, x in enumerate(np.linspace(15.0, 65.0, 5)):
        pts[22 + i] = [x, -55.0]
    for i, y in enumerate(np.linspace(-35.0, 10.0, 4)):
        pts[27 + i] = [yaw_shift * (i / 3.0), y]
    for i, x in enumerate(np.linspace(-18.0, 18.0, 5)):
        pts[31 + i] = [x + yaw_shift, 20.0]

    eye_offsets = [(-12, 0), (-5, -5), (5, -5), (12, 0), (5, 5), (-5, 5)]
    for i, (dx, dy) in enumerate(eye_offsets):
        pts[36 + i] = [-40.0 + dx, -30.0 + dy]
        pts[42 + i] = [40.0 + dx, -30.0 + dy]

    pts[48] = [-mouth_half_w, 55.0]
    pts[54] = [mouth_half_w, 55.0]
    for i in range(49, 54):
        pts[i] = [-mouth_half_w + 2.0 * mouth_half_w * (i - 48) / 6.0, 50.0 - mouth_gap * 0.5]
    for i in range(55, 60):
        pts[i] = [mouth_half_w - 2.0 * mouth_half_w * (i - 54) / 6.0, 60.0 + mouth_gap * 0.5]
    pts[60] = [-24.0, 55.0]
    pts[64] = [24.0, 55.0]
    for top_i, bot_i, x in ((61, 67, -12.0), (62, 66, 0.0), (63, 65, 12.0)):
        pts[top_i] = [x, 55.0 - mouth_gap * 0.5]
        pts[bot_i] = [x, 55.0 + mouth_gap * 0.5]

    rad = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]],
        dtype=np.float32,
    )
    return (pts * float(scale)) @ rot.T + np.array([cx, cy], dtype=np.float32)


class DynamicRefPoolsAndMatchingTest(unittest.TestCase):
    """Tests for pool construction, teeth visibility, signed pose matching, and dwell."""

    def test_pools_reject_isolated_bilabial_outlier_when_calm_span_exists(self):
        """A >=4-frame calm span is selected into closed_pool while a 1-frame /m/ squeeze is dropped."""
        coords = [[400, 300, 600, 520]] * 12
        gaps = [2.0, 2.0, 2.0, 2.0, 2.0, 12.0, 13.0, 12.0, 11.0, 0.0, 13.0, 12.0]
        lms = [_make_landmarks(mouth_gap=g) for g in gaps]

        pools = dynamic_ref.build_mouth_state_pools(coords, lms)
        self.assertEqual(pools["closed_pool"], [0, 1, 2, 3, 4])
        self.assertNotIn(9, pools["closed_pool"])
        self.assertEqual(pools["open_pool"], [5, 6, 7, 8, 10, 11])

    def test_open_pool_filters_for_visible_teeth_when_frames_provided(self):
        """Among equally open MAR frames, toothless dark cavities are filtered out when teeth frames exist."""
        coords = [[400, 300, 600, 520]] * 8
        lms = [
            _make_landmarks(mouth_gap=2.0),
            _make_landmarks(mouth_gap=2.0),
            _make_landmarks(mouth_gap=2.0),
            _make_landmarks(mouth_gap=2.0),
            _make_landmarks(mouth_gap=12.0),  # 4: open but dark toothless cavity
            _make_landmarks(mouth_gap=12.0),  # 5: open with bright neutral teeth
            _make_landmarks(mouth_gap=12.0),  # 6: open with bright neutral teeth
            _make_landmarks(mouth_gap=12.0),  # 7: open with bright neutral teeth
        ]
        frames = [np.full((600, 800, 3), 165, dtype=np.uint8) for _ in range(8)]
        # Inner mouth around (x=476..524, y=449..461)
        frames[4][448:462, 475:525] = [25, 25, 30]  # dark cavity only, no teeth
        for idx in (5, 6, 7):
            frames[idx][448:462, 475:525] = [30, 30, 35]
            frames[idx][449:455, 482:518] = [215, 220, 225]  # visible white upper teeth!

        self.assertLess(dynamic_ref.estimate_inner_mouth_teeth_score(frames[4], lms[4]), 0.04)
        self.assertGreater(dynamic_ref.estimate_inner_mouth_teeth_score(frames[5], lms[5]), 0.15)

        pools = dynamic_ref.build_mouth_state_pools(coords, lms, frames=frames)
        self.assertEqual(pools["open_pool"], [5, 6, 7])
        self.assertNotIn(4, pools["open_pool"])

    def test_signed_yaw_distinguishes_left_vs_right_head_turns(self):
        """Left-turned target frame matches left-turned open candidate rather than opposite right turn."""
        coords = [[400, 300, 600, 520]] * 6
        lms = [
            _make_landmarks(mouth_gap=12.0, yaw_shift=-25.0),
            _make_landmarks(mouth_gap=12.0, yaw_shift=25.0),
            _make_landmarks(mouth_gap=2.0, yaw_shift=-24.0),
            _make_landmarks(mouth_gap=2.0, yaw_shift=-24.0),
            _make_landmarks(mouth_gap=2.0, yaw_shift=-24.0),
            _make_landmarks(mouth_gap=2.0, yaw_shift=-24.0),
        ]
        pools = dynamic_ref.build_mouth_state_pools(coords, lms)
        nearest_open, _ = dynamic_ref.select_pose_time_nearest_per_source(6, pools, fps=30.0)
        self.assertEqual(nearest_open[2], 0)

    def test_silent_intro_saturates_time_cost_and_picks_pose_matched_teeth_frame(self):
        """When open frames start 5s later, intro frames pick the pose-closest open frame."""
        n = 200
        coords = [[400, 300, 600, 520]] * n
        lms = []
        for i in range(n):
            if i < 150:
                lms.append(_make_landmarks(mouth_gap=1.8, yaw_shift=0.0))
            elif i < 175:
                lms.append(_make_landmarks(mouth_gap=12.0, yaw_shift=26.0))
            else:
                lms.append(_make_landmarks(mouth_gap=12.0, yaw_shift=0.0))

        pools = dynamic_ref.build_mouth_state_pools(coords, lms)
        nearest_open, _ = dynamic_ref.select_pose_time_nearest_per_source(n, pools, fps=30.0)
        self.assertGreaterEqual(nearest_open[10], 175)

    def test_dwell_stabilizer_suppresses_jitter_but_tracks_urgent_head_turn(self):
        """Minor oscillation holds cur_ref during dwell; large head turn switches immediately."""
        coords = [[400, 300, 600, 520]] * 5
        lms = [
            _make_landmarks(mouth_gap=12.0, yaw_shift=0.0),
            _make_landmarks(mouth_gap=12.0, yaw_shift=1.0),
            _make_landmarks(mouth_gap=12.0, yaw_shift=28.0),
            _make_landmarks(mouth_gap=12.0, yaw_shift=0.5),
            _make_landmarks(mouth_gap=12.0, yaw_shift=27.0),
        ]
        pools = dynamic_ref.build_mouth_state_pools(coords, lms)
        raw_ref = [0, 1, 2, 1, 2]
        seq = [0, 3, 3, 4]
        stab = dynamic_ref.stabilize_ref_sequence(seq, raw_ref, pools, fps=30.0, min_dwell_frames=10)
        self.assertEqual(stab[:3], [0, 0, 0])
        self.assertEqual(stab[3], 2)


class DynamicAudioScheduleAndLatentBlendTest(unittest.TestCase):
    """Tests for exact pause boundaries, lazy O(1) sequence, and interrupted crossfades."""

    def test_exact_pause_duration_threshold_across_frame_rates(self):
        """Pauses <0.48s stay 0.0 and >=0.48s close at both 30fps and 15fps."""
        # At 30fps: 14f = 0.467s (<0.48s -> stays 0.0), 15f = 0.500s (>=0.48s -> closes), 22f = 0.733s (-> closes)
        voiced_30 = [(10, 50), (64, 100), (115, 150), (172, 200)]
        w30 = dynamic_ref.build_dynamic_closed_weights(voiced_30, total_frames=210, fps=30.0)
        self.assertEqual(float(w30[50:64].max()), 0.0)     # 14 frames (0.467s) stays open!
        self.assertAlmostEqual(float(w30[107]), 1.0, places=5)  # 15 frames (0.500s) closes!
        self.assertAlmostEqual(float(w30[161]), 1.0, places=5)  # 22 frames (0.733s) closes!

        # At 15fps: 7f = 0.467s (<0.48s -> stays 0.0), 8f = 0.533s (>=0.48s -> closes)
        voiced_15 = [(5, 20), (27, 40), (48, 60)]
        w15 = dynamic_ref.build_dynamic_closed_weights(voiced_15, total_frames=65, fps=15.0)
        self.assertEqual(float(w15[20:27].max()), 0.0)     # 7 frames (0.467s) stays open!
        self.assertAlmostEqual(float(w15[44]), 1.0, places=5)   # 8 frames (0.533s) closes!

    def test_lazy_dynamic_latent_sequence_and_interrupted_crossfade_continuity(self):
        """Lazy sequence evaluates only indexed frames and decays A & B smoothly on A->B->C."""
        masked = [np.full((1, 4, 2, 2), float(s), dtype=np.float32) for s in range(6)]
        # Urgent back-to-back switch: 0 -> 2 at i=1, immediately interrupted by -> 5 at i=2!
        plan = {
            "source_seq": [0, 1, 2, 3, 4, 5],
            "open_ref_seq": [0, 2, 5, 5, 5, 5],
            "closed_ref_seq": [1, 1, 1, 1, 1, 1],
            "closed_weights": np.zeros(6, dtype=np.float32),
        }
        calls = []

        def fake_encode(src_i, ref_i):
            calls.append((src_i, ref_i))
            return np.full((1, 4, 2, 2), float(100 * ref_i), dtype=np.float32)

        seq = dynamic_ref.build_dynamic_unet_latents(masked, plan, fake_encode, crossfade_frames=3)
        # Lazy guarantee: zero tensor encodings before indexing!
        self.assertEqual(len(calls), 0)
        self.assertEqual(len(seq), 6)

        # At i=1 (0->2, step=1/4): {0: 0.75, 2: 0.25} -> ref value = 50.0
        self.assertAlmostEqual(float(seq[1][0, 4, 0, 0]), 50.0, places=4)
        # At i=2 (interrupted ->5, step=1/4 from {0: 0.75, 2: 0.25}):
        # {0: 0.5625, 2: 0.1875, 5: 0.25} -> ref=2 weight DECREASES (0.25 -> 0.1875), never jumps to 0.75!
        expected_i2 = 0.5625 * 0.0 + 0.1875 * 200.0 + 0.25 * 500.0
        self.assertAlmostEqual(float(seq[2][0, 4, 0, 0]), expected_i2, places=4)
        # By i=5 (past 3 crossfade frames): {5: 1.0} -> 500.0
        self.assertAlmostEqual(float(seq[5][0, 4, 0, 0]), 500.0, places=4)

    def test_resilient_ref_encoder_recovers_from_alignment_and_vae_errors(self):
        """Single-frame warp or VAE encode failure falls back to self crop or last valid latent."""
        frames = [np.full((64, 64, 3), 10 * i, dtype=np.uint8) for i in range(3)]
        lms = [_make_landmarks() for _ in range(3)]
        boxes = [(8, 8, 56, 56)] * 3
        self_crops = [np.full((16, 16, 3), 100 + i, dtype=np.uint8) for i in range(3)]
        fallback_init = np.full((1, 4, 2, 2), -1.0, dtype=np.float32)
        counts = {"aligned": 0, "self": 0, "fallback_self": 0}

        def flaky_align(ref_f, ref_lm, tgt_lm, box, res):
            if int(ref_f[0, 0, 0]) == 10:
                raise RuntimeError("simulated cv2.warpAffine failure")
            return np.full((16, 16, 3), 222, dtype=np.uint8), "aligned"

        def flaky_encode(crop_bgr):
            val = int(crop_bgr[0, 0, 0])
            if val == 102:
                raise RuntimeError("simulated VAE failure on self_crop[2]")
            return np.full((1, 4, 2, 2), float(val), dtype=np.float32)

        enc = dynamic_ref.make_resilient_ref_encoder(
            frames=frames,
            landmarks_list=lms,
            adjusted_boxes_by_source=boxes,
            self_ref_crops_by_source=self_crops,
            resolution=16,
            encode_crop_fn=flaky_encode,
            initial_fallback_latent=fallback_init,
            align_counts=counts,
            align_fn=flaky_align,
            max_cache_size=2,
        )
        # (0, 2) succeeds via aligned crop -> 222.0, updates last_valid to 222.0
        self.assertEqual(float(enc(0, 2)[0, 0, 0, 0]), 222.0)
        # (1, 1) where ref=1 throws align error if cross, or here (0, 1) throws align error -> falls back to self_crop[0]=100.0
        self.assertEqual(float(enc(0, 1)[0, 0, 0, 0]), 100.0)
        # (2, 1): align throws AND self_crop[2]=102 throws -> falls back to last_valid=100.0 without aborting!
        self.assertEqual(float(enc(2, 1)[0, 0, 0, 0]), 100.0)
        self.assertEqual(counts["aligned"], 1)
        self.assertEqual(counts["fallback_self"], 2)


class NeckPreHealingTest(unittest.TestCase):
    """Tests for open-jaw double-chin / Adam's-apple pre-healing and semantic masking."""

    def test_heals_dropped_chin_inside_skin_neck_and_protects_hair_and_bg(self):
        """Open-jaw target heals neck shadow from closed ref while leaving hair/bg untouched."""
        ori = np.full((600, 600, 3), 180, dtype=np.uint8)
        ori[468:480, 270:330] = 70
        ori[:, :230] = [50, 175, 45]
        ori[:, 365:] = [20, 20, 20]

        ref_closed = np.full((600, 600, 3), 180, dtype=np.uint8)
        ref_closed[:, :230] = [50, 175, 45]
        ref_closed[:, 365:] = [20, 20, 20]

        ori_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=18.0)
        ref_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=1.5)
        face_box = (200, 250, 400, 470)

        def fake_warp(img, mat, dsize, flags=None, borderMode=None):
            rx1 = int(round(-mat[0, 2] / mat[0, 0]))
            ry1 = int(round(-mat[1, 2] / mat[1, 1]))
            return img[ry1:ry1 + dsize[1], rx1:rx1 + dsize[0]].copy()

        fake_cv2 = SimpleNamespace(
            warpAffine=fake_warp,
            INTER_LANCZOS4="lanczos4",
            BORDER_REPLICATE="replicate",
        )

        def fake_fp(pil_img, mode="neck"):
            arr = np.asarray(pil_img)
            is_skin = (arr[:, :, 0] > 60) & (arr[:, :, 1] != 175)
            return Image.fromarray((is_skin.astype(np.uint8) * 255))

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            patch = dynamic_ref.compute_healed_neck_roi(
                ori, ori_lm, face_box, ref_closed, ref_lm, fp=fake_fp
            )
            self.assertIsNotNone(patch)
            healed = dynamic_ref.apply_healed_neck_roi(ori, patch)

            self.assertGreater(int(healed[472, 300, 0]), 170)
            np.testing.assert_array_equal(healed[:, :230], ori[:, :230])
            np.testing.assert_array_equal(healed[:, 365:], ori[:, 365:])
            self.assertIsNone(
                dynamic_ref.compute_healed_neck_roi(
                    ref_closed, ref_lm, face_box, ref_closed, ref_lm, fp=fake_fp
                )
            )

    def test_clarity_and_ref_align_accept_dynamic_mode(self):
        """clarity and ref_align validate and plan 'dynamic' cleanly."""
        self.assertEqual(clarity.validate_ref_frame_mode("dynamic"), "dynamic")
        self.assertEqual(clarity.resolve_ref_frame_index(2, 8, "dynamic", auto_ref_index=5), 5)

        coords = [[400, 300, 600, 500]] * 4
        speaking_lms = [
            _make_landmarks(mouth_gap=18.0),
            _make_landmarks(mouth_gap=1.5),
            _make_landmarks(mouth_gap=16.0),
            _make_landmarks(mouth_gap=1.5),
        ]
        plan = ref_align.plan_reference_frame_mode("dynamic", coords, speaking_lms)
        self.assertEqual(plan["effective_mode"], "dynamic")
        self.assertTrue(plan["use_landmark_align"])
        self.assertFalse(plan["closed_template"])


if __name__ == "__main__":
    unittest.main()
