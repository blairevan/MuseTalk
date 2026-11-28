"""Unit tests for dynamic pose-and-time reference pools, audio routing, and neck healing."""

import math
import unittest

import numpy as np
from PIL import Image

from musetalk.utils import clarity, dynamic_ref, ref_align, throat_chin_blend


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
    """Tests for zero-warp same-frame sub-chin healing, stationary neck, and collar guard."""

    def test_heals_dropped_chin_bulge_and_rim_without_touching_neck_or_flanks(self):
        """Open-jaw target heals both bright chin bulge and dark rim from same-frame throat only."""
        # Base neck/throat skin is 175; dropped open-jaw chin has a bright 215 bulge + dark 70 rim!
        ori = np.full((600, 600, 3), 175, dtype=np.uint8)
        ori[452:466, 270:330] = 215  # bright dropped-chin flesh
        ori[466:474, 270:330] = 70   # dark sub-mental crease
        # Distinct neck texture markers below chin_y + 0.04*fh (y >= 480) and on neck flanks
        ori[480:, :] = 172
        ori[495:580, 260:340] = [168, 173, 178]  # vertical throat muscle texture that must NOT move!
        ori[:, :225] = [50, 175, 45]             # green screen on left
        ori[:, 375:] = [20, 20, 20]              # dark hair on right

        # Even if ref_closed_frame has a completely shifted/different pattern (99), it is NEVER warped!
        ref_closed = np.full((600, 600, 3), 99, dtype=np.uint8)

        ori_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=18.0)  # chin_y = 470.0
        ref_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=1.5)
        face_box = (200, 250, 400, 470)  # fw=200, fh=220 -> bot_y = 470 + 7.7 = 478

        def fake_fp(pil_img, mode="neck"):
            arr = np.asarray(pil_img)
            is_skin = (arr[:, :, 0] > 40) & (arr[:, :, 1] != 175)
            return Image.fromarray((is_skin.astype(np.uint8) * 255))

        patch = dynamic_ref.compute_healed_neck_roi(
            ori, ori_lm, face_box, ref_closed, ref_lm, fp=fake_fp
        )
        self.assertIsNotNone(patch)
        healed = dynamic_ref.apply_healed_neck_roi(ori, patch)

        # Both the bright 215 bulge (y=458) and dark 70 crease (y=469) match ~172 throat skin (not 99!)
        self.assertLess(abs(int(healed[458, 300, 0]) - 170), 12)
        self.assertLess(abs(int(healed[469, 300, 0]) - 170), 12)

        # Entire neck below y=480 AND left/right flanks/hair/bg are byte-for-byte IDENTICAL to ori!
        np.testing.assert_array_equal(healed[480:, :], ori[480:, :])
        np.testing.assert_array_equal(healed[:, :230], ori[:, :230])
        np.testing.assert_array_equal(healed[:, 370:], ori[:, 370:])

        # Closed-mouth frame returns None (zero work, zero pixel changes)
        self.assertIsNone(
            dynamic_ref.compute_healed_neck_roi(
                ori, ref_lm, face_box, ref_closed, ref_lm, fp=fake_fp
            )
        )

    def test_high_collar_occlusion_skips_healing_safely(self):
        """When clothing covers the throat sampling strip, compute_healed_neck_roi returns None."""
        ori = np.full((600, 600, 3), 180, dtype=np.uint8)
        ori[478:520, :] = [15, 15, 15]  # high black turtleneck right under chin (y=478..520)
        ori_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=18.0)
        ref_lm = _make_landmarks(cx=300.0, cy=360.0, mouth_gap=1.5)
        face_box = (200, 250, 400, 470)

        def fake_fp(pil_img, mode="neck"):
            arr = np.asarray(pil_img)
            return Image.fromarray(((arr[:, :, 0] > 50).astype(np.uint8) * 255))

        self.assertIsNone(
            dynamic_ref.compute_healed_neck_roi(
                ori, ori_lm, face_box, None, ref_lm, fp=fake_fp
            )
        )

    def test_temporal_chin_heal_strength_carries_missing_landmarks_and_ignores_closed_pitch(self):
        """Closed mouth stays at 0.0 strength; 1-frame missing landmark does not drop open span."""
        closed_lm = _make_landmarks(mouth_gap=1.5)
        open_lm = _make_landmarks(mouth_gap=18.0)
        lms = [closed_lm, closed_lm, closed_lm, open_lm, open_lm, None, open_lm, open_lm]
        ratio, n_mar = dynamic_ref.compute_neutral_chin_stats(lms, closed_pool=[0, 1, 2])
        strengths = dynamic_ref.build_smoothed_chin_heal_strengths(lms, ratio, n_mar)
        self.assertEqual(float(strengths[0]), 0.0)
        self.assertGreater(float(strengths[5]), 0.90)

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


class ThroatLiftAndMoleRestoreTest(unittest.TestCase):
    """Tests for jaw-tracked chin mole restoration, unimodal throat lift, and bounded lookahead."""

    def test_restores_chin_mole_at_shifted_jaw_height_without_touching_lips(self):
        """A dark chin mole on an open-jaw source crop transfers onto a higher generated chin."""
        h, w = 200, 200
        ori_crop = np.full((h, w, 3), 205, dtype=np.uint8)
        res_crop = np.full((h, w, 3), 205, dtype=np.uint8)

        # Source jaw is 10px lower (chin bottom at y=185, mole at y=175, x=90)
        ori_cls = np.full((h, w), 14, dtype=np.uint8)
        ori_cls[:185, 30:170] = 1
        ori_cls[105:130, 65:135] = 12  # lips
        ori_crop[173:178, 88:93] = [120, 130, 145]  # small brown beauty mark / mole!
        # High-contrast skin feature right next to the dilated lip exclusion zone (y=130)
        ori_crop[134:141, 90:110] = [30, 30, 30]

        # Generated jaw is 10px higher (chin bottom at y=175, smooth mole-free UNet skin)
        res_cls = np.full((h, w), 14, dtype=np.uint8)
        res_cls[:175, 30:170] = 1
        res_cls[105:122, 65:135] = 12
        res_crop[105:122, 65:135] = [110, 100, 185]  # generated lip color

        restored = throat_chin_blend.restore_jaw_skin_details(
            res_crop, ori_crop, res_cls, ori_cls, delta_chin_px=10.0
        )
        # Mole shifted up by 10px from y=175 -> y=165!
        self.assertLess(int(restored[165, 90, 0]), 155)
        # Entire generated lip region (including 1px lip border AND dilated exclusion band) stays 100% untouched!
        lip_margin = max(5.0, 0.065 * float(w))
        res_lip_excl = throat_chin_blend._dilate_isotropic(res_cls == 12, lip_margin)
        np.testing.assert_array_equal(restored[res_lip_excl], res_crop[res_lip_excl])

    def test_elastic_throat_lift_moves_cast_shadow_up_and_protects_collar_and_flanks(self):
        """Lifting open-jaw throat moves cast shadow up cleanly with dx==0 and protects sloped clothing/hair."""
        ori = np.full((400, 400, 3), 188, dtype=np.uint8)
        # Dropped cast throat shadow at y=252..262 (when src_chin_y=248, gen_chin_y=238 -> lift=10px)
        ori[252:262, 160:240] = 140
        ori[310:, :] = [20, 20, 20]    # dark collar below y=310
        ori[:, :105] = [40, 180, 40]   # green screen left
        ori[:, 295:] = [25, 25, 25]    # dark hair right

        ctx_cls = np.full((400, 400), 14, dtype=np.uint8)
        ctx_cls[:248, 120:280] = 1
        ctx_cls[310:, :] = 16          # cloth
        ctx_cls[:, :105] = 0           # bg
        ctx_cls[:, 295:] = 17          # hair
        # Steep diagonal V-neck dark collar intrusion right inside the active sampling zone (x=220..245, y>=254)
        for col in range(220, 250):
            collar_y = 254 + (col - 220) // 2
            ori[collar_y:285, col] = [10, 10, 10]
            ctx_cls[collar_y:285, col] = 16

        lifted, did_lift = throat_chin_blend.lift_throat_shadow_same_frame(
            ori,
            face_box=(120, 80, 280, 255),
            chin_x=200.0,
            src_chin_y=248.0,
            gen_chin_y=238.0,
            ctx_cls=ctx_cls,
            ctx_bounds=(0, 0, 400, 400),
        )
        self.assertTrue(did_lift)
        # Shadow moved up from y=256 to y=246, and old y=256 is now lifted brighter throat skin!
        self.assertLess(int(lifted[246, 200, 0]), 152)
        self.assertGreater(int(lifted[256, 200, 0]), 170)
        # Sloped collar pixels stay untouched AND zero dark [10,10,10] collar color bleeds onto skin above it!
        self.assertTrue(np.all(lifted[ctx_cls == 16] == ori[ctx_cls == 16]))
        for col in range(220, 245):
            collar_y = 254 + (col - 220) // 2
            self.assertGreaterEqual(int(lifted[collar_y - 2, col, 0]), 138)
        # Collar, green screen, and hair flanks remain byte-for-byte untouched!
        np.testing.assert_array_equal(lifted[310:, :], ori[310:, :])
        np.testing.assert_array_equal(lifted[:, :110], ori[:, :110])
        np.testing.assert_array_equal(lifted[:, 290:], ori[:, 290:])

    def test_clean_chin_composite_blocks_synthetic_neck_spillover_and_runs_lookahead(self):
        """Generated class-14 neck pixels inside face_box (even 1px below chin) are never pasted."""
        body = np.full((300, 300, 3), 150, dtype=np.uint8)
        res_face = np.full((120, 120, 3), 245, dtype=np.uint8)  # overly bright synthetic neck at bottom!
        res_cls = np.full((120, 120), 14, dtype=np.uint8)
        res_cls[:95, 10:110] = 1
        res_cls[55:72, 40:80] = 12
        ori_cls = np.full((120, 120), 14, dtype=np.uint8)
        ori_cls[:102, 6:114] = 1

        comp = throat_chin_blend.composite_dynamic_chin_clean(
            body,
            res_face,
            (90, 80, 210, 200),
            res_cls,
            ori_cls=ori_cls,
            left_cheek_width=90,
            right_cheek_width=90,
            protect_upper_face=True,
        )
        # Even 1px below generated chin (y=80+95=175..200), EVERY synthetic class-14 row stays exact body=150!
        np.testing.assert_array_equal(comp[175:200, 90:210], body[175:200, 90:210])
        # Upper face (y=95, above boundary 80 + 60 = 140) stays exact body=150!
        self.assertEqual(int(comp[95, 150, 0]), 150)
        # Entire outer cheek silhouette guard (x=90..105 and x=195..210) stays exact body=150!
        np.testing.assert_array_equal(comp[135:180, 90:105], body[135:180, 90:105])
        np.testing.assert_array_equal(comp[135:180, 195:210], body[135:180, 195:210])

        # Chin-tip cloth/bg protection + parsing_mode='jaw' vs 'raw' differentiation:
        # When generated jaw opens lower (y=0..112 is class 1) over source neck (y=96..108 is class 14)
        # and source collar (y=108..120 is class 16):
        res_open_cls = res_cls.copy()
        res_open_cls[:112, 25:95] = 1
        ori_collar_cls = ori_cls.copy()
        ori_collar_cls[96:108, :] = 14
        ori_collar_cls[108:, :] = 16
        comp_jaw = throat_chin_blend.composite_dynamic_chin_clean(
            body, res_face, (90, 80, 210, 200), res_open_cls, ori_cls=ori_collar_cls, parsing_mode="jaw"
        )
        comp_raw = throat_chin_blend.composite_dynamic_chin_clean(
            body, res_face, (90, 80, 210, 200), res_open_cls, ori_cls=ori_collar_cls, parsing_mode="raw"
        )
        # 'jaw' extends onto source neck (y=80+101=181 > 150) but NEVER onto source collar (y=80+108=188 == 150)!
        self.assertGreater(int(comp_jaw[181, 150, 0]), 165)
        np.testing.assert_array_equal(comp_jaw[188:200, 90:210], body[188:200, 90:210])
        # 'raw' strictly stops at source face skin (y>=80+96=176 stays exact 150), even if generated lips overlap neck!
        res_low_lip_cls = res_open_cls.copy()
        res_low_lip_cls[94:104, 42:78] = 12
        comp_raw_low_lip = throat_chin_blend.composite_dynamic_chin_clean(
            body, res_face, (90, 80, 210, 200), res_low_lip_cls, ori_cls=ori_collar_cls, parsing_mode="raw"
        )
        np.testing.assert_array_equal(comp_raw[176:200, 90:210], body[176:200, 90:210])
        np.testing.assert_array_equal(comp_raw_low_lip[176:200, 90:210], body[176:200, 90:210])

        # Asymmetric side-turn where hair (class 17) enters inside the central column (cols 0..42, x=90..132)
        # across all rows including chin-tip rows (y >= 0.78*fh): the full guard_px=10 cheek band (cols 43..52,
        # x=133..142) MUST remain 100% untouched even in 'jaw' mode!
        ori_asym_cls = ori_cls.copy()
        ori_asym_cls[:, :43] = 17
        res_asym_cls = res_open_cls.copy()
        res_asym_cls[:, :40] = 17
        comp_asym = throat_chin_blend.composite_dynamic_chin_clean(
            body, res_face, (90, 80, 210, 200), res_asym_cls, ori_cls=ori_asym_cls, parsing_mode="jaw"
        )
        np.testing.assert_array_equal(comp_asym[:, 90:143], body[:, 90:143])

        # Safety clamp: even if an external CLI caller passes aggressive legacy flags (120/180, 0.08/0.0),
        # clamp_blend_safety_params & outer silhouette guard still keep outer cheeks 100% untouched!
        self.assertEqual(
            throat_chin_blend.clamp_blend_safety_params(120, 120, 0.08),
            (90, 90, 0.15),
        )
        comp_clamped = throat_chin_blend.composite_dynamic_chin_clean(
            body,
            res_face,
            (90, 80, 210, 200),
            res_cls,
            ori_cls=ori_cls,
            left_cheek_width=180,
            right_cheek_width=180,
            side_protect_ratio=0.0,
            protect_upper_face=True,
        )
        np.testing.assert_array_equal(comp_clamped[135:180, 90:105], body[135:180, 90:105])
        np.testing.assert_array_equal(comp_clamped[135:180, 195:210], body[135:180, 195:210])

        # Verify:
        # 1) Bounded 5-slot lookahead frees buffers;
        # 2) Current frame with raw_src_chin_y <= raw_gen_chin_y is NEVER lifted even when neighbors open wide;
        # 3) A single-frame parse exception on frame 4 is isolated and does NOT drop generated mouths on frames 2,3,5,6!
        class MixedFP:
            def __init__(self):
                self.calls = 0

            def parse_classes(self, pil_img):
                w, h = pil_img.size
                arr_in = np.asarray(pil_img)
                if int(arr_in[0, 0, 0]) == 77:
                    self.calls += 1
                    raise RuntimeError("synthetic single-frame parse failure")
                arr = np.full((h, w), 14, dtype=np.uint8)
                # Source context crops have h > 120; generated face crops have h == 120
                if h > 120:
                    # Source frame 0 has CLOSED chin (y=80+82), source frames 1..3 have OPEN chin (y=80+104)
                    chin_h = 82 if int(arr_in[0, 0, 0]) == 170 else 104
                    arr[:chin_h, :] = 1
                else:
                    # Generated chin is at y=88; mouth lips at 55..70
                    arr[:88, 10:90] = 1
                    arr[55:70, 35:65] = 12
                return Image.fromarray(arr)

        frames = [
            np.full((240, 240, 3), 170, dtype=np.uint8),  # src 0: closed jaw (chin=82 <= gen=88 -> NO lift!)
            np.full((240, 240, 3), 175, dtype=np.uint8),  # src 1: open jaw (chin=104 > gen=88 -> lift!)
            np.full((240, 240, 3), 175, dtype=np.uint8),
            np.full((240, 240, 3), 175, dtype=np.uint8),
        ]
        lms = [_make_landmarks(cx=120.0, cy=110.0, scale=0.45) for _ in range(4)]

        def prep(idx):
            fill_val = 77 if idx == 4 else 230
            src_i = 0 if idx == 1 else (1 + (idx % 3))
            return src_i, (70, 50, 170, 170), np.full((120, 100, 3), fill_val, dtype=np.uint8), 1.0

        mixed_fp = MixedFP()
        coord = throat_chin_blend.DynamicChinThroatCoordinator(8, frames, lms, mixed_fp, prep)
        rendered = []
        for idx in range(8):
            out_f = coord.render_frame(idx)
            rendered.append(out_f)
            self.assertEqual(out_f.shape, (240, 240, 3))
            self.assertLessEqual(len(coord._slots), 5)

        # Source 0 (used on frame idx=1, surrounded by open-jaw frames 0 and 2) was NEVER lifted!
        self.assertNotIn(0, coord.lifted_source_frames)
        self.assertIn(1, coord.lifted_source_frames)
        # Broken frame 4 fell back cleanly to its source frame and threw only ONCE,
        # while adjacent frames 2, 3, 5, 6 still rendered generated lower-face pixels (> 215)!
        self.assertEqual(mixed_fp.calls, 1)
        np.testing.assert_array_equal(rendered[4], frames[2])
        for healthy_idx in (2, 3, 5, 6):
            self.assertGreater(int(rendered[healthy_idx][125, 120, 0]), 215)

        # Out-of-bounds box degrades cleanly to the unmodified source frame
        def prep_oob(idx):
            return idx % 4, (-10, 50, 90, 170), np.full((120, 100, 3), 240, dtype=np.uint8), 1.0

        coord_oob = throat_chin_blend.DynamicChinThroatCoordinator(2, frames, lms, MixedFP(), prep_oob)
        np.testing.assert_array_equal(coord_oob.render_frame(0), frames[0])


if __name__ == "__main__":
    unittest.main()
