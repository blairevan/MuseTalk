import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from musetalk.utils.clarity import gaussian_blur_2d


def _fake_gaussian_blur(image, ksize, sigma_x):
    """Deterministic stand-in for cv2.GaussianBlur used only in these tests."""
    sigma = max(ksize[0] / 6.0, 0.5)
    blurred = gaussian_blur_2d(np.asarray(image, dtype=np.float32), sigma)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _load_blending():
    """Import musetalk.utils.blending with a fake cv2 module."""
    fake_cv2 = SimpleNamespace(GaussianBlur=_fake_gaussian_blur)
    with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
        sys.modules.pop("musetalk.utils.blending", None)
        module = importlib.import_module("musetalk.utils.blending")
    return module


def _legacy_alpha(mask_image, ori_shape, face_rect, upper_boundary_ratio, blur_ratio, side_ratio):
    """The pre-refactor mask construction, kept verbatim for comparison."""
    from musetalk.utils.mask_utils import apply_side_protect_mask

    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified = Image.new('L', ori_shape, 0)
    modified.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))
    kernel = int(blur_ratio * ori_shape[0] // 2 * 2) + 1
    alpha = _fake_gaussian_blur(np.array(modified), (kernel, kernel), 0)
    return apply_side_protect_mask(alpha, face_rect, side_ratio)


class LowerFaceAlphaTest(unittest.TestCase):
    """Blend-mask construction shared by the offline and realtime paths."""

    def setUp(self):
        self.blending = _load_blending()
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[40:180, 40:160] = 255
        self.mask_image = Image.fromarray(mask)
        self.ori_shape = (200, 200)
        self.face_rect = (40, 40, 160, 180)

    def test_default_matches_legacy_offline_and_realtime(self):
        """With protection off the mask is identical to the previous code."""
        for blur_ratio in (0.05, 0.1):
            new = self.blending._build_lower_face_alpha(
                self.mask_image, self.ori_shape, self.face_rect, 0.5,
                blur_ratio=blur_ratio, side_protect_ratio=0.08, protect_upper_face=False,
            )
            old = _legacy_alpha(self.mask_image, self.ori_shape, self.face_rect, 0.5, blur_ratio, 0.08)
            np.testing.assert_array_equal(new, old)

    def test_legacy_feather_leaks_above_boundary(self):
        """Documents the problem: the legacy feather reaches above the boundary."""
        old = _legacy_alpha(self.mask_image, self.ori_shape, self.face_rect, 0.5, 0.1, 0.0)
        self.assertGreater(int(old[:100].max()), 0)

    def test_protection_zeroes_upper_rows_and_keeps_lower_alpha(self):
        """With protection on, alpha above the boundary is exactly zero."""
        alpha = self.blending._build_lower_face_alpha(
            self.mask_image, self.ori_shape, self.face_rect, 0.5,
            blur_ratio=0.1, side_protect_ratio=0.08, protect_upper_face=True,
        )
        self.assertEqual(int(alpha[:100].max()), 0)
        self.assertEqual(int(alpha[150, 100]), 255)


class RunCliPassThroughTest(unittest.TestCase):
    """run_cli.py forwards legacy defaults unless options are set."""

    def _run(self, extra_args, has_512_weights=False):
        import run_cli

        captured = {}

        def fake_run(cmd, check):
            captured["cmd"] = cmd

        real_isfile = os.path.isfile
        weights_512 = {run_cli.DEFAULT_512_UNET_PATH, run_cli.DEFAULT_512_CONFIG_PATH}

        def fake_isfile(path):
            if path in weights_512:
                return has_512_weights
            return real_isfile(path)

        argv = ["run_cli.py", "--face", "f.png", "--audio", "a.wav", "--outfile", "/tmp/o.mp4"] + extra_args
        with mock.patch.object(run_cli, "validate_inputs", return_value=True), \
                mock.patch.object(run_cli.os.path, "isfile", side_effect=fake_isfile), \
                mock.patch.object(run_cli.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(sys, "argv", argv):
            run_cli.main()
        cmd = captured["cmd"]
        self.last_cmd = cmd
        self.last_config = cmd[cmd.index("--inference_config") + 1]
        return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}

    def test_each_run_uses_a_private_config_that_is_removed(self):
        """Concurrent tasks get different YAML files, deleted after the run."""
        self._run([])
        first = self.last_config
        self._run([])
        self.assertNotEqual(first, self.last_config)
        self.assertTrue(os.path.basename(first).startswith("temp_cli_task_"))
        self.assertFalse(os.path.exists(first))
        self.assertFalse(os.path.exists(self.last_config))

    def test_defaults_use_adopted_clarity_settings(self):
        """Without local 512 weights, legacy 256 defaults and clarity options apply."""
        opts = self._run([], has_512_weights=False)
        self.assertEqual(opts["--resolution"], "256")
        self.assertEqual(opts["--unet_model_path"], "./models/musetalkV15/unet.pth")
        self.assertEqual(opts["--unet_config"], "./models/musetalkV15/musetalk.json")
        self.assertEqual(opts["--batch_size"], "8")
        self.assertNotIn("--use_float16", self.last_cmd)
        self.assertEqual(opts["--vae_latent_mode"], "mode")
        self.assertEqual(opts["--paste_interpolation"], "lanczos")
        self.assertEqual(opts["--enhance_mode"], "usm")
        self.assertEqual(opts["--protect_upper_face"], "true")
        self.assertEqual(opts["--fix_teeth_whiteness"], "false")
        self.assertEqual(opts["--close_mouth_pauses"], "false")
        self.assertEqual(opts["--pause_top_db"], "38.0")
        self.assertEqual(opts["--use_audio_pe"], "true")
        self.assertEqual(opts["--ref_frame_mode"], "self")

    def test_defaults_keep_256_even_when_512_weights_exist(self):
        """Unmodified v1.5 CLI defaults to robust 256 v1.5 even when local 512 weights exist."""
        opts = self._run([], has_512_weights=True)
        self.assertEqual(opts["--resolution"], "256")
        self.assertEqual(opts["--unet_model_path"], "./models/musetalkV15/unet.pth")
        self.assertEqual(opts["--unet_config"], "./models/musetalkV15/musetalk.json")
        self.assertEqual(opts["--batch_size"], "8")
        self.assertNotIn("--use_float16", self.last_cmd)
        self.assertEqual(opts["--use_audio_pe"], "true")
        self.assertEqual(opts["--ref_frame_mode"], "self")

    def test_explicit_256_and_512_precision_overrides_work_when_512_weights_exist(self):
        """Explicit --resolution 512 with --no_float16/--batch_size overrides defaults cleanly."""
        forced_256 = self._run(["--resolution", "256"], has_512_weights=True)
        self.assertEqual(forced_256["--resolution"], "256")
        self.assertEqual(forced_256["--unet_model_path"], "./models/musetalkV15/unet.pth")
        self.assertEqual(forced_256["--batch_size"], "8")
        self.assertNotIn("--use_float16", self.last_cmd)
        self.assertEqual(forced_256["--use_audio_pe"], "true")
        self.assertEqual(forced_256["--ref_frame_mode"], "self")

        forced_fp32 = self._run(["--resolution", "512", "--batch_size", "1", "--no_float16"], has_512_weights=True)
        self.assertEqual(forced_fp32["--resolution"], "512")
        self.assertEqual(forced_fp32["--batch_size"], "1")
        self.assertNotIn("--use_float16", self.last_cmd)

        v10_opts = self._run(["--model_version", "v1.0"], has_512_weights=True)
        self.assertEqual(v10_opts["--resolution"], "256")

    def test_resolution_512_selects_512_weights_and_disables_audio_pe_by_default(self):
        """Passing --resolution 512 routes to musetalkV15_512, bs=2, fp16, use_audio_pe=false, ref=auto."""
        opts = self._run(["--resolution", "512"])
        self.assertEqual(opts["--resolution"], "512")
        self.assertEqual(opts["--version"], "v15")
        self.assertEqual(opts["--unet_model_path"], "./models/musetalkV15_512/unet.pth")
        self.assertEqual(opts["--unet_config"], "./models/musetalkV15_512/musetalk.json")
        self.assertEqual(opts["--batch_size"], "2")
        self.assertIn("--use_float16", self.last_cmd)
        self.assertEqual(opts["--use_audio_pe"], "false")
        self.assertEqual(opts["--ref_frame_mode"], "auto")

    def test_resolution_512_respects_explicit_model_and_audio_pe_overrides(self):
        """Explicit UNet path, config, use_audio_pe, and ref_frame_mode flags override defaults."""
        opts = self._run([
            "--resolution", "512",
            "--unet_model_path", "/tmp/custom_512.pth",
            "--unet_config", "/tmp/custom.json",
            "--use_audio_pe", "true",
            "--ref_frame_mode", "first",
        ])
        self.assertEqual(opts["--resolution"], "512")
        self.assertEqual(opts["--unet_model_path"], "/tmp/custom_512.pth")
        self.assertEqual(opts["--unet_config"], "/tmp/custom.json")
        self.assertEqual(opts["--use_audio_pe"], "true")
        self.assertEqual(opts["--ref_frame_mode"], "first")

    def test_ref_frame_mode_default_and_override(self):
        """run_cli defaults --ref_frame_mode to self for 256, auto for 512, and forwards explicit modes."""
        default_256 = self._run([])
        self.assertEqual(default_256["--ref_frame_mode"], "self")
        default_512 = self._run(["--resolution", "512"])
        self.assertEqual(default_512["--ref_frame_mode"], "auto")
        offset_opts = self._run(["--resolution", "512", "--ref_frame_mode", "offset"])
        self.assertEqual(offset_opts["--ref_frame_mode"], "offset")
        self_opts = self._run(["--resolution", "512", "--ref_frame_mode", "self"])
        self.assertEqual(self_opts["--ref_frame_mode"], "self")
        auto_256 = self._run(["--ref_frame_mode", "auto"])
        self.assertEqual(auto_256["--ref_frame_mode"], "auto")
        dynamic_256 = self._run(["--ref_frame_mode", "dynamic"])
        self.assertEqual(dynamic_256["--ref_frame_mode"], "dynamic")

    def test_v10_rejects_512_without_explicit_weights(self):
        """v1.0 with --resolution 512 fails fast unless explicit UNet files are supplied."""
        import run_cli

        with self.assertRaises(ValueError):
            run_cli.resolve_model_settings("v1.0", 512)
        resolved = run_cli.resolve_model_settings(
            "v1.0", 512, unet_model_path="/tmp/u.pth", unet_config="/tmp/c.json"
        )
        self.assertEqual(resolved["resolution"], 512)
        self.assertEqual(resolved["unet_model_path"], "/tmp/u.pth")

    def test_legacy_values_are_forwarded(self):
        """Explicit legacy values reach scripts.inference unchanged."""
        opts = self._run([
            "--vae_latent_mode", "sample", "--paste_interpolation", "linear",
            "--enhance_mode", "none", "--protect_upper_face", "false", "--close_mouth_pauses", "true",
        ])
        self.assertEqual(opts["--vae_latent_mode"], "sample")
        self.assertEqual(opts["--paste_interpolation"], "linear")
        self.assertEqual(opts["--enhance_mode"], "none")
        self.assertEqual(opts["--protect_upper_face"], "false")
        self.assertEqual(opts["--close_mouth_pauses"], "true")

    def test_input_video_1080p_normalization_and_visible_cache(self):
        """4K videos generate a non-hidden *_1080p.mp4 sibling cache and reuse it; <=1080p bypasses."""
        import tempfile
        import run_cli

        self.assertFalse(run_cli.needs_input_video_resize(1080, 1920))
        self.assertFalse(run_cli.needs_input_video_resize(1920, 1080))
        self.assertFalse(run_cli.needs_input_video_resize(720, 1280))
        self.assertTrue(run_cli.needs_input_video_resize(2160, 3840))
        self.assertTrue(run_cli.needs_input_video_resize(3840, 2160))
        self.assertTrue(run_cli.needs_input_video_resize(1079, 1920))
        self.assertFalse(run_cli.needs_input_video_resize(2160, 3840, max_long_edge=0))

        # Visible non-hidden cache filename for persistent template; None for dot-temp timeline
        self.assertEqual(
            run_cli.build_normalized_cache_path("/data/digital_human_video_4136480240.mp4"),
            "/data/digital_human_video_4136480240_1080p.mp4",
        )
        # Bijective cache paths: clip_1080p.mp4, clip.mov, and clip_mov.mp4 never collide
        p_already = run_cli.build_normalized_cache_path("/data/clip_1080p.mp4")
        p_mov = run_cli.build_normalized_cache_path("/data/clip.mov")
        p_mov_mp4 = run_cli.build_normalized_cache_path("/data/clip_mov.mp4")
        self.assertNotEqual(p_already, "/data/clip_1080p.mp4")
        self.assertEqual(len({p_already, p_mov, p_mov_mp4}), 3)
        self.assertFalse(run_cli.needs_input_video_resize(1080, 1920, max_long_edge=1921))
        # Cap of 1 cannot satisfy even dimensions <= 1 and must fail before transcoding
        with self.assertRaises(ValueError):
            run_cli.needs_input_video_resize(1080, 1920, max_long_edge=1)
        with self.assertRaises(ValueError):
            run_cli.prepare_normalized_face_input("/data/clip.mp4", "/tmp", max_long_edge=1)
        self.assertIsNone(
            run_cli.build_normalized_cache_path("/data/.video-timeline-2340047046.mp4")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            src_video = os.path.join(tmpdir, "digital_human_video_99.mp4")
            with open(src_video, "wb") as f:
                f.write(b"raw4k")

            ffmpeg_calls = []

            def fake_subproc_run(cmd, check):
                ffmpeg_calls.append(cmd)
                with open(cmd[-1], "wb") as out_f:
                    out_f.write(b"norm1080p")

            def fake_probe(path):
                if path == src_video:
                    return (2160, 3840, 30.0)
                return (1080, 1920, 30.0)

            with mock.patch.object(run_cli, "_probe_video_size", side_effect=fake_probe), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=fake_subproc_run):
                # 1st call creates visible *_1080p.mp4 cache + sidecar via ffmpeg
                eff1, clean1 = run_cli.prepare_normalized_face_input(src_video, tmpdir)
                expected_cache = os.path.join(tmpdir, "digital_human_video_99_1080p.mp4")
                self.assertEqual(eff1, expected_cache)
                self.assertIsNone(clean1)
                self.assertTrue(os.path.isfile(expected_cache))
                self.assertEqual(len(ffmpeg_calls), 1)
                self.assertNotIn("fps=", " ".join(ffmpeg_calls[0]))

                # 2nd call reuses existing *_1080p.mp4 without calling ffmpeg again
                eff2, clean2 = run_cli.prepare_normalized_face_input(src_video, tmpdir)
                self.assertEqual(eff2, expected_cache)
                self.assertIsNone(clean2)
                self.assertEqual(len(ffmpeg_calls), 1)

                # pin_run_snapshot=True returns an immutable per-run snapshot unaffected by later cache replacement
                snap_out = os.path.join(tmpdir, "snap_out")
                eff_snap, clean_snap = run_cli.prepare_normalized_face_input(
                    src_video, snap_out, pin_run_snapshot=True
                )
                self.assertEqual(eff_snap, clean_snap)
                self.assertNotEqual(eff_snap, expected_cache)
                self.assertEqual(len(ffmpeg_calls), 1)

                # If user replaces expected_cache manually in-place while sidecar remains,
                # in-flight snap stays 100% intact AND subsequent source update never clobbers user's file
                with open(expected_cache, "wb") as f:
                    f.write(b"USER_REPLACED_VIDEO")
                with open(eff_snap, "rb") as f:
                    self.assertEqual(f.read(), b"norm1080p")
                os.remove(clean_snap)

                with open(src_video, "wb") as f:
                    f.write(b"updated_raw4k_source")
                eff3, clean3 = run_cli.prepare_normalized_face_input(src_video, os.path.join(tmpdir, "out"))
                self.assertEqual(eff3, clean3)
                with open(expected_cache, "rb") as f:
                    self.assertEqual(f.read(), b"USER_REPLACED_VIDEO")
                os.remove(clean3)

                # Restore owned cache via fresh source & verify os.replace refresh leaves pinned snapshot intact
                fresh_src = os.path.join(tmpdir, "fresh_lecture.mp4")
                with open(fresh_src, "wb") as f:
                    f.write(b"v1_4k")

                def probe_fresh(path):
                    if path == fresh_src:
                        return (2160, 3840, 30.0)
                    return (1080, 1920, 30.0)

                def subproc_v1(cmd, check):
                    with open(cmd[-1], "wb") as out_f:
                        out_f.write(b"TRANSCODED_V1")

                def subproc_v2(cmd, check):
                    with open(cmd[-1], "wb") as out_f:
                        out_f.write(b"TRANSCODED_V2")

                with mock.patch.object(run_cli, "_probe_video_size", side_effect=probe_fresh):
                    with mock.patch.object(run_cli.subprocess, "run", side_effect=subproc_v1):
                        snap_v1, clean_v1 = run_cli.prepare_normalized_face_input(
                            fresh_src, snap_out, pin_run_snapshot=True
                        )
                    # Concurrent job updates fresh_src and refreshes shared cache via os.replace while v1 is in-flight
                    with open(fresh_src, "wb") as f:
                        f.write(b"v2_4k_updated")
                    with mock.patch.object(run_cli.subprocess, "run", side_effect=subproc_v2):
                        snap_v2, clean_v2 = run_cli.prepare_normalized_face_input(
                            fresh_src, snap_out, pin_run_snapshot=True
                        )
                    # Job 1's pinned snapshot STILL holds TRANSCODED_V1 while Job 2's snapshot holds TRANSCODED_V2!
                    with open(snap_v1, "rb") as f:
                        self.assertEqual(f.read(), b"TRANSCODED_V1")
                    with open(snap_v2, "rb") as f:
                        self.assertEqual(f.read(), b"TRANSCODED_V2")
                    os.remove(clean_v1)
                    os.remove(clean_v2)

                    # Short os.write() calls (3 bytes at a time) still copy full snapshot intact
                    real_os_write = run_cli.os.write

                    def partial_write(fd, buf):
                        return real_os_write(fd, buf[:3])

                    short_src = os.path.join(tmpdir, "shortwrite.mp4")
                    with open(short_src, "wb") as f:
                        f.write(b"raw4k_short")
                    with mock.patch.object(run_cli, "_probe_video_size", side_effect=lambda p: (2160, 3840, 30.0) if p == short_src else (1080, 1920, 30.0)), \
                            mock.patch.object(run_cli.os, "write", side_effect=partial_write), \
                            mock.patch.object(run_cli.subprocess, "run", side_effect=subproc_v1):
                        snap_sw, clean_sw = run_cli.prepare_normalized_face_input(
                            short_src, snap_out, pin_run_snapshot=True
                        )
                        with open(snap_sw, "rb") as f:
                            self.assertEqual(f.read(), b"TRANSCODED_V1")
                        # 2nd call hits cache through partial_write snapshot copy and still verifies SHA-256
                        snap_sw2, clean_sw2 = run_cli.prepare_normalized_face_input(
                            short_src, snap_out, pin_run_snapshot=True
                        )
                        with open(snap_sw2, "rb") as f:
                            self.assertEqual(f.read(), b"TRANSCODED_V1")
                        os.remove(clean_sw)
                        os.remove(clean_sw2)

                    # If snapshot copy fails during cache publish/hit, pin_run_snapshot=True still returns private tmp_path
                    fail_copy_src = os.path.join(tmpdir, "failcopy.mp4")
                    with open(fail_copy_src, "wb") as f:
                        f.write(b"raw4k_fc")
                    with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                            mock.patch.object(run_cli, "_copy_regular_file_nofollow", return_value=None), \
                            mock.patch.object(run_cli.subprocess, "run", side_effect=subproc_v1):
                        snap_fc, clean_fc = run_cli.prepare_normalized_face_input(
                            fail_copy_src, snap_out, pin_run_snapshot=True
                        )
                        self.assertEqual(snap_fc, clean_fc)
                        self.assertTrue(snap_fc.startswith(snap_out))
                        os.remove(clean_fc)

            # Symlink at sidecar path pointing to source video must NEVER truncate source video
            sym_src = os.path.join(tmpdir, "symtest.mp4")
            with open(sym_src, "wb") as f:
                f.write(b"PRECIOUS_SOURCE_VIDEO")
            sym_cache = run_cli.build_normalized_cache_path(sym_src)
            sym_meta = run_cli._cache_sidecar_path(sym_cache)
            os.symlink(sym_src, sym_meta)
            out_sub = os.path.join(tmpdir, "out")
            with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=fake_subproc_run):
                eff_sym, clean_sym = run_cli.prepare_normalized_face_input(sym_src, out_sub)
                self.assertEqual(eff_sym, clean_sym)
                with open(sym_src, "rb") as f:
                    self.assertEqual(f.read(), b"PRECIOUS_SOURCE_VIDEO")
                os.remove(clean_sym)

            # Concurrent publish race: if another writer creates cache_path during ffmpeg, first cache stays intact
            race_src = os.path.join(tmpdir, "race.mp4")
            with open(race_src, "wb") as f:
                f.write(b"race4k")
            race_cache = run_cli.build_normalized_cache_path(race_src)

            def racing_subproc(cmd, check):
                with open(cmd[-1], "wb") as out_f:
                    out_f.write(b"second_writer_bytes")
                with open(race_cache, "wb") as first_f:
                    first_f.write(b"FIRST_WRITER_WINS")

            with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=racing_subproc):
                eff_race, clean_race = run_cli.prepare_normalized_face_input(race_src, out_sub)
                self.assertEqual(eff_race, clean_race)
                with open(race_cache, "rb") as f:
                    self.assertEqual(f.read(), b"FIRST_WRITER_WINS")
                os.remove(clean_race)

            # Read-only source directory falls back cleanly to writable out_dir
            ro_src = os.path.join(tmpdir, "readonly_src.mp4")
            with open(ro_src, "wb") as f:
                f.write(b"ro4k")
            real_mkstemp = run_cli.tempfile.mkstemp

            def ro_mkstemp(*args, **kwargs):
                if kwargs.get("dir") == tmpdir:
                    raise PermissionError("read-only asset dir")
                return real_mkstemp(*args, **kwargs)

            with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                    mock.patch.object(run_cli.tempfile, "mkstemp", side_effect=ro_mkstemp), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=fake_subproc_run):
                eff_ro, clean_ro = run_cli.prepare_normalized_face_input(ro_src, out_sub)
                self.assertEqual(eff_ro, clean_ro)
                self.assertTrue(eff_ro.startswith(out_sub))
                os.remove(clean_ro)

            # Unowned pre-existing *_1080p.mp4 without sidecar is never overwritten or reused
            user_src = os.path.join(tmpdir, "lecture.mp4")
            user_existing = os.path.join(tmpdir, "lecture_1080p.mp4")
            with open(user_src, "wb") as f:
                f.write(b"src4k")
            with open(user_existing, "wb") as f:
                f.write(b"user_owned_file")
            with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=fake_subproc_run):
                eff_u, clean_u = run_cli.prepare_normalized_face_input(user_src, out_sub)
                self.assertEqual(eff_u, clean_u)
                self.assertNotEqual(eff_u, user_existing)
                with open(user_existing, "rb") as f:
                    self.assertEqual(f.read(), b"user_owned_file")
                os.remove(clean_u)

            # Transient .video-timeline-* writes a temp file scheduled for cleanup
            dot_src = os.path.join(tmpdir, ".video-timeline-123.mp4")
            with open(dot_src, "wb") as f:
                f.write(b"raw4k")
            with mock.patch.object(run_cli, "_probe_video_size", return_value=(2160, 3840, 30.0)), \
                    mock.patch.object(run_cli.subprocess, "run", side_effect=fake_subproc_run):
                eff_tmp, clean_tmp = run_cli.prepare_normalized_face_input(dot_src, tmpdir)
                self.assertEqual(eff_tmp, clean_tmp)
                self.assertTrue(os.path.isfile(clean_tmp))


if __name__ == "__main__":
    unittest.main()
