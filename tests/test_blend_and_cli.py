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

    def test_auto_selects_512_with_safe_defaults_when_512_weights_exist(self):
        """Unmodified v1.5 CLI auto-enables 512 with bs=2, fp16, pe=false, ref=auto when 512 weights exist."""
        opts = self._run([], has_512_weights=True)
        self.assertEqual(opts["--resolution"], "512")
        self.assertEqual(opts["--unet_model_path"], "./models/musetalkV15_512/unet.pth")
        self.assertEqual(opts["--unet_config"], "./models/musetalkV15_512/musetalk.json")
        self.assertEqual(opts["--batch_size"], "2")
        self.assertIn("--use_float16", self.last_cmd)
        self.assertEqual(opts["--use_audio_pe"], "false")
        self.assertEqual(opts["--ref_frame_mode"], "auto")

    def test_explicit_256_and_512_precision_overrides_work_when_512_weights_exist(self):
        """Explicit --resolution 256 or --no_float16/--batch_size overrides auto defaults."""
        forced_256 = self._run(["--resolution", "256"], has_512_weights=True)
        self.assertEqual(forced_256["--resolution"], "256")
        self.assertEqual(forced_256["--unet_model_path"], "./models/musetalkV15/unet.pth")
        self.assertEqual(forced_256["--batch_size"], "8")
        self.assertNotIn("--use_float16", self.last_cmd)
        self.assertEqual(forced_256["--use_audio_pe"], "true")
        self.assertEqual(forced_256["--ref_frame_mode"], "self")

        forced_fp32 = self._run(["--batch_size", "1", "--no_float16"], has_512_weights=True)
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


if __name__ == "__main__":
    unittest.main()
