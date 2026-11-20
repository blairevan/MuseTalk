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

    def _run(self, extra_args):
        import run_cli

        captured = {}

        def fake_run(cmd, check):
            captured["cmd"] = cmd

        argv = ["run_cli.py", "--face", "f.png", "--audio", "a.wav", "--outfile", "/tmp/o.mp4"] + extra_args
        with mock.patch.object(run_cli, "validate_inputs", return_value=True), \
                mock.patch.object(run_cli.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(sys, "argv", argv):
            run_cli.main()
        cmd = captured["cmd"]
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
        """The four A/B-approved options are on; experimental ones stay off."""
        opts = self._run([])
        self.assertEqual(opts["--vae_latent_mode"], "mode")
        self.assertEqual(opts["--paste_interpolation"], "lanczos")
        self.assertEqual(opts["--enhance_mode"], "usm")
        self.assertEqual(opts["--protect_upper_face"], "true")
        self.assertEqual(opts["--fix_teeth_whiteness"], "false")
        self.assertEqual(opts["--close_mouth_pauses"], "false")
        self.assertEqual(opts["--pause_top_db"], "38.0")
        self.assertEqual(opts["--use_audio_pe"], "true")

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
