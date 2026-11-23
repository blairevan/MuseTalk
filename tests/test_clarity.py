import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from musetalk.utils import clarity


class ResizeGeneratedFaceTest(unittest.TestCase):
    """Interpolation choice when pasting the generated face back."""

    def _fake_cv2(self):
        calls = []

        def resize(image, size, interpolation=None):
            calls.append((size, interpolation))
            width, height = size
            return np.zeros((height, width, 3), dtype=np.uint8)

        fake = SimpleNamespace(resize=resize, INTER_AREA="area", INTER_LANCZOS4="lanczos4")
        return fake, calls

    def test_linear_mode_reproduces_legacy_call(self):
        """Legacy mode calls cv2.resize once without an interpolation flag."""
        fake, calls = self._fake_cv2()
        with mock.patch.dict("sys.modules", {"cv2": fake}):
            out = clarity.resize_generated_face(np.zeros((256, 256, 3)), 400, 420, "linear")
        self.assertEqual(calls, [((400, 420), None)])
        self.assertEqual(out.shape, (420, 400, 3))

    def test_lanczos_mode_enlarges_with_lanczos_only(self):
        """Pure enlargement uses a single LANCZOS4 resize."""
        fake, calls = self._fake_cv2()
        with mock.patch.dict("sys.modules", {"cv2": fake}):
            clarity.resize_generated_face(np.zeros((256, 256, 3)), 400, 420, "lanczos")
        self.assertEqual(calls, [((400, 420), "lanczos4")])

    def test_lanczos_mode_shrinks_with_area_only(self):
        """Pure reduction uses a single INTER_AREA resize."""
        fake, calls = self._fake_cv2()
        with mock.patch.dict("sys.modules", {"cv2": fake}):
            clarity.resize_generated_face(np.zeros((256, 256, 3)), 200, 180, "lanczos")
        self.assertEqual(calls, [((200, 180), "area")])

    def test_lanczos_mode_handles_mixed_axes(self):
        """One axis shrinks and the other grows: area first, then lanczos."""
        fake, calls = self._fake_cv2()
        with mock.patch.dict("sys.modules", {"cv2": fake}):
            clarity.resize_generated_face(np.zeros((256, 256, 3)), 200, 300, "lanczos")
        self.assertEqual(calls, [((200, 256), "area"), ((200, 300), "lanczos4")])

    def test_rejects_invalid_size_and_mode(self):
        """Bad sizes and unknown modes fail loudly."""
        fake, _ = self._fake_cv2()
        with mock.patch.dict("sys.modules", {"cv2": fake}):
            with self.assertRaises(ValueError):
                clarity.resize_generated_face(np.zeros((4, 4, 3)), 0, 4)
            with self.assertRaises(ValueError):
                clarity.resize_generated_face(np.zeros((4, 4, 3)), 4, 4, "cubic")


class UsmTest(unittest.TestCase):
    """Luma unsharp mask restricted to the lip ellipse."""

    def test_params_skip_small_faces_and_scale_with_enlargement(self):
        """No sharpening without enlargement; stronger with larger boxes."""
        self.assertIsNone(clarity.compute_usm_params(200, 200))
        small = clarity.compute_usm_params(300, 300)
        large = clarity.compute_usm_params(600, 600)
        self.assertLess(small[0], large[0])
        self.assertLessEqual(large[0], 0.85)
        self.assertLessEqual(large[1], 2.2)
        self.assertEqual(clarity.compute_usm_params(600, 600, amount_override=0.5)[0], 0.5)

    def test_gaussian_kernel_is_normalized(self):
        """The blur kernel sums to one and is symmetric."""
        kernel = clarity.gaussian_kernel_1d(1.5)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)
        np.testing.assert_allclose(kernel, kernel[::-1])

    def test_blur_preserves_constant_image(self):
        """Blurring a flat image leaves it flat, including the borders."""
        flat = np.full((20, 30), 77.0, dtype=np.float32)
        np.testing.assert_allclose(clarity.gaussian_blur_2d(flat, 2.0), flat, atol=1e-4)

    def test_ellipse_weight_peaks_on_lips_and_is_zero_at_corners(self):
        """Weight is 1 near the lips and 0 on the upper face and corners."""
        weight = clarity.mouth_ellipse_weight(100, 100)
        self.assertAlmostEqual(float(weight[68, 50]), 1.0, places=5)
        self.assertEqual(float(weight[0, 0]), 0.0)
        self.assertEqual(float(weight[:40].max()), 0.0)

    def test_sharpen_increases_lip_contrast_and_leaves_upper_face(self):
        """An edge on the lips gains contrast; the upper half is untouched."""
        face = np.full((400, 400, 3), 120, dtype=np.uint8)
        face[272:, :, :] = 160
        sharpened = clarity.sharpen_lip_region(face)
        self.assertLess(int(sharpened[271, 200, 0]), 120)
        self.assertGreater(int(sharpened[272, 200, 0]), 160)
        np.testing.assert_array_equal(sharpened[:150], face[:150])

    def test_sharpen_skips_flat_image_and_small_faces(self):
        """Flat regions stay identical; non-enlarged faces are returned as is."""
        flat = np.full((400, 400, 3), 90, dtype=np.uint8)
        np.testing.assert_array_equal(clarity.sharpen_lip_region(flat), flat)
        small = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        self.assertIs(clarity.sharpen_lip_region(small), small)

    def test_512_generated_size_scales_usm_correctly(self):
        """At native 512, 480px boxes skip USM and 600px boxes sharpen more gently than at 256."""
        self.assertIsNone(clarity.compute_usm_params(480, 480, generated_size=512))
        self.assertIsNotNone(clarity.compute_usm_params(480, 480, generated_size=256))
        params_256 = clarity.compute_usm_params(600, 600, generated_size=256)
        params_512 = clarity.compute_usm_params(600, 600, generated_size=512)
        self.assertLess(params_512[0], params_256[0])
        self.assertLess(params_512[1], params_256[1])
        face_480 = np.random.default_rng(1).integers(0, 255, (480, 480, 3), dtype=np.uint8)
        self.assertIs(clarity.sharpen_lip_region(face_480, generated_size=512), face_480)
        with self.assertRaises(ValueError):
            clarity.compute_usm_params(512, 512, generated_size=0)


class ResolutionValidationTest(unittest.TestCase):
    """Validation of inference generation resolutions and realtime cache matching."""

    def test_validate_resolution_accepts_256_and_512_only(self):
        """Only integer 256 and 512 are valid generation resolutions."""
        self.assertEqual(clarity.validate_resolution(256), 256)
        self.assertEqual(clarity.validate_resolution(512), 512)
        for bad in (True, False, 0, 128, 384, 1024, "512", 512.0, None):
            with self.assertRaises(ValueError):
                clarity.validate_resolution(bad)

    def test_cached_avatar_resolution_guard(self):
        """Legacy caches default to 256 and cross-resolution reuse fails fast."""
        clarity.validate_cached_avatar_resolution("av1", {}, 256)
        clarity.validate_cached_avatar_resolution("av1", {"resolution": 512}, 512)
        with self.assertRaises(ValueError):
            clarity.validate_cached_avatar_resolution("av1", {}, 512)
        with self.assertRaises(ValueError):
            clarity.validate_cached_avatar_resolution("av1", {"resolution": 512}, 256)


class TeethCorrectionTest(unittest.TestCase):
    """Mouth-interior brightness correction."""

    def test_darkens_bright_grey_pixels_inside_mouth_only(self):
        """Bright colourless pixels inside the mouth darken; outside stays put."""
        face = np.full((64, 64, 3), 230, dtype=np.uint8)
        mouth = np.zeros((64, 64), dtype=np.uint8)
        mouth[40:52, 20:44] = 255
        corrected = clarity.correct_teeth_whiteness(face, mouth)
        self.assertLess(int(corrected[46, 32].mean()), 230)
        self.assertLess(int(corrected[46, 32, 0]), int(corrected[46, 32, 2]))
        np.testing.assert_array_equal(corrected[:30], face[:30])

    def test_skips_without_mouth_or_teeth(self):
        """Missing, empty or toothless mouths return the input unchanged."""
        face = np.full((32, 32, 3), 230, dtype=np.uint8)
        self.assertIs(clarity.correct_teeth_whiteness(face, None), face)
        self.assertIs(clarity.correct_teeth_whiteness(face, np.zeros((32, 32))), face)
        dark = np.full((32, 32, 3), 60, dtype=np.uint8)
        self.assertIs(clarity.correct_teeth_whiteness(dark, np.ones((32, 32))), dark)

    def test_rejects_mismatched_mask(self):
        """A mask of the wrong size is an error, not a silent skip."""
        with self.assertRaises(ValueError):
            clarity.correct_teeth_whiteness(np.zeros((8, 8, 3), np.uint8), np.ones((4, 4)))


class BlendFacesTest(unittest.TestCase):
    """Generated/closed-mouth blending for pauses."""

    def test_blend_endpoints_and_midpoint(self):
        """Weight 0 keeps the generated face, 1 the closed face, 0.5 averages."""
        generated = np.zeros((2, 2, 3), dtype=np.uint8)
        closed = np.full((2, 2, 3), 200, dtype=np.uint8)
        self.assertIs(clarity.blend_faces(generated, closed, 0.0), generated)
        self.assertIs(clarity.blend_faces(generated, closed, 1.0), closed)
        self.assertEqual(int(clarity.blend_faces(generated, closed, 0.5)[0, 0, 0]), 100)
        with self.assertRaises(ValueError):
            clarity.blend_faces(generated, np.zeros((3, 3, 3), np.uint8), 0.5)


if __name__ == "__main__":
    unittest.main()
