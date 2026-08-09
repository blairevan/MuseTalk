import unittest

from musetalk.utils.audio_utils import audio_samples_to_frame_range, parse_bool


class ActiveAudioFrameRangeTest(unittest.TestCase):
    """Tests for mapping detected audio boundaries to video frames."""

    def test_maps_leading_and_trailing_silence_to_frame_range(self):
        """The start rounds down while the end rounds up to preserve speech."""
        frame_range = audio_samples_to_frame_range(
            start_sample=4800,
            end_sample=20800,
            sample_rate=16000,
            fps=25,
        )

        self.assertEqual(frame_range, (7, 33))


class ParseBoolTest(unittest.TestCase):
    """Tests for CLI boolean parsing."""

    def test_parse_bool_accepts_common_values(self):
        """Common case-insensitive boolean spellings should be supported."""
        self.assertTrue(parse_bool("TRUE"))
        self.assertTrue(parse_bool("on"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool("No"))

    def test_parse_bool_rejects_unknown_value(self):
        """Unknown values should fail instead of silently changing behavior."""
        with self.assertRaises(ValueError):
            parse_bool("maybe")


if __name__ == "__main__":
    unittest.main()
