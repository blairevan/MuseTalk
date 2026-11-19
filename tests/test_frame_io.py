import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np

from musetalk.utils import frame_io
from musetalk.utils.sequence_utils import build_ping_pong_cycle


class FrameSequenceTest(unittest.TestCase):
    """Exercise demand-driven reads and source-frame alignment."""

    def test_reads_only_the_requested_frame_and_does_not_cache_pixels(self):
        """Opening a frame sequence must leave unread and released frames out of RAM."""
        imread = mock.Mock(side_effect=lambda _: np.zeros((2, 2, 3), dtype=np.uint8))
        with mock.patch.dict("sys.modules", {"cv2": SimpleNamespace(imread=imread)}):
            frames = frame_io.FrameSequence(["0.png", "1.png", "2.png"])
            self.assertEqual(len(frames), 3)
            imread.assert_not_called()

            image = frames[2]
            image_ref = weakref.ref(image)
            del image
            self.assertIsNone(image_ref())
            imread.assert_called_once_with("2.png")

    def test_reads_original_frames_in_ping_pong_order(self):
        """Each repeated output position must resolve to the corresponding source frame."""
        imread = mock.Mock(side_effect=lambda path: np.full((1, 1, 3), int(path[0])))
        with mock.patch.dict("sys.modules", {"cv2": SimpleNamespace(imread=imread)}):
            frames = frame_io.FrameSequence(["0.png", "1.png", "2.png", "3.png"])
            cycle = build_ping_pong_cycle(list(range(len(frames))))
            pixels = [int(frames[cycle[i % len(cycle)]][0, 0, 0]) for i in range(8)]

        self.assertEqual(pixels, [0, 1, 2, 3, 2, 1, 0, 1])

    def test_reports_an_unreadable_frame_at_its_original_position(self):
        """A failed decode must stop the read with the source path and preserve indexing."""
        imread = mock.Mock(return_value=None)
        with mock.patch.dict("sys.modules", {"cv2": SimpleNamespace(imread=imread)}):
            frames = frame_io.FrameSequence(["0.png", "broken.png", "2.png"])
            with self.assertRaisesRegex(ValueError, "broken\\.png"):
                frames[1]
