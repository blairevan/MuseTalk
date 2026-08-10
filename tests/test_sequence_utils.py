import unittest

from musetalk.utils.sequence_utils import build_ping_pong_cycle


class BuildPingPongCycleTest(unittest.TestCase):
    """Tests for building seamless forward-and-reverse frame cycles."""

    def test_excludes_both_turning_points_from_repetition(self):
        """The cycle should not repeat either endpoint across loop boundaries."""
        cycle = build_ping_pong_cycle([0, 1, 2, 3])

        self.assertEqual(cycle, [0, 1, 2, 3, 2, 1])

    def test_preserves_single_item_cycle(self):
        """A single-item sequence should remain usable without duplication."""
        self.assertEqual(build_ping_pong_cycle([0]), [0])

    def test_preserves_empty_cycle(self):
        """An empty sequence should remain empty."""
        self.assertEqual(build_ping_pong_cycle([]), [])


if __name__ == "__main__":
    unittest.main()
