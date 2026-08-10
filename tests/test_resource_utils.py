import unittest
from types import SimpleNamespace
from unittest import mock

from musetalk.utils.resource_utils import release_torch_resources


class FakeTorchResource:
    """Record device transfers for resource cleanup tests."""

    def __init__(self, error=None):
        """Configure an optional transfer failure."""
        self.devices = []
        self.error = error

    def to(self, device):
        """Record the requested device and optionally fail."""
        self.devices.append(device)
        if self.error is not None:
            raise self.error


class ReleaseTorchResourcesTest(unittest.TestCase):
    """Tests for best-effort GPU resource cleanup between tasks."""

    def test_moves_every_resource_to_cpu_and_clears_cache(self):
        """Every supplied model should leave CUDA before cache cleanup."""
        first = FakeTorchResource()
        second = FakeTorchResource()
        empty_cache = mock.Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=mock.Mock(return_value=True),
                empty_cache=empty_cache,
            )
        )

        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            errors = release_torch_resources(first, None, second)

        self.assertEqual(first.devices, ["cpu"])
        self.assertEqual(second.devices, ["cpu"])
        self.assertEqual(errors, [])
        empty_cache.assert_called_once_with()

    def test_continues_cleanup_after_one_resource_fails(self):
        """A cleanup failure must not retain later resources or mask task errors."""
        failing = FakeTorchResource(RuntimeError("transfer failed"))
        succeeding = FakeTorchResource()
        empty_cache = mock.Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=mock.Mock(return_value=True),
                empty_cache=empty_cache,
            )
        )

        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            errors = release_torch_resources(failing, succeeding)

        self.assertEqual(succeeding.devices, ["cpu"])
        self.assertEqual(len(errors), 1)
        self.assertIn("transfer failed", errors[0])
        empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
