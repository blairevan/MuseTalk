"""Unit tests for chunked and checkpointed training-time VAE decoding."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from musetalk.utils.train_schedule import normalize_adam_betas
from musetalk.utils.vae_decode import (
    decode_latents_for_training,
    resolve_vae_decode_chunk_size,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeTensor:
    """Minimal batch-sliceable tensor stub for torch-free unit tests."""

    def __init__(self, array, requires_grad=True):
        """Initialize the stub with a NumPy array and grad flag."""
        self.array = np.asarray(array, dtype=np.float32)
        self.requires_grad = requires_grad

    @property
    def shape(self):
        """Return the underlying array shape."""
        return self.array.shape

    def __getitem__(self, item):
        """Return a sliced tensor stub preserving ``requires_grad``."""
        return _FakeTensor(self.array[item], requires_grad=self.requires_grad)


class _RecordingVae:
    """Mock VAE that records slice shapes and scales inputs deterministically."""

    def __init__(self):
        """Initialize call recording state."""
        self.call_shapes = []

    def decode(self, latents):
        """Record input batch shape and return ``2 * latents`` as ``.sample``."""
        self.call_shapes.append(tuple(latents.shape))
        return SimpleNamespace(
            sample=_FakeTensor(latents.array * 2.0, requires_grad=latents.requires_grad)
        )


def _build_fake_torch(grad_enabled=True):
    """Build mocked ``torch`` and ``torch.utils.checkpoint`` modules."""
    checkpoint_calls = []
    cat_calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        checkpoint_calls.append(dict(kwargs))
        return fn(*args)

    def fake_cat(tensors, dim=0):
        cat_calls.append((len(tensors), dim))
        return _FakeTensor(
            np.concatenate([tensor.array for tensor in tensors], axis=dim),
            requires_grad=any(tensor.requires_grad for tensor in tensors),
        )

    checkpoint_mod = SimpleNamespace(checkpoint=fake_checkpoint)
    utils_mod = SimpleNamespace(checkpoint=checkpoint_mod)
    torch_mod = SimpleNamespace(
        is_grad_enabled=lambda: grad_enabled,
        cat=fake_cat,
        utils=utils_mod,
    )
    modules = {
        "torch": torch_mod,
        "torch.utils": utils_mod,
        "torch.utils.checkpoint": checkpoint_mod,
    }
    return modules, checkpoint_calls, cat_calls


class ResolveVaeDecodeChunkSizeTest(unittest.TestCase):
    """Validation tests for ``resolve_vae_decode_chunk_size``."""

    def test_disabled_values_normalize_to_none(self):
        """None, False, and 0 disable chunked decoding."""
        for value in (None, False, 0):
            self.assertIsNone(resolve_vae_decode_chunk_size(value))

    def test_positive_integers_are_accepted(self):
        """Positive integers pass through unchanged."""
        for value in (1, 2, 4, 16):
            self.assertEqual(resolve_vae_decode_chunk_size(value), value)

    def test_invalid_values_raise_value_error(self):
        """Booleans (True), negative ints, floats, and strings are rejected."""
        for value in (True, -1, -8, 1.5, "2", [2]):
            with self.assertRaises(ValueError):
                resolve_vae_decode_chunk_size(value)


class DecodeLatentsForTrainingMockTest(unittest.TestCase):
    """Behavioral tests for ``decode_latents_for_training`` without PyTorch."""

    def test_disabled_chunk_size_calls_decode_once_directly(self):
        """Disabled chunking preserves the legacy single-pass call."""
        vae = _RecordingVae()
        latents = _FakeTensor(np.arange(5 * 4, dtype=np.float32).reshape(5, 4))
        for disabled in (None, False, 0):
            vae.call_shapes.clear()
            out = decode_latents_for_training(
                vae, latents, chunk_size=disabled, use_checkpoint=True
            )
            self.assertEqual(vae.call_shapes, [(5, 4)])
            np.testing.assert_allclose(out.array, latents.array * 2.0)

    def test_empty_batch_calls_decode_directly(self):
        """Empty batches bypass slicing and checkpointing."""
        vae = _RecordingVae()
        latents = _FakeTensor(np.zeros((0, 4), dtype=np.float32))
        out = decode_latents_for_training(vae, latents, chunk_size=2, use_checkpoint=True)
        self.assertEqual(vae.call_shapes, [(0, 4)])
        self.assertEqual(out.shape, (0, 4))

    def test_chunked_checkpointed_decode_handles_uneven_tail(self):
        """5 frames with chunk_size=2 split into [2, 2, 1] with use_reentrant=False."""
        vae = _RecordingVae()
        latents = _FakeTensor(np.arange(5 * 4, dtype=np.float32).reshape(5, 4), requires_grad=True)
        modules, ckpt_calls, cat_calls = _build_fake_torch(grad_enabled=True)
        with mock.patch.dict("sys.modules", modules):
            out = decode_latents_for_training(vae, latents, chunk_size=2, use_checkpoint=True)
        self.assertEqual(vae.call_shapes, [(2, 4), (2, 4), (1, 4)])
        self.assertEqual(ckpt_calls, [{"use_reentrant": False}] * 3)
        self.assertEqual(cat_calls, [(3, 0)])
        np.testing.assert_allclose(out.array, latents.array * 2.0)

    def test_single_chunk_skips_concat_but_still_checkpoints(self):
        """When batch_size <= chunk_size, checkpoint still runs and cat is skipped."""
        vae = _RecordingVae()
        latents = _FakeTensor(np.arange(2 * 4, dtype=np.float32).reshape(2, 4), requires_grad=True)
        modules, ckpt_calls, cat_calls = _build_fake_torch(grad_enabled=True)
        with mock.patch.dict("sys.modules", modules):
            out = decode_latents_for_training(vae, latents, chunk_size=4, use_checkpoint=True)
        self.assertEqual(vae.call_shapes, [(2, 4)])
        self.assertEqual(ckpt_calls, [{"use_reentrant": False}])
        self.assertEqual(cat_calls, [])
        np.testing.assert_allclose(out.array, latents.array * 2.0)

    def test_skips_checkpoint_when_disabled_or_no_grad(self):
        """Checkpointing is skipped if use_checkpoint, grad mode, or requires_grad is False."""
        cases = [
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ]
        for use_checkpoint, grad_enabled, requires_grad in cases:
            vae = _RecordingVae()
            latents = _FakeTensor(
                np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
                requires_grad=requires_grad,
            )
            modules, ckpt_calls, cat_calls = _build_fake_torch(grad_enabled=grad_enabled)
            with mock.patch.dict("sys.modules", modules):
                out = decode_latents_for_training(
                    vae, latents, chunk_size=2, use_checkpoint=use_checkpoint
                )
            self.assertEqual(vae.call_shapes, [(2, 3), (2, 3)])
            self.assertEqual(ckpt_calls, [])
            self.assertEqual(cat_calls, [(2, 0)])
            np.testing.assert_allclose(out.array, latents.array * 2.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment")
class DecodeLatentsForTrainingTorchTest(unittest.TestCase):
    """Real PyTorch autograd equivalence tests for frozen VAE + checkpointing."""

    def test_frozen_vae_checkpointed_gradients_match_direct_decode(self):
        """Checkpointed chunked decode matches direct decode outputs and gradients."""
        import torch
        import torch.nn as nn

        class _TinyFrozenVae(nn.Module):
            """Minimal deterministic decoder matching Diffusers ``.decode(z).sample``."""

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(4, 8, kernel_size=3, padding=1),
                    nn.GroupNorm(2, 8),
                    nn.SiLU(),
                    nn.Conv2d(8, 3, kernel_size=3, padding=1),
                )

            def decode(self, z):
                return SimpleNamespace(sample=self.net(z))

        torch.manual_seed(7)
        vae = _TinyFrozenVae().eval()
        vae.requires_grad_(False)

        base = torch.randn(5, 4, 8, 8)
        latents_ref = base.clone().requires_grad_(True)
        latents_ckpt = base.clone().requires_grad_(True)
        weights = torch.randn(5, 3, 8, 8)

        out_ref = vae.decode(latents_ref).sample
        (out_ref * weights).sum().backward()

        out_ckpt = decode_latents_for_training(
            vae, latents_ckpt, chunk_size=2, use_checkpoint=True
        )
        (out_ckpt * weights).sum().backward()

        self.assertTrue(torch.allclose(out_ckpt, out_ref, atol=1e-6, rtol=1e-5))
        self.assertIsNotNone(latents_ckpt.grad)
        self.assertTrue(torch.isfinite(latents_ckpt.grad).all())
        self.assertTrue(torch.allclose(latents_ckpt.grad, latents_ref.grad, atol=1e-6, rtol=1e-5))
        for param in vae.parameters():
            self.assertIsNone(param.grad)

    def test_adamw_with_normalized_betas_loads_under_weights_only_true(self):
        """Optimizer state dicts built with normalized betas load under PyTorch 2.6+ weights_only=True."""
        import io
        import torch

        class _FakeListConfig(list):
            """Custom list subclass simulating OmegaConf ListConfig."""

        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.AdamW([param], lr=float(1e-4), betas=normalize_adam_betas(_FakeListConfig([0.5, 0.999])))
        buf = io.BytesIO()
        torch.save(opt.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["param_groups"][0]["betas"], (0.5, 0.999))


class NormalizeAdamBetasTest(unittest.TestCase):
    """Validation tests for ``normalize_adam_betas``."""

    def test_converts_custom_sequence_to_plain_float_tuple(self):
        """Custom list subclasses become plain Python float 2-tuples."""
        class _FakeListConfig(list):
            pass

        result = normalize_adam_betas(_FakeListConfig([0.5, 0.999]))
        self.assertIs(type(result), tuple)
        self.assertEqual(result, (0.5, 0.999))
        self.assertTrue(all(type(item) is float for item in result))

    def test_rejects_wrong_length_or_string(self):
        """Invalid beta sequences fail fast with ValueError."""
        for bad in ([0.5], [0.5, 0.9, 0.99], "0.5"):
            with self.assertRaises(ValueError):
                normalize_adam_betas(bad)


if __name__ == "__main__":
    unittest.main()
