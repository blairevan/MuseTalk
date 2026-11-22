"""Memory-efficient VAE decoding helpers for high-resolution training."""


def resolve_vae_decode_chunk_size(raw_chunk_size):
    """Validate and normalize ``solver.vae_decode_chunk_size``.

    Args:
        raw_chunk_size (int | bool | None): Configured max frame count decoded
            per slice. ``None``, ``False``, or ``0`` disables chunked decoding
            and preserves the single-call ``vae.decode`` path.

    Returns:
        int | None: Positive chunk size, or ``None`` when disabled.

    Raises:
        ValueError: If ``raw_chunk_size`` is ``True``, non-integer, or negative.
    """
    if raw_chunk_size is None or raw_chunk_size is False:
        return None
    if isinstance(raw_chunk_size, bool) or not isinstance(raw_chunk_size, int):
        raise ValueError(
            "solver.vae_decode_chunk_size must be a non-negative integer or null, "
            f"got {raw_chunk_size!r}"
        )
    if raw_chunk_size < 0:
        raise ValueError(
            f"solver.vae_decode_chunk_size must be >= 0, got {raw_chunk_size}"
        )
    if raw_chunk_size == 0:
        return None
    return raw_chunk_size


def decode_latents_for_training(vae, latents, chunk_size=None, use_checkpoint=False):
    """Decode predicted latents into images, optionally with chunked checkpointing.

    High-resolution multi-frame training (such as 16 frames at 512x512 in FP32)
    retains tens of gigabytes of autograd activations inside the frozen VAE
    decoder when decoded in one pass. Splitting along the batch dimension and
    wrapping each slice in non-reentrant gradient checkpointing discards decoder
    intermediates between slices while computing exact gradients for ``latents``.

    Args:
        vae: Diffusers ``AutoencoderKL`` (or compatible decoder exposing
            ``.decode(z).sample``).
        latents (torch.Tensor): Scaled latent tensor of shape ``(N, C, H, W)``.
        chunk_size (int | bool | None): Max frames decoded per slice. ``None``,
            ``False``, or ``0`` executes the legacy full-batch decode.
        use_checkpoint (bool): When ``True`` and ``chunk_size`` is active, wrap
            each slice in ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
            if autograd is active and ``latents.requires_grad`` is ``True``.

    Returns:
        torch.Tensor: Decoded image tensor of shape ``(N, 3, H*8, W*8)``.
    """
    resolved_chunk_size = resolve_vae_decode_chunk_size(chunk_size)
    if resolved_chunk_size is None or latents.shape[0] == 0:
        return vae.decode(latents).sample

    import torch
    from torch.utils.checkpoint import checkpoint

    should_checkpoint = bool(
        use_checkpoint
        and torch.is_grad_enabled()
        and getattr(latents, "requires_grad", False)
    )

    def _decode_slice(latent_slice):
        return vae.decode(latent_slice).sample

    batch_size = latents.shape[0]
    decoded_chunks = []
    for start in range(0, batch_size, resolved_chunk_size):
        latent_slice = latents[start : start + resolved_chunk_size]
        if should_checkpoint:
            decoded = checkpoint(_decode_slice, latent_slice, use_reentrant=False)
        else:
            decoded = _decode_slice(latent_slice)
        decoded_chunks.append(decoded)

    if len(decoded_chunks) == 1:
        return decoded_chunks[0]
    return torch.cat(decoded_chunks, dim=0)
