"""Learning-rate schedule sizing for the manual gradient-accumulation loop."""


def compute_schedule_steps(max_train_steps, lr_warmup_steps, accumulation_steps, num_processes=1):
    """Translate config step counts into scheduler step counts.

    ``train.py`` advances ``global_step`` on every micro-step and only calls the
    optimizers and schedulers when ``(global_step + 1)`` is a multiple of
    ``accumulation_steps``. Training stops at ``max_train_steps`` micro-steps,
    so exactly ``max_train_steps // accumulation_steps`` updates happen; any
    trailing micro-steps never reach an update. ``max_train_steps`` and
    ``lr_warmup_steps`` are micro-step counts, and the schedulers are sized in
    optimizer updates. Accelerate's prepared scheduler advances
    ``num_processes`` times per call, so the generator schedule is scaled by
    it; the discriminator schedulers are not prepared.

    Args:
        max_train_steps (int): Training length in micro-steps.
        lr_warmup_steps (int): Warm-up length in micro-steps.
        accumulation_steps (int): Micro-steps per optimizer update.
        num_processes (int): Number of distributed processes.

    Returns:
        dict[str, int]: ``optimizer_updates``, ``scheduler_total_steps``,
        ``scheduler_warmup_steps`` and ``discriminator_t_max``.

    Raises:
        ValueError: If any count is invalid or no update would ever run.
    """
    accumulation_steps = int(accumulation_steps)
    num_processes = int(num_processes)
    if accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {accumulation_steps}")
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    if int(max_train_steps) < 1 or int(lr_warmup_steps) < 0:
        raise ValueError("max_train_steps must be >= 1 and lr_warmup_steps >= 0")

    optimizer_updates = int(max_train_steps) // accumulation_steps
    if optimizer_updates < 1:
        raise ValueError(
            f"max_train_steps ({max_train_steps}) is smaller than gradient_accumulation_steps "
            f"({accumulation_steps}); no optimizer update would run"
        )
    if int(max_train_steps) % accumulation_steps:
        print(
            f"Warning: the last {int(max_train_steps) % accumulation_steps} micro-steps never reach "
            "an optimizer update; make max_train_steps a multiple of gradient_accumulation_steps"
        )
    warmup_updates = int(lr_warmup_steps) // accumulation_steps
    return {
        "optimizer_updates": optimizer_updates,
        "scheduler_total_steps": optimizer_updates * num_processes,
        "scheduler_warmup_steps": warmup_updates * num_processes,
        "discriminator_t_max": optimizer_updates,
    }


def validate_deepspeed_gan_config(is_deepspeed, gan_loss, mouth_gan_loss):
    """Reject DeepSpeed with GAN losses because extra backward calls mis-step the engine.

    In ``train.py``, the generator and each enabled discriminator call
    ``accelerator.backward()`` in the same batch. Under Accelerate's DeepSpeed
    engine, each backward call advances the generator engine's step counter,
    desynchronizing gradient accumulation and learning-rate schedules.

    Args:
        is_deepspeed (bool): Whether Accelerate is configured with DeepSpeed.
        gan_loss (float): Weight for the face discriminator loss.
        mouth_gan_loss (float): Weight for the mouth discriminator loss.

    Raises:
        ValueError: If DeepSpeed is active and either GAN loss is positive.
    """
    if is_deepspeed and (float(gan_loss) > 0 or float(mouth_gan_loss) > 0):
        raise ValueError(
            "DeepSpeed is not supported when gan_loss or mouth_gan_loss is enabled because "
            "each discriminator backward() advances the DeepSpeed engine step; launch with "
            "--config_file ./configs/training/gpu_single.yaml instead."
        )


def normalize_adam_betas(betas):
    """Convert configured Adam betas to a plain Python ``(beta1, beta2)`` tuple.

    OmegaConf parses YAML lists as ``ListConfig``. Passing a ``ListConfig`` into
    ``torch.optim.AdamW`` embeds it in the saved optimizer state dict, which
    fails to load under PyTorch 2.6+ ``torch.load(..., weights_only=True)``.

    Args:
        betas (Iterable[float]): Two-element sequence ``(beta1, beta2)``.

    Returns:
        tuple[float, float]: Plain Python float 2-tuple.

    Raises:
        ValueError: If ``betas`` does not contain exactly two numeric values.
    """
    if isinstance(betas, (str, bytes)):
        raise ValueError(f"Adam betas must be a 2-element sequence, got {betas!r}")
    values = tuple(float(beta) for beta in betas)
    if len(values) != 2:
        raise ValueError(f"Adam betas must have length 2, got {len(values)}")
    return values
