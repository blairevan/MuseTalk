"""Crash-safe publishing of resumable training checkpoints."""

import fcntl
import os
import re
import shutil

STAGING_PREFIX = "staging-"
RETIRED_PREFIX = f"{STAGING_PREFIX}old-"
LOCK_FILE_NAME = ".train.lock"
CUSTOM_CHECKPOINT_PATTERN = re.compile(r"^custom_checkpoint_\d+\.pkl$")


def validate_resumable_interval(checkpointing_steps, accumulation_steps):
    """Require resumable checkpoints to fall on gradient-accumulation boundaries.

    A checkpoint saved in the middle of an accumulation window would lose the
    gradients already accumulated in ``.grad``, so resuming would not continue
    the same update.

    Args:
        checkpointing_steps (int): Micro-steps between checkpoints.
        accumulation_steps (int): Micro-steps per optimizer update.

    Raises:
        ValueError: If the interval is not a positive multiple of the accumulation.
    """
    checkpointing_steps = int(checkpointing_steps)
    accumulation_steps = int(accumulation_steps)
    if checkpointing_steps < 1 or checkpointing_steps % accumulation_steps:
        raise ValueError(
            f"checkpointing_steps ({checkpointing_steps}) must be a positive multiple of "
            f"gradient_accumulation_steps ({accumulation_steps}) when save_training_state is enabled"
        )


def acquire_experiment_lock(save_dir):
    """Take an exclusive, non-blocking lock on an experiment directory.

    Two training jobs writing to the same directory would delete each other's
    staging checkpoints and overwrite each other's weights, so the second job
    fails fast instead. The lock is released when the returned file object is
    closed or the process exits.

    Args:
        save_dir (str): Experiment output directory.

    Returns:
        IO: Open lock file; keep a reference for the lifetime of the job.

    Raises:
        RuntimeError: If another process holds the lock.
    """
    os.makedirs(save_dir, exist_ok=True)
    lock_file = open(os.path.join(save_dir, LOCK_FILE_NAME), "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another training job is already using {save_dir}; use a different output_dir or exp_name"
        ) from error
    return lock_file


def restore_retired_checkpoints(save_dir):
    """Put back complete checkpoints left under a retired name by an interrupted replace.

    Args:
        save_dir (str): Experiment output directory.

    Returns:
        list[str]: Names of the restored ``checkpoint-N`` directories.
    """
    if not os.path.isdir(save_dir):
        return []
    restored = []
    for name in sorted(os.listdir(save_dir)):
        if not name.startswith(RETIRED_PREFIX):
            continue
        final_name = name[len(RETIRED_PREFIX):]
        final_path = os.path.join(save_dir, final_name)
        if not os.path.exists(final_path):
            os.replace(os.path.join(save_dir, name), final_path)
            restored.append(final_name)
    return restored


def staging_checkpoint_dir(save_dir, step):
    """Return the temporary directory a checkpoint is written to before publishing.

    The name does not start with ``checkpoint``, so the resume logic in
    ``train.py`` never selects a partially written state.

    Args:
        save_dir (str): Experiment output directory.
        step (int): Global step of the checkpoint.

    Returns:
        str: Staging directory path.
    """
    return os.path.join(save_dir, f"{STAGING_PREFIX}checkpoint-{int(step)}")


def clear_staging_dirs(save_dir):
    """Remove leftover staging directories from interrupted saves.

    Retired copies whose ``checkpoint-N`` is missing are restored first, so a
    complete checkpoint is never deleted here.

    Args:
        save_dir (str): Experiment output directory.

    Returns:
        list[str]: Names of the removed directories.
    """
    if not os.path.isdir(save_dir):
        return []
    restore_retired_checkpoints(save_dir)
    removed = []
    for name in sorted(os.listdir(save_dir)):
        path = os.path.join(save_dir, name)
        if name.startswith(STAGING_PREFIX) and os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(name)
    return removed


def publish_checkpoint(staging_path, final_path):
    """Move a fully written staging directory to its final ``checkpoint-N`` name.

    An existing ``checkpoint-N`` is first renamed to a ``staging-old-*`` name;
    if moving the new state in then fails, the old copy is renamed back. If
    the process dies in between, :func:`restore_retired_checkpoints` puts it
    back before the next resume or save. The published path is therefore
    never a half-deleted directory.

    Args:
        staging_path (str): Directory produced by ``accelerator.save_state``.
        final_path (str): Published ``checkpoint-N`` directory.

    Raises:
        FileNotFoundError: If the staging directory does not exist.
    """
    if not os.path.isdir(staging_path):
        raise FileNotFoundError(f"Staging checkpoint not found: {staging_path}")
    retired_path = None
    if os.path.isdir(final_path):
        retired_path = os.path.join(
            os.path.dirname(final_path), f"{RETIRED_PREFIX}{os.path.basename(final_path)}"
        )
        if os.path.isdir(retired_path):
            shutil.rmtree(retired_path)
        os.replace(final_path, retired_path)
    try:
        os.replace(staging_path, final_path)
    except BaseException:
        if retired_path is not None and not os.path.exists(final_path):
            os.replace(retired_path, final_path)
        raise
    if retired_path is not None:
        shutil.rmtree(retired_path)


def validate_resume_step(step, accumulation_steps):
    """Reject resuming from a checkpoint that is not on an accumulation boundary.

    Such a checkpoint was either saved mid-window by older code or with a
    different accumulation setting; resuming it would apply an update built
    from fewer micro-steps than the current setting expects.

    Args:
        step (int): Global micro-step encoded in the checkpoint name.
        accumulation_steps (int): Current ``gradient_accumulation_steps``.

    Raises:
        ValueError: If ``step`` is not a multiple of ``accumulation_steps``.
    """
    if int(step) % int(accumulation_steps):
        raise ValueError(
            f"Cannot resume from step {step}: it is not a multiple of gradient_accumulation_steps "
            f"({accumulation_steps}). Use a matching accumulation setting or start a new run."
        )


def count_custom_checkpoints(checkpoint_dir):
    """Count Accelerate custom checkpoint files inside a checkpoint directory.

    Matches the ``custom_checkpoint_<index>.pkl`` naming checked by
    ``Accelerator.load_state``.

    Args:
        checkpoint_dir (str): Path to a ``checkpoint-N`` directory.

    Returns:
        int: Number of matching custom checkpoint files.
    """
    if not os.path.isdir(checkpoint_dir):
        return 0
    return sum(1 for name in os.listdir(checkpoint_dir) if CUSTOM_CHECKPOINT_PATTERN.match(name))


def should_load_custom_checkpoints(checkpoint_dir, expected_count):
    """Decide whether to register custom objects before ``Accelerator.load_state``.

    Older checkpoints created before discriminators were registered contain no
    ``custom_checkpoint_*.pkl`` files. Registering discriminators before
    ``load_state`` on such a checkpoint causes Accelerate to raise a count
    mismatch error.

    Args:
        checkpoint_dir (str): Path to the ``checkpoint-N`` directory to resume.
        expected_count (int): Number of discriminator-related objects to register.

    Returns:
        bool: ``True`` if custom objects should be registered before ``load_state``;
        ``False`` when resuming a legacy checkpoint with no custom state files while
        ``expected_count > 0`` (caller should warn, call ``load_state``, and register
        afterwards so subsequent saves include them).

    Raises:
        ValueError: If the checkpoint contains custom state files whose count
            does not match ``expected_count``.
    """
    expected_count = int(expected_count)
    found_count = count_custom_checkpoints(checkpoint_dir)
    if found_count == expected_count:
        return True
    if found_count == 0 and expected_count > 0:
        return False
    raise ValueError(
        f"Checkpoint {checkpoint_dir} contains {found_count} custom state file(s), "
        f"expected {expected_count} for the enabled discriminators."
    )
