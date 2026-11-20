import os
import random
import subprocess
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import numpy as np

from musetalk.data import data_utils
from musetalk.utils import audio_utils
from musetalk.utils.checkpoint_utils import (
    acquire_experiment_lock,
    clear_staging_dirs,
    count_custom_checkpoints,
    publish_checkpoint,
    restore_retired_checkpoints,
    should_load_custom_checkpoints,
    staging_checkpoint_dir,
    validate_resumable_interval,
    validate_resume_step,
)
from musetalk.utils.mask_utils import confine_mask_below
from musetalk.utils.train_schedule import compute_schedule_steps, validate_deepspeed_gan_config


def _write_wav(path, sample_rate=16000, channels=1, sample_width=2, frames=160):
    """Write a silent PCM WAV with the requested format."""
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * channels * sample_width)


class PauseCloseWeightsTest(unittest.TestCase):
    """Mid-sentence pause detection mapped to closed-mouth weights."""

    def test_long_interior_gap_fades_in_and_out(self):
        """A 10-frame gap gets cosine ramps and full weight in the middle."""
        weights = audio_utils.pause_close_weights([(0, 10), (20, 30)], 30, 8, 3)
        self.assertTrue(np.all(weights[:10] == 0))
        self.assertTrue(np.all(weights[20:] == 0))
        self.assertTrue(np.all(weights[13:17] == 1))
        self.assertTrue(0 < weights[10] < weights[11] < weights[12] < 1)
        np.testing.assert_allclose(weights[10:13], weights[17:20][::-1])

    def test_short_gap_and_edge_silence_are_ignored(self):
        """Gaps below the threshold and leading/trailing silence stay at zero."""
        weights = audio_utils.pause_close_weights([(5, 10), (14, 20)], 40, 8, 4)
        self.assertEqual(float(weights.sum()), 0.0)
        self.assertEqual(float(audio_utils.pause_close_weights([(5, 10)], 40).sum()), 0.0)

    def test_minimum_gap_with_default_fade_reaches_full_close(self):
        """An 8-frame gap with 4-frame fades still has fully closed frames."""
        weights = audio_utils.pause_close_weights([(0, 10), (18, 30)], 30, 8, 4)
        self.assertGreaterEqual(int(np.count_nonzero(weights == 1.0)), 1)
        self.assertTrue(np.all(weights[10:18] > 0))

    def test_overlapping_intervals_merge_and_clip(self):
        """Overlapping spans merge; spans past the end are clipped."""
        weights = audio_utils.pause_close_weights([(0, 6), (4, 10), (30, 99)], 35, 8, 2)
        self.assertTrue(np.all(weights[12:28] == 1))
        self.assertEqual(len(weights), 35)

    def test_frame_intervals_from_librosa_split(self):
        """Sample spans map to frame spans with floor/ceil rounding."""
        fake_librosa = SimpleNamespace(
            load=mock.Mock(return_value=(np.ones(32000, dtype=np.float32), 16000)),
            effects=SimpleNamespace(split=mock.Mock(return_value=np.array([[4800, 9600], [16000, 20800]]))),
        )
        with mock.patch.dict("sys.modules", {"librosa": fake_librosa}):
            intervals = audio_utils.get_active_audio_frame_intervals("a.wav", fps=25)
        self.assertEqual(intervals, [(7, 15), (25, 33)])


class EnsureWavTest(unittest.TestCase):
    """Conversion reuse and atomic replacement."""

    def test_mux_command_copies_video_and_encodes_audio(self):
        """The rendered video stream is copied; only audio is encoded."""
        cmd = audio_utils.build_audio_mux_command("a.mp3", "v.mp4", "out.mp4")
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")
        self.assertIn("1:v:0", cmd)
        self.assertIn("0:a:0", cmd)
        self.assertEqual(cmd[-1], "out.mp4")
        self.assertLess(cmd.index("a.mp3"), cmd.index("v.mp4"))

    def test_returns_conforming_wav_without_running_ffmpeg(self):
        """A 16 kHz mono PCM file is used directly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            _write_wav(path)
            with mock.patch.object(audio_utils.subprocess, "run") as run:
                self.assertEqual(audio_utils.ensure_wav(path), path)
            run.assert_not_called()

    def test_converts_other_formats_via_temp_file_and_rename(self):
        """Non-conforming audio is written to a temp file, then renamed; no cache reuse."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            _write_wav(path, sample_rate=48000, channels=2)

            def fake_run(cmd, **_):
                self.assertTrue(cmd[-1].endswith(".tmp.wav"))
                _write_wav(cmd[-1])

            with mock.patch.object(audio_utils.subprocess, "run", side_effect=fake_run):
                target = audio_utils.ensure_wav(path)
            self.assertEqual(target, os.path.join(tmp, "clip_wav_16k.wav"))
            self.assertTrue(audio_utils.is_pcm16_mono_16k(target))
            self.assertEqual(sorted(os.listdir(tmp)), ["clip.wav", "clip_wav_16k.wav"])

            with mock.patch.object(audio_utils.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(audio_utils.ensure_wav(path), target)
            run.assert_called_once()

    def test_sources_sharing_a_base_name_get_distinct_targets(self):
        """foo.mp3 and foo.m4a never write to the same converted file."""
        with tempfile.TemporaryDirectory() as tmp:
            sources = [os.path.join(tmp, name) for name in ("foo.mp3", "foo.m4a")]
            for source in sources:
                with open(source, "wb") as handle:
                    handle.write(b"compressed audio")

            def fake_run(cmd, **_):
                _write_wav(cmd[-1])

            with mock.patch.object(audio_utils.subprocess, "run", side_effect=fake_run):
                targets = [audio_utils.ensure_wav(source) for source in sources]
            self.assertEqual(
                [os.path.basename(target) for target in targets],
                ["foo_mp3_16k.wav", "foo_m4a_16k.wav"],
            )

    def test_failed_conversion_leaves_no_temp_file(self):
        """An ffmpeg failure removes its temp file and propagates."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp3")
            with open(path, "wb") as handle:
                handle.write(b"not audio")
            error = audio_utils.subprocess.CalledProcessError(1, "ffmpeg")
            with mock.patch.object(audio_utils.subprocess, "run", side_effect=error):
                with self.assertRaises(audio_utils.subprocess.CalledProcessError):
                    audio_utils.ensure_wav(path)
            self.assertEqual(os.listdir(tmp), ["clip.mp3"])

    def test_refuses_to_overwrite_input(self):
        """An explicit target equal to the input is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            _write_wav(path, sample_rate=48000)
            with self.assertRaises(ValueError):
                audio_utils.ensure_wav(path, target_path=path)


class ConvertedWavTest(unittest.TestCase):
    """Private, self-cleaning decode copies used by inference and training."""

    def test_conforming_input_is_used_directly(self):
        """No conversion and no extra file for 16 kHz mono PCM input."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            _write_wav(path)
            with mock.patch.object(audio_utils.subprocess, "run") as run:
                with audio_utils.converted_wav(path) as converted:
                    self.assertEqual(converted, path)
            run.assert_not_called()

    def test_conversion_is_private_and_removed_afterwards(self):
        """Nothing is written next to the source; the temp copy is deleted on exit."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp3")
            with open(path, "wb") as handle:
                handle.write(b"compressed audio")

            def fake_run(cmd, **_):
                _write_wav(cmd[-1])

            with mock.patch.object(audio_utils.subprocess, "run", side_effect=fake_run):
                with audio_utils.converted_wav(path) as first, audio_utils.converted_wav(path) as second:
                    self.assertNotEqual(first, second)
                    self.assertTrue(audio_utils.is_pcm16_mono_16k(first))
                    self.assertNotEqual(os.path.dirname(first), tmp)
            self.assertFalse(os.path.exists(first))
            self.assertFalse(os.path.exists(second))
            self.assertEqual(os.listdir(tmp), ["clip.mp3"])

    def test_failed_conversion_removes_temp_copy(self):
        """An ffmpeg failure propagates and leaves no temp file behind."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp3")
            with open(path, "wb") as handle:
                handle.write(b"not audio")
            created = []

            def failing_run(cmd, **_):
                created.append(cmd[-1])
                raise audio_utils.subprocess.CalledProcessError(1, "ffmpeg")

            with mock.patch.object(audio_utils.subprocess, "run", side_effect=failing_run):
                with self.assertRaises(audio_utils.subprocess.CalledProcessError):
                    with audio_utils.converted_wav(path):
                        pass
            self.assertFalse(os.path.exists(created[0]))


class DataUtilsTest(unittest.TestCase):
    """Dataset list parsing, mask bounds and RNG isolation."""

    def test_reads_every_line_without_skipping_a_header(self):
        """The first clip in a header-less list is kept."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("clip000_a.json\n\n# note\nclip001_b.json extra\n")
            self.assertEqual(data_utils.read_meta_list(path), ["clip000_a.json", "clip001_b.json"])

    def test_face_size_config_check(self):
        """min_face_size above image_size is rejected only when enabled."""
        data_utils.check_face_size_config(512, 512, True)
        data_utils.check_face_size_config(256, 400, False)
        with self.assertRaises(ValueError):
            data_utils.check_face_size_config(256, 400, True)

    def test_whisper_window_matches_legacy_inside_first_window(self):
        """Starts in the first 30 s keep the original window choice."""
        self.assertEqual(data_utils.whisper_window(100), (0, 100))
        self.assertEqual(data_utils.whisper_window(725), (4, 625))

    def test_whisper_window_follows_clip_past_30_seconds(self):
        """Later starts use the window they fall in, not the clip start."""
        self.assertEqual(data_utils.whisper_window(750), (30, 0))
        self.assertEqual(data_utils.whisper_window(1000), (30, 250))
        self.assertEqual(data_utils.whisper_window(1490), (34, 640))
        for start in (0, 699, 700, 749, 750, 1499, 2250):
            _, local = data_utils.whisper_window(start)
            self.assertGreaterEqual(local, 0)
            self.assertLessEqual(2 * (local + 50), 1500)

    def test_mouth_bounds_clip_to_image_size_not_face_size(self):
        """Bounds may extend past a smaller face box up to the crop size."""
        lips = np.array([[300.0, 440.0], [420.0, 470.0]])
        bounds = data_utils.mouth_mask_bounds(lips, 10, 0, 512)
        self.assertEqual(bounds[2], 430.0)
        self.assertLessEqual(bounds[3], 512.0)
        clipped = data_utils.mouth_mask_bounds(np.array([[480.0, 500.0], [520.0, 530.0]]), 10, 0, 512)
        self.assertEqual(clipped[2], 512.0)

    def test_seeded_state_is_reproducible_and_restored(self):
        """Same seed gives same draws; outer RNG streams continue unchanged."""
        random.seed(1)
        np.random.seed(1)
        expected_outer = (random.random(), float(np.random.rand()))
        random.seed(1)
        np.random.seed(1)
        with data_utils.seeded_random_state(7):
            first = (random.random(), float(np.random.rand()))
        with data_utils.seeded_random_state(7):
            second = (random.random(), float(np.random.rand()))
        self.assertEqual(first, second)
        self.assertEqual((random.random(), float(np.random.rand())), expected_outer)


class ConfineMaskTest(unittest.TestCase):
    """Upper-face alpha protection."""

    def test_zeroes_rows_above_boundary_only(self):
        """Rows above the boundary become zero; the rest keep their alpha."""
        mask = np.full((6, 4), 200, dtype=np.uint8)
        confined = confine_mask_below(mask, 3)
        self.assertEqual(int(confined[:3].max()), 0)
        np.testing.assert_array_equal(confined[3:], mask[3:])
        self.assertEqual(int(mask.min()), 200)


class ScheduleStepsTest(unittest.TestCase):
    """Scheduler sizing for the manual accumulation loop."""

    def test_no_accumulation_keeps_legacy_sizes(self):
        """With accum=1 on one process the sizes equal the config values."""
        steps = compute_schedule_steps(250000, 1000, 1, 1)
        self.assertEqual(steps["scheduler_total_steps"], 250000)
        self.assertEqual(steps["scheduler_warmup_steps"], 1000)

    def test_accumulation_sizes_in_optimizer_updates(self):
        """accum=8 sizes schedulers by updates instead of 8x the micro-steps."""
        steps = compute_schedule_steps(250000, 1000, 8, 1)
        self.assertEqual(steps["optimizer_updates"], 31250)
        self.assertEqual(steps["scheduler_total_steps"], 31250)
        self.assertEqual(steps["scheduler_warmup_steps"], 125)
        self.assertEqual(steps["discriminator_t_max"], 31250)
        self.assertEqual(compute_schedule_steps(100, 10, 1, 2)["scheduler_total_steps"], 200)

    def test_rejects_invalid_counts(self):
        """Zero accumulation or processes is an error."""
        with self.assertRaises(ValueError):
            compute_schedule_steps(100, 10, 0, 1)
        with self.assertRaises(ValueError):
            compute_schedule_steps(100, 10, 1, 0)

    def test_non_divisible_length_counts_only_executed_updates(self):
        """10 micro-steps with accum=4 run 2 updates, so the schedule has 2 steps."""
        with mock.patch("builtins.print"):
            steps = compute_schedule_steps(10, 4, 4, 1)
        self.assertEqual(steps["optimizer_updates"], 2)
        self.assertEqual(steps["scheduler_total_steps"], 2)
        self.assertEqual(steps["scheduler_warmup_steps"], 1)

    def test_rejects_length_shorter_than_one_update(self):
        """Fewer micro-steps than one accumulation window would never update."""
        with self.assertRaises(ValueError):
            compute_schedule_steps(3, 0, 4, 1)

    def test_rejects_deepspeed_when_gan_enabled(self):
        """DeepSpeed + GAN raises and points to gpu_single.yaml; non-DeepSpeed or no GAN passes."""
        validate_deepspeed_gan_config(False, 0.01, 0.01)
        validate_deepspeed_gan_config(True, 0, 0)
        with self.assertRaisesRegex(ValueError, "gpu_single.yaml"):
            validate_deepspeed_gan_config(True, 0.01, 0)
        with self.assertRaisesRegex(ValueError, "gpu_single.yaml"):
            validate_deepspeed_gan_config(True, 0, 0.01)


class CheckpointPublishTest(unittest.TestCase):
    """Resumable checkpoints are only visible once fully written."""

    def test_staging_name_is_ignored_by_resume_filter(self):
        """train.py resumes from names starting with 'checkpoint'; staging must not match."""
        staging = staging_checkpoint_dir("/exp", 500)
        self.assertFalse(os.path.basename(staging).startswith("checkpoint"))

    def test_resumable_interval_must_align_with_accumulation(self):
        """Checkpoints inside an accumulation window are rejected."""
        validate_resumable_interval(500, 1)
        validate_resumable_interval(512, 8)
        with self.assertRaises(ValueError):
            validate_resumable_interval(10, 4)
        with self.assertRaises(ValueError):
            validate_resumable_interval(0, 1)

    def test_resume_step_must_align_with_current_accumulation(self):
        """A checkpoint saved with another accumulation setting is not resumed."""
        validate_resume_step(12, 4)
        with self.assertRaises(ValueError):
            validate_resume_step(10, 4)

    def test_interrupted_replace_keeps_the_old_checkpoint_published(self):
        """If moving the new state in fails, the previous checkpoint-N is renamed back."""
        with tempfile.TemporaryDirectory() as tmp:
            staging = staging_checkpoint_dir(tmp, 500)
            final = os.path.join(tmp, "checkpoint-500")
            os.makedirs(staging)
            os.makedirs(final)
            with open(os.path.join(final, "old.bin"), "w") as handle:
                handle.write("old")
            real_replace = os.replace
            calls = []

            def flaky_replace(src, dst):
                calls.append((src, dst))
                if len(calls) == 2:
                    raise OSError("disk full")
                return real_replace(src, dst)

            with mock.patch("musetalk.utils.checkpoint_utils.os.replace", side_effect=flaky_replace):
                with self.assertRaises(OSError):
                    publish_checkpoint(staging, final)
            self.assertEqual(os.listdir(final), ["old.bin"])

    def test_retired_copy_left_by_a_crash_is_restored_not_deleted(self):
        """A complete retired checkpoint is put back before staging leftovers are cleared."""
        with tempfile.TemporaryDirectory() as tmp:
            retired = os.path.join(tmp, "staging-old-checkpoint-500")
            os.makedirs(retired)
            with open(os.path.join(retired, "model.bin"), "w") as handle:
                handle.write("complete")
            os.makedirs(staging_checkpoint_dir(tmp, 500))
            self.assertEqual(restore_retired_checkpoints(tmp), ["checkpoint-500"])
            self.assertEqual(os.listdir(os.path.join(tmp, "checkpoint-500")), ["model.bin"])
            os.makedirs(retired)
            clear_staging_dirs(tmp)
            self.assertEqual(sorted(os.listdir(tmp)), ["checkpoint-500"])

    def test_second_job_cannot_lock_the_same_experiment(self):
        """A running job's directory lock makes a second job fail fast."""
        with tempfile.TemporaryDirectory() as tmp:
            first = acquire_experiment_lock(tmp)
            try:
                code = (
                    "import sys; sys.path.insert(0, %r)\n"
                    "from musetalk.utils.checkpoint_utils import acquire_experiment_lock\n"
                    "try:\n    acquire_experiment_lock(%r)\nexcept RuntimeError:\n    sys.exit(3)\n"
                ) % (os.getcwd(), tmp)
                result = subprocess.run([sys.executable, "-c", code])
                self.assertEqual(result.returncode, 3)
            finally:
                first.close()
            acquire_experiment_lock(tmp).close()

    def test_publish_moves_complete_state_and_replaces_old_copy(self):
        """Publishing renames staging to checkpoint-N, replacing a stale copy."""
        with tempfile.TemporaryDirectory() as tmp:
            staging = staging_checkpoint_dir(tmp, 500)
            final = os.path.join(tmp, "checkpoint-500")
            os.makedirs(staging)
            os.makedirs(final)
            with open(os.path.join(final, "stale.bin"), "w") as handle:
                handle.write("old")
            with open(os.path.join(staging, "model.safetensors"), "w") as handle:
                handle.write("new")
            publish_checkpoint(staging, final)
            self.assertEqual(os.listdir(final), ["model.safetensors"])
            self.assertFalse(os.path.exists(staging))

    def test_clear_removes_only_staging_dirs(self):
        """Leftovers from interrupted saves go away; published checkpoints stay."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(staging_checkpoint_dir(tmp, 100))
            os.makedirs(os.path.join(tmp, "checkpoint-100"))
            removed = clear_staging_dirs(tmp)
            self.assertEqual(removed, ["staging-checkpoint-100"])
            self.assertEqual(os.listdir(tmp), ["checkpoint-100"])
            with self.assertRaises(FileNotFoundError):
                publish_checkpoint(staging_checkpoint_dir(tmp, 100), os.path.join(tmp, "checkpoint-100"))

    def test_custom_checkpoint_detection_handles_legacy_matching_and_mismatch(self):
        """Legacy checkpoints without custom files skip pre-load registration; mismatches raise."""
        with tempfile.TemporaryDirectory() as tmp:
            # Empty checkpoint (legacy): ok for 0 expected, skips pre-load when >0 expected
            self.assertEqual(count_custom_checkpoints(tmp), 0)
            self.assertTrue(should_load_custom_checkpoints(tmp, 0))
            self.assertFalse(should_load_custom_checkpoints(tmp, 6))

            # Non-matching filenames are ignored
            with open(os.path.join(tmp, "custom_checkpoint_extra.pkl"), "w") as handle:
                handle.write("ignored")
            with open(os.path.join(tmp, "model.safetensors"), "w") as handle:
                handle.write("weights")
            self.assertEqual(count_custom_checkpoints(tmp), 0)
            self.assertFalse(should_load_custom_checkpoints(tmp, 3))

            # Matching count loads custom state; mismatched non-zero count fails fast
            for index in range(3):
                with open(os.path.join(tmp, f"custom_checkpoint_{index}.pkl"), "w") as handle:
                    handle.write("state")
            self.assertEqual(count_custom_checkpoints(tmp), 3)
            self.assertTrue(should_load_custom_checkpoints(tmp, 3))
            with self.assertRaises(ValueError):
                should_load_custom_checkpoints(tmp, 6)
            with self.assertRaises(ValueError):
                should_load_custom_checkpoints(tmp, 0)


if __name__ == "__main__":
    unittest.main()
