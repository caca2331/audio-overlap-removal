import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.signal
import soundfile as sf

from audio_overlap_removal import (
    AlignmentSegment,
    cancellation_profile,
    discover_alignment_segments,
    process_audio,
    remove_reference,
    scan_reference,
)
from audio_overlap_removal.alignment import (
    _align_reference,
    _best_scaled_match,
    _GlobalMatcher,
    _ReferenceAlignment,
)
from audio_overlap_removal.cancellation import (
    _cancel_chunk,
    _complex_reference_cancel,
    _estimate_complex_transfer,
    _estimate_gain_envelope,
)
from audio_overlap_removal.cli import _build_parser, _run_cli
from audio_overlap_removal.fingerprint import (
    FingerprintIndex,
    fingerprint_blocks,
)
from audio_overlap_removal.media import (
    FFMPEG_DIR_ENV,
    _atomic_soundfile,
    _audio_channel_count,
    _child_env,
    _decode_stereo,
    _iter_decode_mono_low,
    _output_settings,
    _paths_refer_to_same_file,
    _processing_channel_count,
    _tool,
)
from audio_overlap_removal.models import _clip_alignment_segments
from audio_overlap_removal.parallel import _bounded_ordered_map
from audio_overlap_removal.pipeline import (
    _accepts_cancellation,
    _fit_offset_trajectory,
    _merge_passthrough_spans,
)


class AudioOverlapRemovalTests(unittest.TestCase):
    def test_atomic_output_preserves_existing_file_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "clean.wav")
            output.write_bytes(b"original")

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with _atomic_soundfile(
                    output,
                    samplerate=8_000,
                    channels=1,
                    format="WAV",
                    subtype="PCM_24",
                ) as sink:
                    sink.write(np.zeros(100, dtype=np.float32))
                    raise RuntimeError("synthetic failure")

            self.assertEqual(output.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_output_format_follows_extension(self) -> None:
        self.assertEqual(_output_settings(Path("clean.FLAC")), ("FLAC", "PCM_24"))
        self.assertEqual(_output_settings(Path("clean.wav")), ("WAV", "PCM_24"))
        with self.assertRaisesRegex(ValueError, "Unsupported output extension"):
            _output_settings(Path("clean.mp3"))

    def test_equivalent_paths_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory, "input.wav")
            media.touch()
            equivalent = Path(directory, ".", "input.wav")
            self.assertTrue(_paths_refer_to_same_file(media, equivalent))

    def test_process_audio_rejects_an_input_as_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixture input"):
            process_audio(
                "mixture.wav",
                "reference.wav",
                "mixture.wav",
                alignment_segments=[],
            )

    def test_process_audio_validates_chunk_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_sec"):
            process_audio(
                "mixture.wav",
                "reference.wav",
                "output.flac",
                alignment_segments=[],
                chunk_sec=0.0,
            )

    def test_process_audio_rejects_non_finite_expert_controls(self) -> None:
        with patch(
            "audio_overlap_removal.pipeline._media_duration",
            return_value=1.0,
        ):
            for option in (
                "context_sec",
                "search_sec",
                "cleanup_strength",
                "center_cleanup_strength",
            ):
                with self.subTest(option=option), self.assertRaises(ValueError):
                    process_audio(
                        "mixture.wav",
                        "reference.wav",
                        "output.flac",
                        alignment_segments=[],
                        **{option: float("nan")},
                    )

    def test_cli_uses_start_and_end_as_media_range(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mixture.wav",
                "reference.wav",
                "output.flac",
                "--start",
                "120",
                "--end",
                "360",
            ]
        )

        self.assertEqual(args.start, 120.0)
        self.assertEqual(args.end, 360.0)
        self.assertEqual(args.strength, 1.0)
        self.assertEqual(args.workers, 4)
        self.assertFalse(hasattr(args, "offset"))
        self.assertFalse(hasattr(args, "duration"))

    def test_clipped_segment_preserves_offset_slope(self) -> None:
        segment = AlignmentSegment(
            mixture_start=10.0,
            mixture_end=30.0,
            offset_sec=4.0,
            median_score=0.8,
            offset_slope=0.01,
        )

        clipped = _clip_alignment_segments([segment], 15.0, 20.0)

        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped[0].mixture_start, 15.0)
        self.assertEqual(clipped[0].mixture_end, 20.0)
        for timestamp in (15.0, 17.5, 20.0):
            self.assertAlmostEqual(
                clipped[0].offset_at(timestamp),
                segment.offset_at(timestamp),
            )

    def test_cli_forwards_media_range_to_high_level_api(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mixture.wav",
                "reference.wav",
                "output.flac",
                "--start",
                "120",
                "--end",
                "360",
            ]
        )
        segments = [AlignmentSegment(120.0, 360.0, 20.0, 0.9)]
        with (
            patch(
                "audio_overlap_removal.cli.scan_reference",
                return_value=segments,
            ) as scan,
            patch("audio_overlap_removal.cli.process_audio") as process,
        ):
            _run_cli(args)

        scan.assert_called_once_with(
            "mixture.wav",
            "reference.wav",
            start=120.0,
            end=360.0,
            workers=4,
        )
        process.assert_called_once_with(
            "mixture.wav",
            "reference.wav",
            "output.flac",
            alignment_segments=segments,
            chunk_sec=30.0,
            search_sec=0.25,
            strength=1.0,
            cleanup_strength=None,
            center_strength=None,
            center_cleanup_strength=None,
            silence_cleanup_strength=None,
            adaptive_time_warp=True,
            momentum=True,
            output_start=0.0,
            output_end=None,
            report_path=None,
            sr=48_000,
            workers=4,
        )

    def test_cli_scan_only_writes_segments_and_skips_processing(self) -> None:
        parser = _build_parser()
        segment = AlignmentSegment(
            120.0,
            360.0,
            20.0,
            0.9,
            anchor_times=(120.0, 240.0),
            anchor_offsets=(20.0, 20.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            segments_path = str(Path(directory, "segments.json"))
            args = parser.parse_args(
                [
                    "mixture.wav",
                    "reference.wav",
                    "--scan-only",
                    "--segments-out",
                    segments_path,
                ]
            )
            with (
                patch(
                    "audio_overlap_removal.cli.scan_reference",
                    return_value=[segment],
                ),
                patch("audio_overlap_removal.cli.process_audio") as process,
            ):
                _run_cli(args)
            process.assert_not_called()

            reload_args = parser.parse_args(
                [
                    "mixture.wav",
                    "reference.wav",
                    "output.flac",
                    "--segments",
                    segments_path,
                ]
            )
            with (
                patch("audio_overlap_removal.cli.scan_reference") as scan,
                patch("audio_overlap_removal.cli.process_audio") as process,
            ):
                _run_cli(reload_args)

        scan.assert_not_called()
        # A round trip through the file must preserve the anchor trajectory,
        # not just the headline offset.
        self.assertEqual(process.call_args.kwargs["alignment_segments"], [segment])

    def test_cli_requires_an_output_path_unless_scanning_only(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["mixture.wav", "reference.wav"])

        with self.assertRaisesRegex(ValueError, "output path is required"):
            _run_cli(args)

    def test_scan_reference_limits_discovery_and_clips_segments(self) -> None:
        discovered = [AlignmentSegment(100.0, 400.0, 20.0, 0.9)]
        with (
            patch(
                "audio_overlap_removal.pipeline._media_duration",
                return_value=600.0,
            ),
            patch(
                "audio_overlap_removal.pipeline.discover_alignment_segments",
                return_value=discovered,
            ) as discover,
        ):
            segments = scan_reference(
                "mixture.wav",
                "reference.wav",
                start=120.0,
                end=360.0,
            )

        discover.assert_called_once_with(
            "mixture.wav",
            "reference.wav",
            workers=4,
            mixture_start_sec=120.0,
            mixture_duration_sec=240.0,
            reference_duration_sec=600.0,
        )
        self.assertEqual(segments[0].mixture_start, 120.0)
        self.assertEqual(segments[0].mixture_end, 360.0)

    def test_scan_reference_rejects_media_over_24_hours(self) -> None:
        with patch(
            "audio_overlap_removal.pipeline._media_duration",
            side_effect=(24.0 * 60.0 * 60.0 + 1.0, 60.0),
        ):
            with self.assertRaisesRegex(ValueError, "24-hour"):
                scan_reference("mixture.wav", "reference.wav")

    def test_remove_reference_composes_scan_and_processing(self) -> None:
        segments = [AlignmentSegment(120.0, 360.0, 20.0, 0.9)]
        with (
            patch(
                "audio_overlap_removal.pipeline.scan_reference",
                return_value=segments,
            ) as scan,
            patch("audio_overlap_removal.pipeline.process_audio") as process,
        ):
            returned = remove_reference(
                "mixture.wav",
                "reference.wav",
                "output.flac",
                start=120.0,
                end=360.0,
                strength=1.0,
                workers=2,
            )

        self.assertIs(returned, segments)
        scan.assert_called_once_with(
            "mixture.wav",
            "reference.wav",
            start=120.0,
            end=360.0,
            workers=2,
        )
        self.assertEqual(process.call_args.kwargs["alignment_segments"], segments)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg is required for the decode integration test.",
    )
    def test_ffmpeg_decode_accepts_multichannel_input(self) -> None:
        sr = 8_000
        frames = sr // 4
        time_axis = np.arange(frames, dtype=np.float32) / sr
        channels = np.column_stack(
            [
                0.05 * np.sin(2 * np.pi * frequency * time_axis)
                for frequency in (110, 220, 330, 440, 550, 660)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "surround.wav")
            sf.write(source, channels, sr)

            self.assertEqual(_audio_channel_count(str(source)), 6)
            self.assertEqual(_processing_channel_count(6), 2)
            decoded = _decode_stereo(str(source), 0.0, frames / sr, sr)

        self.assertEqual(decoded.ndim, 2)
        self.assertEqual(decoded.shape[1], 2)
        self.assertGreaterEqual(len(decoded), frames - 1)

    @unittest.skipUnless(
        shutil.which("ffmpeg"),
        "FFmpeg is required for the streaming decode integration test.",
    )
    def test_streaming_low_rate_decode_yields_bounded_blocks(self) -> None:
        sr = 8_000
        frames = 2_003
        time_axis = np.arange(frames, dtype=np.float32) / sr
        audio = 0.1 * np.sin(2.0 * np.pi * 440.0 * time_axis)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "mono.wav")
            sf.write(source, audio, sr, subtype="FLOAT")
            blocks = list(
                _iter_decode_mono_low(
                    str(source),
                    sr,
                    block_frames=137,
                )
            )

        self.assertTrue(blocks)
        self.assertTrue(all(len(block) <= 137 for block in blocks))
        np.testing.assert_allclose(
            np.concatenate(blocks),
            audio,
            atol=2e-6,
            rtol=0.0,
        )

    @unittest.skipUnless(
        shutil.which("ffmpeg"),
        "FFmpeg is required for the streaming error integration test.",
    )
    def test_streaming_decode_reports_invalid_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "invalid.wav")
            source.write_bytes(b"not audio")
            with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                list(_iter_decode_mono_low(str(source), 1_000))

    def test_fingerprint_index_returns_media_id_and_compact_candidates(self) -> None:
        sr = 500
        rng = np.random.default_rng(405)
        first = rng.standard_normal(30 * sr).astype(np.float32)
        second = rng.standard_normal(30 * sr).astype(np.float32)
        first_track = fingerprint_blocks(
            [first],
            "first",
            sr=sr,
            query_sec=2.0,
        )
        second_track = fingerprint_blocks(
            [second],
            "second",
            sr=sr,
            query_sec=2.0,
        )
        index = FingerprintIndex([first_track, second_track])

        candidate = index.query(second_track.features[20], candidates=4)[0]

        self.assertEqual(candidate.media_id, "second")
        self.assertAlmostEqual(candidate.time_sec, second_track.times[20])
        self.assertGreater(candidate.score, 0.999)
        bytes_per_window = index.memory_bytes / index.entry_count
        estimated_24_hour_index = bytes_per_window * (24 * 60 * 60 / 0.25)
        self.assertLess(estimated_24_hour_index, 512 * 1024 * 1024)

    def test_alignment_strategy_preserves_short_file_fft_path(self) -> None:
        with (
            patch(
                "audio_overlap_removal.alignment._discover_full_alignment_segments",
                return_value=[],
            ) as full,
            patch(
                "audio_overlap_removal.alignment._discover_indexed_alignment_segments",
                return_value=[],
            ) as indexed,
        ):
            discover_alignment_segments(
                "mixture",
                "reference",
                mixture_duration_sec=4.0 * 60.0 * 60.0,
                reference_duration_sec=4.0 * 60.0 * 60.0,
            )

        full.assert_called_once()
        indexed.assert_not_called()

    def test_alignment_strategy_indexes_media_over_four_hours(self) -> None:
        with (
            patch(
                "audio_overlap_removal.alignment._discover_full_alignment_segments",
                return_value=[],
            ) as full,
            patch(
                "audio_overlap_removal.alignment._discover_indexed_alignment_segments",
                return_value=[],
            ) as indexed,
        ):
            discover_alignment_segments(
                "mixture",
                "reference",
                mixture_duration_sec=4.0 * 60.0 * 60.0 + 1.0,
                reference_duration_sec=60.0,
            )

        indexed.assert_called_once()
        full.assert_not_called()

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg is required for the output integration test.",
    )
    def test_wav_output_is_really_wav_and_preserves_mono(self) -> None:
        sr = 8_000
        frames = sr // 4
        time_axis = np.arange(frames, dtype=np.float32) / sr
        audio = 0.1 * np.sin(2 * np.pi * 220 * time_axis)
        with tempfile.TemporaryDirectory() as directory:
            mixture = Path(directory, "mixture.wav")
            reference = Path(directory, "reference.wav")
            output = Path(directory, "clean.wav")
            sf.write(mixture, audio, sr)
            sf.write(reference, audio, sr)

            process_audio(
                str(mixture),
                str(reference),
                str(output),
                alignment_segments=[],
                chunk_sec=frames / sr,
                sr=sr,
            )
            info = sf.info(output)

        self.assertEqual(info.format, "WAV")
        self.assertEqual(info.channels, 1)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg is required for the range output integration test.",
    )
    def test_only_matched_range_is_changed_in_complete_output(self) -> None:
        sr = 8_000
        frames = sr
        time_axis = np.arange(frames, dtype=np.float32) / sr
        audio = 0.1 * np.sin(2 * np.pi * 220 * time_axis)
        diagnostics = {
            "alignment_score": 1.0,
            "aligned_start": 0.0,
            "gain_p05": 1.0,
            "gain_median": 1.0,
            "gain_p95": 1.0,
            "side_corr_median": 1.0,
            "foreground_guard": 1.0,
            "side_residual_ratio": 0.0,
            "cleanup_output_ratio": 0.0,
        }

        def cancel_to_silence(mixture, *args, **kwargs):
            return np.zeros(len(mixture), dtype=np.float32), diagnostics.copy()

        with tempfile.TemporaryDirectory() as directory:
            mixture = Path(directory, "mixture.wav")
            reference = Path(directory, "reference.wav")
            output = Path(directory, "clean.wav")
            sf.write(mixture, audio, sr)
            sf.write(reference, audio, sr)

            with patch(
                "audio_overlap_removal.pipeline._cancel_chunk",
                side_effect=cancel_to_silence,
            ):
                process_audio(
                    str(mixture),
                    str(reference),
                    str(output),
                    alignment_segments=[AlignmentSegment(0.25, 0.50, 0.0, 1.0)],
                    chunk_sec=1.0,
                    sr=sr,
                )
            source_audio, _ = sf.read(mixture, dtype="float32")
            output_audio, _ = sf.read(output, dtype="float32")

        np.testing.assert_allclose(
            output_audio[: int(0.25 * sr)],
            source_audio[: int(0.25 * sr)],
            atol=2e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            output_audio[int(0.50 * sr) :],
            source_audio[int(0.50 * sr) :],
            atol=2e-6,
            rtol=0.0,
        )
        self.assertLess(
            float(np.max(np.abs(output_audio[int(0.30 * sr) : int(0.45 * sr)]))),
            2e-6,
        )

    def test_bounded_parallel_map_runs_concurrently_in_input_order(
        self,
    ) -> None:
        barrier = threading.Barrier(3)

        def work(value: int) -> int:
            barrier.wait(timeout=2.0)
            return value * value

        output = list(_bounded_ordered_map(work, [3, 2, 1], workers=3))

        self.assertEqual(output, [9, 4, 1])

    def test_global_matcher_reuses_reference_without_changing_result(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        reference = rng.standard_normal(12_000).astype(np.float32)
        query = reference[4_321:5_321] + (0.02 * rng.standard_normal(1_000)).astype(
            np.float32
        )

        expected_index, expected_score = _best_scaled_match(reference, query)
        matcher = _GlobalMatcher(reference, max_query_frames=1_010)
        actual_index, actual_score = matcher.best_scaled_match(query)

        self.assertEqual(actual_index, expected_index)
        self.assertAlmostEqual(actual_score, expected_score, places=10)

    def test_strength_profile_has_stable_landmarks_and_open_upper_range(
        self,
    ) -> None:
        conservative = cancellation_profile(0.0)
        v8 = cancellation_profile(1.0)
        beyond_v9 = cancellation_profile(2.5)

        self.assertEqual(conservative.cleanup_strength, 0.0)
        self.assertEqual(conservative.center_strength, 0.5)
        self.assertEqual(v8.cleanup_strength, 1.0)
        self.assertEqual(v8.center_strength, 1.0)
        self.assertEqual(v8.center_cleanup_strength, 1.0)
        self.assertEqual(beyond_v9.cleanup_strength, 1.0)
        self.assertEqual(beyond_v9.center_cleanup_strength, 2.5)
        with self.assertRaises(ValueError):
            cancellation_profile(-0.01)

    def test_gain_envelope_detects_pause_instead_of_interpolating_it(
        self,
    ) -> None:
        sr = 8_000
        frames = 4 * sr
        rng = np.random.default_rng(101)
        reference = scipy.signal.lfilter(
            [1.0], [1.0, -0.85], rng.standard_normal(frames)
        ).astype(np.float32)
        gain = np.full(frames, 0.7, dtype=np.float32)
        gain[sr : 2 * sr] = 0.0
        mixture = gain * reference

        estimated, _ = _estimate_gain_envelope(mixture, reference, sr)

        self.assertGreater(float(np.median(estimated[: sr // 2])), 0.65)
        self.assertLess(
            float(np.median(estimated[int(1.2 * sr) : int(1.8 * sr)])),
            0.02,
        )
        self.assertGreater(float(np.median(estimated[3 * sr :])), 0.65)

    def test_gain_envelope_correlations_match_direct_windows(self) -> None:
        sr = 2_000
        rng = np.random.default_rng(102)
        reference = rng.standard_normal(3 * sr).astype(np.float32)
        mixture = (0.6 * reference + 0.2 * rng.standard_normal(3 * sr)).astype(
            np.float32
        )

        _, correlations = _estimate_gain_envelope(mixture, reference, sr)

        window = int(round(0.25 * sr))
        hop = int(round(0.05 * sr))
        expected = []
        for center in np.arange(0, len(mixture), hop):
            start = max(0, center - window // 2)
            end = min(len(mixture), center + window // 2)
            mix = mixture[start:end].astype(np.float64)
            ref = reference[start:end].astype(np.float64)
            cross = np.dot(mix, ref)
            expected.append(
                cross
                / np.sqrt(
                    (np.dot(mix, mix) + 1e-20)
                    * (np.dot(ref, ref) + 1e-20)
                )
            )

        np.testing.assert_allclose(correlations, expected, rtol=1e-10, atol=1e-12)

    def test_local_time_warp_tracks_small_speed_jitter(self) -> None:
        sr = 8_000
        frames = 6 * sr
        pad = sr // 2
        rng = np.random.default_rng(202)
        reference = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.88],
            rng.standard_normal((frames + 2 * pad + 128, 2)),
            axis=0,
        ).astype(np.float32)
        sample = np.arange(frames, dtype=np.float64)
        source_positions = (
            pad
            + sample
            + 0.0006 * sample
            + 3.0 * np.sin(2 * np.pi * sample / (2.5 * sr))
        )
        axis = np.arange(len(reference), dtype=np.float64)
        warped = np.column_stack(
            [
                np.interp(source_positions, axis, reference[:, channel])
                for channel in range(2)
            ]
        ).astype(np.float32)
        foreground = (0.02 * np.sin(2 * np.pi * 190 * sample / sr)).astype(np.float32)
        mixture = warped + foreground[:, np.newaxis]

        alignment = _align_reference(mixture, reference, sr)
        aligned = alignment.reference
        relative_error = np.sqrt(np.mean((aligned - warped) ** 2) / np.mean(warped**2))

        self.assertTrue(alignment.covered)
        self.assertGreater(alignment.score, 0.7)
        self.assertLess(relative_error, 0.12)

    def test_adaptive_short_anchors_improve_steady_rate_drift(self) -> None:
        sr = 8_000
        frames = 8 * sr
        pad = sr // 2
        rng = np.random.default_rng(212)
        reference = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.88],
            rng.standard_normal((frames + 2 * pad + 512, 2)),
            axis=0,
        ).astype(np.float32)
        sample = np.arange(frames, dtype=np.float64)
        source_positions = pad + 1.005 * sample
        axis = np.arange(len(reference), dtype=np.float64)
        warped = np.column_stack(
            [
                np.interp(source_positions, axis, reference[:, channel])
                for channel in range(2)
            ]
        ).astype(np.float32)
        foreground = (
            0.04
            * scipy.signal.lfilter([1.0], [1.0, -0.70], rng.standard_normal(frames))
        ).astype(np.float32)
        diagnostics: dict[str, float] = {}

        aligned = _align_reference(
            warped + foreground[:, np.newaxis],
            reference,
            sr,
            diagnostics,
        ).reference
        long_aligned = _align_reference(
            warped + foreground[:, np.newaxis],
            reference,
            sr,
            adaptive_time_warp=False,
        ).reference
        relative_error = np.sqrt(np.mean((aligned - warped) ** 2) / np.mean(warped**2))
        long_error = np.sqrt(np.mean((long_aligned - warped) ** 2) / np.mean(warped**2))

        self.assertEqual(diagnostics["short_warp_considered"], 1.0)
        self.assertEqual(diagnostics["short_warp_accepted"], 1.0)
        self.assertLess(relative_error, 0.06)
        self.assertLess(relative_error, 0.4 * long_error)

    def test_adaptive_short_anchors_bound_fast_jitter(self) -> None:
        sr = 8_000
        frames = 8 * sr
        pad = sr // 2
        rng = np.random.default_rng(213)
        reference = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.88],
            rng.standard_normal((frames + 2 * pad + 512, 2)),
            axis=0,
        ).astype(np.float32)
        sample = np.arange(frames, dtype=np.float64)
        source_positions = (
            pad + 1.0006 * sample + 3.0 * np.sin(2 * np.pi * sample / (0.25 * sr))
        )
        axis = np.arange(len(reference), dtype=np.float64)
        warped = np.column_stack(
            [
                np.interp(source_positions, axis, reference[:, channel])
                for channel in range(2)
            ]
        ).astype(np.float32)
        foreground = (
            0.04
            * scipy.signal.lfilter([1.0], [1.0, -0.70], rng.standard_normal(frames))
        ).astype(np.float32)
        diagnostics: dict[str, float] = {}

        aligned = _align_reference(
            warped + foreground[:, np.newaxis],
            reference,
            sr,
            diagnostics,
        ).reference
        relative_error = np.sqrt(np.mean((aligned - warped) ** 2) / np.mean(warped**2))

        self.assertEqual(diagnostics["short_warp_considered"], 1.0)
        self.assertEqual(diagnostics["short_warp_accepted"], 1.0)
        self.assertLess(relative_error, 0.15)

        extreme_positions = (
            pad + 1.0006 * sample + 30.0 * np.sin(2 * np.pi * sample / (0.25 * sr))
        )
        extreme = np.column_stack(
            [
                np.interp(extreme_positions, axis, reference[:, channel])
                for channel in range(2)
            ]
        ).astype(np.float32)
        extreme_diagnostics: dict[str, float] = {}
        _align_reference(
            extreme + foreground[:, np.newaxis],
            reference,
            sr,
            extreme_diagnostics,
        )
        self.assertEqual(extreme_diagnostics["short_warp_accepted"], 0.0)

    def test_segment_discovery_reacquires_after_pause_and_replay(
        self,
    ) -> None:
        sr = 500
        rng = np.random.default_rng(303)
        # The mixture is deliberately longer than the reference so replay
        # recovery cannot rely on the initial offset-derived end time.
        reference = rng.standard_normal(12 * sr).astype(np.float32)
        mixture = (0.01 * rng.standard_normal(24 * sr)).astype(np.float32)

        def copy_region(
            mixture_start: float,
            mixture_end: float,
            reference_start: float,
        ) -> None:
            count = int(round((mixture_end - mixture_start) * sr))
            mix_start = int(round(mixture_start * sr))
            ref_start = int(round(reference_start * sr))
            mixture[mix_start : mix_start + count] += reference[
                ref_start : ref_start + count
            ]

        copy_region(2.0, 8.0, 0.0)
        copy_region(10.0, 16.0, 6.0)
        copy_region(16.0, 22.0, 2.0)

        def decode(
            path: str,
            requested_sr: int,
            start_sec: float,
            duration_sec: float | None,
        ) -> np.ndarray:
            self.assertEqual(requested_sr, sr)
            self.assertEqual(start_sec, 0.0)
            self.assertIsNone(duration_sec)
            return mixture.copy() if path == "mixture" else reference.copy()

        def discover(workers: int):
            with patch(
                "audio_overlap_removal.alignment._decode_mono_low",
                side_effect=decode,
            ):
                return discover_alignment_segments(
                    "mixture",
                    "reference",
                    align_sr=sr,
                    global_step_sec=1.0,
                    query_sec=1.0,
                    local_step_sec=0.5,
                    local_search_sec=0.10,
                    min_score=0.50,
                    reacquire_after_sec=0.5,
                    reacquire_interval_sec=0.5,
                    workers=workers,
                )

        segments = discover(1)
        parallel_segments = discover(4)

        offsets = [segment.offset_sec for segment in segments]
        self.assertTrue(any(abs(value - 2.0) < 0.15 for value in offsets))
        self.assertTrue(any(abs(value - 4.0) < 0.15 for value in offsets))
        self.assertTrue(any(abs(value - 14.0) < 0.15 for value in offsets))
        self.assertEqual(parallel_segments, segments)

    def test_indexed_discovery_reacquires_after_pause_and_replay(self) -> None:
        sr = 500
        rng = np.random.default_rng(406)
        reference = rng.standard_normal(24 * sr).astype(np.float32)
        mixture = (0.05 * rng.standard_normal(30 * sr)).astype(np.float32)

        def copy_region(
            mixture_start: float,
            mixture_end: float,
            reference_start: float,
        ) -> None:
            frames = int(round((mixture_end - mixture_start) * sr))
            mixture_index = int(round(mixture_start * sr))
            reference_index = int(round(reference_start * sr))
            mixture[mixture_index : mixture_index + frames] += reference[
                reference_index : reference_index + frames
            ]

        copy_region(2.0, 8.0, 0.0)
        copy_region(10.0, 16.0, 6.0)
        copy_region(16.0, 22.0, 2.0)
        reference_track = fingerprint_blocks(
            [reference],
            "reference",
            sr=sr,
            query_sec=2.0,
        )
        mixture_track = fingerprint_blocks(
            [mixture],
            "mixture",
            sr=sr,
            query_sec=2.0,
        )

        def build_track(path: str, media_id: str, **kwargs):
            self.assertEqual(kwargs["sr"], sr)
            return reference_track if media_id == "reference" else mixture_track

        def discover(workers: int) -> list[AlignmentSegment]:
            with patch(
                "audio_overlap_removal.alignment.fingerprint_media",
                side_effect=build_track,
            ):
                return discover_alignment_segments(
                    "mixture",
                    "reference",
                    align_sr=sr,
                    global_step_sec=1.0,
                    query_sec=2.0,
                    local_step_sec=0.5,
                    local_search_sec=0.30,
                    min_score=0.18,
                    reacquire_after_sec=0.5,
                    reacquire_interval_sec=0.5,
                    workers=workers,
                    mixture_duration_sec=30.0,
                    reference_duration_sec=24.0,
                    max_in_memory_sec=0.0,
                )

        segments = discover(1)
        parallel_segments = discover(4)
        offsets = [segment.offset_sec for segment in segments]

        self.assertTrue(any(abs(value - 2.0) < 0.30 for value in offsets))
        self.assertTrue(any(abs(value - 4.0) < 0.30 for value in offsets))
        self.assertTrue(any(abs(value - 14.0) < 0.30 for value in offsets))
        self.assertEqual(parallel_segments, segments)

    def test_indexed_discovery_rejects_unrelated_media(self) -> None:
        sr = 500
        rng = np.random.default_rng(407)
        reference_track = fingerprint_blocks(
            [rng.standard_normal(60 * sr).astype(np.float32)],
            "reference",
            sr=sr,
            query_sec=2.0,
        )
        mixture_track = fingerprint_blocks(
            [rng.standard_normal(60 * sr).astype(np.float32)],
            "mixture",
            sr=sr,
            query_sec=2.0,
        )

        def build_track(path: str, media_id: str, **kwargs):
            return reference_track if media_id == "reference" else mixture_track

        with patch(
            "audio_overlap_removal.alignment.fingerprint_media",
            side_effect=build_track,
        ):
            segments = discover_alignment_segments(
                "mixture",
                "reference",
                align_sr=sr,
                global_step_sec=1.0,
                query_sec=2.0,
                local_step_sec=0.5,
                workers=2,
                mixture_duration_sec=60.0,
                reference_duration_sec=60.0,
                max_in_memory_sec=0.0,
            )

        self.assertEqual(segments, [])

    def test_compact_long_media_smoke_matches_fft_across_state_changes(
        self,
    ) -> None:
        """Force both strategies on 96 seconds that model hours of state changes."""
        sr = 500
        reference_seconds = 70
        mixture_seconds = 96
        rng = np.random.default_rng(600)

        reference_time = np.arange(reference_seconds * sr) / sr
        reference = scipy.signal.sosfilt(
            scipy.signal.butter(
                4,
                [50.0, 210.0],
                btype="bandpass",
                fs=sr,
                output="sos",
            ),
            rng.standard_normal(len(reference_time)),
        )
        reference += 0.35 * np.sin(
            2.0
            * np.pi
            * (80.0 + 18.0 * np.sin(2.0 * np.pi * 0.017 * reference_time))
            * reference_time
        )
        reference *= 0.65 + 0.35 * np.square(
            np.sin(2.0 * np.pi * 0.11 * reference_time)
        )
        for transient_time in np.arange(1.0, reference_seconds, 3.7):
            index = int(round(transient_time * sr))
            frames = min(25, len(reference) - index)
            reference[index : index + frames] += 2.0 * scipy.signal.windows.hann(frames)
        reference = (reference / np.std(reference)).astype(np.float32)

        mixture_time = np.arange(mixture_seconds * sr) / sr
        foreground = (
            0.18
            * np.sin(
                2.0
                * np.pi
                * (115.0 + 22.0 * np.sin(2.0 * np.pi * 0.7 * mixture_time))
                * mixture_time
            )
            * (0.4 + 0.6 * np.square(np.sin(2.0 * np.pi * 2.3 * mixture_time)))
        )
        mixture = (foreground + 0.04 * rng.standard_normal(len(mixture_time))).astype(
            np.float32
        )
        colored_reference = scipy.signal.sosfilt(
            scipy.signal.butter(
                2,
                170.0,
                btype="lowpass",
                fs=sr,
                output="sos",
            ),
            reference,
        ).astype(np.float32)

        def add_reference(
            mixture_start: float,
            mixture_end: float,
            reference_start: float,
        ) -> None:
            frames = int(round((mixture_end - mixture_start) * sr))
            mixture_index = int(round(mixture_start * sr))
            reference_index = int(round(reference_start * sr))
            time_axis = np.arange(frames) / sr
            dynamic_gain = 0.5 * (0.75 + 0.25 * np.sin(2.0 * np.pi * 0.09 * time_axis))
            mixture[mixture_index : mixture_index + frames] += (
                dynamic_gain
                * colored_reference[reference_index : reference_index + frames]
            )

        # Continuous playback, a pause/resume, then a backwards replay.
        add_reference(5.0, 25.0, 0.0)
        add_reference(32.0, 52.0, 20.0)
        add_reference(58.0, 78.0, 5.0)

        def decode(path: str, requested_sr: int, start_sec, duration_sec):
            self.assertEqual(requested_sr, sr)
            return mixture.copy() if path == "mixture" else reference.copy()

        common_options = {
            "align_sr": sr,
            "global_step_sec": 2.0,
            "query_sec": 2.0,
            "local_step_sec": 0.5,
            "local_search_sec": 0.30,
            "reacquire_after_sec": 1.0,
            "reacquire_interval_sec": 1.0,
            "workers": 2,
            "mixture_duration_sec": float(mixture_seconds),
            "reference_duration_sec": float(reference_seconds),
        }
        with patch(
            "audio_overlap_removal.alignment._decode_mono_low",
            side_effect=decode,
        ):
            fft_segments = discover_alignment_segments(
                "mixture",
                "reference",
                min_score=0.25,
                max_in_memory_sec=10_000.0,
                **common_options,
            )

        reference_track = fingerprint_blocks(
            [reference],
            "reference",
            sr=sr,
            query_sec=2.0,
        )
        mixture_track = fingerprint_blocks(
            [mixture],
            "mixture",
            sr=sr,
            query_sec=2.0,
        )

        def build_track(path: str, media_id: str, **kwargs):
            return reference_track if media_id == "reference" else mixture_track

        with patch(
            "audio_overlap_removal.alignment.fingerprint_media",
            side_effect=build_track,
        ):
            indexed_segments = discover_alignment_segments(
                "mixture",
                "reference",
                min_score=0.18,
                max_in_memory_sec=0.0,
                **common_options,
            )

        expected_offsets = (5.0, 12.0, 53.0)
        self.assertEqual(len(fft_segments), len(expected_offsets))
        self.assertEqual(len(indexed_segments), len(expected_offsets))
        for expected, fft_segment, indexed_segment in zip(
            expected_offsets,
            fft_segments,
            indexed_segments,
        ):
            self.assertAlmostEqual(fft_segment.offset_sec, expected, delta=0.30)
            self.assertAlmostEqual(indexed_segment.offset_sec, expected, delta=0.30)
            self.assertAlmostEqual(
                indexed_segment.offset_sec,
                fft_segment.offset_sec,
                delta=0.30,
            )

        for active_time in (10.0, 40.0, 65.0):
            self.assertTrue(
                any(
                    segment.mixture_start <= active_time < segment.mixture_end
                    for segment in indexed_segments
                )
            )
        for passthrough_time in (28.0, 55.0, 85.0):
            self.assertFalse(
                any(
                    segment.mixture_start <= passthrough_time < segment.mixture_end
                    for segment in indexed_segments
                )
            )

    def test_segment_discovery_preserves_absolute_scan_timestamps(
        self,
    ) -> None:
        sr = 500
        rng = np.random.default_rng(304)
        reference = rng.standard_normal(8 * sr).astype(np.float32)
        mixture = reference.copy()

        def decode(
            path: str,
            requested_sr: int,
            start_sec: float,
            duration_sec: float | None,
        ) -> np.ndarray:
            self.assertEqual(requested_sr, sr)
            if path == "mixture":
                self.assertEqual(start_sec, 100.0)
                self.assertEqual(duration_sec, 9.0)
                return mixture.copy()
            self.assertEqual(start_sec, 0.0)
            self.assertIsNone(duration_sec)
            return reference.copy()

        with patch(
            "audio_overlap_removal.alignment._decode_mono_low",
            side_effect=decode,
        ):
            segments = discover_alignment_segments(
                "mixture",
                "reference",
                align_sr=sr,
                global_step_sec=1.0,
                query_sec=1.0,
                local_step_sec=0.5,
                min_score=0.5,
                mixture_start_sec=100.0,
                mixture_duration_sec=8.0,
                workers=2,
            )

        self.assertTrue(segments)
        self.assertGreaterEqual(segments[0].mixture_start, 100.0)
        self.assertAlmostEqual(segments[0].offset_sec, 100.0, places=2)

    def test_small_balance_and_volume_changes_preserve_center_target(
        self,
    ) -> None:
        sr = 8_000
        frames = 4 * sr
        time = np.arange(frames) / sr
        rng = np.random.default_rng(404)
        reference_mid = scipy.signal.lfilter(
            [1.0], [1.0, -0.90], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_mid *= 0.30 / np.std(reference_mid)
        reference_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.75], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_side *= 0.12 / np.std(reference_side)
        reference = np.column_stack(
            [
                reference_mid + reference_side,
                reference_mid - reference_side,
            ]
        ).astype(np.float32)
        target = (
            0.15
            * np.sin(2 * np.pi * (160 + 10 * np.sin(2 * np.pi * 0.2 * time)) * time)
        ).astype(np.float32)
        common_gain = 0.65 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.4 * time))
        balance = 0.04 * np.sin(2 * np.pi * 0.23 * time)
        left_gain = common_gain * (1.0 + balance)
        right_gain = common_gain * (1.0 - balance)
        mixture = np.column_stack(
            [
                target + left_gain * reference[:, 0],
                target + right_gain * reference[:, 1],
            ]
        ).astype(np.float32)
        pad = np.zeros((sr // 4, 2), dtype=np.float32)

        output, diagnostics = _cancel_chunk(
            mixture,
            np.vstack([pad, reference, pad]),
            sr,
            cleanup_strength=0.0,
        )
        relative_error = np.sqrt(np.mean((output - target) ** 2) / np.mean(target**2))

        self.assertGreater(diagnostics["alignment_score"], 0.9)
        self.assertLess(relative_error, 0.05)

    def test_truncated_reference_is_passed_through(self) -> None:
        sr = 8_000
        rng = np.random.default_rng(505)
        mixture = rng.standard_normal((2 * sr, 2)).astype(np.float32)
        short_reference = rng.standard_normal((sr, 2)).astype(np.float32)

        output, diagnostics = _cancel_chunk(
            mixture,
            short_reference,
            sr,
            cleanup_strength=1.0,
            center_strength=1.0,
            center_cleanup_strength=2.0,
            silence_cleanup_strength=1.0,
        )

        expected = 0.5 * (mixture[:, 0] + mixture[:, 1])
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(diagnostics["alignment_score"], 0.0)
        self.assertEqual(diagnostics["insufficient_reference_passthrough"], 1.0)

        stereo_output, _ = _cancel_chunk(
            mixture,
            short_reference,
            sr,
            cleanup_strength=1.0,
            center_strength=1.0,
            center_cleanup_strength=2.0,
            silence_cleanup_strength=1.0,
            mixture_channels=2,
            reference_channels=2,
            output_channels=2,
        )
        np.testing.assert_array_equal(stereo_output, mixture)

    def test_complex_transfer_reconstructs_known_filter(self) -> None:
        rng = np.random.default_rng(11)
        reference = (
            rng.standard_normal((64, 200)) + 1j * rng.standard_normal((64, 200))
        ).astype(np.complex64)
        expected_transfer = 0.42 * np.exp(0.35j)
        mixture = expected_transfer * reference

        transfer, coherence = _estimate_complex_transfer(
            mixture, reference, sigma=(1.0, 5.0)
        )
        transfer_only, omitted_coherence = _estimate_complex_transfer(
            mixture,
            reference,
            sigma=(1.0, 5.0),
            estimate_coherence=False,
        )
        reconstruction = transfer * reference
        relative_error = np.sqrt(
            np.mean(np.abs(mixture - reconstruction) ** 2)
            / np.mean(np.abs(mixture) ** 2)
        )

        self.assertLess(relative_error, 0.02)
        self.assertGreater(float(np.median(coherence)), 0.98)
        np.testing.assert_array_equal(transfer_only, transfer)
        self.assertIsNone(omitted_coherence)

    def test_stereo_side_control_recovers_centered_target(self) -> None:
        sr = 16_000
        duration = 4
        frames = sr * duration
        rng = np.random.default_rng(7)
        time = np.arange(frames) / sr

        reference = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.94],
            rng.standard_normal((frames, 2)),
            axis=0,
        ).astype(np.float32)
        reference /= np.max(np.abs(reference))
        target = (
            0.12
            * np.sin(2 * np.pi * (170 + 20 * np.sin(2 * np.pi * 0.3 * time)) * time)
            * (np.sin(2 * np.pi * 2.1 * time) > 0)
        ).astype(np.float32)
        gain = (0.25 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.17 * time)) ** 2).astype(
            np.float32
        )
        mixture = np.column_stack(
            [
                target + gain * reference[:, 0],
                target + gain * reference[:, 1],
            ]
        ).astype(np.float32)
        pad = np.zeros((int(0.2 * sr), 2), dtype=np.float32)
        reference_search = np.vstack([pad, reference, pad])

        output, diagnostics = _cancel_chunk(
            mixture, reference_search, sr, cleanup_strength=0.0
        )

        relative_error = np.sqrt(np.mean((target - output) ** 2) / np.mean(target**2))
        self.assertEqual(len(output), len(target))
        self.assertGreater(diagnostics["alignment_score"], 0.8)
        self.assertLess(relative_error, 0.02)

        host_mid = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.85],
            rng.standard_normal(frames),
        ).astype(np.float32)
        host_side = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.80],
            rng.standard_normal(frames),
        ).astype(np.float32)
        host_mid *= 0.15 / np.std(host_mid)
        host_side *= 0.50 / np.std(host_side)
        host_bgm = np.column_stack([host_mid + host_side, host_mid - host_side])
        mixture_with_bgm = mixture + host_bgm
        raw_with_bgm, _ = _cancel_chunk(
            mixture_with_bgm,
            reference_search,
            sr,
            cleanup_strength=0.0,
            center_strength=1.0,
        )
        guarded_with_bgm, bgm_diagnostics = _cancel_chunk(
            mixture_with_bgm,
            reference_search,
            sr,
            cleanup_strength=1.0,
            center_strength=1.0,
            center_cleanup_strength=1.0,
            silence_cleanup_strength=1.0,
        )
        guard_difference = np.sqrt(
            np.mean((guarded_with_bgm - raw_with_bgm) ** 2) / np.mean(raw_with_bgm**2)
        )
        self.assertLess(bgm_diagnostics["foreground_guard"], 0.15)
        self.assertLess(guard_difference, 0.01)

    def test_center_prior_reduces_colored_center_reference(self) -> None:
        sr = 16_000
        frames = 4 * sr
        time = np.arange(frames) / sr
        rng = np.random.default_rng(42)

        center_reference = (
            0.3
            * np.sin(2 * np.pi * (220 + 20 * np.sin(2 * np.pi * 0.7 * time)) * time)
            * (0.4 + 0.6 * (np.sin(2 * np.pi * 2.3 * time) > 0))
        ).astype(np.float32)
        mid_bed = scipy.signal.lfilter(
            [1.0], [1.0, -0.8], rng.standard_normal(frames)
        ).astype(np.float32)
        mid_bed *= 0.03 / np.std(mid_bed)
        reference_mid = center_reference + mid_bed

        reference_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.7], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_side *= 0.08 / np.std(reference_side)
        target = (
            0.25
            * np.sin(2 * np.pi * (173 + 13 * np.sin(2 * np.pi * 0.31 * time)) * time)
        ).astype(np.float32)

        mixture_mid = target + scipy.signal.lfilter(
            [0.7, 0.18, -0.06], [1.0], reference_mid
        ).astype(np.float32)
        mixture_side = scipy.signal.lfilter([0.52, 0.02], [1.0], reference_side).astype(
            np.float32
        )
        mixture_side += (0.03 * rng.standard_normal(frames)).astype(np.float32)
        scalar_gain = np.full(frames, 0.52, dtype=np.float32)

        conservative, _, _ = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            scalar_gain,
            cleanup_strength=0.0,
            center_strength=0.0,
            sr=sr,
        )
        balanced, _, _ = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            scalar_gain,
            cleanup_strength=0.0,
            center_strength=0.5,
            sr=sr,
        )
        conservative_error = np.sqrt(
            np.mean((conservative - target) ** 2) / np.mean(target**2)
        )
        balanced_error = np.sqrt(np.mean((balanced - target) ** 2) / np.mean(target**2))

        self.assertLess(balanced_error, 0.85 * conservative_error)

        protected, _, _ = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            scalar_gain,
            cleanup_strength=1.0,
            center_strength=1.0,
            sr=sr,
            center_cleanup_strength=1.0,
            silence_cleanup_strength=0.0,
        )
        foreground_guarded, _, _ = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            scalar_gain,
            cleanup_strength=1.0,
            center_strength=1.0,
            sr=sr,
            center_cleanup_strength=1.0,
            silence_cleanup_strength=1.0,
        )
        guard_delta = np.sqrt(
            np.mean((foreground_guarded - protected) ** 2) / np.mean(protected**2)
        )
        self.assertLess(guard_delta, 0.01)

    def test_filtered_path_does_not_take_scalar_fast_path(self) -> None:
        sr = 16_000
        frames = 4 * sr
        rng = np.random.default_rng(606)
        reference_mid = scipy.signal.lfilter(
            [1.0], [1.0, -0.91], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.73], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_mid *= 0.20 / np.std(reference_mid)
        reference_side *= 0.10 / np.std(reference_side)
        target = scipy.signal.lfilter(
            [1.0], [1.0, -0.82], rng.standard_normal(frames)
        ).astype(np.float32)
        target *= 0.08 / np.std(target)
        path = [0.62, 0.18, -0.08, 0.04]
        mixture_mid = target + scipy.signal.lfilter(path, [1.0], reference_mid)
        mixture_side = scipy.signal.lfilter(path, [1.0], reference_side)
        scalar_gain = np.full(frames, 0.62, dtype=np.float32)

        output, _, _ = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            scalar_gain,
            cleanup_strength=0.0,
            center_strength=0.5,
            sr=sr,
        )
        scalar_output = mixture_mid - scalar_gain * reference_mid
        complex_error = np.sqrt(np.mean((output - target) ** 2))
        scalar_error = np.sqrt(np.mean((scalar_output - target) ** 2))

        self.assertLess(complex_error, 0.50 * scalar_error)

    def test_stereo_output_downmix_matches_mono_output(self) -> None:
        sr = 8_000
        frames = 4 * sr
        rng = np.random.default_rng(707)
        reference = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.82],
            rng.standard_normal((frames, 2)),
            axis=0,
        ).astype(np.float32)
        reference *= 0.16 / np.std(reference)
        target_mid = scipy.signal.lfilter(
            [1.0], [1.0, -0.88], rng.standard_normal(frames)
        ).astype(np.float32)
        target_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.70], rng.standard_normal(frames)
        ).astype(np.float32)
        target_mid *= 0.06 / np.std(target_mid)
        target_side *= 0.03 / np.std(target_side)
        target = np.column_stack([target_mid + target_side, target_mid - target_side])
        mixture = target + 0.55 * reference
        pad = np.zeros((sr // 4, 2), dtype=np.float32)
        reference_search = np.vstack([pad, reference, pad])

        mono, _ = _cancel_chunk(
            mixture,
            reference_search,
            sr,
            cleanup_strength=0.0,
            mixture_channels=2,
            reference_channels=2,
            output_channels=1,
        )
        stereo, _ = _cancel_chunk(
            mixture,
            reference_search,
            sr,
            cleanup_strength=0.0,
            mixture_channels=2,
            reference_channels=2,
            output_channels=2,
        )

        self.assertEqual(stereo.shape, (frames, 2))
        np.testing.assert_allclose(stereo.mean(axis=1), mono, atol=2e-7, rtol=0.0)

    def test_mono_reference_preserves_stereo_mixture_side(self) -> None:
        sr = 8_000
        frames = 4 * sr
        rng = np.random.default_rng(808)
        reference_mono = scipy.signal.lfilter(
            [1.0], [1.0, -0.84], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_mono *= 0.18 / np.std(reference_mono)
        reference = np.column_stack([reference_mono, reference_mono])
        target_mid = scipy.signal.lfilter(
            [1.0], [1.0, -0.90], rng.standard_normal(frames)
        ).astype(np.float32)
        target_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.65], rng.standard_normal(frames)
        ).astype(np.float32)
        target_mid *= 0.05 / np.std(target_mid)
        target_side *= 0.04 / np.std(target_side)
        mixture = np.column_stack(
            [
                target_mid + target_side + 0.6 * reference_mono,
                target_mid - target_side + 0.6 * reference_mono,
            ]
        ).astype(np.float32)
        pad = np.zeros((sr // 4, 2), dtype=np.float32)

        output, _ = _cancel_chunk(
            mixture,
            np.vstack([pad, reference, pad]),
            sr,
            cleanup_strength=0.0,
            mixture_channels=2,
            reference_channels=1,
            output_channels=2,
        )

        output_mid = output.mean(axis=1)
        output_side = 0.5 * (output[:, 0] - output[:, 1])
        relative_mid_error = np.sqrt(
            np.mean((output_mid - target_mid) ** 2) / np.mean(target_mid**2)
        )
        np.testing.assert_allclose(output_side, target_side, atol=2e-7, rtol=0.0)
        self.assertLess(relative_mid_error, 0.10)

    def test_mono_mixture_routes_cancel_both_reference_layouts(self) -> None:
        sr = 8_000
        frames = 4 * sr
        rng = np.random.default_rng(909)
        reference_mid = scipy.signal.lfilter(
            [1.0], [1.0, -0.86], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_side = scipy.signal.lfilter(
            [1.0], [1.0, -0.72], rng.standard_normal(frames)
        ).astype(np.float32)
        reference_mid *= 0.18 / np.std(reference_mid)
        reference_side *= 0.08 / np.std(reference_side)
        stereo_reference = np.column_stack(
            [
                reference_mid + reference_side,
                reference_mid - reference_side,
            ]
        )
        mono_reference = np.column_stack([reference_mid, reference_mid])
        target = scipy.signal.lfilter(
            [1.0], [1.0, -0.91], rng.standard_normal(frames)
        ).astype(np.float32)
        target *= 0.04 / np.std(target)
        mixture_mono = target + 0.58 * reference_mid
        decoded_mixture = np.column_stack([mixture_mono, mixture_mono])
        pad = np.zeros((sr // 4, 2), dtype=np.float32)

        for reference, reference_channels in (
            (mono_reference, 1),
            (stereo_reference, 2),
        ):
            with self.subTest(reference_channels=reference_channels):
                output, _ = _cancel_chunk(
                    decoded_mixture,
                    np.vstack([pad, reference, pad]),
                    sr,
                    cleanup_strength=0.0,
                    mixture_channels=1,
                    reference_channels=reference_channels,
                    output_channels=1,
                )
                relative_error = np.sqrt(
                    np.mean((output - target) ** 2) / np.mean(target**2)
                )
                self.assertEqual(output.ndim, 1)
                self.assertLess(relative_error, 0.10)


def _write_drifting_fixture(
    directory: Path,
    sr: int,
    true_offset: float,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Write a mixture whose matched span carries a known reference offset."""
    rng = np.random.default_rng(1234)
    reference = scipy.signal.lfilter(
        [1.0],
        [1.0, -0.85],
        rng.standard_normal((30 * sr, 2)),
        axis=0,
    ).astype(np.float32)
    # Keep peaks well inside full scale: the 24-bit output would otherwise
    # clip and stop being a bit-exact copy of a passed-through chunk.
    reference *= 0.12 / np.std(reference)
    frames = 24 * sr
    time = np.arange(frames) / sr
    target = (0.05 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    mixture = np.column_stack([target, target])
    start = int(round(4.0 * sr))
    end = int(round(20.0 * sr))
    reference_start = int(round((4.0 - true_offset) * sr))
    mixture[start:end] += 0.8 * reference[
        reference_start : reference_start + (end - start)
    ]
    mixture_path = Path(directory, "mixture.wav")
    reference_path = Path(directory, "reference.wav")
    sf.write(mixture_path, mixture, sr, subtype="FLOAT")
    sf.write(reference_path, reference, sr, subtype="FLOAT")
    return str(mixture_path), str(reference_path), mixture, target


class FFmpegDiscoveryTests(unittest.TestCase):
    """Resolution order for ffmpeg/ffprobe in packaged and source runs."""

    def setUp(self) -> None:
        _tool.cache_clear()
        self.addCleanup(_tool.cache_clear)
        environment = patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(FFMPEG_DIR_ENV, None)

    @staticmethod
    def _create_tool(directory: Path) -> Path:
        executable = directory / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        executable.write_bytes(b"")
        executable.chmod(0o755)
        return executable

    def test_env_override_outranks_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self._create_tool(Path(directory))
            os.environ[FFMPEG_DIR_ENV] = directory
            with patch(
                "audio_overlap_removal.media.shutil.which",
                return_value="/from/path/ffmpeg",
            ):
                self.assertEqual(_tool("ffmpeg"), str(executable))

    def test_path_outranks_guessed_install_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._create_tool(Path(directory))
            with (
                patch(
                    "audio_overlap_removal.media._fallback_tool_dirs",
                    return_value=[Path(directory)],
                ),
                patch(
                    "audio_overlap_removal.media.shutil.which",
                    return_value="/from/path/ffmpeg",
                ),
            ):
                self.assertEqual(_tool("ffmpeg"), "/from/path/ffmpeg")

    def test_guessed_install_location_is_used_when_path_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self._create_tool(Path(directory))
            with (
                patch(
                    "audio_overlap_removal.media._fallback_tool_dirs",
                    return_value=[Path(directory, "missing"), Path(directory)],
                ),
                patch(
                    "audio_overlap_removal.media.shutil.which",
                    return_value=None,
                ),
            ):
                self.assertEqual(_tool("ffmpeg"), str(executable))

    def test_child_env_is_inherited_when_running_from_source(self) -> None:
        self.assertIsNone(_child_env())

    def test_frozen_child_env_drops_the_bundled_loader_path(self) -> None:
        os.environ["LD_LIBRARY_PATH"] = "/bundle/_internal"
        with patch(
            "audio_overlap_removal.media._frozen_dir", return_value=Path("/bundle")
        ):
            self.assertNotIn("LD_LIBRARY_PATH", _child_env())

    def test_frozen_child_env_restores_the_original_loader_path(self) -> None:
        os.environ["LD_LIBRARY_PATH"] = "/bundle/_internal"
        os.environ["LD_LIBRARY_PATH_ORIG"] = "/opt/mine/lib"
        with patch(
            "audio_overlap_removal.media._frozen_dir", return_value=Path("/bundle")
        ):
            env = _child_env()
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/mine/lib")
        self.assertNotIn("LD_LIBRARY_PATH_ORIG", env)

    def test_bare_name_is_returned_when_nothing_matches(self) -> None:
        with (
            patch(
                "audio_overlap_removal.media._fallback_tool_dirs",
                return_value=[],
            ),
            patch("audio_overlap_removal.media.shutil.which", return_value=None),
        ):
            expected = "ffprobe.exe" if os.name == "nt" else "ffprobe"
            self.assertEqual(_tool("ffprobe"), expected)


class ChunkOffsetMomentumTests(unittest.TestCase):
    def test_offset_trajectory_replaces_the_single_slope_model(self) -> None:
        segment = AlignmentSegment(
            mixture_start=0.0,
            mixture_end=100.0,
            offset_sec=5.0,
            median_score=0.9,
            offset_slope=0.0,
            anchor_times=(0.0, 50.0, 100.0),
            anchor_offsets=(5.0, 5.4, 5.2),
        )

        self.assertAlmostEqual(segment.offset_at(25.0), 5.2)
        self.assertAlmostEqual(segment.offset_at(75.0), 5.3)
        # The straight-line model would predict 5.0 everywhere, i.e. 400ms off
        # in the middle - far more than a chunk-local search buffer.
        self.assertAlmostEqual(segment.offset_at(50.0), 5.4)

        clipped = _clip_alignment_segments([segment], 40.0, 60.0)[0]
        for timestamp in (40.0, 50.0, 60.0):
            self.assertAlmostEqual(
                clipped.offset_at(timestamp),
                segment.offset_at(timestamp),
            )

    def test_discovery_keeps_the_anchor_trajectory(self) -> None:
        sr = 500
        rng = np.random.default_rng(3141)
        reference = rng.standard_normal(20 * sr).astype(np.float32)
        mixture = reference.copy()

        def decode(path, requested_sr, start_sec, duration_sec):
            return mixture.copy() if path == "mixture" else reference.copy()

        with patch(
            "audio_overlap_removal.alignment._decode_mono_low",
            side_effect=decode,
        ):
            segments = discover_alignment_segments(
                "mixture",
                "reference",
                align_sr=sr,
                global_step_sec=1.0,
                query_sec=1.0,
                local_step_sec=0.5,
                min_score=0.5,
                workers=1,
            )

        self.assertTrue(segments)
        self.assertGreater(len(segments[0].anchor_times), 3)
        self.assertEqual(
            len(segments[0].anchor_times),
            len(segments[0].anchor_offsets),
        )

    def test_prior_resolves_a_repeated_passage(self) -> None:
        sr = 8_000
        rng = np.random.default_rng(2718)
        passage = scipy.signal.lfilter(
            [1.0],
            [1.0, -0.88],
            rng.standard_normal((4 * sr, 2)),
            axis=0,
        ).astype(np.float32)
        passage *= 0.2 / np.std(passage)
        foreground = (
            0.3
            * scipy.signal.lfilter([1.0], [1.0, -0.7], rng.standard_normal(4 * sr))
        ).astype(np.float32)
        mixture = passage + foreground[:, np.newaxis]
        pad = np.zeros((sr, 2), dtype=np.float32)
        gap = (0.01 * rng.standard_normal((sr, 2))).astype(np.float32)
        # The decoy repeat carries the foreground too, so an unconstrained
        # search prefers it over the passage the chunk actually came from.
        reference_search = np.vstack([pad, passage, gap, mixture])
        true_start = len(pad)
        decoy_start = len(pad) + len(passage) + len(gap)

        unconstrained = _align_reference(mixture, reference_search, sr)
        guided = _align_reference(
            mixture,
            reference_search,
            sr,
            predicted_start=float(true_start),
            search_radius_sec=0.25,
        )

        self.assertAlmostEqual(unconstrained.start, decoy_start, delta=0.01 * sr)
        self.assertAlmostEqual(guided.start, true_start, delta=0.01 * sr)
        self.assertTrue(guided.covered)

    def test_uncovered_alignment_passes_through_instead_of_raising(self) -> None:
        sr = 8_000
        rng = np.random.default_rng(999)
        mixture = rng.standard_normal((sr, 2)).astype(np.float32)
        reference_search = rng.standard_normal((2 * sr, 2)).astype(np.float32)

        def uncovered(*args, **kwargs):
            diagnostics = args[3] if len(args) > 3 else kwargs.get("diagnostics")
            if diagnostics is not None:
                diagnostics["coverage_deficit_start_sec"] = 0.0
                diagnostics["coverage_deficit_end_sec"] = 0.4
            return _ReferenceAlignment(None, 0.9, 0, False)

        with patch(
            "audio_overlap_removal.cancellation._align_reference",
            side_effect=uncovered,
        ):
            output, diagnostics = _cancel_chunk(
                mixture,
                reference_search,
                sr,
                cleanup_strength=1.0,
            )

        np.testing.assert_array_equal(output, 0.5 * (mixture[:, 0] + mixture[:, 1]))
        self.assertEqual(diagnostics["coverage_passthrough"], 1.0)
        self.assertEqual(diagnostics["coverage_deficit_end_sec"], 0.4)

    def test_trajectory_fit_rejects_outliers_and_fills_gaps(self) -> None:
        fitted = _fit_offset_trajectory(
            [
                (0, 1.0, 0.9),
                (1, 1.1, 0.9),
                (2, 1.2, 0.9),
                (3, 9.9, 0.9),
                (4, 1.4, 0.9),
                (5, None, 0.0),
            ]
        )

        self.assertAlmostEqual(fitted[0].offset, 1.0)
        self.assertAlmostEqual(fitted[3].offset, 1.3)
        self.assertAlmostEqual(fitted[5].offset, 1.4)
        self.assertTrue(fitted[0].confident)
        # A borrowed offset must not also lend its neighbours' confidence.
        self.assertFalse(fitted[3].confident)
        self.assertFalse(fitted[5].confident)

    def test_trajectory_fit_gives_up_without_confident_probes(self) -> None:
        self.assertEqual(
            _fit_offset_trajectory([(0, 1.0, 0.1), (1, None, 0.0)]),
            {},
        )

    def test_cancellation_needs_two_of_three_votes(self) -> None:
        sr = 8_000
        quiet_but_removing = {
            "alignment_score": 0.05,
            "aligned_start": 1_000.0,
            "control_reduction_db": 4.0,
        }
        nothing_removed = {
            "alignment_score": 0.05,
            "aligned_start": 1_000.0,
            "control_reduction_db": 0.1,
        }
        strong_but_runaway = {
            "alignment_score": 0.90,
            "aligned_start": 40_000.0,
            "control_reduction_db": 9.0,
        }

        # A low correlation score no longer vetoes a chunk on its own: the
        # momentum probe and the measured reduction can carry it.
        self.assertTrue(
            _accepts_cancellation(quiet_but_removing, 1_100.0, sr, 0.5, True)
        )
        self.assertFalse(
            _accepts_cancellation(quiet_but_removing, 1_100.0, sr, 0.5, False)
        )
        self.assertFalse(
            _accepts_cancellation(nothing_removed, 1_100.0, sr, 0.5, True)
        )
        # Landing far outside the predicted window is a veto, not one vote.
        self.assertFalse(
            _accepts_cancellation(strong_but_runaway, 1_100.0, sr, 0.5, True)
        )
        # An outright strong correlation is sufficient on its own.
        self.assertTrue(
            _accepts_cancellation(
                {**nothing_removed, "alignment_score": 0.90},
                1_100.0,
                sr,
                0.5,
                False,
            )
        )
        self.assertFalse(
            _accepts_cancellation(
                {**quiet_but_removing, "coverage_passthrough": 1.0},
                1_100.0,
                sr,
                0.5,
                True,
            )
        )

    def test_passthrough_spans_merge_when_contiguous(self) -> None:
        self.assertEqual(
            _merge_passthrough_spans([(0.0, 30.0), (30.0, 60.0), (90.0, 120.0)]),
            [(0.0, 60.0), (90.0, 120.0)],
        )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg is required for the offset recovery integration tests.",
)
class OffsetRecoveryIntegrationTests(unittest.TestCase):
    sr = 8_000
    true_offset = 2.0
    matched = (4.0, 20.0)

    def run_pipeline(
        self,
        declared_offset: float,
        *,
        momentum: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory() as directory:
            mixture_path, reference_path, mixture, target = _write_drifting_fixture(
                Path(directory), self.sr, self.true_offset
            )
            output = Path(directory, "clean.wav")
            process_audio(
                mixture_path,
                reference_path,
                str(output),
                alignment_segments=[
                    AlignmentSegment(*self.matched, declared_offset, 0.9)
                ],
                chunk_sec=4.0,
                search_sec=0.25,
                strength=0.0,
                momentum=momentum,
                sr=self.sr,
                workers=2,
            )
            written, _ = sf.read(output, dtype="float32")
        return written, mixture, target

    def matched_slice(self, audio: np.ndarray) -> np.ndarray:
        start = int(round((self.matched[0] + 1.0) * self.sr))
        end = int(round((self.matched[1] - 1.0) * self.sr))
        return audio[start:end]

    def residual_ratio(self, written, mixture, target) -> float:
        """How much of the removable reference survives in the matched span."""
        stereo_target = np.column_stack([target, target])
        residual = self.matched_slice(written) - self.matched_slice(stereo_target)
        removable = self.matched_slice(mixture) - self.matched_slice(stereo_target)
        return float(np.sqrt(np.mean(residual**2) / np.mean(removable**2)))

    def assert_reference_removed(self, written, mixture, target) -> None:
        self.assertLess(self.residual_ratio(written, mixture, target), 0.25)

    def test_momentum_recovers_a_wrong_segment_offset(self) -> None:
        written, mixture, target = self.run_pipeline(
            self.true_offset + 0.8,
            momentum=True,
        )

        self.assert_reference_removed(written, mixture, target)

    def test_widened_retry_recovers_without_momentum(self) -> None:
        written, mixture, target = self.run_pipeline(
            self.true_offset + 0.8,
            momentum=False,
        )

        self.assert_reference_removed(written, mixture, target)

    def test_output_range_writes_only_the_requested_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mixture_path, reference_path, mixture, target = _write_drifting_fixture(
                Path(directory), self.sr, self.true_offset
            )
            output = Path(directory, "clean.flac")
            report = Path(directory, "report.jsonl")
            process_audio(
                mixture_path,
                reference_path,
                str(output),
                alignment_segments=[
                    AlignmentSegment(*self.matched, self.true_offset, 0.9)
                ],
                chunk_sec=4.0,
                strength=0.0,
                output_start=self.matched[0],
                output_end=self.matched[1],
                report_path=str(report),
                sr=self.sr,
                workers=2,
            )
            written, _ = sf.read(output, dtype="float32")
            records = [
                json.loads(line) for line in report.read_text().splitlines() if line
            ]

        self.assertEqual(len(written), int(round(16.0 * self.sr)))
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["start_sec"], self.matched[0])
        self.assertEqual(records[-1]["end_sec"], self.matched[1])
        self.assertEqual({record["mode"] for record in records}, {"cancelled"})
        self.assertGreater(records[0]["control_reduction_db"], 1.0)

    def test_unrecoverable_offset_passes_through_the_whole_file(self) -> None:
        written, mixture, target = self.run_pipeline(
            self.true_offset + 8.0,
            momentum=True,
        )

        # No chunk may abort the run, and an unmatched chunk must reach the
        # output untouched rather than half-cancelled.
        self.assertEqual(len(written), len(mixture))
        self.assertGreater(self.residual_ratio(written, mixture, target), 0.99)
        np.testing.assert_allclose(written, mixture, atol=2e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
