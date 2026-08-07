"""File-level scan and cancellation orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.ndimage
import scipy.signal

from .alignment import _best_scaled_match, discover_alignment_segments
from .cancellation import (
    _cancel_chunk,
    _passthrough_channels,
    _passthrough_diagnostics,
)
from .media import (
    DEFAULT_SR,
    MAX_MEDIA_DURATION_SEC,
    MIN_SAMPLE_RATE,
    _atomic_soundfile,
    _audio_channel_count,
    _decode_mono_low,
    _decode_stereo,
    _media_duration,
    _output_settings,
    _paths_refer_to_same_file,
    _processing_channel_count,
)
from .models import (
    AlignmentSegment,
    _clip_alignment_segments,
    cancellation_profile,
)
from .parallel import _bounded_ordered_map

_MOMENTUM_ALIGN_SR = 4_000
_MOMENTUM_PROBE_SEC = 8.0
_MOMENTUM_SEARCH_SEC = 2.0
_MOMENTUM_MIN_SCORE = 0.25
_MOMENTUM_TOLERANCE_SEC = 0.5
_MIN_ALIGNMENT_SCORE = 0.20
_STRONG_ALIGNMENT_SCORE = 0.60
_MIN_REDUCTION_DB = 1.0
_MAX_RETRY_RADIUS_SEC = 30.0
_MOMENTUM_BAND = scipy.signal.butter(
    4,
    [80.0, 1_800.0],
    btype="bandpass",
    fs=_MOMENTUM_ALIGN_SR,
    output="sos",
)


@dataclass(frozen=True)
class _ProcessingChunk:
    position: float
    core_duration: float
    context_start: float
    context_duration: float
    active: AlignmentSegment | None


def _iter_processing_chunks(
    duration_sec: float,
    alignment_segments: list[AlignmentSegment],
    chunk_sec: float,
    context_sec: float,
    range_start: float = 0.0,
    range_end: float | None = None,
) -> Iterator[_ProcessingChunk]:
    """Yield chunk descriptors without retaining one object per output chunk.

    ``range_start``/``range_end`` restrict what is written, not what may be
    read: alignment context still reaches outside the range when the matched
    segment extends past it.
    """
    written_end = duration_sec if range_end is None else min(duration_sec, range_end)
    position = range_start
    while position < written_end - 1e-9:
        active = next(
            (
                segment
                for segment in alignment_segments
                if segment.mixture_start <= position < segment.mixture_end
            ),
            None,
        )
        next_active_start = min(
            (
                segment.mixture_start
                for segment in alignment_segments
                if segment.mixture_start > position
            ),
            default=written_end,
        )
        piece_end = (
            min(written_end, active.mixture_end)
            if active is not None
            else min(written_end, next_active_start)
        )
        core_duration = min(chunk_sec, piece_end - position)
        if core_duration <= 1e-9:
            position = piece_end
            continue

        context_floor = (
            max(0.0, active.mixture_start) if active is not None else position
        )
        context_ceiling = (
            min(duration_sec, active.mixture_end)
            if active is not None
            else position + core_duration
        )
        context_start = max(context_floor, position - context_sec)
        context_end = min(
            context_ceiling,
            position + core_duration + context_sec,
        )
        yield _ProcessingChunk(
            position=position,
            core_duration=core_duration,
            context_start=context_start,
            context_duration=context_end - context_start,
            active=active,
        )
        position += core_duration


def _measure_chunk_offset(
    mixture_path: str,
    reference_path: str,
    chunk: _ProcessingChunk,
) -> tuple[float, float] | None:
    """Estimate one chunk's true offset from a cheap low-rate probe."""
    active = chunk.active
    if active is None:
        return None
    probe_sec = min(_MOMENTUM_PROBE_SEC, chunk.core_duration)
    center = chunk.position + 0.5 * chunk.core_duration
    probe_start = max(0.0, center - 0.5 * probe_sec)
    predicted_offset = active.offset_at(center)
    reference_start = max(0.0, probe_start - predicted_offset - _MOMENTUM_SEARCH_SEC)
    mixture_low = _decode_mono_low(
        mixture_path,
        _MOMENTUM_ALIGN_SR,
        probe_start,
        probe_sec,
    )
    reference_low = _decode_mono_low(
        reference_path,
        _MOMENTUM_ALIGN_SR,
        reference_start,
        probe_sec + 2.0 * _MOMENTUM_SEARCH_SEC,
    )
    if len(mixture_low) < _MOMENTUM_ALIGN_SR or len(reference_low) < len(mixture_low):
        return None
    # Match on the same band the chunk-local aligner uses, so playback EQ and
    # codec differences do not weaken the probe that seeds it.
    mixture_low = scipy.signal.sosfiltfilt(_MOMENTUM_BAND, mixture_low)
    reference_low = scipy.signal.sosfiltfilt(_MOMENTUM_BAND, reference_low)
    index, score = _best_scaled_match(reference_low, mixture_low)
    measured_offset = probe_start - (reference_start + index / _MOMENTUM_ALIGN_SR)
    return measured_offset, float(score)


class _MeasuredOffset(NamedTuple):
    """A chunk's probed offset and whether the probe itself was convincing."""

    offset: float
    confident: bool


def _fit_offset_trajectory(
    measurements: list[tuple[int, float | None, float]],
) -> dict[int, _MeasuredOffset]:
    """Turn one segment's raw chunk probes into a smooth, outlier-free track.

    Chunk offsets are physically continuous, so a chunk that probes badly
    (a quiet passage, a repeated musical phrase) is far better served by its
    neighbours than by its own best guess.
    """
    indices = np.array([item[0] for item in measurements], dtype=np.float64)
    values = np.array(
        [np.nan if item[1] is None else item[1] for item in measurements],
        dtype=np.float64,
    )
    scores = np.array([item[2] for item in measurements], dtype=np.float64)
    confident = np.isfinite(values) & (scores >= _MOMENTUM_MIN_SCORE)
    if np.count_nonzero(confident) < 2:
        return {}
    confident_values = values[confident]
    baseline = scipy.ndimage.median_filter(
        confident_values,
        size=min(5, len(confident_values)),
        mode="nearest",
    )
    kept = np.abs(confident_values - baseline) <= _MOMENTUM_TOLERANCE_SEC
    if np.count_nonzero(kept) < 2:
        return {}
    kept_indices = indices[confident][kept]
    fitted = np.interp(indices, kept_indices, baseline[kept])
    measured = set(kept_indices.tolist())
    return {
        int(index): _MeasuredOffset(float(offset), index in measured)
        for index, offset in zip(indices, fitted)
    }


def _momentum_offsets(
    mixture_path: str,
    reference_path: str,
    chunks: Iterator[tuple[int, _ProcessingChunk]],
    workers: int,
) -> dict[int, _MeasuredOffset]:
    """Measure per-chunk offsets ahead of cancellation, one segment at a time."""
    matched = [item for item in chunks if item[1].active is not None]
    if not matched:
        return {}
    print(
        f"Probing {len(matched)} matched chunks at {_MOMENTUM_ALIGN_SR} Hz "
        "for offset momentum..."
    )
    probes = list(
        _bounded_ordered_map(
            lambda item: _measure_chunk_offset(mixture_path, reference_path, item[1]),
            matched,
            workers,
        )
    )
    offsets: dict[int, _MeasuredOffset] = {}
    group: list[tuple[int, float | None, float]] = []
    group_segment: AlignmentSegment | None = None
    for (index, chunk), probe in zip(matched, probes):
        if chunk.active is not group_segment and group:
            offsets.update(_fit_offset_trajectory(group))
            group = []
        group_segment = chunk.active
        group.append((index, None, 0.0) if probe is None else (index, *probe))
    if group:
        offsets.update(_fit_offset_trajectory(group))
    drift = [
        offsets[index].offset - chunk.active.offset_at(chunk.position)
        for index, chunk in matched
        if index in offsets and chunk.active is not None
    ]
    if drift:
        measured = sum(1 for offset in offsets.values() if offset.confident)
        print(
            f"  momentum tracked {measured}/{len(matched)} chunks directly "
            f"(median {1_000.0 * float(np.median(drift)):+.0f}ms, "
            f"max {1_000.0 * float(np.max(np.abs(drift))):.0f}ms "
            "from the segment model)"
        )
    return offsets


def _accepts_cancellation(
    diagnostics: dict[str, float],
    predicted_start: float,
    sr: int,
    tolerance_sec: float,
    momentum_confident: bool,
) -> bool:
    """Vote on a cancelled chunk using three independent kinds of evidence.

    The chunk-local correlation score sags on quiet passages, the low-rate
    momentum probe is unavailable where a chunk had to borrow its offset from
    neighbours, and the measured energy reduction is naturally zero while the
    removable media is paused. Any two agreeing is enough, and an outright
    strong correlation carries a chunk on its own.

    Landing far from the prediction is a veto rather than a vote: the coarse
    search is deliberately biased towards the prior, so agreement is weak
    evidence while a large disagreement still means the alignment ran away.
    """
    if diagnostics.get("coverage_passthrough") or diagnostics.get(
        "insufficient_reference_passthrough"
    ):
        return False
    deviation = abs(diagnostics.get("aligned_start", 0.0) - predicted_start) / sr
    if deviation > tolerance_sec:
        return False
    score = diagnostics.get("alignment_score", 0.0)
    votes = (
        2
        if score >= _STRONG_ALIGNMENT_SCORE
        else 1
        if score >= _MIN_ALIGNMENT_SCORE
        else 0
    )
    votes += momentum_confident
    votes += diagnostics.get("control_reduction_db", 0.0) >= _MIN_REDUCTION_DB
    return votes >= 2


def _chunk_mode(diagnostics: dict[str, float] | None) -> str:
    if diagnostics is None:
        return "unmatched"
    if diagnostics.get("low_confidence_passthrough"):
        return "low-confidence"
    return "cancelled"


def _merge_passthrough_spans(
    spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= 1e-6:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def scan_reference(
    mixture_path: str,
    reference_path: str,
    *,
    start: float = 0.0,
    end: float | None = None,
    workers: int = 4,
    scan_mode: str = "auto",
) -> list[AlignmentSegment]:
    """Locate reference-bearing segments inside a mixture time range."""
    if not np.isfinite(start) or start < 0.0:
        raise ValueError("start must be non-negative and finite.")
    if end is not None and not np.isfinite(end):
        raise ValueError("end must be finite.")

    mixture_duration = _media_duration(mixture_path)
    reference_duration = _media_duration(reference_path)
    for label, duration in (
        ("mixture", mixture_duration),
        ("reference", reference_duration),
    ):
        if duration > MAX_MEDIA_DURATION_SEC + 1e-3:
            raise ValueError(
                f"{label} duration ({duration:.3f}s) exceeds the supported "
                f"24-hour limit."
            )
    if start >= mixture_duration:
        raise ValueError(
            f"start ({start:.3f}s) is outside the mixture ({mixture_duration:.3f}s)."
        )
    scan_end = mixture_duration if end is None else end
    if scan_end <= start:
        raise ValueError("end must be greater than start.")
    if scan_end > mixture_duration + 1e-3:
        raise ValueError(
            f"end ({scan_end:.3f}s) is outside the mixture ({mixture_duration:.3f}s)."
        )
    scan_end = min(scan_end, mixture_duration)

    segments = discover_alignment_segments(
        mixture_path,
        reference_path,
        workers=workers,
        mixture_start_sec=start,
        mixture_duration_sec=scan_end - start,
        reference_duration_sec=reference_duration,
        scan_mode=scan_mode,
    )
    return _clip_alignment_segments(segments, start, scan_end)


def remove_reference(
    mixture_path: str,
    reference_path: str,
    output_path: str,
    *,
    start: float = 0.0,
    end: float | None = None,
    chunk_sec: float = 30.0,
    context_sec: float = 1.0,
    search_sec: float = 0.25,
    strength: float = 1.0,
    cleanup_strength: float | None = None,
    center_strength: float | None = None,
    center_cleanup_strength: float | None = None,
    silence_cleanup_strength: float | None = None,
    adaptive_time_warp: bool = True,
    momentum: bool = True,
    output_start: float = 0.0,
    output_end: float | None = None,
    report_path: str | None = None,
    sr: int = DEFAULT_SR,
    workers: int = 4,
    scan_mode: str = "auto",
) -> list[AlignmentSegment]:
    """Scan a range, remove matched reference audio, and write the full mixture."""
    segments = scan_reference(
        mixture_path,
        reference_path,
        start=start,
        end=end,
        workers=workers,
        scan_mode=scan_mode,
    )
    process_audio(
        mixture_path,
        reference_path,
        output_path,
        alignment_segments=segments,
        chunk_sec=chunk_sec,
        context_sec=context_sec,
        search_sec=search_sec,
        strength=strength,
        cleanup_strength=cleanup_strength,
        center_strength=center_strength,
        center_cleanup_strength=center_cleanup_strength,
        silence_cleanup_strength=silence_cleanup_strength,
        adaptive_time_warp=adaptive_time_warp,
        momentum=momentum,
        output_start=output_start,
        output_end=output_end,
        report_path=report_path,
        sr=sr,
        workers=workers,
    )
    return segments


def process_audio(
    mixture_path: str,
    reference_path: str,
    output_path: str,
    alignment_segments: list[AlignmentSegment],
    chunk_sec: float = 30.0,
    context_sec: float = 1.0,
    search_sec: float = 0.25,
    strength: float = 1.0,
    cleanup_strength: float | None = None,
    center_strength: float | None = None,
    center_cleanup_strength: float | None = None,
    silence_cleanup_strength: float | None = None,
    adaptive_time_warp: bool = True,
    momentum: bool = True,
    output_start: float = 0.0,
    output_end: float | None = None,
    report_path: str | None = None,
    sr: int = DEFAULT_SR,
    workers: int = 4,
) -> None:
    if not np.isfinite(chunk_sec) or chunk_sec <= 0.0:
        raise ValueError("chunk_sec must be positive and finite.")
    if (
        not np.isfinite(context_sec)
        or context_sec < 0.0
        or not np.isfinite(search_sec)
        or search_sec < 0.0
    ):
        raise ValueError("context_sec and search_sec must be non-negative.")
    if sr < MIN_SAMPLE_RATE:
        raise ValueError(f"sample rate must be at least {MIN_SAMPLE_RATE} Hz.")
    output = Path(output_path)
    if _paths_refer_to_same_file(output, mixture_path):
        raise ValueError("Output path must differ from the mixture input path.")
    if _paths_refer_to_same_file(output, reference_path):
        raise ValueError("Output path must differ from the reference input path.")
    output_format, output_subtype = _output_settings(output)
    duration_sec = _media_duration(mixture_path)
    if not np.isfinite(output_start) or output_start < 0.0:
        raise ValueError("output_start must be non-negative and finite.")
    if output_end is not None and not np.isfinite(output_end):
        raise ValueError("output_end must be finite.")
    written_end = duration_sec if output_end is None else min(duration_sec, output_end)
    if written_end <= output_start:
        raise ValueError("output_end must be greater than output_start.")

    profile = cancellation_profile(strength)
    cleanup_strength = (
        profile.cleanup_strength if cleanup_strength is None else cleanup_strength
    )
    center_strength = (
        profile.center_strength if center_strength is None else center_strength
    )
    center_cleanup_strength = (
        profile.center_cleanup_strength
        if center_cleanup_strength is None
        else center_cleanup_strength
    )
    silence_cleanup_strength = (
        profile.silence_cleanup_strength
        if silence_cleanup_strength is None
        else silence_cleanup_strength
    )
    if not np.isfinite(cleanup_strength) or cleanup_strength < 0.0:
        raise ValueError("cleanup_strength must be non-negative and finite.")
    if not 0.0 <= center_strength <= 1.0:
        raise ValueError("center_strength must be between 0 and 1.")
    if not np.isfinite(center_cleanup_strength) or center_cleanup_strength < 0.0:
        raise ValueError("center_cleanup_strength must be non-negative and finite.")
    if not 0.0 <= silence_cleanup_strength <= 1.0:
        raise ValueError("silence_cleanup_strength must be between 0 and 1.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    mixture_native_channels = _audio_channel_count(mixture_path)
    reference_native_channels = _audio_channel_count(reference_path)
    mixture_channels = _processing_channel_count(mixture_native_channels)
    reference_channels = _processing_channel_count(reference_native_channels)
    if mixture_native_channels > 2:
        print(
            f"Mixture has {mixture_native_channels} channels; "
            "FFmpeg will downmix it to stereo."
        )
    if reference_native_channels > 2:
        print(
            f"Reference has {reference_native_channels} channels; "
            "FFmpeg will downmix it to stereo."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    alignment_segments = _clip_alignment_segments(
        sorted(alignment_segments, key=lambda segment: segment.mixture_start),
        0.0,
        duration_sec,
    )

    def iter_indexed_chunks() -> Iterator[tuple[int, _ProcessingChunk]]:
        return enumerate(
            _iter_processing_chunks(
                duration_sec,
                alignment_segments,
                chunk_sec,
                context_sec,
                output_start,
                written_end,
            )
        )

    measured_offsets = (
        _momentum_offsets(
            mixture_path,
            reference_path,
            iter_indexed_chunks(),
            workers,
        )
        if momentum
        else {}
    )

    def cancel_attempt(
        chunk: _ProcessingChunk,
        mixture: np.ndarray,
        offset: float,
        radius: float,
    ) -> tuple[np.ndarray, dict[str, float], float]:
        predicted_reference_start = chunk.context_start - offset
        window_start = max(0.0, predicted_reference_start - radius)
        window_end = predicted_reference_start + chunk.context_duration + radius
        predicted_start = (predicted_reference_start - window_start) * sr
        if window_end - window_start < chunk.context_duration:
            # The offset places this chunk before the start of the reference,
            # so there is no window to decode at all.
            return (
                _passthrough_channels(mixture, mixture_channels),
                _passthrough_diagnostics("insufficient_reference_passthrough"),
                predicted_start,
            )
        reference_search = _decode_stereo(
            reference_path,
            window_start,
            window_end - window_start,
            sr,
            reference_channels,
        )
        cleaned, diagnostics = _cancel_chunk(
            mixture,
            reference_search,
            sr,
            cleanup_strength,
            center_strength,
            center_cleanup_strength,
            silence_cleanup_strength,
            adaptive_time_warp,
            mixture_channels,
            reference_channels,
            mixture_channels,
            predicted_start=predicted_start,
            search_radius_sec=radius,
        )
        return cleaned, diagnostics, predicted_start

    def process_chunk(
        indexed_chunk: tuple[int, _ProcessingChunk],
    ) -> tuple[_ProcessingChunk, np.ndarray, dict[str, float] | None]:
        index, chunk = indexed_chunk
        mixture = _decode_stereo(
            mixture_path,
            chunk.context_start,
            chunk.context_duration,
            sr,
            mixture_channels,
        )
        original = _passthrough_channels(mixture, mixture_channels)
        if len(mixture) == 0 or chunk.active is None:
            cleaned = original
            diagnostics = None
        else:
            measured = measured_offsets.get(index)
            offset = (
                measured.offset
                if measured is not None
                else chunk.active.offset_at(
                    chunk.context_start + 0.5 * chunk.context_duration
                )
            )
            radius = search_sec
            retried = False
            while True:
                cleaned, diagnostics, predicted_start = cancel_attempt(
                    chunk, mixture, offset, radius
                )
                if retried:
                    diagnostics["retry_radius_sec"] = radius
                if _accepts_cancellation(
                    diagnostics,
                    predicted_start,
                    sr,
                    2.0 * radius,
                    measured is not None and measured.confident,
                ):
                    break
                if retried:
                    cleaned = original
                    diagnostics["low_confidence_passthrough"] = 1.0
                    break
                # One widened retry recovers both recoverable failures: a
                # window that did not span the aligned chunk, and a prior too
                # far off to be found inside it.
                # The measured deficit is what the first attempt's warp needed;
                # a wider window can warp a little further still, so leave real
                # headroom on top of it.
                deficit = diagnostics.get(
                    "coverage_deficit_start_sec", 0.0
                ) + diagnostics.get("coverage_deficit_end_sec", 0.0)
                radius = min(
                    _MAX_RETRY_RADIUS_SEC,
                    max(4.0 * search_sec, deficit + 4.0 * search_sec),
                )
                retried = True

        core_start = int(round((chunk.position - chunk.context_start) * sr))
        core_frames = int(round(chunk.core_duration * sr))
        core = cleaned[core_start : core_start + core_frames]
        original_core = original[core_start : core_start + core_frames]
        if len(core) < core_frames:
            missing = core_frames - len(core)
            core = (
                np.pad(core, (0, missing))
                if core.ndim == 1
                else np.pad(core, ((0, missing), (0, 0)))
            )
            original_core = (
                np.pad(original_core, (0, missing))
                if original_core.ndim == 1
                else np.pad(original_core, ((0, missing), (0, 0)))
            )

        was_processed = diagnostics is not None and not diagnostics.get(
            "low_confidence_passthrough"
        )
        if was_processed and len(core):
            # Blend only inside the declared matched segment. Pass-through
            # samples outside it remain untouched.
            fade_frames = min(len(core), max(16, int(round(0.01 * sr))))
            if abs(chunk.position - chunk.active.mixture_start) < 1e-6:
                weight = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
                if core.ndim == 2:
                    weight = weight[:, np.newaxis]
                core[:fade_frames] = (
                    original_core[:fade_frames] * (1.0 - weight)
                    + core[:fade_frames] * weight
                )
            chunk_end = chunk.position + chunk.core_duration
            if abs(chunk_end - chunk.active.mixture_end) < 1e-6:
                weight = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
                if core.ndim == 2:
                    weight = weight[:, np.newaxis]
                core[-fade_frames:] = (
                    original_core[-fade_frames:] * (1.0 - weight)
                    + core[-fade_frames:] * weight
                )
        return chunk, core, diagnostics

    # The report is written as it goes and flushed per chunk, so an interrupted
    # run still says exactly how far it got and what each chunk decided.
    report = (
        Path(report_path).open("w", encoding="utf-8")
        if report_path is not None
        else nullcontext(None)
    )
    with report as report_file, _atomic_soundfile(
        output,
        samplerate=sr,
        channels=mixture_channels,
        format=output_format,
        subtype=output_subtype,
    ) as sink:
        previous_sample: float | np.ndarray | None = None
        passthrough_spans: list[tuple[float, float]] = []
        for chunk, core, diagnostics in _bounded_ordered_map(
            process_chunk, iter_indexed_chunks(), workers
        ):
            if report_file is not None:
                report_file.write(
                    json.dumps(
                        {
                            "start_sec": chunk.position,
                            "end_sec": chunk.position + chunk.core_duration,
                            "mode": _chunk_mode(diagnostics),
                            **(diagnostics or {}),
                        }
                    )
                    + "\n"
                )
                report_file.flush()
            was_processed = diagnostics is not None and not diagnostics.get(
                "low_confidence_passthrough"
            )
            if previous_sample is not None and len(core) and was_processed:
                # Independent codec seeks and reference discontinuities can
                # otherwise create a one-sample step at a chunk boundary.
                ramp_frames = min(len(core), max(16, int(round(0.01 * sr))))
                correction = previous_sample - core[0]
                ramp = np.linspace(1.0, 0.0, ramp_frames, dtype=np.float32)
                core[:ramp_frames] += (
                    correction * ramp
                    if core.ndim == 1
                    else ramp[:, np.newaxis] * correction[np.newaxis, :]
                )
            sink.write(core)
            if len(core):
                previous_sample = float(core[-1]) if core.ndim == 1 else core[-1].copy()

            if diagnostics is None:
                print(
                    f"{chunk.position:8.2f}-"
                    f"{chunk.position + chunk.core_duration:8.2f}s "
                    "pass-through (no reference match)"
                )
            else:
                if diagnostics.get("low_confidence_passthrough"):
                    passthrough_spans.append(
                        (chunk.position, chunk.position + chunk.core_duration)
                    )
                mode = (
                    " low-confidence pass-through"
                    if diagnostics.get("low_confidence_passthrough")
                    else ""
                )
                warp = (
                    " warp=short"
                    if diagnostics.get("short_warp_accepted")
                    else " warp=long-validated"
                    if diagnostics.get("short_warp_considered")
                    else ""
                )
                retry = (
                    f" retry-radius={diagnostics['retry_radius_sec']:.2f}s"
                    if "retry_radius_sec" in diagnostics
                    else ""
                )
                print(
                    f"{chunk.position:8.2f}-"
                    f"{chunk.position + chunk.core_duration:8.2f}s "
                    f"align={diagnostics['alignment_score']:.3f} "
                    f"gain={diagnostics['gain_median']:.3f} "
                    f"[{diagnostics['gain_p05']:.3f},"
                    f"{diagnostics['gain_p95']:.3f}] "
                    f"side-res={diagnostics['side_residual_ratio']:.3f}"
                    f" cleanup={diagnostics['cleanup_output_ratio']:.3f}"
                    f" reduction={diagnostics.get('control_reduction_db', 0.0):+.1f}dB"
                    f"{warp}"
                    f"{retry}"
                    f"{mode}"
                )

    for start, end in _merge_passthrough_spans(passthrough_spans):
        print(
            f"Low-confidence pass-through inside a matched segment: "
            f"{start:.2f}-{end:.2f}s"
        )

    elapsed = time.time() - started
    written_sec = written_end - output_start
    span = (
        ""
        if written_sec >= duration_sec - 1e-6
        else f", mixture {output_start:.2f}-{written_end:.2f}s"
    )
    print(
        f"Wrote {output} ({written_sec:.2f}s{span}, "
        f"{'mono' if mixture_channels == 1 else 'stereo'}) "
        f"in {elapsed:.1f}s "
        f"(RTF={elapsed / max(written_sec, 1e-9):.3f}, "
        f"workers={workers})."
    )
