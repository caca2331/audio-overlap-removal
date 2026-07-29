"""File-level scan and cancellation orchestration."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import discover_alignment_segments
from .cancellation import _cancel_chunk, _passthrough_channels
from .media import (
    DEFAULT_SR,
    MAX_MEDIA_DURATION_SEC,
    MIN_SAMPLE_RATE,
    _atomic_soundfile,
    _audio_channel_count,
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
) -> Iterator[_ProcessingChunk]:
    """Yield chunk descriptors without retaining one object per output chunk."""
    position = 0.0
    while position < duration_sec - 1e-9:
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
            default=duration_sec,
        )
        piece_end = (
            min(duration_sec, active.mixture_end)
            if active is not None
            else min(duration_sec, next_active_start)
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


def scan_reference(
    mixture_path: str,
    reference_path: str,
    *,
    start: float = 0.0,
    end: float | None = None,
    workers: int = 4,
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
    sr: int = DEFAULT_SR,
    workers: int = 4,
) -> list[AlignmentSegment]:
    """Scan a range, remove matched reference audio, and write the full mixture."""
    segments = scan_reference(
        mixture_path,
        reference_path,
        start=start,
        end=end,
        workers=workers,
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

    chunks = _iter_processing_chunks(
        duration_sec,
        alignment_segments,
        chunk_sec,
        context_sec,
    )

    def process_chunk(
        chunk: _ProcessingChunk,
    ) -> tuple[_ProcessingChunk, np.ndarray, dict[str, float] | None]:
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
            predicted_reference_start = chunk.context_start - chunk.active.offset_at(
                chunk.context_start
            )
            reference_search_start = max(0.0, predicted_reference_start - search_sec)
            reference_search = _decode_stereo(
                reference_path,
                reference_search_start,
                chunk.context_duration + 2.0 * search_sec,
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
            )
            if diagnostics["alignment_score"] < 0.20:
                cleaned = original
                diagnostics["low_confidence_passthrough"] = 1.0

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

    with _atomic_soundfile(
        output,
        samplerate=sr,
        channels=mixture_channels,
        format=output_format,
        subtype=output_subtype,
    ) as sink:
        previous_sample: float | np.ndarray | None = None
        for chunk, core, diagnostics in _bounded_ordered_map(
            process_chunk, chunks, workers
        ):
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
                print(
                    f"{chunk.position:8.2f}-"
                    f"{chunk.position + chunk.core_duration:8.2f}s "
                    f"align={diagnostics['alignment_score']:.3f} "
                    f"gain={diagnostics['gain_median']:.3f} "
                    f"[{diagnostics['gain_p05']:.3f},"
                    f"{diagnostics['gain_p95']:.3f}] "
                    f"side-res={diagnostics['side_residual_ratio']:.3f}"
                    f" cleanup={diagnostics['cleanup_output_ratio']:.3f}"
                    f"{warp}"
                    f"{mode}"
                )

    elapsed = time.time() - started
    print(
        f"Wrote {output} ({duration_sec:.2f}s, "
        f"{'mono' if mixture_channels == 1 else 'stereo'}) "
        f"in {elapsed:.1f}s "
        f"(RTF={elapsed / max(duration_sec, 1e-9):.3f}, "
        f"workers={workers})."
    )
