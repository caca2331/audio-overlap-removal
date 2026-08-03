"""Public data models and cancellation profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentSegment:
    mixture_start: float
    mixture_end: float
    offset_sec: float
    median_score: float
    offset_slope: float = 0.0
    anchor_times: tuple[float, ...] = ()
    anchor_offsets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.anchor_times) != len(self.anchor_offsets):
            raise ValueError("anchor_times and anchor_offsets must be equal length.")

    def offset_at(self, mixture_time: float) -> float:
        """Return C-time minus B-time at one mixture timestamp.

        A discovered segment keeps the anchor trajectory measured during the
        scan. Real playback drift is rarely a straight line over tens of
        minutes, and a single slope leaves the middle of a long segment tens
        or hundreds of milliseconds off, which is enough to push chunk-local
        alignment outside its reference search buffer.
        """
        if self.anchor_times:
            return float(
                np.interp(mixture_time, self.anchor_times, self.anchor_offsets)
            )
        center = 0.5 * (self.mixture_start + self.mixture_end)
        return self.offset_sec + self.offset_slope * (mixture_time - center)


@dataclass(frozen=True)
class CancellationProfile:
    """Resolved expert controls for one quality/removal slider position."""

    cleanup_strength: float
    center_strength: float
    center_cleanup_strength: float
    silence_cleanup_strength: float


def cancellation_profile(strength: float) -> CancellationProfile:
    """Map one open-ended control to the four internal cancellation controls.

    Zero is the conservative subtraction-only mode, one reproduces the v8
    aggressive settings, and values above one select the v9-style increasingly
    aggressive centered cleanup.  The scale intentionally remains open-ended:
    callers may trade more audible damage for ASR-oriented media removal.
    """
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be non-negative.")
    common = min(1.0, strength)
    return CancellationProfile(
        cleanup_strength=common,
        center_strength=0.5 + 0.5 * common,
        center_cleanup_strength=strength,
        silence_cleanup_strength=common,
    )


def _segment_to_dict(segment: AlignmentSegment) -> dict:
    """Serialise a segment, including the anchor trajectory behind offset_at."""
    return {
        "mixture_start": segment.mixture_start,
        "mixture_end": segment.mixture_end,
        "offset_sec": segment.offset_sec,
        "median_score": segment.median_score,
        "offset_slope": segment.offset_slope,
        "anchor_times": list(segment.anchor_times),
        "anchor_offsets": list(segment.anchor_offsets),
    }


def _segment_from_dict(payload: dict) -> AlignmentSegment:
    """Rebuild a segment, tolerating hand-written files without a trajectory."""
    try:
        return AlignmentSegment(
            mixture_start=float(payload["mixture_start"]),
            mixture_end=float(payload["mixture_end"]),
            offset_sec=float(payload["offset_sec"]),
            median_score=float(payload.get("median_score", 1.0)),
            offset_slope=float(payload.get("offset_slope", 0.0)),
            anchor_times=tuple(
                float(time) for time in payload.get("anchor_times", ())
            ),
            anchor_offsets=tuple(
                float(offset) for offset in payload.get("anchor_offsets", ())
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid alignment segment: {payload!r}") from error


def _clipped_trajectory(
    segment: AlignmentSegment,
    start_sec: float,
    end_sec: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Restrict an anchor trajectory to a range without moving any offset."""
    if not segment.anchor_times:
        return (), ()
    times = np.asarray(segment.anchor_times, dtype=np.float64)
    interior = times[(times > start_sec) & (times < end_sec)]
    kept_times = (start_sec, *(float(time) for time in interior), end_sec)
    return kept_times, tuple(segment.offset_at(time) for time in kept_times)


def _clip_alignment_segments(
    segments: list[AlignmentSegment],
    start_sec: float,
    end_sec: float,
) -> list[AlignmentSegment]:
    """Clip matches to a declared media range without changing their timing."""
    if start_sec < 0.0 or end_sec <= start_sec:
        raise ValueError("media range must satisfy 0 <= start < end.")
    clipped: list[AlignmentSegment] = []
    for segment in segments:
        clipped_start = max(start_sec, segment.mixture_start)
        clipped_end = min(end_sec, segment.mixture_end)
        if clipped_end <= clipped_start:
            continue
        center = 0.5 * (clipped_start + clipped_end)
        anchor_times, anchor_offsets = _clipped_trajectory(
            segment,
            clipped_start,
            clipped_end,
        )
        clipped.append(
            AlignmentSegment(
                mixture_start=clipped_start,
                mixture_end=clipped_end,
                offset_sec=segment.offset_at(center),
                median_score=segment.median_score,
                offset_slope=segment.offset_slope,
                anchor_times=anchor_times,
                anchor_offsets=anchor_offsets,
            )
        )
    return clipped
