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

    def offset_at(self, mixture_time: float) -> float:
        """Return C-time minus B-time, including steady playback-rate drift."""
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
        clipped.append(
            AlignmentSegment(
                mixture_start=clipped_start,
                mixture_end=clipped_end,
                offset_sec=segment.offset_at(center),
                median_score=segment.median_score,
                offset_slope=segment.offset_slope,
            )
        )
    return clipped
