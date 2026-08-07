"""Global discovery and chunk-local reference alignment."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal

from .fingerprint import (
    FingerprintCandidate,
    FingerprintIndex,
    FingerprintTrack,
    fingerprint_media,
)
from .media import MIN_ALIGNMENT_SAMPLE_RATE, _decode_mono_low, _media_duration
from .models import AlignmentSegment
from .parallel import _bounded_ordered_map


class _GlobalMatcher:
    """Reuse reference-side FFT and energy data across global queries."""

    def __init__(self, reference: np.ndarray, max_query_frames: int):
        if max_query_frames < 1:
            raise ValueError("max_query_frames must be positive.")
        self.reference = reference.astype(np.float64, copy=False)
        self.reference_frames = len(self.reference)
        self.fft_frames = scipy.fft.next_fast_len(
            self.reference_frames + max_query_frames - 1
        )
        self.reference_fft = scipy.fft.rfft(self.reference, n=self.fft_frames)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.reference)))
        self.cumulative_sq = np.concatenate(
            ([0.0], np.cumsum(self.reference * self.reference))
        )

    def _normalized_match(self, query: np.ndarray) -> tuple[int, float]:
        query64 = query.astype(np.float64, copy=False)
        query64 = query64 - np.mean(query64)
        query_frames = len(query64)
        if query_frames == 0 or self.reference_frames < query_frames:
            raise ValueError("Reference search window is shorter than query.")

        query_fft = scipy.fft.rfft(query64[::-1], n=self.fft_frames)
        convolution = scipy.fft.irfft(self.reference_fft * query_fft, n=self.fft_frames)
        numerator = convolution[query_frames - 1 : self.reference_frames]
        rolling_sum = self.cumulative[query_frames:] - self.cumulative[:-query_frames]
        rolling_energy = (
            self.cumulative_sq[query_frames:]
            - self.cumulative_sq[:-query_frames]
            - rolling_sum * rolling_sum / query_frames
        )
        rolling_energy = np.maximum(rolling_energy, 1e-20)
        denominator = np.sqrt(rolling_energy * np.sum(query64 * query64))
        correlation = numerator / np.maximum(denominator, 1e-20)
        best = int(np.argmax(correlation))
        return best, float(correlation[best])

    def best_scaled_match(self, query: np.ndarray) -> tuple[int, float]:
        best_index, best_score = self._normalized_match(query)
        if best_score >= 0.30:
            return best_index, best_score
        for scale in (0.995, 1.005, 0.99, 1.01):
            frames = int(round(len(query) * scale))
            if frames <= 0 or frames > self.reference_frames:
                continue
            scaled = scipy.signal.resample(query, frames)
            index, score = self._normalized_match(scaled)
            if score > best_score:
                best_index, best_score = index, score
        return best_index, best_score


def _validate_alignment_options(
    *,
    align_sr: int,
    global_step_sec: float,
    query_sec: float,
    local_step_sec: float,
    local_search_sec: float,
    min_score: float,
    reacquire_after_sec: float,
    reacquire_interval_sec: float,
    workers: int,
    mixture_start_sec: float,
    mixture_duration_sec: float | None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if align_sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(f"align_sr must be at least {MIN_ALIGNMENT_SAMPLE_RATE}.")
    positive_timings = (
        global_step_sec,
        query_sec,
        local_step_sec,
        reacquire_interval_sec,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_timings):
        raise ValueError("alignment step and query durations must be positive.")
    if not np.isfinite(local_search_sec) or local_search_sec < 0.0:
        raise ValueError("local_search_sec must be non-negative.")
    if not np.isfinite(reacquire_after_sec) or reacquire_after_sec < 0.0:
        raise ValueError("reacquisition timings are invalid.")
    if not np.isfinite(min_score) or not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1.")
    if not np.isfinite(mixture_start_sec) or mixture_start_sec < 0.0:
        raise ValueError("mixture_start_sec must be non-negative.")
    if mixture_duration_sec is not None and (
        not np.isfinite(mixture_duration_sec) or mixture_duration_sec <= 0.0
    ):
        raise ValueError("mixture_duration_sec must be positive.")


def _discover_full_alignment_segments(
    mixture_path: str,
    reference_path: str,
    align_sr: int = 1_000,
    global_step_sec: float = 60.0,
    query_sec: float = 8.0,
    local_step_sec: float = 5.0,
    local_search_sec: float = 0.25,
    min_score: float = 0.18,
    reacquire_after_sec: float = 10.0,
    reacquire_interval_sec: float = 15.0,
    workers: int = 4,
    mixture_start_sec: float = 0.0,
    mixture_duration_sec: float | None = None,
) -> list[AlignmentSegment]:
    """Find matching regions and recover after pauses, seeks, and replays."""
    _validate_alignment_options(
        align_sr=align_sr,
        global_step_sec=global_step_sec,
        query_sec=query_sec,
        local_step_sec=local_step_sec,
        local_search_sec=local_search_sec,
        min_score=min_score,
        reacquire_after_sec=reacquire_after_sec,
        reacquire_interval_sec=reacquire_interval_sec,
        workers=workers,
        mixture_start_sec=mixture_start_sec,
        mixture_duration_sec=mixture_duration_sec,
    )
    # Full-reference FFTs become memory-bandwidth-bound beyond two concurrent
    # queries on typical desktop CPUs. Keep cancellation workers independent.
    scan_workers = min(workers, 2)
    print(f"Scanning alignment at {align_sr} Hz (workers={scan_workers})...")
    decoded_mixture_duration = (
        mixture_duration_sec + query_sec if mixture_duration_sec is not None else None
    )
    decode_jobs = (
        (mixture_path, mixture_start_sec, decoded_mixture_duration),
        (reference_path, 0.0, None),
    )
    mixture, reference = _bounded_ordered_map(
        lambda job: _decode_mono_low(job[0], align_sr, job[1], job[2]),
        decode_jobs,
        scan_workers,
    )
    query_frames = int(round(query_sec * align_sr))
    if len(mixture) < query_frames or len(reference) < query_frames:
        return []
    high = min(450.0, 0.45 * align_sr)
    sos = scipy.signal.butter(
        4, [60.0, high], btype="bandpass", fs=align_sr, output="sos"
    )
    mixture = scipy.signal.sosfiltfilt(sos, mixture).astype(np.float32)
    reference = scipy.signal.sosfiltfilt(sos, reference).astype(np.float32)
    global_matcher = _GlobalMatcher(reference, int(np.ceil(1.01 * query_frames)))

    def global_match(mixture_time: float) -> tuple[float, int, float]:
        start = int(round(mixture_time * align_sr))
        query = mixture[start : start + query_frames]
        reference_index, score = global_matcher.best_scaled_match(query)
        return mixture_time, reference_index, score

    # Mixture array positions remain relative to the requested scan range;
    # offsets and returned segment timestamps use the original media timeline.
    seed: tuple[float, float, float] | None = None
    max_mixture_start = (len(mixture) - query_frames) / align_sr
    seed_times = np.arange(0.0, max_mixture_start, global_step_sec)
    for batch_start in range(0, len(seed_times), scan_workers):
        batch = seed_times[batch_start : batch_start + scan_workers]
        results = list(_bounded_ordered_map(global_match, batch, scan_workers))
        for mixture_time, reference_index, score in results:
            if score >= max(0.25, min_score):
                reference_time = reference_index / align_sr
                seed = (mixture_time, reference_time, score)
                break
        if seed is not None:
            break

    if seed is None:
        return []

    seed_absolute_time = mixture_start_sec + seed[0]
    seed_offset = seed_absolute_time - seed[1]
    print(
        f"  seed C={seed_absolute_time:.1f}s B={seed[1]:.3f}s "
        f"offset={seed_offset:.3f}s score={seed[2]:.3f}"
    )

    earliest = max(
        0.0,
        seed_offset - mixture_start_sec - local_search_sec,
    )
    # Do not cap the mixture timeline using the initial offset: pauses and
    # replays can make C longer than the remaining portion of B.
    latest = max_mixture_start
    anchors: list[tuple[float, float, float]] = []
    search_frames = int(round(local_search_sec * align_sr))
    tracked_offset = seed_offset
    last_match_time = seed[0]
    last_reacquire_time = seed[0] - reacquire_interval_sec
    prefetched_global: dict[float, tuple[int, float]] = {}
    first_local_time = np.ceil(earliest / local_step_sec) * local_step_sec
    local_times = np.arange(first_local_time, latest, local_step_sec)
    for local_index, mixture_time in enumerate(local_times):
        mixture_time = float(mixture_time)
        query_start = int(round(mixture_time * align_sr))
        query = mixture[query_start : query_start + query_frames]
        absolute_time = mixture_start_sec + mixture_time
        predicted_reference = int(round((absolute_time - tracked_offset) * align_sr))
        search_start = max(0, predicted_reference - search_frames)
        search_end = min(
            len(reference),
            predicted_reference + query_frames + search_frames,
        )
        search = reference[search_start:search_end]
        if len(search) >= len(query):
            local_index, score = _best_scaled_match(search, query)
            reference_time = (search_start + local_index) / align_sr
            if score >= min_score:
                tracked_offset = absolute_time - reference_time
                anchors.append((absolute_time, tracked_offset, score))
                last_match_time = mixture_time
                prefetched_global.clear()
                continue

        # A pause, seek, replay, or truncation invalidates the local prediction.
        # Search globally only after the tracker has genuinely been lost, then
        # rate-limit the expensive reacquisition while unmatched.
        should_reacquire = (
            mixture_time - last_match_time >= reacquire_after_sec
            and mixture_time - last_reacquire_time >= reacquire_interval_sec
        )
        if should_reacquire:
            last_reacquire_time = mixture_time
            cached = prefetched_global.pop(mixture_time, None)
            if cached is None:
                candidate_times = [mixture_time]
                next_due = mixture_time + reacquire_interval_sec
                for future_time in local_times[local_index + 1 :]:
                    future_time = float(future_time)
                    if future_time + 1e-9 < next_due:
                        continue
                    candidate_times.append(future_time)
                    next_due = future_time + reacquire_interval_sec
                    if len(candidate_times) >= scan_workers:
                        break
                results = list(
                    _bounded_ordered_map(global_match, candidate_times, scan_workers)
                )
                for result_time, result_index, result_score in results[1:]:
                    prefetched_global[result_time] = (
                        result_index,
                        result_score,
                    )
                _, global_index, global_score = results[0]
            else:
                global_index, global_score = cached
            if global_score >= max(0.25, min_score):
                reference_time = global_index / align_sr
                tracked_offset = absolute_time - reference_time
                anchors.append((absolute_time, tracked_offset, global_score))
                last_match_time = mixture_time
                prefetched_global.clear()

    segments = _segments_from_anchors(
        anchors,
        local_step_sec=local_step_sec,
        query_sec=query_sec,
    )
    _print_alignment_segments(segments)
    return segments


def _segments_from_anchors(
    anchors: list[tuple[float, float, float]],
    *,
    local_step_sec: float,
    query_sec: float,
) -> list[AlignmentSegment]:
    """Convert timestamp/offset/score anchors into stable matched regions."""
    if not anchors:
        return []
    gap_limit = max(3.0 * local_step_sec, query_sec + local_step_sec)
    groups: list[list[tuple[float, float, float]]] = []
    current = [anchors[0]]
    for anchor in anchors[1:]:
        if anchor[0] - current[-1][0] > gap_limit:
            groups.append(current)
            current = []
        current.append(anchor)
    groups.append(current)

    segments: list[AlignmentSegment] = []
    for group in groups:
        if len(group) < 3:
            continue
        offsets = np.array([item[1] for item in group])
        kernel = min(5, len(offsets) if len(offsets) % 2 else len(offsets) - 1)
        smooth = (
            scipy.signal.medfilt(offsets, kernel_size=kernel)
            if kernel >= 3
            else offsets
        )
        offset_steps = np.diff(smooth)
        if len(offset_steps) >= 3:
            local_trend = scipy.ndimage.median_filter(
                offset_steps, size=5, mode="nearest"
            )
        else:
            local_trend = np.full_like(
                offset_steps,
                np.median(offset_steps) if len(offset_steps) else 0.0,
            )
        # Subtract the local slope so a small steady speed difference does not
        # fragment the track into five-second pieces. Timestamp jumps remain.
        change_indices = list(
            np.where(np.abs(offset_steps - local_trend) >= 0.02)[0] + 1
        )
        boundaries = [0, *change_indices, len(group)]
        for part_index, (first, last) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            part = group[first:last]
            if len(part) < 2:
                continue
            # Keep the median-filtered offsets rather than the raw ones: a
            # single mismatched anchor must not steer a chunk-local search.
            part_offsets = np.asarray(smooth[first:last], dtype=np.float64)
            part_scores = np.array([item[2] for item in part])
            part_times = np.array([item[0] for item in part])
            start = part[0][0]
            end = (
                group[boundaries[part_index + 1]][0]
                if part_index + 1 < len(boundaries) - 1
                else part[-1][0] + query_sec
            )
            center_time = 0.5 * (start + end)
            if len(part) >= 3 and np.ptp(part_times) > 0:
                slope, intercept = np.polyfit(part_times - center_time, part_offsets, 1)
                slope = float(np.clip(slope, -0.05, 0.05))
                modeled_offset = float(intercept)
            else:
                slope = 0.0
                modeled_offset = float(np.median(part_offsets))
            segments.append(
                AlignmentSegment(
                    mixture_start=float(start),
                    mixture_end=float(end),
                    offset_sec=modeled_offset,
                    median_score=float(np.median(part_scores)),
                    offset_slope=slope,
                    anchor_times=tuple(float(time) for time in part_times),
                    anchor_offsets=tuple(
                        float(offset) for offset in part_offsets
                    ),
                )
            )

    return segments


def _print_alignment_segments(segments: list[AlignmentSegment]) -> None:
    for segment in segments:
        # The trajectory deviation is how far the kept anchors sit from the
        # straight-line model, i.e. how much a single slope would mispredict.
        deviation = 0.0
        if segment.anchor_times:
            times = np.asarray(segment.anchor_times)
            modeled = segment.offset_sec + segment.offset_slope * (
                times - 0.5 * (segment.mixture_start + segment.mixture_end)
            )
            offsets = np.asarray(segment.anchor_offsets)
            deviation = float(np.max(np.abs(offsets - modeled)))
        print(
            f"  match C={segment.mixture_start:.1f}-"
            f"{segment.mixture_end:.1f}s "
            f"offset={segment.offset_sec:.3f}s "
            f"speed={(1.0 - segment.offset_slope):.6f}x "
            f"median-score={segment.median_score:.3f} "
            f"traj-dev={1_000.0 * deviation:.0f}ms"
        )


def _sequence_global_match(
    mixture_track: FingerprintTrack,
    reference_track: FingerprintTrack,
    index: FingerprintIndex,
    mixture_time: float,
    query_sec: float,
) -> FingerprintCandidate | None:
    feature = mixture_track.feature_near(mixture_time)
    if feature is None:
        return None
    candidates = index.query(feature, candidates=24)
    reranked: list[FingerprintCandidate] = []
    for candidate in candidates:
        scores: list[float] = []
        for delta in (0.0, query_sec, 2.0 * query_sec):
            mixture_feature = mixture_track.feature_near(mixture_time + delta)
            reference_feature = reference_track.feature_near(candidate.time_sec + delta)
            if mixture_feature is not None and reference_feature is not None:
                scores.append(float(mixture_feature @ reference_feature))
        if scores:
            reranked.append(
                FingerprintCandidate(
                    media_id=candidate.media_id,
                    time_sec=candidate.time_sec,
                    score=float(min(scores[0], np.mean(scores))),
                )
            )
    return (
        min(
            reranked,
            key=lambda match: (-match.score, match.media_id, match.time_sec),
        )
        if reranked
        else None
    )


def _refine_indexed_offset(
    mixture_path: str,
    reference_path: str,
    mixture_time: float,
    grid_offset: float,
    *,
    align_sr: int,
    probe_sec: float,
    radius_sec: float,
    min_score: float,
) -> float | None:
    """Pin a fingerprint offset to the waveform, or None if it cannot be.

    Index times land on the fingerprint hop grid, so an offset read straight
    off them is up to half a hop out. That error is not harmless: it survives
    into cancellation, and it makes the tracker alternate between neighbouring
    grid points, which splits one match into hundreds of segments.
    """
    reference_time = mixture_time - grid_offset
    search_start = max(0.0, reference_time - radius_sec)
    query = _decode_mono_low(
        mixture_path, align_sr, start_sec=mixture_time, duration_sec=probe_sec
    )
    search = _decode_mono_low(
        reference_path,
        align_sr,
        start_sec=search_start,
        duration_sec=probe_sec + 2.0 * radius_sec,
    )
    if len(query) < align_sr or len(search) <= len(query):
        return None
    index, score = _normalized_match(search, query)
    if score < min_score:
        return None
    fractional = _fractional_match_index(search, query, index)
    refined = mixture_time - (search_start + fractional / align_sr)
    # The search window is clamped at the start of the media, so the peak can
    # land outside the intended radius. A refinement is only ever a correction
    # to the grid offset; anything further out is not one, and because the
    # tracker carries the refined value forward, one bad result would persist.
    if abs(refined - grid_offset) > radius_sec:
        return None
    return refined


def _discover_indexed_alignment_segments(
    mixture_path: str,
    reference_path: str,
    *,
    align_sr: int,
    global_step_sec: float,
    query_sec: float,
    local_step_sec: float,
    local_search_sec: float,
    min_score: float,
    reacquire_after_sec: float,
    reacquire_interval_sec: float,
    workers: int,
    mixture_start_sec: float,
    mixture_duration_sec: float | None,
) -> list[AlignmentSegment]:
    """Discover long-media matches from streamed compact fingerprints."""
    scan_workers = min(workers, 2)
    print(
        f"Scanning long-media alignment at {align_sr} Hz "
        f"(fingerprint index, workers={scan_workers})..."
    )
    decoded_mixture_duration = (
        mixture_duration_sec + query_sec if mixture_duration_sec is not None else None
    )
    jobs = (
        (
            reference_path,
            "reference",
            0.0,
            None,
        ),
        (
            mixture_path,
            "mixture",
            mixture_start_sec,
            decoded_mixture_duration,
        ),
    )
    reference_track, mixture_track = _bounded_ordered_map(
        lambda job: fingerprint_media(
            job[0],
            job[1],
            sr=align_sr,
            start_sec=job[2],
            duration_sec=job[3],
            query_sec=query_sec,
        ),
        jobs,
        scan_workers,
    )
    if not len(reference_track.times) or not len(mixture_track.times):
        return []
    index = FingerprintIndex([reference_track])
    print(
        f"  indexed {len(reference_track.times):,} reference windows "
        f"({index.memory_bytes / (1024 * 1024):.1f} MiB compact arrays)"
    )

    # Both sit above the chance floor of the fingerprint, measured at 0.29 on
    # real material. The global search compares against every reference window
    # and so sees a much heavier tail than the local one, which is constrained
    # by a prior; it keeps the wider margin.
    global_threshold = max(0.42, 0.34 + 0.25 * min_score)
    local_threshold = max(0.31, 0.24 + 0.25 * min_score)
    first_mixture_time = float(mixture_track.times[0])
    last_mixture_time = float(mixture_track.times[-1])
    available_span = last_mixture_time - first_mixture_time

    seed: tuple[float, float, float] | None = None
    for relative_time in np.arange(0.0, available_span + 1e-9, global_step_sec):
        mixture_time = first_mixture_time + float(relative_time)
        candidate = _sequence_global_match(
            mixture_track,
            reference_track,
            index,
            mixture_time,
            query_sec,
        )
        if candidate is not None and candidate.score >= global_threshold:
            seed = (mixture_time, candidate.time_sec, candidate.score)
            break
    if seed is None:
        return []

    seed_offset = seed[0] - seed[1]

    hop_sec = reference_track.hop_sec

    # Refinement needs to decode; when that is impossible it stays off for the
    # whole scan, so a rejected refinement always means "the waveform
    # disagrees" rather than "the waveform could not be read".
    refinable = True

    def refine(mixture_time: float, grid_offset: float) -> float | None:
        nonlocal refinable
        if not refinable:
            return None
        try:
            return _refine_indexed_offset(
                mixture_path,
                reference_path,
                mixture_time,
                grid_offset,
                align_sr=align_sr,
                probe_sec=query_sec,
                radius_sec=1.5 * hop_sec,
                min_score=min_score,
            )
        except (RuntimeError, ValueError):
            refinable = False
            return None

    # Probe one window inside the match rather than at the seed itself: at the
    # boundary the mixture has not started carrying the reference yet, and the
    # search window is clamped by the start of the media, so the peak there is
    # not a measurement of the offset.
    seed_probe = min(seed[0] + query_sec, last_mixture_time)
    refined_seed = refine(seed_probe, seed_offset)
    if refined_seed is not None:
        seed_offset = refined_seed
    print(
        f"  seed C={seed[0]:.1f}s B={seed[1]:.3f}s "
        f"offset={seed_offset:.3f}s score={seed[2]:.3f}"
    )
    earliest_relative = max(
        0.0,
        seed_offset
        - mixture_start_sec
        - max(local_search_sec, reference_track.hop_sec),
    )
    first_local_relative = np.ceil(earliest_relative / local_step_sec) * local_step_sec
    local_times = np.arange(
        first_mixture_time + first_local_relative,
        last_mixture_time + 1e-9,
        local_step_sec,
    )
    anchors: list[tuple[float, float, float]] = []
    tracked_offset = seed_offset
    last_match_time = seed[0]
    last_reacquire_time = seed[0] - reacquire_interval_sec
    local_radius = max(local_search_sec, 1.1 * reference_track.hop_sec)

    for mixture_time_value in local_times:
        mixture_time = float(mixture_time_value)
        feature = mixture_track.feature_near(mixture_time)
        if feature is None:
            continue
        predicted_reference = mixture_time - tracked_offset
        local = index.best_near(
            "reference",
            feature,
            predicted_reference,
            local_radius,
        )
        if local is not None and local.score >= local_threshold:
            grid_offset = mixture_time - local.time_sec
            # Within one hop the index cannot tell whether the alignment moved,
            # so keep the refined offset rather than letting the anchor snap to
            # whichever grid point happened to win.
            if abs(grid_offset - tracked_offset) > 1.1 * hop_sec:
                refined = refine(mixture_time, grid_offset)
                tracked_offset = grid_offset if refined is None else refined
            anchors.append((mixture_time, tracked_offset, local.score))
            last_match_time = mixture_time
            continue

        should_reacquire = (
            mixture_time - last_match_time >= reacquire_after_sec
            and mixture_time - last_reacquire_time >= reacquire_interval_sec
        )
        if not should_reacquire:
            continue
        last_reacquire_time = mixture_time
        candidate = _sequence_global_match(
            mixture_track,
            reference_track,
            index,
            mixture_time,
            query_sec,
        )
        if candidate is not None and candidate.score >= global_threshold:
            grid_offset = mixture_time - candidate.time_sec
            refined = refine(mixture_time, grid_offset)
            if refined is None and refinable:
                # A reacquire claims the alignment jumped, with no prior to
                # constrain it. When the waveform can be read and disagrees,
                # the claim is wrong; accepting it would drag the rest of the
                # track onto a false offset.
                continue
            tracked_offset = grid_offset if refined is None else refined
            anchors.append((mixture_time, tracked_offset, candidate.score))
            last_match_time = mixture_time

    segments = _segments_from_anchors(
        anchors,
        local_step_sec=local_step_sec,
        query_sec=query_sec,
    )
    _print_alignment_segments(segments)
    return segments


def discover_alignment_segments(
    mixture_path: str,
    reference_path: str,
    align_sr: int = 1_000,
    global_step_sec: float = 60.0,
    query_sec: float = 8.0,
    local_step_sec: float = 5.0,
    local_search_sec: float = 0.25,
    min_score: float = 0.18,
    reacquire_after_sec: float = 10.0,
    reacquire_interval_sec: float = 15.0,
    workers: int = 4,
    mixture_start_sec: float = 0.0,
    mixture_duration_sec: float | None = None,
    reference_duration_sec: float | None = None,
    max_in_memory_sec: float = 4.0 * 60.0 * 60.0,
) -> list[AlignmentSegment]:
    """Find matching regions and recover after pauses, seeks, and replays."""
    _validate_alignment_options(
        align_sr=align_sr,
        global_step_sec=global_step_sec,
        query_sec=query_sec,
        local_step_sec=local_step_sec,
        local_search_sec=local_search_sec,
        min_score=min_score,
        reacquire_after_sec=reacquire_after_sec,
        reacquire_interval_sec=reacquire_interval_sec,
        workers=workers,
        mixture_start_sec=mixture_start_sec,
        mixture_duration_sec=mixture_duration_sec,
    )
    if not np.isfinite(max_in_memory_sec) or max_in_memory_sec < 0.0:
        raise ValueError("max_in_memory_sec must be non-negative and finite.")
    if reference_duration_sec is not None and (
        not np.isfinite(reference_duration_sec) or reference_duration_sec <= 0.0
    ):
        raise ValueError("reference_duration_sec must be positive and finite.")
    if reference_duration_sec is None:
        try:
            reference_duration_sec = _media_duration(reference_path)
        except FileNotFoundError:
            # Synthetic/custom decoders may not refer to filesystem media.
            # The caller can pass a duration to select the indexed strategy.
            pass
    use_index = (
        mixture_duration_sec is not None and mixture_duration_sec > max_in_memory_sec
    ) or (
        reference_duration_sec is not None
        and reference_duration_sec > max_in_memory_sec
    )
    if use_index:
        return _discover_indexed_alignment_segments(
            mixture_path,
            reference_path,
            align_sr=align_sr,
            global_step_sec=global_step_sec,
            query_sec=query_sec,
            local_step_sec=local_step_sec,
            local_search_sec=local_search_sec,
            min_score=min_score,
            reacquire_after_sec=reacquire_after_sec,
            reacquire_interval_sec=reacquire_interval_sec,
            workers=workers,
            mixture_start_sec=mixture_start_sec,
            mixture_duration_sec=mixture_duration_sec,
        )
    return _discover_full_alignment_segments(
        mixture_path,
        reference_path,
        align_sr=align_sr,
        global_step_sec=global_step_sec,
        query_sec=query_sec,
        local_step_sec=local_step_sec,
        local_search_sec=local_search_sec,
        min_score=min_score,
        reacquire_after_sec=reacquire_after_sec,
        reacquire_interval_sec=reacquire_interval_sec,
        workers=workers,
        mixture_start_sec=mixture_start_sec,
        mixture_duration_sec=mixture_duration_sec,
    )


def _normalized_match(search: np.ndarray, query: np.ndarray) -> tuple[int, float]:
    """Return the best valid query start and normalized correlation."""
    query64 = query.astype(np.float64, copy=False)
    search64 = search.astype(np.float64, copy=False)
    query64 = query64 - np.mean(query64)
    n = len(query64)
    if n == 0 or len(search64) < n:
        raise ValueError("Reference search window is shorter than the query.")

    numerator = scipy.signal.correlate(search64, query64, mode="valid", method="fft")
    cumulative = np.concatenate(([0.0], np.cumsum(search64)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(search64 * search64)))
    rolling_sum = cumulative[n:] - cumulative[:-n]
    rolling_energy = (
        cumulative_sq[n:] - cumulative_sq[:-n] - rolling_sum * rolling_sum / n
    )
    rolling_energy = np.maximum(rolling_energy, 1e-20)
    denominator = np.sqrt(rolling_energy * np.sum(query64 * query64))
    correlation = numerator / np.maximum(denominator, 1e-20)
    best = int(np.argmax(correlation))
    return best, float(correlation[best])


def _best_scaled_match(
    search: np.ndarray,
    query: np.ndarray,
) -> tuple[int, float]:
    """Match normally, with a bounded speed search when correlation is weak."""
    best_index, best_score = _normalized_match(search, query)
    if best_score >= 0.30:
        return best_index, best_score
    for scale in (0.995, 1.005, 0.99, 1.01):
        frames = int(round(len(query) * scale))
        if frames <= 0 or frames > len(search):
            continue
        scaled = scipy.signal.resample(query, frames)
        index, score = _normalized_match(search, scaled)
        if score > best_score:
            best_index, best_score = index, score
    return best_index, best_score


class _CoarseCandidate(NamedTuple):
    """One Mid/Side coarse alignment hypothesis for a chunk."""

    adjusted_score: float
    score: float
    index: int
    query_feature: np.ndarray
    search_feature: np.ndarray
    query_low: np.ndarray
    search_low: np.ndarray


class _ReferenceAlignment(NamedTuple):
    """Chunk-local alignment result, including its coverage verdict."""

    reference: np.ndarray | None
    score: float
    start: int
    covered: bool


_PROBE_RADIUS_LADDER = (1.0, 4.0, 16.0)
_PROBE_ACCEPT_SCORE = 0.30
_PROBE_WIDEN_MARGIN = 0.05


def _prior_constrained_match(
    search: np.ndarray,
    query: np.ndarray,
    predicted_start: float | None,
    radius: float,
) -> tuple[int, float]:
    """Match near a predicted position, widening only when the prior fails.

    The caller already knows where the query should land to within ``radius``.
    An unconstrained ``argmax`` over the complete search window turns any
    repeated passage into a confident wrong answer, so trust the prior first
    and only pay for a wider window when the constrained score is weak.
    """
    if predicted_start is None:
        return _best_scaled_match(search, query)
    best: tuple[int, float] | None = None
    for scale in _PROBE_RADIUS_LADDER:
        window_radius = radius * scale
        start = int(max(0.0, np.floor(predicted_start - window_radius)))
        end = int(
            min(
                float(len(search)),
                np.ceil(predicted_start + len(query) + window_radius),
            )
        )
        if end - start < len(query):
            continue
        index, score = _best_scaled_match(search[start:end], query)
        if best is None or score > best[1] + _PROBE_WIDEN_MARGIN:
            best = (start + index, score)
        if best[1] >= _PROBE_ACCEPT_SCORE or (start == 0 and end == len(search)):
            break
    if best is None:
        return _best_scaled_match(search, query)
    return best


def _fractional_match_index(
    search: np.ndarray,
    query: np.ndarray,
    index: int,
) -> float:
    """Refine an integer correlation peak with a three-point parabola."""
    if index <= 0 or index + len(query) >= len(search):
        return float(index)
    query64 = query.astype(np.float64, copy=False)
    query64 = query64 - np.mean(query64)
    query_energy = float(np.dot(query64, query64))
    scores: list[float] = []
    for candidate in (index - 1, index, index + 1):
        window = search[candidate : candidate + len(query)].astype(
            np.float64, copy=False
        )
        window = window - np.mean(window)
        scores.append(
            float(np.dot(window, query64))
            / np.sqrt((float(np.dot(window, window)) + 1e-20) * (query_energy + 1e-20))
        )
    left, center, right = scores
    curvature = left - 2.0 * center + right
    if curvature >= -1e-12:
        return float(index)
    delta = 0.5 * (left - right) / curvature
    return float(index) + float(np.clip(delta, -0.5, 0.5))


def _align_reference(
    mixture: np.ndarray,
    reference_search: np.ndarray,
    sr: int,
    diagnostics: dict[str, float] | None = None,
    adaptive_time_warp: bool = True,
    predicted_start: float | None = None,
    search_radius_sec: float = 0.25,
) -> _ReferenceAlignment:
    """Align and locally time-warp a reference to one mixture chunk.

    ``predicted_start`` is where ``mixture[0]`` is expected inside
    ``reference_search``, in samples. Pass ``None`` only when no prior exists;
    the coarse search is then unconstrained and can lock onto a repeated
    passage elsewhere in the window.
    """
    if not np.isfinite(search_radius_sec) or search_radius_sec <= 0.0:
        raise ValueError("search_radius_sec must be positive and finite.")
    down = max(1, sr // 4_000)
    low_sr = sr / down
    high = min(1_800.0, 0.45 * low_sr)
    sos = scipy.signal.butter(
        4, [80.0, high], btype="bandpass", fs=low_sr, output="sos"
    )

    # Mid is often contaminated by the centered foreground voice, while Side
    # can be weak in a nearly mono reference. Evaluate both consistently and
    # use whichever provides the stronger match.
    feature_pairs = [
        (
            0.5 * (mixture[:, 0] + mixture[:, 1]),
            0.5 * (reference_search[:, 0] + reference_search[:, 1]),
        ),
        (
            0.5 * (mixture[:, 0] - mixture[:, 1]),
            0.5 * (reference_search[:, 0] - reference_search[:, 1]),
        ),
    ]
    prior_low = None if predicted_start is None else predicted_start / down
    radius_low = max(search_radius_sec * low_sr, 4.0)

    def prior_penalty(coarse_index: int) -> float:
        """Discount a candidate that disagrees with the caller's prediction."""
        if prior_low is None:
            return 0.0
        deviation = abs(coarse_index - prior_low) / radius_low
        return 0.10 * float(np.clip(deviation - 1.0, 0.0, 4.0))

    candidates: list[_CoarseCandidate] = []
    for query_feature, search_feature in feature_pairs:
        query_low = scipy.signal.resample_poly(query_feature, 1, down)
        search_low = scipy.signal.resample_poly(search_feature, 1, down)
        query_low = scipy.signal.sosfiltfilt(sos, query_low)
        search_low = scipy.signal.sosfiltfilt(sos, search_low)
        # A short center probe stays correlated even when the complete chunk
        # contains a small speed drift. Its hit gives a base offset for local
        # anchors across the chunk.
        probe_frames = min(len(query_low), max(int(round(4.0 * low_sr)), 256))
        probe_start = max(0, (len(query_low) - probe_frames) // 2)
        coarse_hit, coarse_score = _prior_constrained_match(
            search_low,
            query_low[probe_start : probe_start + probe_frames],
            None if prior_low is None else prior_low + probe_start,
            radius_low,
        )
        coarse_index = coarse_hit - probe_start
        candidates.append(
            _CoarseCandidate(
                adjusted_score=coarse_score - prior_penalty(coarse_index),
                score=coarse_score,
                index=coarse_index,
                query_feature=query_feature,
                search_feature=search_feature,
                query_low=query_low,
                search_low=search_low,
            )
        )

    best_candidate = max(candidates, key=lambda item: item.adjusted_score)
    coarse_score = best_candidate.score
    coarse_index = best_candidate.index
    query_feature = best_candidate.query_feature
    search_feature = best_candidate.search_feature
    query_low = best_candidate.query_low
    search_low = best_candidate.search_low
    if diagnostics is not None and prior_low is not None:
        diagnostics["coarse_prior_error_sec"] = (coarse_index - prior_low) / low_sr

    def collect_native_anchors(
        anchor_sec: float,
        minimum_low_frames: int,
    ) -> list[tuple[float, float, float]]:
        anchor_frames = min(
            len(query_low),
            max(int(round(anchor_sec * low_sr)), minimum_low_frames),
        )
        anchor_starts = np.arange(
            0,
            max(1, len(query_low) - anchor_frames + 1),
            anchor_frames,
        )
        anchor_starts = np.unique(
            np.append(anchor_starts, max(0, len(query_low) - anchor_frames))
        )
        local_radius = max(int(round(0.20 * low_sr)), 8)
        coarse_anchors: list[tuple[int, int, float]] = []
        for query_start in anchor_starts:
            predicted = coarse_index + int(query_start)
            search_start = max(0, predicted - local_radius)
            search_end = min(
                len(search_low),
                predicted + anchor_frames + local_radius,
            )
            if search_end - search_start < anchor_frames:
                continue
            hit, score = _best_scaled_match(
                search_low[search_start:search_end],
                query_low[query_start : query_start + anchor_frames],
            )
            if score >= max(0.12, 0.45 * coarse_score):
                coarse_anchors.append((int(query_start), search_start + hit, score))

        native_anchors: list[tuple[float, float, float]] = []
        fine_radius = max(2 * down, 16)
        native_anchor_frames = min(len(mixture), anchor_frames * down)
        for query_start_low, reference_start_low, score in coarse_anchors:
            query_start = min(
                query_start_low * down,
                max(0, len(mixture) - native_anchor_frames),
            )
            predicted = reference_start_low * down
            search_start = max(0, predicted - fine_radius)
            search_end = min(
                len(search_feature),
                predicted + native_anchor_frames + fine_radius,
            )
            if search_end - search_start < native_anchor_frames:
                continue
            query_window = query_feature[
                query_start : query_start + native_anchor_frames
            ]
            search_window = search_feature[search_start:search_end]
            fine_hit, fine_score = _normalized_match(search_window, query_window)
            fractional_hit = _fractional_match_index(
                search_window, query_window, fine_hit
            )
            native_anchors.append(
                (
                    query_start + 0.5 * native_anchor_frames,
                    search_start + fractional_hit - query_start,
                    max(score, fine_score),
                )
            )
        return native_anchors

    def positions_from_anchors(
        anchors: list[tuple[float, float, float]],
        median_size: int,
    ) -> tuple[np.ndarray, float]:
        if not anchors:
            start = int(round(coarse_index * down))
            return start + np.arange(len(mixture)), coarse_score
        anchor_positions = np.array([item[0] for item in anchors], dtype=np.float64)
        anchor_offsets = np.array([item[1] for item in anchors], dtype=np.float64)
        if len(anchor_offsets) >= 3:
            anchor_offsets = scipy.ndimage.median_filter(
                anchor_offsets, size=median_size, mode="nearest"
            )
        sample_positions = np.arange(len(mixture), dtype=np.float64)
        local_offsets = np.interp(
            sample_positions,
            anchor_positions,
            anchor_offsets,
            left=anchor_offsets[0],
            right=anchor_offsets[-1],
        )
        if len(anchor_offsets) >= 2:
            left_slope = (anchor_offsets[1] - anchor_offsets[0]) / (
                anchor_positions[1] - anchor_positions[0]
            )
            right_slope = (anchor_offsets[-1] - anchor_offsets[-2]) / (
                anchor_positions[-1] - anchor_positions[-2]
            )
            before = sample_positions < anchor_positions[0]
            after = sample_positions > anchor_positions[-1]
            local_offsets[before] = anchor_offsets[0] + left_slope * (
                sample_positions[before] - anchor_positions[0]
            )
            local_offsets[after] = anchor_offsets[-1] + right_slope * (
                sample_positions[after] - anchor_positions[-1]
            )
        return (
            sample_positions + local_offsets,
            float(np.median([item[2] for item in anchors])),
        )

    def has_warp_evidence(
        anchors: list[tuple[float, float, float]],
    ) -> bool:
        if len(anchors) < 5:
            return False
        positions = np.array([item[0] for item in anchors])
        offsets = np.array([item[1] for item in anchors])
        offsets = scipy.ndimage.median_filter(offsets, size=5, mode="nearest")
        slope, intercept = np.polyfit(positions, offsets, 1)
        residual = offsets - (slope * positions + intercept)
        drift = abs(slope) * np.ptp(positions)
        jitter = np.percentile(residual, 90) - np.percentile(residual, 10)
        threshold = max(4.0 * down, 4.0)
        if diagnostics is not None:
            diagnostics["long_anchor_drift_samples"] = float(drift)
            diagnostics["long_anchor_jitter_samples"] = float(jitter)
        return drift >= threshold or jitter >= 1.5 * threshold

    def validation_score(source_positions: np.ndarray) -> tuple[float, float]:
        window = max(128, int(round(0.125 * sr)))
        hop = max(window, int(round(0.25 * sr)))
        starts = np.arange(window // 2, len(mixture) - window, hop)
        scores: list[float] = []
        search_axis = np.arange(len(search_feature), dtype=np.float64)
        for start in starts:
            end = int(start + window)
            aligned_window = np.interp(
                source_positions[start:end],
                search_axis,
                search_feature,
            )
            query_window = query_feature[start:end]
            query_window = query_window - np.mean(query_window)
            aligned_window = aligned_window - np.mean(aligned_window)
            denominator = np.sqrt(
                np.dot(query_window, query_window)
                * np.dot(aligned_window, aligned_window)
                + 1e-20
            )
            scores.append(float(np.dot(query_window, aligned_window) / denominator))
        if not scores:
            return 0.0, 0.0
        return float(np.median(scores)), float(np.percentile(scores, 10))

    long_anchors = collect_native_anchors(0.25, 256)
    source_positions, alignment_score = positions_from_anchors(
        long_anchors, median_size=3
    )
    if adaptive_time_warp and has_warp_evidence(long_anchors):
        short_anchors = collect_native_anchors(0.016, 32)
        short_positions, short_score = positions_from_anchors(
            short_anchors, median_size=5
        )
        source_rate = np.diff(short_positions)
        monotonic = (
            len(source_rate) > 0
            and np.percentile(source_rate, 1) >= 0.95
            and np.percentile(source_rate, 99) <= 1.05
        )
        long_validation = validation_score(source_positions)
        short_validation = validation_score(short_positions)
        in_bounds = short_positions[0] >= -1.0 and short_positions[-1] <= len(
            reference_search
        )
        accept_short = (
            monotonic
            and in_bounds
            and len(short_anchors) >= len(long_anchors)
            and short_validation[0] >= 0.80
            and short_validation[0] >= long_validation[0] + 0.001
            and short_validation[1] >= long_validation[1] - 0.03
        )
        if diagnostics is not None:
            diagnostics["short_warp_considered"] = 1.0
            diagnostics["short_warp_accepted"] = float(accept_short)
            diagnostics["long_validation_score"] = long_validation[0]
            diagnostics["short_validation_score"] = short_validation[0]
        if accept_short:
            source_positions = short_positions
            alignment_score = short_score

    aligned_start = int(round(source_positions[0]))
    # Running off either end of the decoded window is a recoverable condition,
    # not a fatal one: the caller can re-decode a wider window using the
    # reported deficit, or fall back to pass-through for this chunk alone.
    deficit_start = max(0.0, -1.0 - float(source_positions[0]))
    deficit_end = max(0.0, float(source_positions[-1]) - len(reference_search))
    if diagnostics is not None:
        diagnostics["coverage_deficit_start_sec"] = deficit_start / sr
        diagnostics["coverage_deficit_end_sec"] = deficit_end / sr
    if deficit_start > 0.0 or deficit_end > 0.0:
        return _ReferenceAlignment(None, alignment_score, aligned_start, False)

    source_positions = np.clip(source_positions, 0.0, len(reference_search) - 1.0)
    sample_axis = np.arange(len(reference_search), dtype=np.float64)
    aligned = np.column_stack(
        [
            np.interp(source_positions, sample_axis, reference_search[:, ch])
            for ch in range(reference_search.shape[1])
        ]
    ).astype(np.float32)
    return _ReferenceAlignment(aligned, alignment_score, aligned_start, True)
