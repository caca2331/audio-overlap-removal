"""Low-memory reference cancellation for real mono/stereo recordings.

Stereo inputs retain Mid/Side information through cancellation, while mono
inputs use a guarded Mid-only path. Output channel count follows the original
mixture.

The implementation intentionally works in short independently decoded chunks:
peak memory scales with ``chunk_sec * workers`` instead of total media
duration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal
import soundfile as sf

DEFAULT_SR = 48_000
MIN_SAMPLE_RATE = 1_000
MIN_ALIGNMENT_SAMPLE_RATE = 200
_OUTPUT_FORMATS = {
    ".flac": ("FLAC", "PCM_24"),
    ".wav": ("WAV", "PCM_24"),
}


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
class _ProcessingChunk:
    position: float
    core_duration: float
    context_start: float
    context_duration: float
    active: AlignmentSegment | None


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


def _run_media_command(
    command: list[str], *, text: bool = False
) -> subprocess.CompletedProcess:
    """Run FFmpeg/FFprobe and preserve the useful diagnostic on failure."""
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{command[0]} was not found. Install FFmpeg and ensure both "
            "ffmpeg and ffprobe are available on PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = (stderr or "").strip()
        message = f"{command[0]} failed with exit code {error.returncode}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from error


def _probe_audio(path: str) -> dict:
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,duration:format=duration",
        "-of",
        "json",
        str(media),
    ]
    result = _run_media_command(command, text=True)
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"ffprobe returned invalid metadata for {path!r}."
        ) from error
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream was found in {path!r}.")
    return payload


def _media_duration(path: str) -> float:
    payload = _probe_audio(path)
    candidates = [
        (payload.get("format") or {}).get("duration"),
        (payload.get("streams") or [{}])[0].get("duration"),
    ]
    for candidate in candidates:
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(duration) and duration > 0.0:
            return duration
    raise ValueError(
        f"Could not determine the duration of {path!r}; pass --duration explicitly."
    )


def _audio_channel_count(path: str) -> int:
    payload = _probe_audio(path)
    try:
        channels = int(payload["streams"][0]["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not determine the audio channel count of {path!r}."
        ) from error
    if channels < 1:
        raise ValueError(f"Invalid audio channel count in {path!r}: {channels}.")
    return channels


def _processing_channel_count(native_channels: int) -> int:
    """Keep mono intact and let FFmpeg downmix every wider layout to stereo."""
    return 1 if native_channels == 1 else 2


def _output_settings(path: Path) -> tuple[str, str]:
    try:
        return _OUTPUT_FORMATS[path.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output extension {path.suffix or '<none>'!r}; "
            f"use one of: {supported}."
        ) from error


def _paths_refer_to_same_file(first: str | Path, second: str | Path) -> bool:
    left = Path(first)
    right = Path(second)
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return left.resolve(strict=False) == right.resolve(strict=False)


@contextmanager
def _atomic_soundfile(
    output: Path,
    *,
    samplerate: int,
    channels: int,
    format: str,
    subtype: str,
):
    """Replace the destination only after the temporary output closes cleanly."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".part",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with sf.SoundFile(
            temporary,
            mode="w",
            samplerate=samplerate,
            channels=channels,
            format=format,
            subtype=subtype,
        ) as sink:
            yield sink
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_ordered_map(function, items, workers: int):
    """Run at most ``workers`` jobs concurrently and yield in input order."""
    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="reference-cancel",
    ) as executor:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def _decode_mono_low(
    path: str,
    sr: int,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> np.ndarray:
    if sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(
            f"decode sample rate must be at least {MIN_ALIGNMENT_SAMPLE_RATE} Hz."
        )
    if start_sec < 0.0:
        raise ValueError("decode start must be non-negative.")
    if duration_sec is not None and duration_sec <= 0.0:
        raise ValueError("decode duration must be positive.")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
    ]
    if start_sec > 0.0:
        command.extend(["-ss", f"{start_sec:.9f}"])
    command.extend(
        [
            "-i",
            path,
            "-map",
            "0:a:0",
        ]
    )
    if duration_sec is not None:
        command.extend(["-t", f"{duration_sec:.9f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    result = _run_media_command(command)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


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
    workers: int = 1,
    mixture_start_sec: float = 0.0,
    mixture_duration_sec: float | None = None,
) -> list[AlignmentSegment]:
    """Find matching regions and recover after pauses, seeks, and replays."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if align_sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(f"align_sr must be at least {MIN_ALIGNMENT_SAMPLE_RATE}.")
    if global_step_sec <= 0.0 or query_sec <= 0.0 or local_step_sec <= 0.0:
        raise ValueError("alignment step and query durations must be positive.")
    if local_search_sec < 0.0:
        raise ValueError("local_search_sec must be non-negative.")
    if reacquire_after_sec < 0.0 or reacquire_interval_sec <= 0.0:
        raise ValueError("reacquisition timings are invalid.")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1.")
    if mixture_start_sec < 0.0:
        raise ValueError("mixture_start_sec must be non-negative.")
    if mixture_duration_sec is not None and mixture_duration_sec <= 0.0:
        raise ValueError("mixture_duration_sec must be positive.")
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

    if not anchors:
        return []

    # First split no-match gaps, then split stable regions at real timestamp
    # discontinuities. Median filtering prevents one bad anchor from creating a
    # false segment boundary.
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
            part_offsets = np.array([item[1] for item in part])
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
                )
            )

    for segment in segments:
        print(
            f"  match C={segment.mixture_start:.1f}-"
            f"{segment.mixture_end:.1f}s "
            f"offset={segment.offset_sec:.3f}s "
            f"speed={(1.0 - segment.offset_slope):.6f}x "
            f"median-score={segment.median_score:.3f}"
        )
    return segments


def _decode_stereo(
    path: str,
    start_sec: float,
    duration_sec: float,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    if sr < MIN_SAMPLE_RATE:
        raise ValueError(f"sample rate must be at least {MIN_SAMPLE_RATE} Hz.")
    if start_sec < 0.0:
        raise ValueError("decode start must be non-negative.")
    if duration_sec <= 0.0:
        raise ValueError("decode duration must be positive.")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.9f}",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-t",
        f"{duration_sec:.9f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = _run_media_command(command)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(audio) % 2:
        audio = audio[:-1]
    return audio.reshape(-1, 2)


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
) -> tuple[np.ndarray, float, int]:
    """Align and locally time-warp a reference to one mixture chunk."""
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
    candidates: list[
        tuple[
            float,
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
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
        coarse_hit, coarse_score = _best_scaled_match(
            search_low,
            query_low[probe_start : probe_start + probe_frames],
        )
        coarse_index = coarse_hit - probe_start
        candidates.append(
            (
                coarse_score,
                coarse_index,
                query_feature,
                search_feature,
                query_low,
                search_low,
            )
        )

    (
        coarse_score,
        coarse_index,
        query_feature,
        search_feature,
        query_low,
        search_low,
    ) = max(candidates, key=lambda item: item[0])

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
    if source_positions[0] < -1.0 or source_positions[-1] > len(reference_search):
        raise ValueError("Aligned reference does not cover the complete chunk.")
    source_positions = np.clip(source_positions, 0.0, len(reference_search) - 1.0)
    sample_axis = np.arange(len(reference_search), dtype=np.float64)
    aligned = np.column_stack(
        [
            np.interp(source_positions, sample_axis, reference_search[:, ch])
            for ch in range(reference_search.shape[1])
        ]
    ).astype(np.float32)
    return aligned, alignment_score, aligned_start


def _estimate_gain_envelope(
    mixture_side: np.ndarray,
    reference_side: np.ndarray,
    sr: int,
    window_sec: float = 0.25,
    hop_sec: float = 0.05,
    min_correlation: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    window = max(32, int(round(window_sec * sr)))
    hop = max(16, int(round(hop_sec * sr)))
    centers = np.arange(0, len(mixture_side), hop)
    gains = np.zeros(len(centers), dtype=np.float64)
    correlations = np.zeros(len(centers), dtype=np.float64)
    relative_levels = np.zeros(len(centers), dtype=np.float64)

    for index, center in enumerate(centers):
        start = max(0, center - window // 2)
        end = min(len(mixture_side), center + window // 2)
        mix = mixture_side[start:end].astype(np.float64, copy=False)
        ref = reference_side[start:end].astype(np.float64, copy=False)
        cross = float(np.dot(mix, ref))
        ref_energy = float(np.dot(ref, ref))
        mix_energy = float(np.dot(mix, mix))
        gains[index] = cross / (ref_energy + 1e-20)
        correlations[index] = cross / np.sqrt(
            (mix_energy + 1e-20) * (ref_energy + 1e-20)
        )
        relative_levels[index] = np.sqrt((mix_energy + 1e-20) / (ref_energy + 1e-20))

    gains = np.clip(gains, 0.0, 1.5)
    confident = correlations >= min_correlation
    typical_gain = (
        float(np.median(gains[confident]))
        if np.any(confident)
        else float(np.median(gains[gains > 0.0]))
        if np.any(gains > 0.0)
        else 0.0
    )
    # Do not interpolate straight through a real playback pause. A low-
    # correlation window with almost no mixture-side energy relative to the
    # active reference is evidence that the removable media is muted, not that
    # its gain is merely temporarily unobservable.
    paused = (
        ~confident
        & (typical_gain > 0.0)
        & (relative_levels <= max(0.03, 0.20 * typical_gain))
    )
    if np.count_nonzero(confident) >= 2:
        positions = np.arange(len(gains))
        gains = np.interp(positions, positions[confident], gains[confident])
    elif np.count_nonzero(confident) == 1:
        gains.fill(gains[confident][0])
    else:
        gains.fill(0.0)

    gains[paused] = 0.0
    if len(gains) >= 3:
        gains = scipy.signal.medfilt(gains, kernel_size=3)
        # A short soft edge avoids a click while preserving long zero-gain
        # pause interiors.
        gains = scipy.ndimage.gaussian_filter1d(gains, sigma=0.75, mode="nearest")
    sample_gain = np.interp(np.arange(len(mixture_side)), centers, gains)
    return sample_gain.astype(np.float32), correlations


def _stft(audio: np.ndarray, sr: int) -> np.ndarray:
    n_fft = 2_048
    hop = 512
    return scipy.signal.stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )[2]


def _istft(spectrum: np.ndarray, length: int, sr: int) -> np.ndarray:
    n_fft = 2_048
    hop = 512
    audio = scipy.signal.istft(
        spectrum,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )[1]
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return audio[:length].astype(np.float32)


def _smooth_complex(spectrum: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(
        spectrum.real, sigma
    ) + 1j * scipy.ndimage.gaussian_filter(spectrum.imag, sigma)


def _smooth_power(spectrum: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(np.abs(spectrum) ** 2, sigma)


def _estimate_complex_transfer(
    mixture: np.ndarray,
    reference: np.ndarray,
    sigma: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    cross = _smooth_complex(mixture * np.conj(reference), sigma)
    reference_power = scipy.ndimage.gaussian_filter(np.abs(reference) ** 2, sigma)
    mixture_power = scipy.ndimage.gaussian_filter(np.abs(mixture) ** 2, sigma)
    regularizer = 0.01 * np.median(reference_power, axis=1, keepdims=True)
    transfer = cross / (reference_power + regularizer + 1e-14)
    transfer_magnitude = np.abs(transfer)
    transfer *= np.minimum(1.0, 1.5 / (transfer_magnitude + 1e-12))
    coherence = np.clip(
        np.abs(cross) ** 2 / (reference_power * mixture_power + 1e-14),
        0.0,
        1.0,
    )
    return transfer, coherence


def _complex_reference_cancel(
    mixture_mid: np.ndarray,
    mixture_side: np.ndarray,
    reference_mid: np.ndarray,
    reference_side: np.ndarray,
    scalar_gain: np.ndarray,
    cleanup_strength: float,
    center_strength: float,
    sr: int,
    center_cleanup_strength: float = 0.0,
    silence_cleanup_strength: float = 0.0,
    cleanup_floor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, float]:
    mixture_mid_stft = _stft(mixture_mid, sr)
    mixture_side_stft = _stft(mixture_side, sr)
    reference_mid_stft = _stft(reference_mid, sr)
    reference_side_stft = _stft(reference_side, sr)

    # Mid contains the foreground, so its direct estimate is trustworthy only
    # in coherent bins. Side is mostly foreground-free and provides the safer
    # prior elsewhere.
    mid_transfer, mid_coherence = _estimate_complex_transfer(
        mixture_mid_stft, reference_mid_stft, sigma=(1.0, 10.0)
    )
    side_transfer, _ = _estimate_complex_transfer(
        mixture_side_stft, reference_side_stft, sigma=(1.0, 5.0)
    )
    scalar_side_error = np.sqrt(
        np.sum(
            np.square(
                mixture_side - scalar_gain * reference_side,
                dtype=np.float64,
            )
        )
        / (np.sum(np.square(mixture_side, dtype=np.float64)) + 1e-20)
    )
    if scalar_side_error < 0.10:
        # An exact or nearly exact digital mix needs no foreground-contaminated
        # Mid estimate. The time-domain envelope is more accurate than a
        # smoothed STFT transfer for this special case.
        output = mixture_mid - scalar_gain * reference_mid
        side_residual = mixture_side - scalar_gain * reference_side
        return (
            output.astype(np.float32),
            side_residual.astype(np.float32),
            1.0,
        )
    else:
        coherence_weight = np.clip((mid_coherence - 0.3) / 0.4, 0.0, 1.0)
        coherence_weight = scipy.ndimage.gaussian_filter(
            coherence_weight, sigma=(0.5, 1.0)
        )
        direct_mid_prediction = mid_transfer * reference_mid_stft
        side_mid_prediction = side_transfer * reference_mid_stft
        scalar_mid_prediction = _stft(scalar_gain * reference_mid, sr)

        # A centered reference component (dialogue or singing) can have almost
        # no Side energy. In those bins Side cannot identify a complex
        # transfer, so fall back to the foreground-independent scalar envelope
        # instead of silently preserving the centered media voice.
        support_sigma = (1.5, 3.0)
        mid_reference_power = scipy.ndimage.gaussian_filter(
            np.abs(reference_mid_stft) ** 2, support_sigma
        )
        side_reference_power = scipy.ndimage.gaussian_filter(
            np.abs(reference_side_stft) ** 2, support_sigma
        )
        side_support = side_reference_power / (
            mid_reference_power + side_reference_power + 1e-14
        )
        side_weight = np.clip(side_support / 0.10, 0.0, 1.0)
        safe_mid_prediction = (
            side_weight * side_mid_prediction
            + (1.0 - side_weight) * scalar_mid_prediction
        )
        center_evidence = np.clip((0.10 - side_support) / 0.10, 0.0, 1.0)
        reference_activity = np.clip(
            mid_reference_power
            / (3.0 * np.median(mid_reference_power, axis=1, keepdims=True) + 1e-14),
            0.0,
            1.0,
        )
        coherence_weight = np.maximum(
            coherence_weight,
            center_strength * center_evidence * reference_activity,
        )
        predicted_mid_stft = (
            coherence_weight * direct_mid_prediction
            + (1.0 - coherence_weight) * safe_mid_prediction
        )

    predicted_side_stft = side_transfer * reference_side_stft
    mid_residual_stft = mixture_mid_stft - predicted_mid_stft
    side_residual_stft = mixture_side_stft - predicted_side_stft
    raw_residual_power = float(np.sum(np.abs(mid_residual_stft) ** 2))

    if cleanup_strength > 0:
        sigma = (1.5, 3.0)
        side_artifact_power = _smooth_power(side_residual_stft, sigma)
        predicted_mid_power = _smooth_power(predicted_mid_stft, sigma)
        predicted_side_power = _smooth_power(predicted_side_stft, sigma)
        mid_side_ratio = np.clip(
            predicted_mid_power / (predicted_side_power + 1e-14),
            0.1,
            12.0,
        )
        estimated_artifact_power = side_artifact_power * mid_side_ratio
        mid_power = _smooth_power(mid_residual_stft, sigma)
        cleanup_gain = np.sqrt(
            np.clip(
                1.0 - cleanup_strength * estimated_artifact_power / (mid_power + 1e-14),
                cleanup_floor * cleanup_floor,
                1.0,
            )
        )
        cleanup_gain = scipy.ndimage.gaussian_filter(cleanup_gain, sigma=(0.7, 1.0))
        mid_residual_stft *= cleanup_gain

    if center_cleanup_strength > 0:
        # Phase/coloration errors can leave a recognizable "ghost" even after
        # complex subtraction. Side-derived cleanup cannot see a centered
        # reference voice, so attenuate only bins where the reference itself
        # is active and center-dominant. This intentionally trades a small
        # amount of double-talk transparency for less media-vocal residue.
        sigma = (1.5, 3.0)
        residual_power = _smooth_power(mid_residual_stft, sigma)
        predicted_power = _smooth_power(predicted_mid_stft, sigma)
        center_aggression = np.clip(center_cleanup_strength - 1.0, 0.0, 1.0)
        activity_scale = 1.5 - 0.75 * center_aggression
        center_activity = np.clip(
            mid_reference_power
            / (
                activity_scale * np.median(mid_reference_power, axis=1, keepdims=True)
                + 1e-14
            ),
            0.0,
            1.0,
        )
        center_support_limit = 0.25 + 0.25 * center_aggression
        center_dominance = np.clip(
            (center_support_limit - side_support) / center_support_limit,
            0.0,
            1.0,
        )
        media_dominance = np.power(
            predicted_power / (predicted_power + residual_power + 1e-14),
            1.0 / (1.0 + center_aggression),
        )
        center_mask = center_activity * center_dominance * media_dominance
        center_gain = np.sqrt(
            np.clip(
                1.0 - center_cleanup_strength * center_mask,
                0.10,
                1.0,
            )
        )
        center_gain = scipy.ndimage.gaussian_filter(center_gain, sigma=(0.7, 1.0))
        mid_residual_stft *= center_gain

    if silence_cleanup_strength > 0:
        # Aggregate evidence across frequency before expanding the mask.
        # Unrelated foreground (host speech or host-side BGM) raises the
        # unexplained-energy ratio and protects the entire nearby time span.
        # When the mixture is well explained by the removable reference, the
        # remaining phase-incoherent ghost can be suppressed more aggressively.
        sigma = (1.5, 3.0)
        residual_power = _smooth_power(mid_residual_stft, sigma)
        predicted_power = _smooth_power(predicted_mid_stft, sigma)
        frequencies = np.fft.rfftfreq(2_048, d=1.0 / sr)
        foreground_band = (frequencies >= 100.0) & (
            frequencies <= min(8_000.0, 0.48 * sr)
        )
        unexplained_frame = np.sum(residual_power[foreground_band], axis=0)
        explained_frame = np.sum(predicted_power[foreground_band], axis=0)
        unexplained_ratio = unexplained_frame / (
            unexplained_frame + explained_frame + 1e-14
        )
        foreground_presence = np.clip((unexplained_ratio - 0.10) / 0.35, 0.0, 1.0)
        foreground_presence = scipy.ndimage.gaussian_filter1d(
            foreground_presence, sigma=3.0
        )
        foreground_presence = scipy.ndimage.maximum_filter1d(
            foreground_presence, size=21, mode="nearest"
        )
        foreground_absence = 1.0 - foreground_presence

        broad_activity = np.clip(
            mid_reference_power
            / (0.75 * np.median(mid_reference_power, axis=1, keepdims=True) + 1e-14),
            0.0,
            1.0,
        )
        broad_center_dominance = np.clip((0.50 - side_support) / 0.50, 0.0, 1.0)
        media_dominance = np.sqrt(
            predicted_power / (predicted_power + residual_power + 1e-14)
        )
        silence_mask = (
            foreground_absence[np.newaxis, :]
            * broad_activity
            * broad_center_dominance
            * media_dominance
        )
        silence_gain = np.sqrt(
            np.clip(
                1.0 - silence_cleanup_strength * silence_mask,
                0.03,
                1.0,
            )
        )
        silence_gain = scipy.ndimage.gaussian_filter(silence_gain, sigma=(0.7, 1.0))
        mid_residual_stft *= silence_gain

    cleanup_ratio = float(
        np.sqrt(np.sum(np.abs(mid_residual_stft) ** 2) / (raw_residual_power + 1e-20))
    )
    output = _istft(mid_residual_stft, len(mixture_mid), sr)
    side_residual = _istft(side_residual_stft, len(mixture_side), sr)
    return output, side_residual, cleanup_ratio


def _format_output_channels(
    mid: np.ndarray,
    side: np.ndarray,
    output_channels: int,
) -> np.ndarray:
    if output_channels == 1:
        return mid.astype(np.float32, copy=False)
    if output_channels == 2:
        return np.column_stack([mid + side, mid - side]).astype(np.float32, copy=False)
    raise ValueError("output_channels must be 1 or 2.")


def _passthrough_channels(
    mixture: np.ndarray,
    output_channels: int,
) -> np.ndarray:
    if output_channels == 1:
        return (0.5 * (mixture[:, 0] + mixture[:, 1])).astype(np.float32, copy=False)
    if output_channels == 2:
        return mixture.astype(np.float32, copy=False)
    raise ValueError("output_channels must be 1 or 2.")


def _mono_reference_cancel(
    mixture_mid: np.ndarray,
    reference_mid: np.ndarray,
    scalar_gain: np.ndarray,
    cleanup_strength: float,
    center_strength: float,
    sr: int,
    center_cleanup_strength: float,
    silence_cleanup_strength: float,
) -> tuple[np.ndarray, float]:
    """Cancel without Side, preferring the foreground-safe scalar model."""
    scalar_output = mixture_mid - scalar_gain * reference_mid
    residual_stft = _stft(scalar_output, sr)
    reference_stft = _stft(reference_mid, sr)
    _, residual_coherence = _estimate_complex_transfer(
        residual_stft, reference_stft, sigma=(1.0, 10.0)
    )
    reference_power = np.abs(reference_stft) ** 2
    active = reference_power > np.median(reference_power, axis=1, keepdims=True)
    coherence_score = (
        float(np.median(residual_coherence[active])) if np.any(active) else 0.0
    )
    if coherence_score < 0.15:
        return scalar_output.astype(np.float32), 1.0

    # A coherent residual indicates EQ/FIR coloration that a scalar envelope
    # cannot represent. Reuse the complex path with Mid as its own control;
    # this is deliberately gated because it is more exposed to foreground
    # double-talk than the normal stereo Side-controlled route.
    output, _, cleanup_ratio = _complex_reference_cancel(
        mixture_mid,
        mixture_mid,
        reference_mid,
        reference_mid,
        scalar_gain,
        cleanup_strength,
        center_strength,
        sr,
        center_cleanup_strength,
        silence_cleanup_strength,
    )
    return output, cleanup_ratio


def _cancel_chunk(
    mixture: np.ndarray,
    reference_search: np.ndarray,
    sr: int,
    cleanup_strength: float,
    center_strength: float = 0.5,
    center_cleanup_strength: float = 0.0,
    silence_cleanup_strength: float = 0.0,
    adaptive_time_warp: bool = True,
    mixture_channels: int = 2,
    reference_channels: int = 2,
    output_channels: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    if mixture_channels not in (1, 2):
        raise ValueError("mixture_channels must be 1 or 2.")
    if reference_channels not in (1, 2):
        raise ValueError("reference_channels must be 1 or 2.")
    if output_channels not in (1, 2):
        raise ValueError("output_channels must be 1 or 2.")
    if mixture_channels == 1 and output_channels != 1:
        raise ValueError("A mono mixture cannot produce a stereo output.")

    mixture_mid = 0.5 * (mixture[:, 0] + mixture[:, 1])
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    if len(reference_search) < len(mixture):
        # This occurs naturally after a truncated reference when a caller
        # supplied a fixed offset. Treat it as a no-match region instead of
        # passing an empty/short vector into the alignment filters.
        return _passthrough_channels(mixture, output_channels), {
            "alignment_score": 0.0,
            "aligned_start": 0.0,
            "gain_p05": 0.0,
            "gain_median": 0.0,
            "gain_p95": 0.0,
            "side_corr_median": 0.0,
            "foreground_guard": 0.0,
            "side_residual_ratio": 1.0,
            "cleanup_output_ratio": 1.0,
            "insufficient_reference_passthrough": 1.0,
        }

    alignment_diagnostics: dict[str, float] = {}
    reference, alignment_score, aligned_start = _align_reference(
        mixture,
        reference_search,
        sr,
        alignment_diagnostics,
        adaptive_time_warp,
    )
    reference_mid = 0.5 * (reference[:, 0] + reference[:, 1])
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])

    # Side is unavailable if either original input was mono. In that case use
    # the common Mid signal as the cancellation control. For a stereo mixture,
    # preserve its original Side exactly: a mono removable source cannot
    # contribute spatial difference information.
    mono_route = mixture_channels == 1 or reference_channels == 1
    control_mixture = mixture_mid if mono_route else mixture_side
    control_reference = reference_mid if mono_route else reference_side
    gain, correlations = _estimate_gain_envelope(control_mixture, control_reference, sr)
    valid_corr = correlations[np.isfinite(correlations)]
    side_corr_median = float(np.median(valid_corr)) if len(valid_corr) else 0.0
    # A second, unrelated stereo source appears as double-talk in Side. Keep
    # reference subtraction active, but fade out all residual suppression so
    # host-side music is not mistaken for removable-media artifacts.
    foreground_guard = float(np.clip((side_corr_median - 0.20) / 0.20, 0.0, 1.0))
    if mono_route:
        output_mid, cleanup_ratio = _mono_reference_cancel(
            mixture_mid,
            reference_mid,
            gain,
            cleanup_strength * foreground_guard,
            center_strength,
            sr,
            center_cleanup_strength * foreground_guard,
            silence_cleanup_strength * foreground_guard,
        )
        modeled_side_residual = mixture_side
    else:
        output_mid, modeled_side_residual, cleanup_ratio = _complex_reference_cancel(
            mixture_mid,
            mixture_side,
            reference_mid,
            reference_side,
            gain,
            cleanup_strength * foreground_guard,
            center_strength,
            sr,
            center_cleanup_strength * foreground_guard,
            silence_cleanup_strength * foreground_guard,
        )
    side_residual = mixture_side if mono_route else modeled_side_residual
    output = _format_output_channels(output_mid, side_residual, output_channels)

    diagnostics = {
        "alignment_score": alignment_score,
        "aligned_start": float(aligned_start),
        "gain_p05": float(np.percentile(gain, 5)),
        "gain_median": float(np.median(gain)),
        "gain_p95": float(np.percentile(gain, 95)),
        "side_corr_median": side_corr_median,
        "foreground_guard": foreground_guard,
        "side_residual_ratio": (
            1.0
            if mono_route
            else float(
                np.sqrt(
                    np.mean(side_residual * side_residual)
                    / (np.mean(mixture_side * mixture_side) + 1e-20)
                )
            )
        ),
        "cleanup_output_ratio": float(cleanup_ratio),
        **alignment_diagnostics,
    }
    return output, diagnostics


def process_range(
    mixture_path: str,
    reference_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    offset_sec: float | None = None,
    alignment_segments: list[AlignmentSegment] | None = None,
    chunk_sec: float = 30.0,
    context_sec: float = 1.0,
    search_sec: float = 0.25,
    strength: float = 0.0,
    cleanup_strength: float | None = None,
    center_strength: float | None = None,
    center_cleanup_strength: float | None = None,
    silence_cleanup_strength: float | None = None,
    adaptive_time_warp: bool = True,
    sr: int = DEFAULT_SR,
    workers: int = 1,
) -> None:
    if start_sec < 0.0:
        raise ValueError("start_sec must be non-negative.")
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive and finite.")
    if not np.isfinite(chunk_sec) or chunk_sec <= 0.0:
        raise ValueError("chunk_sec must be positive and finite.")
    if context_sec < 0.0 or search_sec < 0.0:
        raise ValueError("context_sec and search_sec must be non-negative.")
    if sr < MIN_SAMPLE_RATE:
        raise ValueError(f"sample rate must be at least {MIN_SAMPLE_RATE} Hz.")
    output = Path(output_path)
    if _paths_refer_to_same_file(output, mixture_path):
        raise ValueError("Output path must differ from the mixture input path.")
    if _paths_refer_to_same_file(output, reference_path):
        raise ValueError("Output path must differ from the reference input path.")
    output_format, output_subtype = _output_settings(output)

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
    if cleanup_strength < 0.0:
        raise ValueError("cleanup_strength must be non-negative.")
    if not 0.0 <= center_strength <= 1.0:
        raise ValueError("center_strength must be between 0 and 1.")
    if center_cleanup_strength < 0.0:
        raise ValueError("center_cleanup_strength must be non-negative.")
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
    if alignment_segments is None:
        if offset_sec is None:
            raise ValueError("Either offset_sec or alignment_segments is required.")
        alignment_segments = [
            AlignmentSegment(
                mixture_start=start_sec,
                mixture_end=start_sec + duration_sec,
                offset_sec=offset_sec,
                median_score=1.0,
            )
        ]
    alignment_segments = sorted(
        alignment_segments, key=lambda segment: segment.mixture_start
    )

    chunks: list[_ProcessingChunk] = []
    position = start_sec
    end_sec = start_sec + duration_sec
    while position < end_sec - 1e-9:
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
            default=end_sec,
        )
        piece_end = (
            min(end_sec, active.mixture_end)
            if active is not None
            else min(end_sec, next_active_start)
        )
        core_duration = min(chunk_sec, piece_end - position)
        if core_duration <= 1e-9:
            position = piece_end
            continue

        context_floor = (
            max(start_sec, active.mixture_start) if active is not None else position
        )
        context_ceiling = (
            min(end_sec, active.mixture_end)
            if active is not None
            else position + core_duration
        )
        context_start = max(context_floor, position - context_sec)
        context_end = min(context_ceiling, position + core_duration + context_sec)
        chunks.append(
            _ProcessingChunk(
                position=position,
                core_duration=core_duration,
                context_start=context_start,
                context_duration=context_end - context_start,
                active=active,
            )
        )
        position += core_duration

    def process_chunk(
        chunk: _ProcessingChunk,
    ) -> tuple[_ProcessingChunk, np.ndarray, dict[str, float] | None]:
        mixture = _decode_stereo(
            mixture_path,
            chunk.context_start,
            chunk.context_duration,
            sr,
        )
        if len(mixture) == 0 or chunk.active is None:
            cleaned = _passthrough_channels(mixture, mixture_channels)
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
                cleaned = _passthrough_channels(mixture, mixture_channels)
                diagnostics["low_confidence_passthrough"] = 1.0

        core_start = int(round((chunk.position - chunk.context_start) * sr))
        core_frames = int(round(chunk.core_duration * sr))
        core = cleaned[core_start : core_start + core_frames]
        if len(core) < core_frames:
            missing = core_frames - len(core)
            core = (
                np.pad(core, (0, missing))
                if core.ndim == 1
                else np.pad(core, ((0, missing), (0, 0)))
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
            if previous_sample is not None and len(core):
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cancel a known reference from a real mixture. Wider inputs are "
            "downmixed to stereo."
        )
    )
    parser.add_argument("mixture", help="Mixture media containing the target audio.")
    parser.add_argument("reference", help="Known removable reference media.")
    parser.add_argument("output", help="24-bit .flac or .wav output path.")
    parser.add_argument(
        "--offset",
        type=float,
        help="Mixture time minus reference time. Omit to scan automatically.",
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--chunk", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel alignment queries and cancellation chunks. Higher "
            "values use more CPU and roughly proportional temporary memory."
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.0,
        help=(
            "Quality/removal trade-off: 0=preserve, 1=v8 aggressive, "
            ">1=ASR-oriented extra removal (open-ended)."
        ),
    )
    parser.add_argument(
        "--cleanup-strength",
        type=float,
        help="Expert override for side-derived residual cleanup.",
    )
    parser.add_argument(
        "--center-strength",
        type=float,
        help=(
            "Expert override for direct cancellation of active centered reference bins."
        ),
    )
    parser.add_argument(
        "--center-cleanup-strength",
        type=float,
        help=("Expert override for centered residue suppression; may exceed 1."),
    )
    parser.add_argument(
        "--silence-cleanup-strength",
        type=float,
        help=(
            "Expert override for cleanup when little unrelated foreground is detected."
        ),
    )
    parser.add_argument(
        "--disable-adaptive-warp",
        action="store_true",
        help=(
            "Use only the conservative 250ms alignment anchors; disable "
            "validated 16ms refinement for detected speed drift."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    if not np.isfinite(args.strength) or args.strength < 0:
        raise ValueError("--strength must be non-negative.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.start < 0.0:
        raise ValueError("--start must be non-negative.")
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0.0
    ):
        raise ValueError("--duration must be positive and finite.")
    if not np.isfinite(args.chunk) or args.chunk <= 0.0:
        raise ValueError("--chunk must be positive and finite.")
    if args.sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(f"--sample-rate must be at least {MIN_SAMPLE_RATE} Hz.")
    if args.cleanup_strength is not None and args.cleanup_strength < 0:
        raise ValueError("--cleanup-strength must be non-negative.")
    if args.center_strength is not None and not 0.0 <= args.center_strength <= 1.0:
        raise ValueError("--center-strength must be between 0 and 1.")
    if args.center_cleanup_strength is not None and args.center_cleanup_strength < 0.0:
        raise ValueError("--center-cleanup-strength must be non-negative.")
    if (
        args.silence_cleanup_strength is not None
        and not 0.0 <= args.silence_cleanup_strength <= 1.0
    ):
        raise ValueError("--silence-cleanup-strength must be between 0 and 1.")
    mixture_duration = _media_duration(args.mixture)
    if args.start >= mixture_duration:
        raise ValueError(
            f"--start ({args.start:.3f}s) is outside the mixture "
            f"({mixture_duration:.3f}s)."
        )
    available_duration = mixture_duration - args.start
    if args.duration is not None and args.duration > available_duration + 1e-3:
        raise ValueError(
            "Requested range ends after the mixture; at most "
            f"{available_duration:.3f}s is available from --start."
        )
    duration = args.duration if args.duration is not None else available_duration
    segments = (
        None
        if args.offset is not None
        else discover_alignment_segments(
            args.mixture,
            args.reference,
            workers=args.workers,
            mixture_start_sec=args.start,
            mixture_duration_sec=duration,
        )
    )
    process_range(
        args.mixture,
        args.reference,
        args.output,
        start_sec=args.start,
        duration_sec=duration,
        offset_sec=args.offset,
        alignment_segments=segments,
        chunk_sec=args.chunk,
        strength=args.strength,
        cleanup_strength=args.cleanup_strength,
        center_strength=args.center_strength,
        center_cleanup_strength=args.center_cleanup_strength,
        silence_cleanup_strength=args.silence_cleanup_strength,
        adaptive_time_warp=not args.disable_adaptive_warp,
        sr=args.sample_rate,
        workers=args.workers,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _run_cli(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
