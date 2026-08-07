"""Streaming audio fingerprints and an in-memory candidate index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.spatial

from .media import MIN_ALIGNMENT_SAMPLE_RATE, _iter_decode_mono_low

# A 0.5 s hop halves the number of signatures, which pays for twice the bands
# at the same index size, and on real material that trades well: measured
# against a real broadcast reference under a real host voice, 32x16 at this
# hop misses 58% of windows at its own chance floor where 16x16 at a 0.25 s
# hop misses 63%, for the same 2.8 GB over 24 hours. Anchors are five seconds
# apart regardless, and sub-hop offset error is refined away.
DEFAULT_FINGERPRINT_HOP_SEC = 0.50
DEFAULT_FINGERPRINT_FRAME_SEC = 0.50
# Wider than the original 8x8. Extra dimensions do not raise the score of a
# true match -- on real material they lower it slightly -- but they push
# chance collisions down faster, so the threshold can come down with them and
# recall improves net. The old 8x8 floor of 0.45 sat above the 0.40 local
# threshold, leaving that gate no margin over coincidence at all.
DEFAULT_FINGERPRINT_BANDS = 32
DEFAULT_FINGERPRINT_TEMPORAL_BINS = 16
# Raising this does not help: measured, extending the band to 900 or 1800 Hz
# lowers the true-match score. The robust contour lives in the low band.
DEFAULT_FINGERPRINT_HIGH_HZ = 450.0


@dataclass(frozen=True)
class FingerprintCandidate:
    """One approximate media/time match returned by a fingerprint index."""

    media_id: str
    time_sec: float
    score: float


@dataclass(frozen=True)
class FingerprintTrack:
    """Compact query-window signatures for one media item."""

    media_id: str
    times: np.ndarray
    features: np.ndarray
    hop_sec: float
    query_sec: float

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("fingerprint times must be one-dimensional.")
        if self.features.ndim != 2:
            raise ValueError("fingerprint features must be two-dimensional.")
        if len(self.times) != len(self.features):
            raise ValueError("fingerprint times and features must have equal length.")
        if not np.all(np.isfinite(self.times)) or not np.all(
            np.isfinite(self.features)
        ):
            raise ValueError("fingerprint data must be finite.")
        if np.any(np.diff(self.times) < 0.0):
            raise ValueError("fingerprint times must be sorted.")
        if self.hop_sec <= 0.0 or self.query_sec <= 0.0:
            raise ValueError("fingerprint timings must be positive.")

    def feature_near(
        self,
        time_sec: float,
        *,
        tolerance_sec: float | None = None,
    ) -> np.ndarray | None:
        """Return the nearest signature when it lies inside the tolerance."""
        if not len(self.times):
            return None
        position = int(np.searchsorted(self.times, time_sec))
        candidates = [
            index for index in (position - 1, position) if 0 <= index < len(self.times)
        ]
        index = min(candidates, key=lambda item: abs(self.times[item] - time_sec))
        tolerance = 0.51 * self.hop_sec if tolerance_sec is None else tolerance_sec
        if abs(self.times[index] - time_sec) > tolerance:
            return None
        return self.features[index]


class FingerprintIndex:
    """Search compact fingerprints across one or more media IDs."""

    def __init__(self, tracks: list[FingerprintTrack]):
        usable = [track for track in tracks if len(track.times)]
        if not usable:
            raise ValueError("at least one non-empty fingerprint track is required.")
        media_ids = [track.media_id for track in usable]
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("fingerprint media IDs must be unique.")
        dimensions = {track.features.shape[1] for track in usable}
        if len(dimensions) != 1:
            raise ValueError("all fingerprint tracks must use the same feature size.")

        self._tracks = {track.media_id: track for track in usable}
        self._media_ids = tuple(self._tracks)
        self._features = np.ascontiguousarray(
            np.concatenate([track.features for track in usable]),
            dtype=np.float32,
        )
        self._times = np.concatenate([track.times for track in usable]).astype(
            np.float64,
            copy=False,
        )
        self._media_codes = np.concatenate(
            [
                np.full(len(track.times), code, dtype=np.uint32)
                for code, track in enumerate(usable)
            ]
        )
        self._tree = scipy.spatial.cKDTree(
            self._features,
            compact_nodes=True,
            balanced_tree=True,
        )

    @property
    def memory_bytes(self) -> int:
        """Estimate bytes retained by tracks, lookup arrays, and tree data."""
        total = self._features.nbytes + self._times.nbytes + self._media_codes.nbytes
        total += sum(
            track.times.nbytes + track.features.nbytes
            for track in self._tracks.values()
        )
        if not np.shares_memory(self._tree.data, self._features):
            total += self._tree.data.nbytes
        total += self._tree.indices.nbytes
        return total

    @property
    def entry_count(self) -> int:
        """Return the number of indexed media/time windows."""
        return len(self._times)

    def query(
        self,
        feature: np.ndarray,
        *,
        candidates: int = 12,
    ) -> list[FingerprintCandidate]:
        """Return deterministic nearest candidates across all indexed media."""
        if candidates < 1:
            raise ValueError("candidates must be positive.")
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 1 or feature.shape[0] != self._features.shape[1]:
            raise ValueError("query feature has the wrong dimensions.")
        if not np.all(np.isfinite(feature)):
            raise ValueError("query feature must be finite.")
        if np.linalg.norm(feature) < 1e-6:
            return []
        count = min(candidates, len(self._features))
        distances, indices = self._tree.query(feature, k=count, workers=1)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        matches = [
            FingerprintCandidate(
                media_id=self._media_ids[int(self._media_codes[index])],
                time_sec=float(self._times[index]),
                score=float(np.clip(1.0 - 0.5 * distance * distance, -1.0, 1.0)),
            )
            for distance, index in zip(distances, indices)
        ]
        return sorted(
            matches,
            key=lambda match: (-match.score, match.media_id, match.time_sec),
        )

    def best_near(
        self,
        media_id: str,
        feature: np.ndarray,
        predicted_time_sec: float,
        radius_sec: float,
    ) -> FingerprintCandidate | None:
        """Find the strongest candidate inside a predicted local time window."""
        if radius_sec < 0.0:
            raise ValueError("radius_sec must be non-negative.")
        track = self._tracks[media_id]
        if np.linalg.norm(feature) < 1e-6:
            return None
        first = int(np.searchsorted(track.times, predicted_time_sec - radius_sec))
        last = int(
            np.searchsorted(
                track.times,
                predicted_time_sec + radius_sec,
                side="right",
            )
        )
        if first >= last:
            return None
        scores = track.features[first:last] @ feature
        relative = int(np.argmax(scores))
        index = first + relative
        return FingerprintCandidate(
            media_id=media_id,
            time_sec=float(track.times[index]),
            score=float(scores[relative]),
        )


def fingerprint_media(
    path: str,
    media_id: str,
    *,
    sr: int = 1_000,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    query_sec: float = 8.0,
    hop_sec: float = DEFAULT_FINGERPRINT_HOP_SEC,
    frame_sec: float = DEFAULT_FINGERPRINT_FRAME_SEC,
    bands: int = DEFAULT_FINGERPRINT_BANDS,
    temporal_bins: int = DEFAULT_FINGERPRINT_TEMPORAL_BINS,
    high_hz: float = DEFAULT_FINGERPRINT_HIGH_HZ,
) -> FingerprintTrack:
    """Stream a media file and return compact, volume-invariant signatures."""
    blocks = _iter_decode_mono_low(
        path,
        sr,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    return fingerprint_blocks(
        blocks,
        media_id,
        sr=sr,
        start_sec=start_sec,
        query_sec=query_sec,
        hop_sec=hop_sec,
        frame_sec=frame_sec,
        bands=bands,
        temporal_bins=temporal_bins,
        high_hz=high_hz,
    )


def fingerprint_blocks(
    blocks,
    media_id: str,
    *,
    sr: int,
    start_sec: float = 0.0,
    query_sec: float = 8.0,
    hop_sec: float = DEFAULT_FINGERPRINT_HOP_SEC,
    frame_sec: float = DEFAULT_FINGERPRINT_FRAME_SEC,
    bands: int = DEFAULT_FINGERPRINT_BANDS,
    temporal_bins: int = DEFAULT_FINGERPRINT_TEMPORAL_BINS,
    high_hz: float = DEFAULT_FINGERPRINT_HIGH_HZ,
) -> FingerprintTrack:
    """Build fingerprints from an iterable without retaining decoded audio."""
    if sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(
            f"fingerprint sample rate must be at least {MIN_ALIGNMENT_SAMPLE_RATE}."
        )
    if not np.isfinite(start_sec) or start_sec < 0.0:
        raise ValueError("fingerprint start must be non-negative and finite.")
    if (
        not np.isfinite(query_sec)
        or not np.isfinite(hop_sec)
        or not np.isfinite(frame_sec)
        or query_sec <= 0.0
        or hop_sec <= 0.0
        or frame_sec <= 0.0
    ):
        raise ValueError("fingerprint timings must be positive and finite.")
    if frame_sec > query_sec or hop_sec > frame_sec:
        raise ValueError("fingerprint timings must satisfy hop <= frame <= query.")
    if bands < 4:
        raise ValueError("fingerprint bands must be at least 4.")
    if temporal_bins < 2:
        raise ValueError("fingerprint temporal_bins must be at least 2.")

    frame_frames = int(round(frame_sec * sr))
    hop_frames = int(round(hop_sec * sr))
    if frame_frames < 2 or hop_frames < 1:
        raise ValueError("fingerprint frame or hop is too short.")
    signature_frames = 1 + int(round((query_sec - frame_sec) / hop_sec))
    effective_temporal_bins = min(temporal_bins, signature_frames)
    feature_dimensions = effective_temporal_bins * bands
    high_hz = min(high_hz, 0.45 * sr)
    if high_hz <= 60.0:
        raise ValueError("fingerprint sample rate leaves no usable frequency band.")
    band_edges = np.geomspace(60.0, high_hz, bands + 1)
    frequencies = np.fft.rfftfreq(frame_frames, d=1.0 / sr)
    band_masks = []
    for index, (low, high) in enumerate(zip(band_edges[:-1], band_edges[1:])):
        mask = (frequencies >= low) & (
            frequencies <= high if index == bands - 1 else frequencies < high
        )
        if not np.any(mask):
            mask[np.argmin(np.abs(frequencies - np.sqrt(low * high)))] = True
        band_masks.append(mask)
    window = np.hanning(frame_frames).astype(np.float32)

    buffer = np.empty(0, dtype=np.float32)
    feature_blocks: list[np.ndarray] = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if not len(block):
            continue
        buffer = np.concatenate((buffer, block))
        frame_count = 1 + (len(buffer) - frame_frames) // hop_frames
        if frame_count <= 0:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(buffer, frame_frames)[
            : frame_count * hop_frames : hop_frames
        ]
        spectra = np.fft.rfft(frames * window, axis=1)
        power = np.square(np.abs(spectra))
        features = np.column_stack(
            [
                np.log(np.maximum(np.mean(power[:, mask], axis=1), 1e-12))
                for mask in band_masks
            ]
        )
        features -= np.mean(features, axis=1, keepdims=True)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        feature_blocks.append((features / np.maximum(norms, 1e-12)).astype(np.float32))
        buffer = buffer[frame_count * hop_frames :]

    if not feature_blocks:
        return FingerprintTrack(
            media_id=media_id,
            times=np.empty(0, dtype=np.float64),
            features=np.empty((0, feature_dimensions), dtype=np.float32),
            hop_sec=hop_sec,
            query_sec=query_sec,
        )

    frame_features = np.concatenate(feature_blocks)
    signature_count = len(frame_features) - signature_frames + 1
    if signature_count <= 0:
        return FingerprintTrack(
            media_id=media_id,
            times=np.empty(0, dtype=np.float64),
            features=np.empty((0, feature_dimensions), dtype=np.float32),
            hop_sec=hop_sec,
            query_sec=query_sec,
        )

    cumulative = np.vstack(
        (
            np.zeros((1, bands), dtype=np.float64),
            np.cumsum(frame_features, axis=0, dtype=np.float64),
        )
    )
    starts = np.arange(signature_count)
    boundaries = np.rint(
        np.linspace(0, signature_frames, effective_temporal_bins + 1)
    ).astype(int)
    signatures = np.concatenate(
        [
            (cumulative[starts + last] - cumulative[starts + first]) / (last - first)
            for first, last in zip(boundaries[:-1], boundaries[1:])
        ],
        axis=1,
    )
    signatures = signatures.reshape(
        signature_count,
        effective_temporal_bins,
        bands,
    )
    signatures -= np.mean(signatures, axis=1, keepdims=True)
    signatures = signatures.reshape(signature_count, feature_dimensions)
    signature_norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    signatures = (signatures / np.maximum(signature_norms, 1e-12)).astype(np.float32)
    times = start_sec + starts.astype(np.float64) * hop_sec
    return FingerprintTrack(
        media_id=media_id,
        times=times,
        features=signatures,
        hop_sec=hop_sec,
        query_sec=query_sec,
    )
