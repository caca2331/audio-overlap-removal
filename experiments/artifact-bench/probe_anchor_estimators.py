"""Per-anchor offset estimators on the real drift fixture, against the truth.

Rebuilds the bench fixture for one chunk, places 0.25 s anchors at the true
positions (so only the estimator is measured, not the search) and compares
argmax+parabola with plateau-centroid variants on Mid and Side.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bench  # noqa: E402

from audio_overlap_removal.alignment import (  # noqa: E402
    _fractional_match_index,
    _normalized_match,
)
from audio_overlap_removal.media import _decode_stereo  # noqa: E402

SR = bench.SR
ANCHOR = int(0.25 * SR)
RADIUS = 24


def correlation_curve(search: np.ndarray, query: np.ndarray) -> np.ndarray:
    q = query.astype(np.float64) - np.mean(query)
    s = search.astype(np.float64)
    num = scipy.signal.correlate(s, q, mode="valid", method="fft")
    n = len(q)
    cum = np.concatenate(([0.0], np.cumsum(s)))
    cum2 = np.concatenate(([0.0], np.cumsum(s * s)))
    rs = cum[n:] - cum[:-n]
    en = np.maximum(cum2[n:] - cum2[:-n] - rs * rs / n, 1e-20)
    return num / np.maximum(np.sqrt(en * np.sum(q * q)), 1e-20)


def plateau_center(curve: np.ndarray, index: int, fraction: float) -> float:
    threshold = curve[index] - fraction * (curve[index] - np.median(curve))
    weights = np.clip(curve - threshold, 0.0, None)
    left = index
    while left > 0 and weights[left - 1] > 0:
        left -= 1
    right = index
    while right < len(curve) - 1 and weights[right + 1] > 0:
        right += 1
    lags = np.arange(left, right + 1)
    w = weights[left : right + 1]
    return float(np.sum(w * lags) / np.sum(w))


def wide_parabola(curve: np.ndarray, index: int, half: int) -> float:
    lo = max(0, index - half)
    hi = min(len(curve), index + half + 1)
    x = np.arange(lo, hi, dtype=np.float64)
    a, b, _ = np.polyfit(x - index, curve[lo:hi], 2)
    if a >= 0:
        return float(index)
    return float(index - b / (2 * a))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-start", type=float, default=400.0)
    parser.add_argument("--chain", default="drift")
    parser.add_argument("--chunk-start", type=float, default=35.0)
    parser.add_argument("--chunk-sec", type=float, default=28.0)
    args = parser.parse_args()
    stages = set() if args.chain == "none" else set(args.chain.split(","))
    speed = bench.SPEED if "drift" in stages else 1.0
    duration = 60.0
    reference = _decode_stereo(
        str(bench.REFERENCE), args.media_start - bench.LEAD_SEC, duration + 2 * bench.LEAD_SEC, SR, 2
    )
    lead = int(bench.LEAD_SEC * SR)
    with tempfile.TemporaryDirectory() as temporary:
        media = bench.broadcast_chain(reference[lead : lead + int(duration * SR)], Path(temporary), "m", stages)
    total = int((duration * speed + 2 * bench.LEAD_SEC) * SR)
    foreground = _decode_stereo(str(bench.MIXTURE), 200.0, total / SR + 1.0, SR, 2)[:total]
    foreground *= np.sqrt(np.mean(media**2)) / (np.sqrt(np.mean(foreground**2)) + 1e-12)
    mixture = foreground.copy()
    mixture[lead : lead + len(media)] += media

    chunk_start = int(args.chunk_start * SR)
    chunk_len = int(args.chunk_sec * SR)
    chunk = mixture[chunk_start : chunk_start + chunk_len]
    truth = lead + (np.arange(chunk_start, chunk_start + chunk_len) - lead) / speed
    features = {
        "mid": (0.5 * (chunk[:, 0] + chunk[:, 1]), 0.5 * (reference[:, 0] + reference[:, 1])),
        "side": (0.5 * (chunk[:, 0] - chunk[:, 1]), 0.5 * (reference[:, 0] - reference[:, 1])),
    }
    estimators = {
        "argmax+parabola": None,
        "centroid 25%": lambda c, i: plateau_center(c, i, 0.25),
        "centroid 50%": lambda c, i: plateau_center(c, i, 0.5),
        "parabola ±3": lambda c, i: wide_parabola(c, i, 3),
        "parabola ±6": lambda c, i: wide_parabola(c, i, 6),
    }
    for name, (query_feature, search_feature) in features.items():
        errors = {key: [] for key in estimators}
        scores = []
        errors["energy-centroid pos"] = []
        errors["rate-compensated"] = []
        errors["rate-comp + centroid"] = []
        rate = 1.0 / speed  # reference samples per mixture sample, assumed known
        for query_start in range(0, chunk_len - ANCHOR + 1, ANCHOR):
            centre = query_start + 0.5 * ANCHOR
            true_offset = truth[query_start] + 0.5 * ANCHOR / speed - centre  # offset at centre
            predicted = int(round(truth[query_start]))
            search_start = predicted - RADIUS
            search = search_feature[search_start : predicted + ANCHOR + RADIUS]
            query = query_feature[query_start : query_start + ANCHOR]
            idx, score = _normalized_match(search, query)
            scores.append(score)
            curve = correlation_curve(search, query)
            for key, fn in estimators.items():
                est = _fractional_match_index(search, query, idx) if fn is None else fn(curve, idx)
                errors[key].append(search_start + est - query_start - true_offset)
            # (1) keep the offset, move the anchor to where the energy is
            est = _fractional_match_index(search, query, idx)
            q64 = query.astype(np.float64)
            weights = q64 * q64
            centroid = float(np.sum(weights * np.arange(ANCHOR)) / (np.sum(weights) + 1e-20))
            t_c = query_start + centroid
            true_at_centroid = truth[query_start] + centroid / speed - t_c
            errors["energy-centroid pos"].append(search_start + est - query_start - true_at_centroid)
            # (2) resample the query to the reference rate before correlating
            import scipy.ndimage
            count = int(np.floor((ANCHOR - 1) * rate))
            positions = np.arange(count) / rate
            q_rc = scipy.ndimage.map_coordinates(q64, [positions], order=3, mode="nearest")
            idx_rc, _ = _normalized_match(search, q_rc)
            est_rc = _fractional_match_index(search, q_rc, idx_rc)
            p0 = search_start + est_rc  # reference position of the query start
            errors["rate-compensated"].append(p0 + 0.5 * ANCHOR * rate - centre - true_offset)
            w_rc = q_rc * q_rc
            c_rc = float(np.sum(w_rc * np.arange(count)) / (np.sum(w_rc) + 1e-20))
            t_c2 = query_start + c_rc / rate
            true_at_c2 = truth[query_start] + (c_rc / rate) / speed - t_c2
            errors["rate-comp + centroid"].append(p0 + c_rc - t_c2 - true_at_c2)
        print(f"{name}: {len(scores)} anchors, score median {np.median(scores):.3f}")
        for key, err in errors.items():
            err = np.array(err)
            print(
                f"  {key:16s} rms {np.sqrt(np.mean(err**2)):6.2f}  p50|e| {np.median(np.abs(err)):5.2f}  "
                f"p90|e| {np.percentile(np.abs(err), 90):5.2f}  max|e| {np.max(np.abs(err)):6.2f}"
            )


if __name__ == "__main__":
    main()
