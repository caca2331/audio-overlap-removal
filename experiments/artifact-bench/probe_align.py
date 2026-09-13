"""Measure the chunk aligner's position error against a known warp.

Builds the bench's drift fixture, runs ``_align_reference`` on one chunk with
the true prior, captures the source positions it hands to the interpolator and
compares them with the truth sample by sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bench  # noqa: E402

from audio_overlap_removal import alignment  # noqa: E402
from audio_overlap_removal.media import _decode_stereo  # noqa: E402

SR = bench.SR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-start", type=float, default=400.0)
    parser.add_argument("--chain", default="drift")
    parser.add_argument("--chunk-start", type=float, default=35.0)
    parser.add_argument("--chunk-sec", type=float, default=32.0)
    parser.add_argument("--disable-adaptive-warp", action="store_true")
    parser.add_argument("--rate-prior", type=float, default=1.0)
    args = parser.parse_args()
    stages = set() if args.chain == "none" else set(args.chain.split(","))
    speed = bench.SPEED if "drift" in stages else 1.0

    duration = 60.0
    reference = _decode_stereo(
        str(bench.REFERENCE), args.media_start - bench.LEAD_SEC, duration + 2 * bench.LEAD_SEC, SR, 2
    )
    lead = int(bench.LEAD_SEC * SR)
    media_clean = reference[lead : lead + int(duration * SR)]
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        media = bench.broadcast_chain(media_clean, Path(temporary), "media", stages)
    total = int((duration * speed + 2 * bench.LEAD_SEC) * SR)
    foreground = _decode_stereo(str(bench.MIXTURE), 200.0, total / SR + 1.0, SR, 2)[:total]
    foreground *= np.sqrt(np.mean(media**2)) / (np.sqrt(np.mean(foreground**2)) + 1e-12)
    mixture = foreground.copy()
    mixture[lead : lead + len(media)] += media

    chunk_start = int(args.chunk_start * SR)
    chunk_len = int(args.chunk_sec * SR)
    chunk = mixture[chunk_start : chunk_start + chunk_len]
    # truth: mixture sample m maps to reference sample lead + (m - lead) / speed
    truth = lead + (np.arange(chunk_start, chunk_start + chunk_len) - lead) / speed
    radius = int(0.25 * SR)
    window_start = int(np.floor(truth[0])) - radius
    window_end = int(np.ceil(truth[-1])) + radius
    window = reference[window_start:window_end]
    predicted_start = truth[0] - window_start

    captured: dict[str, np.ndarray] = {}
    original = scipy.ndimage.map_coordinates

    def spy(input, coordinates, *a, **k):  # noqa: A002
        captured["positions"] = np.asarray(coordinates[0], dtype=np.float64)
        return original(input, coordinates, *a, **k)

    alignment.scipy.ndimage.map_coordinates = spy
    diagnostics: dict[str, float] = {}
    try:
        result = alignment._align_reference(
            chunk,
            window,
            SR,
            diagnostics,
            adaptive_time_warp=not args.disable_adaptive_warp,
            predicted_start=predicted_start,
            search_radius_sec=0.25,
            predicted_rate=args.rate_prior,
        )
    finally:
        alignment.scipy.ndimage.map_coordinates = original
    positions = captured["positions"] + window_start
    error = positions - truth
    print(f"covered={result.covered} score={result.score:.3f}")
    print({k: round(v, 4) for k, v in diagnostics.items()})
    print(
        f"position error (samples): median {np.median(error):+.2f}  "
        f"p05 {np.percentile(error, 5):+.2f}  p95 {np.percentile(error, 95):+.2f}  "
        f"max|e| {np.max(np.abs(error)):.2f}"
    )
    anchor = int(0.25 * SR)
    centres = np.arange(anchor // 2, min(chunk_len, int(6.0 * SR)), anchor)
    print("first anchors, error at anchor centres (samples):",
          " ".join(f"{error[c]:+.1f}" for c in centres))
    step = 2 * SR
    for start in range(0, chunk_len, step):
        e = error[start : start + step]
        print(f"  t={args.chunk_start + start / SR:6.1f}s  err median {np.median(e):+7.2f}  p95 {np.percentile(np.abs(e), 95):7.2f}")
    if result.reference is not None:
        true_media = chunk - foreground[chunk_start : chunk_start + chunk_len]
        aligned = result.reference
        # Compare against the media actually present, scaled by the chain gain.
        gain = np.sum(aligned * true_media) / (np.sum(aligned * aligned) + 1e-20)
        residual = true_media - gain * aligned
        print(
            f"aligned vs true media (scalar fit gain {gain:.3f}): "
            f"{10 * np.log10(np.sum(true_media**2) / (np.sum(residual**2) + 1e-20)):.2f} dB"
        )


if __name__ == "__main__":
    main()
