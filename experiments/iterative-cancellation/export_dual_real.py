"""Export selected dual-model output for the real listening clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage
import soundfile as sf

import explore_dual_model as dual
import explore_iterative as iterative
import explore_two_pass as base
import test_exact_pipeline as exact
from audio_overlap_removal.cancellation import (
    _estimate_complex_transfer,
    _istft,
    _stft,
)
from audio_overlap_removal.media import _decode_stereo

SR = 48_000
base.SR = SR
iterative.SR = SR
exact.SR = SR
dual.SR = SR
ROOT = Path(__file__).parent
CLIPS = [
    ("early", 400.0, 81.533),
    ("middle", 2200.192, 81.533),
    ("late", 3719.992, 81.463),
]


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def related_rms(output: np.ndarray, reference: np.ndarray) -> float:
    output_stft = _stft(output, SR)
    reference_stft = _stft(reference.mean(axis=1), SR)
    transfer, coherence = _estimate_complex_transfer(
        output_stft, reference_stft, (1.0, 10.0)
    )
    confidence = scipy.ndimage.gaussian_filter(
        np.clip((coherence - 0.55) / 0.35, 0.0, 1.0),
        sigma=(0.5, 1.0),
    )
    related = _istft(
        confidence * transfer * reference_stft, len(output), SR
    )
    return rms(related)


def main() -> None:
    report = {}
    comparison = []
    silence = np.zeros(SR, dtype=np.float32)
    for label, start, offset in CLIPS:
        mixture = _decode_stereo(
            str(base.ASSETS / "miyako.webm"), start, 12.0, SR
        )
        reference = _decode_stereo(
            str(base.ASSETS / "sr.webm"), start - offset, 12.0, SR
        )
        frames = min(len(mixture), len(reference))
        mixture = mixture[:frames]
        reference = reference[:frames]
        current = base.production_output(mixture, reference)
        selected, diagnostics = dual.complete_output(
            mixture,
            reference,
            floor=0.90,
            coherence_floor=0.30,
            dominance_floor=0.30,
        )
        sf.write(
            ROOT / f"{label}-dual-model.flac",
            selected,
            SR,
            subtype="PCM_24",
        )
        comparison.extend([current, silence, selected, 2 * silence])
        current_related = related_rms(current, reference)
        selected_related = related_rms(selected, reference)
        report[label] = {
            **diagnostics,
            "difference_rms_ratio": rms(selected - current)
            / (rms(current) + 1e-12),
            "current_output_rms": rms(current),
            "dual_output_rms": rms(selected),
            "current_reference_related_rms": current_related,
            "dual_reference_related_rms": selected_related,
            "reference_related_change": selected_related
            / (current_related + 1e-12)
            - 1.0,
        }
    sf.write(
        ROOT / "all-dual-comparisons.flac",
        np.concatenate(comparison),
        SR,
        subtype="PCM_24",
    )
    (ROOT / "dual-real-results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
