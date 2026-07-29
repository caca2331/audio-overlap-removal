"""Export short current-vs-two-pass comparisons from the original material."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage
import soundfile as sf

import explore_two_pass as explore
from audio_overlap_removal.cancellation import (
    _complex_reference_cancel,
    _estimate_complex_transfer,
    _estimate_gain_envelope,
    _istft,
    _stft,
)
from audio_overlap_removal.media import _decode_stereo

SR = 48_000
DURATION = 12.0
explore.SR = SR
ROOT = Path(__file__).parent
MIXTURE = explore.ASSETS / "miyako.webm"
REFERENCE = explore.ASSETS / "sr.webm"
CLIPS = [
    ("early", 400.0, 81.533),
    ("middle", 2200.192, 81.533),
    ("late", 3719.992, 81.463),
]


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def current_strength_one(
    mixture: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    mixture_mid = mixture.mean(axis=1)
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_mid = reference.mean(axis=1)
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    gain, correlations = _estimate_gain_envelope(
        mixture_side, reference_side, SR
    )
    valid = correlations[np.isfinite(correlations)]
    correlation = float(np.median(valid)) if len(valid) else 0.0
    guard = float(np.clip((correlation - 0.20) / 0.20, 0.0, 1.0))
    output, _, _ = _complex_reference_cancel(
        mixture_mid,
        mixture_side,
        reference_mid,
        reference_side,
        gain,
        cleanup_strength=guard,
        center_strength=1.0,
        sr=SR,
        center_cleanup_strength=guard,
        silence_cleanup_strength=guard,
    )
    return output, guard


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


def write_audio(path: Path, audio: np.ndarray) -> None:
    sf.write(path, np.clip(audio, -1.0, 1.0), SR, subtype="PCM_24")


def main() -> None:
    report = {}
    comparisons = []
    silence = np.zeros(SR, dtype=np.float32)
    for label, mixture_start, offset in CLIPS:
        mixture = _decode_stereo(
            str(MIXTURE), mixture_start, DURATION, SR
        )
        reference = _decode_stereo(
            str(REFERENCE), mixture_start - offset, DURATION, SR
        )
        frames = min(len(mixture), len(reference))
        mixture = mixture[:frames]
        reference = reference[:frames]
        linear, side_residual, prediction = explore.first_pass(
            mixture, reference, center_strength=1.0
        )
        side_refined = explore.second_pass(
            linear,
            side_residual,
            prediction,
            reference,
            alpha=0.75,
            mid_share=0.0,
            max_relative=0.25,
        )
        nonlinear_refined, nonlinear_validation = (
            explore.nonlinear_second_pass(
                linear, side_residual, prediction, reference
            )
        )
        correction = (linear - side_refined) + (
            linear - nonlinear_refined
        )
        baseline, guard = current_strength_one(mixture, reference)
        refined = (baseline - correction).astype(np.float32)

        mixture_mid = mixture.mean(axis=1)
        write_audio(ROOT / f"{label}-mixture.flac", mixture_mid)
        write_audio(ROOT / f"{label}-current.flac", baseline)
        write_audio(ROOT / f"{label}-two-pass.flac", refined)
        # Within each group: mixture, current output, proposed output.
        comparisons.extend(
            [mixture_mid, silence, baseline, silence, refined, 2 * silence]
        )
        report[label] = {
            "mixture_start_sec": mixture_start,
            "reference_start_sec": mixture_start - offset,
            "foreground_guard": guard,
            "nonlinear_validation_improvement": nonlinear_validation,
            "correction_to_current_rms_ratio": rms(correction)
            / (rms(baseline) + 1e-12),
            "current_reference_related_rms": related_rms(
                baseline, reference
            ),
            "two_pass_reference_related_rms": related_rms(
                refined, reference
            ),
        }
    write_audio(ROOT / "all-comparisons.flac", np.concatenate(comparisons))
    (ROOT / "real-results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
