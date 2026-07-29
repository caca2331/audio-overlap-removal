"""Compare conservative first-pass models and repeated Side residual repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal

import explore_two_pass as base
from audio_overlap_removal.cancellation import (
    _estimate_complex_transfer,
    _estimate_gain_envelope,
    _istft,
    _stft,
)
from audio_overlap_removal.media import _decode_stereo

SR = 16_000
DURATION = 14.0
FRAMES = int(SR * DURATION)
base.SR = SR
ROOT = Path(__file__).parent


def conservative_first_pass(
    mixture: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate B without using mixture Mid to learn its transfer."""
    mixture_mid = mixture.mean(axis=1)
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_mid = reference.mean(axis=1)
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    gain, _ = _estimate_gain_envelope(mixture_side, reference_side, SR)
    scalar_error = base.rms(mixture_side - gain * reference_side) / (
        base.rms(mixture_side) + 1e-12
    )
    if scalar_error < 0.10:
        prediction_mid = gain * reference_mid
        prediction_side = gain * reference_side
        return (
            (mixture_mid - prediction_mid).astype(np.float32),
            (mixture_side - prediction_side).astype(np.float32),
            prediction_mid.astype(np.float32),
        )

    mixture_side_stft = _stft(mixture_side, SR)
    reference_mid_stft = _stft(reference_mid, SR)
    reference_side_stft = _stft(reference_side, SR)
    side_transfer, _ = _estimate_complex_transfer(
        mixture_side_stft, reference_side_stft, (1.0, 5.0)
    )
    side_prediction_mid = side_transfer * reference_mid_stft
    scalar_prediction_mid = _stft(gain * reference_mid, SR)

    sigma = (1.5, 3.0)
    mid_power = scipy.ndimage.gaussian_filter(
        np.abs(reference_mid_stft) ** 2, sigma
    )
    side_power = scipy.ndimage.gaussian_filter(
        np.abs(reference_side_stft) ** 2, sigma
    )
    side_support = side_power / (mid_power + side_power + 1e-14)
    side_weight = np.clip(side_support / 0.10, 0.0, 1.0)
    prediction_mid_stft = (
        side_weight * side_prediction_mid
        + (1.0 - side_weight) * scalar_prediction_mid
    )
    prediction_side_stft = side_transfer * reference_side_stft
    prediction_mid = _istft(
        prediction_mid_stft, len(mixture_mid), SR
    )
    prediction_side = _istft(
        prediction_side_stft, len(mixture_side), SR
    )
    return (
        (mixture_mid - prediction_mid).astype(np.float32),
        (mixture_side - prediction_side).astype(np.float32),
        prediction_mid.astype(np.float32),
    )


def one_side_iteration(
    residual_mid: np.ndarray,
    residual_side: np.ndarray,
    prediction_mid: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.75,
    max_relative: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    reference_mid = reference.mean(axis=1)
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    residual_side_stft = _stft(residual_side, SR)
    reference_mid_stft = _stft(reference_mid, SR)
    reference_side_stft = _stft(reference_side, SR)
    prediction_mid_stft = _stft(prediction_mid, SR)

    delta, coherence = _estimate_complex_transfer(
        residual_side_stft, reference_side_stft, (1.0, 5.0)
    )
    confidence = scipy.ndimage.gaussian_filter(
        np.clip((coherence - 0.20) / 0.50, 0.0, 1.0),
        sigma=(0.5, 1.0),
    )
    mid_power = scipy.ndimage.gaussian_filter(
        np.abs(reference_mid_stft) ** 2, (1.5, 3.0)
    )
    side_power = scipy.ndimage.gaussian_filter(
        np.abs(reference_side_stft) ** 2, (1.5, 3.0)
    )
    side_support = side_power / (mid_power + side_power + 1e-14)
    side_weight = np.clip(side_support / 0.10, 0.0, 1.0)

    correction_mid_stft = (
        side_weight * confidence * delta * reference_mid_stft
    )
    correction_side_stft = confidence * delta * reference_side_stft
    magnitude = np.abs(correction_mid_stft)
    correction_mid_stft *= np.minimum(
        1.0,
        max_relative * np.abs(prediction_mid_stft) / (magnitude + 1e-12),
    )
    correction_mid = alpha * _istft(
        correction_mid_stft, len(residual_mid), SR
    )
    correction_side = alpha * _istft(
        correction_side_stft, len(residual_side), SR
    )
    before_side = base.rms(residual_side)
    after_side = base.rms(residual_side - correction_side)
    diagnostics = {
        "mid_correction_ratio": base.rms(correction_mid)
        / (base.rms(residual_mid) + 1e-12),
        "side_improvement": 1.0 - after_side / (before_side + 1e-12),
        "coherence_median": float(np.median(coherence)),
    }
    return (
        (residual_mid - correction_mid).astype(np.float32),
        (residual_side - correction_side).astype(np.float32),
        (prediction_mid + correction_mid).astype(np.float32),
        diagnostics,
    )


def one_mid_iteration(
    residual_mid: np.ndarray,
    prediction_mid: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float,
    coherence_floor: float,
    max_relative: float,
) -> tuple[np.ndarray, dict[str, float]]:
    reference_mid_stft = _stft(reference.mean(axis=1), SR)
    residual_stft = _stft(residual_mid, SR)
    prediction_stft = _stft(prediction_mid, SR)
    delta, coherence = _estimate_complex_transfer(
        residual_stft, reference_mid_stft, (1.0, 10.0)
    )
    confidence = np.clip(
        (coherence - coherence_floor) / (0.95 - coherence_floor),
        0.0,
        1.0,
    )
    residual_power = scipy.ndimage.gaussian_filter(
        np.abs(residual_stft) ** 2, (1.5, 3.0)
    )
    prediction_power = scipy.ndimage.gaussian_filter(
        np.abs(prediction_stft) ** 2, (1.5, 3.0)
    )
    # Only revisit bins in which the conservative model already established
    # that removable media dominates. This prevents a new Mid model from
    # freely fitting foreground-only regions.
    explained = prediction_power / (
        prediction_power + residual_power + 1e-14
    )
    confidence *= np.clip((explained - 0.60) / 0.25, 0.0, 1.0)
    confidence = scipy.ndimage.gaussian_filter(
        confidence, sigma=(0.5, 1.0)
    )
    correction_stft = confidence * delta * reference_mid_stft
    magnitude = np.abs(correction_stft)
    correction_stft *= np.minimum(
        1.0,
        max_relative * np.abs(prediction_stft) / (magnitude + 1e-12),
    )
    correction = alpha * _istft(
        correction_stft, len(residual_mid), SR
    )
    return (residual_mid - correction).astype(np.float32), {
        "mid_correction_ratio": base.rms(correction)
        / (base.rms(residual_mid) + 1e-12),
        "mid_coherence_median": float(np.median(coherence)),
    }


def build_scenarios() -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    reference = _decode_stereo(
        str(base.ASSETS / "sr.webm"), 318.467, DURATION, SR
    )[:FRAMES]
    reference = base.scale_rms(reference, 0.18)
    voice = _decode_stereo(
        str(base.ASSETS / "ya.mp3"), 3.0, DURATION, SR
    )[:FRAMES]
    voice_mid = base.scale_rms(voice.mean(axis=1), 0.10)
    foreground_voice = np.column_stack([voice_mid, voice_mid])
    unrelated = _decode_stereo(
        str(base.ASSETS / "sr.webm"), 900.0, DURATION, SR
    )[:FRAMES]
    unrelated = base.scale_rms(unrelated, 0.07)
    time = np.arange(FRAMES) / SR

    common = base.common_filter(reference, [0.62, 0.18, -0.08, 0.04])
    moving = base.common_filter(reference, [0.70, 0.12, -0.04])
    moving *= (
        0.45 + 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.37 * time))
    )[:, None]
    switched = np.empty_like(reference)
    middle = FRAMES // 2
    switched[:middle] = base.common_filter(
        reference[:middle], [0.72, 0.16, -0.05]
    )
    switched[middle:] = base.common_filter(
        reference[middle:], [0.42, -0.20, 0.13, 0.08]
    )
    linear = base.common_filter(reference, [0.82, 0.10, -0.05])
    clipped = (0.24 * np.tanh(linear / 0.24)).astype(np.float32)
    asymmetric = np.column_stack(
        [
            scipy.signal.lfilter([0.70, 0.12, -0.04], [1.0], reference[:, 0]),
            scipy.signal.lfilter([0.52, -0.08, 0.10], [1.0], reference[:, 1]),
        ]
    ).astype(np.float32)
    return reference, {
        "common_fir": (common, foreground_voice),
        "moving_gain": (moving, foreground_voice),
        "path_switch": (switched, foreground_voice),
        "soft_clip": (clipped, foreground_voice),
        "asymmetric_lr": (asymmetric, foreground_voice),
        "side_doubletalk": (common, foreground_voice + unrelated),
        "music_doubletalk": (common, unrelated),
        "voice_only": (np.zeros_like(reference), foreground_voice),
        "music_only": (np.zeros_like(reference), unrelated),
    }


def safe_metrics(
    output: np.ndarray, target: np.ndarray, media_mid: np.ndarray
) -> dict[str, float]:
    if base.rms(media_mid) < 1e-9:
        media_mid = np.full_like(target, 0.18)
    return base.metrics(output, target, media_mid)


def main() -> None:
    reference, scenarios = build_scenarios()
    results = {}
    for name, (media, foreground) in scenarios.items():
        mixture = media + foreground
        target = foreground.mean(axis=1)
        media_mid = media.mean(axis=1)
        current, current_side, current_prediction = base.first_pass(
            mixture, reference
        )
        residual_mid, residual_side, prediction = conservative_first_pass(
            mixture, reference
        )
        scenario = {
            "current_strength0": safe_metrics(current, target, media_mid),
            "conservative_0": safe_metrics(
                residual_mid, target, media_mid
            ),
        }
        current_loop_mid = current
        current_loop_side = current_side
        current_loop_prediction = current_prediction
        for iteration in range(1, 7):
            (
                current_loop_mid,
                current_loop_side,
                current_loop_prediction,
                current_diagnostics,
            ) = one_side_iteration(
                current_loop_mid,
                current_loop_side,
                current_loop_prediction,
                reference,
            )
            scenario[f"current_loop_{iteration}"] = {
                **safe_metrics(current_loop_mid, target, media_mid),
                **current_diagnostics,
            }
        staged_mid = None
        for iteration in range(1, 7):
            residual_mid, residual_side, prediction, diagnostics = (
                one_side_iteration(
                    residual_mid, residual_side, prediction, reference
                )
            )
            scenario[f"conservative_{iteration}"] = {
                **safe_metrics(residual_mid, target, media_mid),
                **diagnostics,
            }
            if iteration == 2:
                staged_mid, mid_diagnostics = one_mid_iteration(
                    residual_mid,
                    prediction,
                    reference,
                    alpha=0.5,
                    coherence_floor=0.70,
                    max_relative=0.15,
                )
                scenario["staged"] = {
                    **safe_metrics(staged_mid, target, media_mid),
                    **mid_diagnostics,
                }
        results[name] = scenario

    real_mixture = _decode_stereo(
        str(base.ASSETS / "miyako.webm"), 2200.192, DURATION, SR
    )
    real_reference = _decode_stereo(
        str(base.ASSETS / "sr.webm"), 2118.659, DURATION, SR
    )
    frames = min(len(real_mixture), len(real_reference))
    real_mixture = real_mixture[:frames]
    real_reference = real_reference[:frames]
    real_mid, real_side, real_prediction = conservative_first_pass(
        real_mixture, real_reference
    )
    real_iterations = {
        "0": {"output_rms": base.rms(real_mid)}
    }
    for iteration in range(1, 7):
        real_mid, real_side, real_prediction, diagnostics = one_side_iteration(
            real_mid, real_side, real_prediction, real_reference
        )
        real_iterations[str(iteration)] = {
            "output_rms": base.rms(real_mid),
            **diagnostics,
        }
    results["real_middle"] = real_iterations

    path = ROOT / "iterative-results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("strength=0 suppression / foreground, then conservative loop 0..6")
    for name, scenario in results.items():
        if name == "real_middle":
            continue
        entries = [scenario["current_strength0"]] + [
            scenario[f"conservative_{index}"] for index in range(7)
        ]
        values = " ".join(
            f"{entry['suppression_db']:6.2f}/{entry['foreground_db']:5.2f}"
            for entry in entries
        )
        print(f"{name:18} {values}")
    print("\nreal middle per-iteration correction ratio / side improvement")
    for key, value in real_iterations.items():
        if key == "0":
            print(f"{key}: initial RMS {value['output_rms']:.6f}")
        else:
            print(
                f"{key}: {value['mid_correction_ratio']:.6f} /"
                f" {value['side_improvement']:.6f}"
            )
    print("\nstaged: conservative + two Side passes + one gated Mid pass")
    for name, scenario in results.items():
        if name == "real_middle":
            continue
        current = scenario["current_strength0"]
        staged = scenario["staged"]
        print(
            f"{name:18} {current['suppression_db']:6.2f}/"
            f"{current['foreground_db']:5.2f} ->"
            f" {staged['suppression_db']:6.2f}/"
            f"{staged['foreground_db']:5.2f}"
        )
    print("\ncurrent strength=0 model followed by repeated Side passes")
    for name, scenario in results.items():
        if name == "real_middle":
            continue
        entries = [scenario["current_strength0"]] + [
            scenario[f"current_loop_{index}"] for index in range(1, 7)
        ]
        print(
            f"{name:18} "
            + " ".join(
                f"{entry['suppression_db']:6.2f}/{entry['foreground_db']:5.2f}"
                for entry in entries
            )
        )


if __name__ == "__main__":
    main()
