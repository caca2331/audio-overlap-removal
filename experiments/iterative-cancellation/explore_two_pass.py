"""Offline exploration of a constrained residual-reference second pass."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

PROJECT = Path(__file__).parents[2]
ASSETS = Path(
    os.environ.get(
        "AUDIO_OVERLAP_ASSETS",
        PROJECT.parent
        / "Agent-Task-Playground"
        / "audio-removal"
        / "extra-asset",
    )
)
sys.path.insert(0, str(PROJECT))

from audio_overlap_removal.cancellation import (  # noqa: E402
    _complex_reference_cancel,
    _estimate_complex_transfer,
    _estimate_gain_envelope,
    _istft,
    _smooth_power,
    _stft,
)
from audio_overlap_removal.media import _decode_stereo  # noqa: E402

SR = 16_000
DURATION = 14.0
FRAMES = int(SR * DURATION)


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def scale_rms(value: np.ndarray, wanted: float) -> np.ndarray:
    return (value * (wanted / (rms(value) + 1e-12))).astype(np.float32)


def common_filter(stereo: np.ndarray, taps: list[float]) -> np.ndarray:
    return scipy.signal.lfilter(taps, [1.0], stereo, axis=0).astype(np.float32)


def first_pass(
    mixture: np.ndarray,
    reference: np.ndarray,
    *,
    center_strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mixture_mid = mixture.mean(axis=1)
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_mid = reference.mean(axis=1)
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    gain, _ = _estimate_gain_envelope(mixture_side, reference_side, SR)
    residual_mid, residual_side, _ = _complex_reference_cancel(
        mixture_mid,
        mixture_side,
        reference_mid,
        reference_side,
        gain,
        cleanup_strength=0.0,
        center_strength=center_strength,
        sr=SR,
    )
    return residual_mid, residual_side, mixture_mid - residual_mid


def production_output(mixture: np.ndarray, reference: np.ndarray) -> np.ndarray:
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
    return output


def second_pass(
    residual_mid: np.ndarray,
    residual_side: np.ndarray,
    first_prediction: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float,
    mid_share: float,
    max_relative: float,
) -> np.ndarray:
    """Estimate only a bounded correction to the first-pass transfer."""
    reference_mid = reference.mean(axis=1)
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    residual_mid_stft = _stft(residual_mid, SR)
    residual_side_stft = _stft(residual_side, SR)
    reference_mid_stft = _stft(reference_mid, SR)
    reference_side_stft = _stft(reference_side, SR)
    prediction_stft = _stft(first_prediction, SR)

    side_delta, side_coherence = _estimate_complex_transfer(
        residual_side_stft, reference_side_stft, (1.0, 5.0)
    )
    mid_delta, mid_coherence = _estimate_complex_transfer(
        residual_mid_stft, reference_mid_stft, (1.0, 10.0)
    )

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

    side_confidence = np.clip((side_coherence - 0.20) / 0.50, 0.0, 1.0)
    side_confidence = scipy.ndimage.gaussian_filter(
        side_confidence, sigma=(0.5, 1.0)
    )
    side_correction = side_confidence * side_delta * reference_mid_stft

    # The centered fallback is deliberately much stricter. It is enabled only
    # where the residual remains highly reference-coherent and the first pass
    # already explains most energy, then capped relative to that prediction.
    residual_power = _smooth_power(residual_mid_stft, support_sigma)
    prediction_power = _smooth_power(prediction_stft, support_sigma)
    explained = prediction_power / (
        prediction_power + residual_power + 1e-14
    )
    mid_confidence = np.clip((mid_coherence - 0.70) / 0.25, 0.0, 1.0)
    mid_confidence *= np.clip((explained - 0.55) / 0.30, 0.0, 1.0)
    mid_confidence = scipy.ndimage.gaussian_filter(
        mid_confidence, sigma=(0.5, 1.0)
    )
    mid_correction = mid_confidence * mid_delta * reference_mid_stft

    correction = (
        side_weight * side_correction
        + (1.0 - side_weight) * mid_share * mid_correction
    )
    correction_magnitude = np.abs(correction)
    correction *= np.minimum(
        1.0,
        max_relative * np.abs(prediction_stft) / (correction_magnitude + 1e-12),
    )
    return (
        residual_mid - alpha * _istft(correction, len(residual_mid), SR)
    ).astype(np.float32)


def nonlinear_second_pass(
    residual_mid: np.ndarray,
    residual_side: np.ndarray,
    first_prediction: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.5,
    max_relative: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Fit memoryless odd residual terms from Side and transfer them to Mid."""
    scale = max(float(np.std(reference)), 1e-5)
    normalized = reference / scale
    side_features = np.column_stack(
        [
            0.5 * (normalized[:, 0] ** 3 - normalized[:, 1] ** 3),
            0.5 * (normalized[:, 0] ** 5 - normalized[:, 1] ** 5),
        ]
    )
    mid_features = np.column_stack(
        [
            0.5 * (normalized[:, 0] ** 3 + normalized[:, 1] ** 3),
            0.5 * (normalized[:, 0] ** 5 + normalized[:, 1] ** 5),
        ]
    )
    # Alternating one-second blocks provide a cheap persistence check. Chance
    # correlation or unrelated Side content should not generalize across folds.
    blocks = np.arange(len(residual_side)) // SR
    improvements = []
    for fold in range(2):
        train = blocks % 2 == fold
        valid = ~train
        gram = side_features[train].T @ side_features[train]
        ridge = 1e-4 * np.trace(gram) / len(gram)
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(len(gram)),
            side_features[train].T @ residual_side[train],
        )
        predicted = side_features[valid] @ coefficients
        before = np.sum(np.square(residual_side[valid], dtype=np.float64))
        after = np.sum(
            np.square(residual_side[valid] - predicted, dtype=np.float64)
        )
        improvements.append(1.0 - after / (before + 1e-20))
    validation_improvement = float(min(improvements))
    if validation_improvement < 0.01:
        return residual_mid.copy(), validation_improvement

    gram = side_features.T @ side_features
    ridge = 1e-4 * np.trace(gram) / len(gram)
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        side_features.T @ residual_side,
    )
    correction = mid_features @ coefficients
    correction_stft = _stft(correction.astype(np.float32), SR)
    prediction_stft = _stft(first_prediction, SR)
    magnitude = np.abs(correction_stft)
    correction_stft *= np.minimum(
        1.0,
        max_relative * np.abs(prediction_stft) / (magnitude + 1e-12),
    )
    correction = _istft(correction_stft, len(residual_mid), SR)
    return (residual_mid - alpha * correction).astype(np.float32), validation_improvement


def metrics(
    output: np.ndarray, target: np.ndarray, media_mid: np.ndarray
) -> dict[str, float]:
    start = 2 * SR
    output = output[start:]
    target = target[start:]
    media_mid = media_mid[start:]
    error = output - target
    design = np.column_stack([target, media_mid])
    coefficients, _, _, _ = np.linalg.lstsq(design, output, rcond=None)
    unexplained = output - design @ coefficients
    return {
        "suppression_db": float(
            20 * np.log10(rms(media_mid) / (rms(error) + 1e-12) + 1e-12)
        ),
        "foreground_db": float(
            20 * np.log10(abs(coefficients[0]) + 1e-12)
        ),
        "media_db": float(20 * np.log10(abs(coefficients[1]) + 1e-12)),
        "unexplained_db": float(
            20
            * np.log10(rms(unexplained) / (rms(media_mid) + 1e-12) + 1e-12)
        ),
        "error_rms": rms(error),
    }


def main() -> None:
    reference = _decode_stereo(
        str(ASSETS / "sr.webm"), 318.467, DURATION, SR
    )[:FRAMES]
    reference = scale_rms(reference, 0.18)
    voice = _decode_stereo(
        str(ASSETS / "ya.mp3"), 3.0, DURATION, SR
    )[:FRAMES]
    voice_mid = scale_rms(voice.mean(axis=1), 0.10)
    foreground_voice = np.column_stack([voice_mid, voice_mid])
    unrelated = _decode_stereo(
        str(ASSETS / "sr.webm"), 900.0, DURATION, SR
    )[:FRAMES]
    unrelated = scale_rms(unrelated, 0.07)
    time = np.arange(FRAMES) / SR

    common = common_filter(reference, [0.62, 0.18, -0.08, 0.04])
    moving_gain = common_filter(reference, [0.70, 0.12, -0.04])
    moving_gain *= (
        0.45 + 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.37 * time))
    )[:, None]
    switched = np.empty_like(reference)
    middle = FRAMES // 2
    switched[:middle] = common_filter(
        reference[:middle], [0.72, 0.16, -0.05]
    )
    switched[middle:] = common_filter(
        reference[middle:], [0.42, -0.20, 0.13, 0.08]
    )
    linear = common_filter(reference, [0.82, 0.10, -0.05])
    clipped = (0.24 * np.tanh(linear / 0.24)).astype(np.float32)
    asymmetric = np.column_stack(
        [
            scipy.signal.lfilter([0.70, 0.12, -0.04], [1.0], reference[:, 0]),
            scipy.signal.lfilter([0.52, -0.08, 0.10], [1.0], reference[:, 1]),
        ]
    ).astype(np.float32)

    scenarios = {
        "common_fir": (common, foreground_voice),
        "moving_gain": (moving_gain, foreground_voice),
        "path_switch": (switched, foreground_voice),
        "soft_clip": (clipped, foreground_voice),
        "asymmetric_lr": (asymmetric, foreground_voice),
        "side_doubletalk": (common, foreground_voice + unrelated),
        "music_doubletalk": (common, unrelated),
        "voice_only": (np.zeros_like(reference), foreground_voice),
        "music_only": (np.zeros_like(reference), unrelated),
    }
    settings = [
        (0.25, 0.0, 0.15),
        (0.50, 0.0, 0.15),
        (0.50, 0.10, 0.15),
        (0.50, 0.20, 0.25),
        (0.75, 0.10, 0.25),
    ]
    results: dict[str, dict] = {}
    for name, (media, foreground) in scenarios.items():
        mixture = media + foreground
        baseline, side_residual, prediction = first_pass(mixture, reference)
        target = foreground.mean(axis=1)
        media_mid = media.mean(axis=1)
        # A zero-media scenario has no meaningful "suppression" denominator.
        if rms(media_mid) < 1e-9:
            media_mid = np.full_like(target, rms(reference.mean(axis=1)))
        scenario_result = {"baseline": metrics(baseline, target, media_mid)}
        for alpha, mid_share, max_relative in settings:
            key = f"a{alpha:g}_m{mid_share:g}_c{max_relative:g}"
            refined = second_pass(
                baseline,
                side_residual,
                prediction,
                reference,
                alpha=alpha,
                mid_share=mid_share,
                max_relative=max_relative,
            )
            scenario_result[key] = metrics(refined, target, media_mid)
        nonlinear, validation_improvement = nonlinear_second_pass(
            baseline, side_residual, prediction, reference
        )
        scenario_result["nonlinear"] = {
            **metrics(nonlinear, target, media_mid),
            "validation_improvement": validation_improvement,
        }
        production = production_output(mixture, reference)
        selected_side = second_pass(
            baseline,
            side_residual,
            prediction,
            reference,
            alpha=0.75,
            mid_share=0.0,
            max_relative=0.25,
        )
        production_refined = production - (baseline - selected_side) - (
            baseline - nonlinear
        )
        scenario_result["production"] = metrics(
            production, target, media_mid
        )
        scenario_result["production_two_pass"] = metrics(
            production_refined, target, media_mid
        )
        results[name] = scenario_result

    output_path = Path(__file__).with_name("synthetic-results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_path)
    keys = ["baseline"] + [
        f"a{alpha:g}_m{mid_share:g}_c{max_relative:g}"
        for alpha, mid_share, max_relative in settings
    ] + ["nonlinear"]
    print("scenario".ljust(18), *(key.rjust(18) for key in keys))
    for name, values in results.items():
        print(
            name.ljust(18),
            *(
                f"{values[key]['suppression_db']:7.2f}/"
                f"{values[key]['foreground_db']:6.2f}".rjust(18)
                for key in keys
            ),
        )
    print("\ncurrent strength=1 output versus proposed two-pass")
    for name, values in results.items():
        before = values["production"]
        after = values["production_two_pass"]
        print(
            f"{name:18} suppression {before['suppression_db']:7.2f}"
            f" -> {after['suppression_db']:7.2f} dB,"
            f" foreground {before['foreground_db']:6.2f}"
            f" -> {after['foreground_db']:6.2f} dB"
        )

    # Export one ground-truth case to make numerical regressions audible.
    media, foreground = scenarios["path_switch"]
    mixture = media + foreground
    baseline, side_residual, prediction = first_pass(mixture, reference)
    refined = second_pass(
        baseline,
        side_residual,
        prediction,
        reference,
        alpha=0.5,
        mid_share=0.0,
        max_relative=0.15,
    )
    sf.write(Path(__file__).with_name("synthetic-target.flac"), foreground.mean(1), SR)
    sf.write(Path(__file__).with_name("synthetic-baseline.flac"), baseline, SR)
    sf.write(Path(__file__).with_name("synthetic-refined.flac"), refined, SR)


if __name__ == "__main__":
    main()
