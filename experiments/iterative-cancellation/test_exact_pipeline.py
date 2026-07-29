"""End-to-end test: linear model, adaptive Side loop, one final cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage

import explore_iterative as iterative
import explore_two_pass as base
from audio_overlap_removal.cancellation import (
    _estimate_gain_envelope,
    _istft,
    _smooth_power,
    _stft,
)

SR = 16_000
base.SR = SR
iterative.SR = SR
ROOT = Path(__file__).parent


def foreground_guard(mixture: np.ndarray, reference: np.ndarray) -> float:
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    _, correlations = _estimate_gain_envelope(
        mixture_side, reference_side, SR
    )
    valid = correlations[np.isfinite(correlations)]
    median = float(np.median(valid)) if len(valid) else 0.0
    return float(np.clip((median - 0.20) / 0.20, 0.0, 1.0))


def final_cleanup(
    residual_mid: np.ndarray,
    residual_side: np.ndarray,
    prediction_mid: np.ndarray,
    prediction_side: np.ndarray,
    reference: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply the production strength-1 cleanup masks exactly once."""
    if strength <= 0.0:
        return residual_mid.astype(np.float32, copy=True)
    mid_residual_stft = _stft(residual_mid, SR)
    side_residual_stft = _stft(residual_side, SR)
    predicted_mid_stft = _stft(prediction_mid, SR)
    predicted_side_stft = _stft(prediction_side, SR)
    reference_mid_stft = _stft(reference.mean(axis=1), SR)
    reference_side_stft = _stft(
        0.5 * (reference[:, 0] - reference[:, 1]), SR
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
    sigma = (1.5, 3.0)

    if strength > 0:
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
                1.0
                - strength
                * estimated_artifact_power
                / (mid_power + 1e-14),
                0.2**2,
                1.0,
            )
        )
        cleanup_gain = scipy.ndimage.gaussian_filter(
            cleanup_gain, sigma=(0.7, 1.0)
        )
        mid_residual_stft *= cleanup_gain

        residual_power = _smooth_power(mid_residual_stft, sigma)
        predicted_power = _smooth_power(predicted_mid_stft, sigma)
        center_activity = np.clip(
            mid_reference_power
            / (
                1.5
                * np.median(mid_reference_power, axis=1, keepdims=True)
                + 1e-14
            ),
            0.0,
            1.0,
        )
        center_dominance = np.clip(
            (0.25 - side_support) / 0.25, 0.0, 1.0
        )
        media_dominance = predicted_power / (
            predicted_power + residual_power + 1e-14
        )
        center_mask = (
            center_activity * center_dominance * media_dominance
        )
        center_gain = np.sqrt(
            np.clip(1.0 - strength * center_mask, 0.10, 1.0)
        )
        center_gain = scipy.ndimage.gaussian_filter(
            center_gain, sigma=(0.7, 1.0)
        )
        mid_residual_stft *= center_gain

        residual_power = _smooth_power(mid_residual_stft, sigma)
        predicted_power = _smooth_power(predicted_mid_stft, sigma)
        frequencies = np.fft.rfftfreq(2_048, d=1.0 / SR)
        band = (frequencies >= 100.0) & (
            frequencies <= min(8_000.0, 0.48 * SR)
        )
        unexplained = np.sum(residual_power[band], axis=0)
        explained = np.sum(predicted_power[band], axis=0)
        unexplained_ratio = unexplained / (
            unexplained + explained + 1e-14
        )
        foreground_presence = np.clip(
            (unexplained_ratio - 0.10) / 0.35, 0.0, 1.0
        )
        foreground_presence = scipy.ndimage.gaussian_filter1d(
            foreground_presence, sigma=3.0
        )
        foreground_presence = scipy.ndimage.maximum_filter1d(
            foreground_presence, size=21, mode="nearest"
        )
        foreground_absence = 1.0 - foreground_presence
        broad_activity = np.clip(
            mid_reference_power
            / (
                0.75
                * np.median(mid_reference_power, axis=1, keepdims=True)
                + 1e-14
            ),
            0.0,
            1.0,
        )
        broad_center = np.clip(
            (0.50 - side_support) / 0.50, 0.0, 1.0
        )
        media_dominance = np.sqrt(
            predicted_power / (
                predicted_power + residual_power + 1e-14
            )
        )
        silence_mask = (
            foreground_absence[np.newaxis, :]
            * broad_activity
            * broad_center
            * media_dominance
        )
        silence_gain = np.sqrt(
            np.clip(1.0 - strength * silence_mask, 0.03, 1.0)
        )
        silence_gain = scipy.ndimage.gaussian_filter(
            silence_gain, sigma=(0.7, 1.0)
        )
        mid_residual_stft *= silence_gain

    return _istft(mid_residual_stft, len(residual_mid), SR)


def structured_output(
    mixture: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    # center_strength is a linear-model parameter, so retain the strength-1
    # value while disabling all nonlinear cleanup during model estimation.
    residual_mid, residual_side, prediction_mid = base.first_pass(
        mixture, reference, center_strength=1.0
    )
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    gain, _ = _estimate_gain_envelope(
        mixture_side, reference_side, SR
    )
    scalar_side_error = base.rms(
        mixture_side - gain * reference_side
    ) / (base.rms(mixture_side) + 1e-12)
    prediction_side = mixture_side - residual_side
    applied = 0
    last_diagnostics = {
        "mid_correction_ratio": 0.0,
        "side_improvement": 0.0,
    }
    for _ in range(2):
        (
            candidate_mid,
            candidate_side,
            candidate_prediction,
            diagnostics,
        ) = iterative.one_side_iteration(
            residual_mid,
            residual_side,
            prediction_mid,
            reference,
        )
        if (
            diagnostics["mid_correction_ratio"] < 0.001
            or diagnostics["side_improvement"] < 0.0015
        ):
            break
        prediction_side += residual_side - candidate_side
        residual_mid = candidate_mid
        residual_side = candidate_side
        prediction_mid = candidate_prediction
        last_diagnostics = diagnostics
        applied += 1
    guard = foreground_guard(mixture, reference)
    if applied == 0:
        # A rejected loop must be bit-for-bit behaviorally equivalent to the
        # current path; do not force an unnecessary STFT reconstruction.
        output = base.production_output(mixture, reference)
    elif scalar_side_error < 0.10:
        # Preserve the production fast path: a nearly exact digital mix is
        # already handled by subtraction and intentionally skips all cleanup.
        output = residual_mid
    else:
        output = final_cleanup(
            residual_mid,
            residual_side,
            prediction_mid,
            prediction_side,
            reference,
            guard,
        )
    return output, {
        "iterations": float(applied),
        "guard": guard,
        "scalar_side_error": scalar_side_error,
        **last_diagnostics,
    }


def main() -> None:
    reference, scenarios = iterative.build_scenarios()
    results = {}
    print("current strength=1 -> exact structured pipeline")
    for name, (media, foreground) in scenarios.items():
        mixture = media + foreground
        target = foreground.mean(axis=1)
        media_mid = media.mean(axis=1)
        current = base.production_output(mixture, reference)
        structured, diagnostics = structured_output(
            mixture, reference
        )
        current_metrics = iterative.safe_metrics(
            current, target, media_mid
        )
        structured_metrics = iterative.safe_metrics(
            structured, target, media_mid
        )
        results[name] = {
            "current": current_metrics,
            "structured": structured_metrics,
            "diagnostics": diagnostics,
        }
        print(
            f"{name:18} "
            f"{current_metrics['suppression_db']:6.2f}/"
            f"{current_metrics['foreground_db']:5.2f} -> "
            f"{structured_metrics['suppression_db']:6.2f}/"
            f"{structured_metrics['foreground_db']:5.2f} "
            f"loops={int(diagnostics['iterations'])}"
        )
    path = ROOT / "exact-pipeline-results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
