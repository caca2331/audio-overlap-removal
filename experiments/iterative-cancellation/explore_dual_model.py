"""Explore a B_safe/B_full risk-gated linear prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage

import explore_iterative as iterative
import explore_two_pass as base
import test_exact_pipeline as exact
from audio_overlap_removal.cancellation import (
    _estimate_complex_transfer,
    _estimate_gain_envelope,
    _istft,
    _stft,
)

SR = 16_000
base.SR = SR
iterative.SR = SR
exact.SR = SR
ROOT = Path(__file__).parent


def dual_linear_output(
    mixture: np.ndarray,
    reference: np.ndarray,
    *,
    floor: float,
    coherence_floor: float,
    dominance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Blend the risky full-minus-safe prediction only where evidence agrees."""
    mixture_mid = mixture.mean(axis=1)
    full_residual, side_residual, full_prediction = base.first_pass(
        mixture, reference, center_strength=1.0
    )
    safe_residual, _, safe_prediction = (
        iterative.conservative_first_pass(mixture, reference)
    )
    risky_prediction = full_prediction - safe_prediction

    safe_residual_stft = _stft(safe_residual, SR)
    safe_prediction_stft = _stft(safe_prediction, SR)
    risky_stft = _stft(risky_prediction, SR)
    reference_stft = _stft(reference.mean(axis=1), SR)
    _, coherence = _estimate_complex_transfer(
        safe_residual_stft, reference_stft, (1.0, 10.0)
    )
    sigma = (1.5, 3.0)
    residual_power = scipy.ndimage.gaussian_filter(
        np.abs(safe_residual_stft) ** 2, sigma
    )
    prediction_power = scipy.ndimage.gaussian_filter(
        np.abs(safe_prediction_stft) ** 2, sigma
    )
    dominance = prediction_power / (
        prediction_power + residual_power + 1e-14
    )
    coherence_evidence = np.clip(
        (coherence - coherence_floor) / (0.95 - coherence_floor),
        0.0,
        1.0,
    )
    dominance_evidence = np.clip(
        (dominance - dominance_floor) / (0.90 - dominance_floor),
        0.0,
        1.0,
    )
    evidence = scipy.ndimage.gaussian_filter(
        coherence_evidence * dominance_evidence,
        sigma=(0.5, 1.0),
    )
    gate = floor + (1.0 - floor) * evidence
    accepted_risky = _istft(
        gate * risky_stft, len(mixture_mid), SR
    )
    prediction = safe_prediction + accepted_risky
    residual = mixture_mid - prediction
    diagnostics = {
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.median(gate)),
        "gate_p95": float(np.percentile(gate, 95)),
        "risky_acceptance_rms": base.rms(accepted_risky)
        / (base.rms(risky_prediction) + 1e-12),
        "full_safe_difference_ratio": base.rms(risky_prediction)
        / (base.rms(full_prediction) + 1e-12),
        "full_residual_rms": base.rms(full_residual),
    }
    return (
        residual.astype(np.float32),
        side_residual,
        prediction.astype(np.float32),
        diagnostics,
    )


def complete_output(
    mixture: np.ndarray,
    reference: np.ndarray,
    *,
    floor: float,
    coherence_floor: float,
    dominance_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
    reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
    gain, _ = _estimate_gain_envelope(
        mixture_side, reference_side, SR
    )
    scalar_side_error = base.rms(
        mixture_side - gain * reference_side
    ) / (base.rms(mixture_side) + 1e-12)
    if scalar_side_error < 0.10:
        # Preserve the production digital-mix fast path. The separate
        # Side-residual loop already covers useful refinement in this case.
        return base.production_output(mixture, reference), {
            "fast_path": 1.0,
            "scalar_side_error": scalar_side_error,
        }
    residual, side_residual, prediction, diagnostics = dual_linear_output(
        mixture,
        reference,
        floor=floor,
        coherence_floor=coherence_floor,
        dominance_floor=dominance_floor,
    )
    prediction_side = mixture_side - side_residual
    guard = exact.foreground_guard(mixture, reference)
    output = exact.final_cleanup(
        residual,
        side_residual,
        prediction,
        prediction_side,
        reference,
        guard,
    )
    return output, {
        **diagnostics,
        "guard": guard,
        "fast_path": 0.0,
        "scalar_side_error": scalar_side_error,
    }


def build_extended_scenarios():
    reference, scenarios = iterative.build_scenarios()
    common, voice = scenarios["common_fir"]
    voice_mid = voice.mean(axis=1)
    loud_voice = np.column_stack([2.0 * voice_mid, 2.0 * voice_mid])
    quiet_voice = np.column_stack([0.5 * voice_mid, 0.5 * voice_mid])
    scenarios["common_loud_voice"] = (common, loud_voice)
    scenarios["common_quiet_voice"] = (common, quiet_voice)
    return reference, scenarios


def main() -> None:
    reference, scenarios = build_extended_scenarios()
    settings = [
        (0.00, 0.30, 0.30),
        (0.10, 0.30, 0.30),
        (0.25, 0.30, 0.30),
        (0.25, 0.50, 0.40),
        (0.50, 0.40, 0.35),
        (0.75, 0.40, 0.35),
    ]
    results = {}
    for name, (media, foreground) in scenarios.items():
        mixture = media + foreground
        target = foreground.mean(axis=1)
        media_mid = media.mean(axis=1)
        current = base.production_output(mixture, reference)
        safe_residual, safe_side, safe_prediction = (
            iterative.conservative_first_pass(mixture, reference)
        )
        mixture_side = 0.5 * (mixture[:, 0] - mixture[:, 1])
        reference_side = 0.5 * (reference[:, 0] - reference[:, 1])
        gain, _ = _estimate_gain_envelope(
            mixture_side, reference_side, SR
        )
        scalar_side_error = base.rms(
            mixture_side - gain * reference_side
        ) / (base.rms(mixture_side) + 1e-12)
        if scalar_side_error < 0.10:
            safe_complete = current
        else:
            safe_complete = exact.final_cleanup(
                safe_residual,
                safe_side,
                safe_prediction,
                mixture_side - safe_side,
                reference,
                exact.foreground_guard(mixture, reference),
            )
        scenario = {
            "current": iterative.safe_metrics(
                current, target, media_mid
            ),
            "safe": iterative.safe_metrics(
                safe_complete, target, media_mid
            ),
        }
        for floor, coherence_floor, dominance_floor in settings:
            key = f"f{floor:g}_c{coherence_floor:g}_d{dominance_floor:g}"
            output, diagnostics = complete_output(
                mixture,
                reference,
                floor=floor,
                coherence_floor=coherence_floor,
                dominance_floor=dominance_floor,
            )
            scenario[key] = {
                **iterative.safe_metrics(output, target, media_mid),
                **diagnostics,
            }
        results[name] = scenario

    path = ROOT / "dual-model-results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    keys = ["current", "safe"] + [
        f"f{floor:g}_c{coherence_floor:g}_d{dominance_floor:g}"
        for floor, coherence_floor, dominance_floor in settings
    ]
    print("suppression / foreground dB")
    print("scenario".ljust(20), *(key.rjust(18) for key in keys))
    for name, scenario in results.items():
        print(
            name.ljust(20),
            *(
                f"{scenario[key]['suppression_db']:6.2f}/"
                f"{scenario[key]['foreground_db']:5.2f}".rjust(18)
                for key in keys
            ),
        )


if __name__ == "__main__":
    main()
