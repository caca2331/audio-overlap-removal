"""In-memory DSP for reference cancellation."""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.signal

from .alignment import _align_reference


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
        # This occurs naturally when a discovered segment reaches beyond a
        # truncated reference. Treat it as a no-match region instead of passing
        # an empty/short vector into the alignment filters.
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
