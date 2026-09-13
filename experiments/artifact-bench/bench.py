"""Ground-truth benchmark for the cancellation stage on real media.

The reference is a window of ``sr.webm``; the mixture is that window pushed
through a synthetic broadcast chain (EQ, balance, 0.1% speed drift, gain
envelope with a ducking dip, Opus re-encode) and added to a stretch of
``miyako.webm`` that carries no reference media, so the foreground is real
host audio and the residual against it is measurable.

Usage:

    python experiments/artifact-bench/bench.py --label baseline
    python experiments/artifact-bench/bench.py --label cubic --codec-mix

Rows are printed and appended as JSON to ``--out``; the same fixtures are
regenerated deterministically, so two labels compare the code, not the data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio_overlap_removal import AlignmentSegment, process_audio  # noqa: E402
from audio_overlap_removal.media import _decode_stereo, _tool  # noqa: E402
from audio_overlap_removal.result import _RunResult  # noqa: E402

SR = 48_000
MIXTURE = ROOT / "extra-asset" / "miyako.webm"
REFERENCE = ROOT / "extra-asset" / "sr.webm"
LEAD_SEC = 5.0
SPEED = 1.001
BANDS = ((0.0, 1_000.0), (1_000.0, 4_000.0), (4_000.0, 8_000.0), (8_000.0, 20_000.0))


def opus_roundtrip(audio: np.ndarray, bitrate: str, workdir: Path, name: str) -> np.ndarray:
    wav = workdir / f"{name}.wav"
    opus = workdir / f"{name}.opus"
    sf.write(wav, audio, SR, subtype="FLOAT")
    subprocess.run(
        [
            _tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(wav),
            "-c:a", "libopus", "-b:a", bitrate, "-vbr", "on", str(opus),
        ],
        check=True,
        capture_output=True,
    )
    decoded = _decode_stereo(str(opus), 0.0, len(audio) / SR + 1.0, SR, 2)
    if len(decoded) < len(audio):
        decoded = np.pad(decoded, ((0, len(audio) - len(decoded)), (0, 0)))
    return decoded[: len(audio)]


CHAIN_STAGES = ("eq", "balance", "drift", "gain", "codec")


def broadcast_chain(
    reference: np.ndarray, workdir: Path, name: str, stages: set[str]
) -> np.ndarray:
    """Turn the clean reference window into what a stream would carry.

    ``stages`` selects which degradations apply, so the residual can be
    attributed to one of them at a time.
    """
    shaped = reference.astype(np.float64)
    if "eq" in stages:
        # Gentle, asymmetric EQ.
        taps = scipy.signal.firwin2(
            63, [0.0, 0.05, 0.3, 0.6, 1.0], [1.0, 1.1, 0.9, 1.05, 0.8]
        )
        shaped = scipy.signal.lfilter(taps, [1.0], shaped, axis=0)
    if "balance" in stages:
        shaped[:, 0] *= 1.03
        shaped[:, 1] *= 0.97
    if "drift" in stages:
        # 0.1% slower playback: every sample position becomes fractional.
        shaped = scipy.signal.resample_poly(shaped, 1001, 1000, axis=0)
    else:
        shaped = shaped * 1.0
    t = np.arange(len(shaped)) / SR
    gain = np.full(len(shaped), 0.55)
    if "gain" in stages:
        # Fader movement and a ducking dip in the middle.
        gain = 0.55 + 0.15 * np.sin(2.0 * np.pi * t / 7.0)
        dip = np.clip((t - 25.0) / 0.2, 0.0, 1.0) * np.clip((37.0 - t) / 0.2, 0.0, 1.0)
        gain *= 1.0 - 0.5 * dip
    shaped *= gain[:, np.newaxis]
    if "codec" not in stages:
        return shaped.astype(np.float32)
    return opus_roundtrip(shaped.astype(np.float32), "96k", workdir, name)


def band_energy(x: np.ndarray, low: float, high: float) -> float:
    if low <= 0.0:
        sos = scipy.signal.butter(4, high, btype="lowpass", fs=SR, output="sos")
    elif high >= 0.49 * SR:
        sos = scipy.signal.butter(4, low, btype="highpass", fs=SR, output="sos")
    else:
        sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=SR, output="sos")
    y = scipy.signal.sosfiltfilt(sos, x)
    return float(np.sum(y * y))


def db(numerator: float, denominator: float) -> float:
    return 10.0 * np.log10((numerator + 1e-20) / (denominator + 1e-20))


def decompose(residual: np.ndarray, foreground: np.ndarray, media: np.ndarray) -> dict:
    """Split the residual into foreground gain error, media leak and the rest."""
    window = SR
    fg_change = []
    fg_energy = []
    leak = 0.0
    rest = 0.0
    for start in range(0, len(residual) - window + 1, window):
        sl = slice(start, start + window)
        basis = np.column_stack([foreground[sl], media[sl]])
        coef, *_ = np.linalg.lstsq(basis, residual[sl], rcond=None)
        fg_change.append(20.0 * np.log10(abs(1.0 + coef[0]) + 1e-9))
        fg_energy.append(float(np.sum(foreground[sl] ** 2)))
        leak += float(np.sum((coef[1] * media[sl]) ** 2))
        rest += float(np.sum((residual[sl] - basis @ coef) ** 2))
    fg_change = np.array(fg_change)
    fg_energy = np.array(fg_energy)
    # Quiet host seconds are where the silence cleanup bites hardest, but a
    # 6 dB loss on near-silence is not the same risk as 6 dB on speech.
    loud = fg_energy >= np.median(fg_energy)
    return {
        "foreground_gain_db_median": float(np.median(fg_change)),
        "foreground_gain_db_p05": float(np.percentile(fg_change, 5)),
        "foreground_gain_db_p05_loud": float(np.percentile(fg_change[loud], 5)),
        "leak_energy": leak,
        "incoherent_energy": rest,
    }


def run_case(
    media_start: float,
    duration: float,
    foreground_start: float,
    codec_mix: bool,
    strength: float,
    workdir: Path,
    stages: set[str],
    adaptive_time_warp: bool = True,
    keep_result: bool = False,
) -> dict:
    reference = _decode_stereo(
        str(REFERENCE), media_start - LEAD_SEC, duration + 2 * LEAD_SEC, SR, 2
    )
    media_clean = reference[int(LEAD_SEC * SR) : int((LEAD_SEC + duration) * SR)]
    media = broadcast_chain(media_clean, workdir, f"media-{int(media_start)}", stages)
    speed = SPEED if "drift" in stages else 1.0
    total = int((duration * speed + 2 * LEAD_SEC) * SR)
    foreground = _decode_stereo(str(MIXTURE), foreground_start, total / SR + 1.0, SR, 2)[:total]
    foreground = foreground * (
        np.sqrt(np.mean(media[:, 0] ** 2 + media[:, 1] ** 2))
        / (np.sqrt(np.mean(foreground[:, 0] ** 2 + foreground[:, 1] ** 2)) + 1e-12)
    )
    mixture = foreground.copy()
    lead = int(LEAD_SEC * SR)
    mixture[lead : lead + len(media)] += media
    if codec_mix:
        mixture = opus_roundtrip(mixture.astype(np.float32), "128k", workdir, "mixture")
    mixture_path = workdir / "mixture.wav"
    reference_path = workdir / "reference.wav"
    output_path = workdir / "output.wav"
    sf.write(mixture_path, mixture.astype(np.float32), SR, subtype="FLOAT")
    sf.write(reference_path, reference.astype(np.float32), SR, subtype="FLOAT")

    media_end = LEAD_SEC + len(media) / SR
    slope = 1.0 - 1.0 / speed
    center = 0.5 * (LEAD_SEC + media_end)
    segment = AlignmentSegment(
        mixture_start=LEAD_SEC,
        mixture_end=media_end,
        offset_sec=(center - LEAD_SEC) * slope,
        median_score=1.0,
        offset_slope=slope,
    )
    started = time.perf_counter()
    result = _RunResult(argv=None) if keep_result else None
    process_audio(
        str(mixture_path),
        str(reference_path),
        str(output_path),
        alignment_segments=[segment],
        strength=strength,
        adaptive_time_warp=adaptive_time_warp,
        workers=1,
        _result=result,
    )
    elapsed = time.perf_counter() - started
    if result is not None:
        result.dump(workdir / "result.json", "complete")
    output, _ = sf.read(output_path, dtype="float32")

    margin = int(2.0 * SR)
    sl = slice(lead + margin, lead + len(media) - margin)
    out_mid = 0.5 * (output[sl, 0] + output[sl, 1]).astype(np.float64)
    fg_mid = 0.5 * (foreground[sl, 0] + foreground[sl, 1]).astype(np.float64)
    media_mid = 0.5 * (media[margin:-margin, 0] + media[margin:-margin, 1]).astype(np.float64)
    out_side = 0.5 * (output[sl, 0] - output[sl, 1]).astype(np.float64)
    fg_side = 0.5 * (foreground[sl, 0] - foreground[sl, 1]).astype(np.float64)
    media_side = 0.5 * (media[margin:-margin, 0] - media[margin:-margin, 1]).astype(np.float64)
    residual = out_mid - fg_mid
    side_residual = out_side - fg_side
    media_energy = float(np.sum(media_mid**2))
    parts = decompose(residual, fg_mid, media_mid)
    row = {
        "media_start": media_start,
        "chain": sorted(stages),
        "codec_mix": codec_mix,
        "strength": strength,
        "elapsed_sec": elapsed,
        "mid_suppression_db": db(media_energy, float(np.sum(residual**2))),
        "side_suppression_db": db(float(np.sum(media_side**2)), float(np.sum(side_residual**2))),
        "leak_db": db(media_energy, parts["leak_energy"]),
        "incoherent_db": db(media_energy, parts["incoherent_energy"]),
        "foreground_gain_db_median": parts["foreground_gain_db_median"],
        "foreground_gain_db_p05": parts["foreground_gain_db_p05"],
        "foreground_gain_db_p05_loud": parts["foreground_gain_db_p05_loud"],
        "bands_db": {
            f"{int(low)}-{int(high)}": db(
                band_energy(media_mid, low, high), band_energy(residual, low, high)
            )
            for low, high in BANDS
        },
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=str(Path(__file__).with_name("results.jsonl")))
    parser.add_argument("--media-start", type=float, nargs="+", default=[400.0, 1500.0])
    parser.add_argument("--foreground-start", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--strength", type=float, nargs="+", default=[1.0])
    parser.add_argument("--codec-mix", action="store_true")
    parser.add_argument("--both", action="store_true", help="clean and codec mixtures")
    parser.add_argument("--keep", help="directory to keep fixtures and outputs")
    parser.add_argument(
        "--chain",
        default=",".join(CHAIN_STAGES),
        help=(
            "comma-separated degradation stages applied to the media: any of "
            f"{', '.join(CHAIN_STAGES)}; 'none' for the clean reference"
        ),
    )
    parser.add_argument("--disable-adaptive-warp", action="store_true")
    args = parser.parse_args()
    stages = set() if args.chain == "none" else set(args.chain.split(","))
    unknown = stages - set(CHAIN_STAGES)
    if unknown:
        parser.error(f"unknown chain stages: {sorted(unknown)}")
    codec_options = [False, True] if args.both else [args.codec_mix]
    rows = []
    for media_start in args.media_start:
        for codec_mix in codec_options:
            for strength in args.strength:
                if args.keep:
                    workdir = Path(args.keep) / f"{int(media_start)}-{'opus' if codec_mix else 'clean'}-s{strength:g}"
                    workdir.mkdir(parents=True, exist_ok=True)
                    row = run_case(media_start, args.duration, args.foreground_start, codec_mix, strength, workdir, stages, not args.disable_adaptive_warp, True)
                else:
                    with tempfile.TemporaryDirectory() as temporary:
                        row = run_case(media_start, args.duration, args.foreground_start, codec_mix, strength, Path(temporary), stages, not args.disable_adaptive_warp)
                row["label"] = args.label
                rows.append(row)
                bands = " ".join(f"{k}:{v:5.1f}" for k, v in row["bands_db"].items())
                chain = "+".join(sorted(stages)) or "none"
                print(
                    f"{args.label:14s} media={media_start:6.0f} {chain:20s} {'opus ' if codec_mix else 'clean'} s={strength:g} "
                    f"mid {row['mid_suppression_db']:5.2f} dB  side {row['side_suppression_db']:5.2f} dB  "
                    f"leak {row['leak_db']:5.1f}  incoh {row['incoherent_db']:5.1f}  "
                    f"fg {row['foreground_gain_db_median']:+5.2f}/{row['foreground_gain_db_p05']:+5.2f}/{row['foreground_gain_db_p05_loud']:+5.2f} dB  "
                    f"bands[{bands}]  {row['elapsed_sec']:.1f}s",
                    flush=True,
                )
    with open(args.out, "a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
