"""Verify a built distribution beyond the smoke test in build.py.

    python packaging/verify.py [--bundle DIR] [--no-source-comparison]

Checks, in order:

1. the cancellation path really runs — `--help` and pass-through chunks never
   reach the scipy calls that a bad `EXCLUDES` entry would break;
2. the frozen build agrees with the source install byte for byte;
3. FFmpeg is found through `AOR_FFMPEG_DIR` when nothing is on `PATH`;
4. a missing FFmpeg produces the guidance message rather than a traceback.

Check 2 is only meaningful when both sides run in the same environment. Pass
`--no-source-comparison` when verifying a container-built bundle from the
host, and run the full script inside the build container as well.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
# The checkout itself, so the discovery check can read the package's own list
# of search directories without the package having to be installed.
sys.path[:0] = [str(ROOT / "packaging"), str(ROOT)]
from build import NAME, _platform_tag  # noqa: E402

SR = 8_000
TRUE_OFFSET = 2.0
MATCHED = (4.0, 20.0)
RESIDUAL_CEILING = 0.01


def _write_fixture(directory: Path) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    """Mirror the drifting fixture the test suite uses for offset recovery."""
    rng = np.random.default_rng(1234)
    reference = scipy.signal.lfilter(
        [1.0], [1.0, -0.85], rng.standard_normal((30 * SR, 2)), axis=0
    ).astype(np.float32)
    # Keep peaks well inside full scale so 24-bit output does not clip.
    reference *= 0.12 / np.std(reference)

    frames = 24 * SR
    time = np.arange(frames) / SR
    target = (0.05 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    mixture = np.column_stack([target, target])
    start, end = int(MATCHED[0] * SR), int(MATCHED[1] * SR)
    reference_start = int((MATCHED[0] - TRUE_OFFSET) * SR)
    mixture[start:end] += 0.8 * reference[reference_start : reference_start + end - start]

    mixture_path = directory / "mixture.wav"
    reference_path = directory / "reference.wav"
    segments_path = directory / "segments.json"
    sf.write(mixture_path, mixture, SR, subtype="FLOAT")
    sf.write(reference_path, reference, SR, subtype="FLOAT")
    segments_path.write_text(
        json.dumps(
            [
                {
                    "mixture_start": MATCHED[0],
                    "mixture_end": MATCHED[1],
                    "offset_sec": TRUE_OFFSET,
                    "median_score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    return mixture_path, reference_path, segments_path, mixture, target


def _job_args(fixture: tuple, output: Path) -> list[str]:
    mixture_path, reference_path, segments_path, _, _ = fixture
    return [
        str(mixture_path),
        str(reference_path),
        str(output),
        "--segments",
        str(segments_path),
        "--sample-rate",
        str(SR),
        "--chunk",
        "4",
        "--strength",
        "0",
        "--workers",
        "2",
    ]


def _residual_ratio(written: np.ndarray, mixture: np.ndarray, target: np.ndarray) -> float:
    """How much of the removable reference survives inside the matched span."""
    start = int((MATCHED[0] + 1.0) * SR)
    end = int((MATCHED[1] - 1.0) * SR)
    stereo_target = np.column_stack([target, target])
    residual = written[start:end] - stereo_target[start:end]
    removable = mixture[start:end] - stereo_target[start:end]
    return float(np.sqrt(np.mean(residual**2) / np.mean(removable**2)))


def _check_cancellation(executable: Path, fixture: tuple, directory: Path) -> np.ndarray:
    output = directory / "frozen.wav"
    subprocess.run(
        [str(executable), *_job_args(fixture, output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    written, _ = sf.read(output, dtype="float32")
    _, _, _, mixture, target = fixture

    ratio = _residual_ratio(written, mixture, target)
    if ratio > RESIDUAL_CEILING:
        raise SystemExit(
            f"FAIL cancellation: residual ratio {ratio:.4f} exceeds {RESIDUAL_CEILING}"
        )

    # Pass-through must be a sample-exact copy, up to the 24-bit output step.
    span = slice(0, int(MATCHED[0] * SR))
    drift = float(np.max(np.abs(written[span] - mixture[span])))
    if drift > 2.0**-23:
        raise SystemExit(f"FAIL pass-through: drifted by {drift:.3e}")

    print(f"  cancellation      residual ratio {ratio:.5f}  (ceiling {RESIDUAL_CEILING})")
    print(f"  pass-through      max drift {drift:.3e}  (24-bit step {2.0**-23:.3e})")
    return written


def _check_matches_source(frozen: np.ndarray, fixture: tuple, directory: Path) -> None:
    output = directory / "source.wav"
    subprocess.run(
        [sys.executable, "-m", "audio_overlap_removal", *_job_args(fixture, output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    written, _ = sf.read(output, dtype="float32")
    if not np.array_equal(frozen, written):
        raise SystemExit(
            "FAIL exactness: frozen output differs from the source install by "
            f"up to {np.max(np.abs(frozen - written)):.3e}"
        )
    print("  exactness         frozen output identical to the source install")


def _stripped_path_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "C:\\Windows\\System32" if os.name == "nt" else "/nonexistent"
    env.pop("AOR_FFMPEG_DIR", None)
    return env


def _check_ffmpeg_discovery(executable: Path, fixture: tuple, directory: Path) -> None:
    from audio_overlap_removal.media import _fallback_tool_dirs, _tool

    ffmpeg = Path(_tool("ffmpeg"))
    if not ffmpeg.is_absolute():
        raise SystemExit("FAIL discovery: FFmpeg is not installed on this machine")

    env = _stripped_path_env()
    env["AOR_FFMPEG_DIR"] = str(ffmpeg.parent)
    subprocess.run(
        [str(executable), *_job_args(fixture, directory / "override.wav")],
        check=True,
        stdout=subprocess.DEVNULL,
        env=env,
    )
    print(f"  discovery         AOR_FFMPEG_DIR honoured with no PATH ({ffmpeg.parent})")

    # The guidance message can only be provoked where no guessed install
    # location happens to hold a copy — which is exactly where Homebrew puts it.
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    reachable = [d for d in _fallback_tool_dirs() if (d / exe).is_file()]
    if reachable:
        print(f"  discovery         missing-FFmpeg message skipped ({reachable[0]} has it)")
        return
    result = subprocess.run(
        [str(executable), *_job_args(fixture, directory / "missing.wav")],
        capture_output=True,
        text=True,
        env=_stripped_path_env(),
    )
    if "AOR_FFMPEG_DIR" not in result.stderr:
        raise SystemExit(f"FAIL discovery: unhelpful error {result.stderr.strip()!r}")
    print("  discovery         missing FFmpeg reports the guidance message")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Directory holding the executable.")
    parser.add_argument(
        "--no-source-comparison",
        action="store_true",
        help="Skip the byte-for-byte check (use when the bundle was built elsewhere).",
    )
    args = parser.parse_args()

    bundle = args.bundle or ROOT / "dist" / _platform_tag() / NAME
    executable = bundle / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    if not executable.is_file():
        raise SystemExit(f"No executable at {executable}")

    print(f"Verifying {executable}")
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        fixture = _write_fixture(directory)
        frozen = _check_cancellation(executable, fixture, directory)
        if args.no_source_comparison:
            print("  exactness         skipped")
        else:
            _check_matches_source(frozen, fixture, directory)
        _check_ffmpeg_discovery(executable, fixture, directory)
    print("All checks passed.")


if __name__ == "__main__":
    main()
