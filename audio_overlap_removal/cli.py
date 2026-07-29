"""Command-line interface."""

from __future__ import annotations

import argparse

import numpy as np

from .media import DEFAULT_SR, MIN_SAMPLE_RATE
from .pipeline import remove_reference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cancel a known reference from a real mixture. Wider inputs are "
            "downmixed to stereo."
        )
    )
    parser.add_argument("mixture", help="Mixture media containing the target audio.")
    parser.add_argument("reference", help="Known removable reference media.")
    parser.add_argument("output", help="24-bit .flac or .wav output path.")
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help=(
            "Start of the mixture range that may contain the reference media. "
            "The complete mixture is still written."
        ),
    )
    parser.add_argument(
        "--end",
        type=float,
        help=(
            "End of the mixture range that may contain the reference media. "
            "Defaults to the end of the mixture."
        ),
    )
    parser.add_argument("--chunk", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel alignment queries and cancellation chunks. Higher "
            "values use more CPU and roughly proportional temporary memory."
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Quality/removal trade-off: 0=preserve, 1=recommended default, "
            ">1=ASR-oriented extra removal (open-ended)."
        ),
    )
    parser.add_argument(
        "--cleanup-strength",
        type=float,
        help="Expert override for side-derived residual cleanup.",
    )
    parser.add_argument(
        "--center-strength",
        type=float,
        help=(
            "Expert override for direct cancellation of active centered reference bins."
        ),
    )
    parser.add_argument(
        "--center-cleanup-strength",
        type=float,
        help=("Expert override for centered residue suppression; may exceed 1."),
    )
    parser.add_argument(
        "--silence-cleanup-strength",
        type=float,
        help=(
            "Expert override for cleanup when little unrelated foreground is detected."
        ),
    )
    parser.add_argument(
        "--disable-adaptive-warp",
        action="store_true",
        help=(
            "Use only the conservative 250ms alignment anchors; disable "
            "validated 16ms refinement for detected speed drift."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    if not np.isfinite(args.strength) or args.strength < 0:
        raise ValueError("--strength must be non-negative.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not np.isfinite(args.start) or args.start < 0.0:
        raise ValueError("--start must be non-negative and finite.")
    if args.end is not None and not np.isfinite(args.end):
        raise ValueError("--end must be finite.")
    if not np.isfinite(args.chunk) or args.chunk <= 0.0:
        raise ValueError("--chunk must be positive and finite.")
    if args.sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(f"--sample-rate must be at least {MIN_SAMPLE_RATE} Hz.")
    if args.cleanup_strength is not None and args.cleanup_strength < 0:
        raise ValueError("--cleanup-strength must be non-negative.")
    if args.center_strength is not None and not 0.0 <= args.center_strength <= 1.0:
        raise ValueError("--center-strength must be between 0 and 1.")
    if args.center_cleanup_strength is not None and args.center_cleanup_strength < 0.0:
        raise ValueError("--center-cleanup-strength must be non-negative.")
    if (
        args.silence_cleanup_strength is not None
        and not 0.0 <= args.silence_cleanup_strength <= 1.0
    ):
        raise ValueError("--silence-cleanup-strength must be between 0 and 1.")
    remove_reference(
        args.mixture,
        args.reference,
        args.output,
        start=args.start,
        end=args.end,
        chunk_sec=args.chunk,
        strength=args.strength,
        cleanup_strength=args.cleanup_strength,
        center_strength=args.center_strength,
        center_cleanup_strength=args.center_cleanup_strength,
        silence_cleanup_strength=args.silence_cleanup_strength,
        adaptive_time_warp=not args.disable_adaptive_warp,
        sr=args.sample_rate,
        workers=args.workers,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _run_cli(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
