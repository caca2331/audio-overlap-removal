"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .alignment import SCAN_MODES
from .logging_setup import LOG_LEVELS, _configure_logging
from .media import DEFAULT_SR, MIN_SAMPLE_RATE
from .models import AlignmentSegment, _segment_from_dict, _segment_to_dict
from .pipeline import process_audio, scan_reference
from .result import ANCHOR_TIERS, _RunResult

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cancel a known reference from a real mixture. Wider inputs are "
            "downmixed to stereo."
        )
    )
    parser.add_argument(
        "mixture",
        help="Mixture media containing the target audio (maximum 24 hours).",
    )
    parser.add_argument(
        "reference",
        help="Known removable reference media (maximum 24 hours).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="24-bit .flac or .wav output path. Omit only with --scan-only.",
    )
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
        default=4,
        help=(
            "Parallel alignment queries and cancellation chunks. Keep this at "
            "or below the CPU core count; each worker also raises peak memory."
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
    parser.add_argument(
        "--disable-momentum",
        action="store_true",
        help=(
            "Skip the low-rate probing pass that measures each chunk's true "
            "offset before cancellation. Faster, but every chunk then relies "
            "on the segment-level offset model alone."
        ),
    )
    parser.add_argument(
        "--search",
        type=float,
        default=0.25,
        help=(
            "Reference search radius per chunk. A chunk that cannot be aligned "
            "inside it is retried once with a wider radius."
        ),
    )
    parser.add_argument(
        "--scan-mode",
        choices=SCAN_MODES,
        default="auto",
        help=(
            "How to locate the reference. 'correlation' compares whole decoded "
            "waveforms and is the more sensitive on short inputs, but its cost "
            "grows quadratically; 'fingerprint' queries a streamed index at "
            "roughly constant cost per hour. 'auto' switches to the index once "
            "either input runs longer than four hours."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Locate the reference and stop before processing any audio.",
    )
    parser.add_argument(
        "--segments-out",
        help=(
            "Write the discovered segments here. Defaults to "
            "<output>-segments.json, unless --segments supplied them."
        ),
    )
    parser.add_argument(
        "--no-segments-out",
        action="store_true",
        help="Do not write the discovered segments.",
    )
    parser.add_argument(
        "--segments",
        help=(
            "Process the segments in this JSON file instead of scanning. "
            "Accepts a file written by --segments-out, optionally edited."
        ),
    )
    parser.add_argument(
        "--log",
        help=(
            "Write the run log here. Defaults to <output>-log.txt beside the "
            "output audio."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a log file. The console is unaffected.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Detail level for both the console and the log file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show only warnings on the console; the log file keeps everything.",
    )
    parser.add_argument(
        "--result",
        help=(
            "Write the machine-readable run result here. Defaults to "
            "<output>-result.json beside the output audio."
        ),
    )
    parser.add_argument(
        "--no-result",
        action="store_true",
        help="Do not write the run result.",
    )
    parser.add_argument(
        "--result-anchors",
        choices=ANCHOR_TIERS,
        default="summary",
        help=(
            "How much of each anchor trajectory the result keeps. 'summary' "
            "records the distribution and the weakest anchors; 'full' adds "
            "every anchor, which is large on long media."
        ),
    )
    parser.add_argument(
        "--output-start",
        type=float,
        default=0.0,
        help=(
            "Write only from this mixture timestamp. Alignment context still "
            "reads outside the written range."
        ),
    )
    parser.add_argument(
        "--output-end",
        type=float,
        help="Write only up to this mixture timestamp.",
    )
    return parser


def _load_segments(path: str) -> list[AlignmentSegment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path!r} must contain a JSON list of segments.")
    return [_segment_from_dict(item) for item in payload]


def _write_segments(path: Path, segments: list[AlignmentSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_segment_to_dict(segment) for segment in segments], indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %d segment(s) to %s.", len(segments), path)


def _sidecar_path(
    explicit: str | None, disabled: bool, base: str, suffix: str
) -> Path | None:
    """Resolve one companion file, defaulting to a name beside the audio."""
    if disabled:
        return None
    if explicit:
        return Path(explicit)
    anchor = Path(base)
    return anchor.with_name(f"{anchor.stem}-{suffix}")


def _run_cli(args: argparse.Namespace) -> None:
    if not np.isfinite(args.strength) or args.strength < 0:
        raise ValueError("--strength must be non-negative.")
    if args.output is None and not args.scan_only:
        raise ValueError("An output path is required unless --scan-only is used.")
    if args.segments and args.scan_only:
        raise ValueError("--segments and --scan-only cannot be combined.")
    if not np.isfinite(args.output_start) or args.output_start < 0.0:
        raise ValueError("--output-start must be non-negative and finite.")
    if args.output_end is not None and not np.isfinite(args.output_end):
        raise ValueError("--output-end must be finite.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not np.isfinite(args.start) or args.start < 0.0:
        raise ValueError("--start must be non-negative and finite.")
    if args.end is not None and not np.isfinite(args.end):
        raise ValueError("--end must be finite.")
    if not np.isfinite(args.chunk) or args.chunk <= 0.0:
        raise ValueError("--chunk must be positive and finite.")
    if not np.isfinite(args.search) or args.search <= 0.0:
        raise ValueError("--search must be positive and finite.")
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

    # Companion files sit beside the output audio, so the common invocation
    # needs no paths at all. --scan-only writes no audio, so the mixture names
    # them instead.
    base = args.mixture if args.scan_only else args.output
    log_path = _sidecar_path(args.log, args.no_log, base, "log.txt")
    result_path = _sidecar_path(args.result, args.no_result, base, "result.json")
    segments_out = _sidecar_path(
        args.segments_out,
        # Segments that came from a file would only be copied back out.
        args.no_segments_out or (bool(args.segments) and not args.segments_out),
        base,
        "segments.json",
    )
    _configure_logging(log_path, args.log_level, args.quiet)

    result = (
        _RunResult(argv=list(sys.argv), anchors=args.result_anchors)
        if result_path is not None
        else None
    )
    status = "scan-only" if args.scan_only else "complete"
    try:
        if result is not None:
            result.record_inputs(args.mixture, args.reference)
            result.record_settings(
                segments_source=(
                    f"file:{args.segments}" if args.segments else "scan"
                )
            )
        if args.segments:
            segments = _load_segments(args.segments)
            logger.info(
                "Loaded %d segment(s) from %s.", len(segments), args.segments
            )
        else:
            segments = scan_reference(
                args.mixture,
                args.reference,
                start=args.start,
                end=args.end,
                workers=args.workers,
                scan_mode=args.scan_mode,
                _result=result,
            )
        if result is not None:
            result.record_segments(segments)
        if segments_out is not None:
            _write_segments(segments_out, segments)
        if args.scan_only:
            return
        process_audio(
            args.mixture,
            args.reference,
            args.output,
            alignment_segments=segments,
            chunk_sec=args.chunk,
            search_sec=args.search,
            strength=args.strength,
            cleanup_strength=args.cleanup_strength,
            center_strength=args.center_strength,
            center_cleanup_strength=args.center_cleanup_strength,
            silence_cleanup_strength=args.silence_cleanup_strength,
            adaptive_time_warp=not args.disable_adaptive_warp,
            momentum=not args.disable_momentum,
            output_start=args.output_start,
            output_end=args.output_end,
            sr=args.sample_rate,
            workers=args.workers,
            _result=result,
        )
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        # A long run that died partway is exactly when the record matters.
        if result is not None:
            result.dump(result_path, status)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _run_cli(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
