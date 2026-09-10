"""The machine-readable record of one run, written beside the output audio."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._version import __version__
from .media import _probe_media_info
from .models import (
    AlignmentSegment,
    _json_float,
    _linear_residuals,
    _merge_passthrough_spans,
)

SCHEMA = "audio-overlap-removal/result@1"
ANCHOR_TIERS = ("none", "summary", "full")

_PACKAGE_LOGGER = "audio_overlap_removal"
_WEAKEST_ANCHORS = 10
_MAX_WARNINGS = 200


class _WarningCollector(logging.Handler):
    """Mirror warnings into the result so it explains itself without the log."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        if len(self._sink) < _MAX_WARNINGS:
            self._sink.append(record.getMessage())


def _timestamp(moment: float) -> str:
    return (
        datetime.fromtimestamp(moment, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _stats(values: np.ndarray, keys: tuple[str, ...]) -> dict[str, float | None]:
    """Summarise a distribution, ignoring the NaN holes left by clipping."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict.fromkeys(keys)
    percentiles = {
        "min": lambda: float(np.min(finite)),
        "max": lambda: float(np.max(finite)),
        "rms": lambda: float(np.sqrt(np.mean(finite * finite))),
        "p05": lambda: float(np.percentile(finite, 5)),
        "median": lambda: float(np.median(finite)),
        "p95": lambda: float(np.percentile(finite, 95)),
    }
    return {key: percentiles[key]() for key in keys}


def _anchor_report(segment: AlignmentSegment, tier: str) -> dict:
    times = np.asarray(segment.anchor_times, dtype=np.float64)
    report: dict = {"count": int(len(times))}
    if tier == "none" or not len(times):
        return report

    offsets = np.asarray(segment.anchor_offsets, dtype=np.float64)
    scores = (
        np.asarray(segment.anchor_scores, dtype=np.float64)
        if segment.anchor_scores
        else np.full(len(times), np.nan)
    )
    gaps = np.diff(times)
    report["interval_sec"] = (
        {"median": float(np.median(gaps))} if len(gaps) else {"median": None}
    )
    report["score"] = _stats(scores, ("min", "p05", "median", "p95"))
    report["residual_sec"] = _stats(_linear_residuals(segment), ("rms", "p95"))
    if len(gaps):
        widest = int(np.argmax(gaps))
        report["sparsest_gap"] = {
            "start": float(times[widest]),
            "end": float(times[widest + 1]),
        }
    else:
        report["sparsest_gap"] = None

    # Only the worst anchors carry information: a well-scored middle says
    # nothing that the distribution above has not already said.
    scored = [
        index for index in range(len(times)) if np.isfinite(scores[index])
    ]
    scored.sort(key=lambda index: (scores[index], times[index]))
    report["weakest"] = [
        {
            "time": float(times[index]),
            "offset": float(offsets[index]),
            "score": float(scores[index]),
        }
        for index in scored[:_WEAKEST_ANCHORS]
    ]

    if tier == "full":
        report["times"] = [float(time) for time in times]
        report["offsets"] = [float(offset) for offset in offsets]
        report["scores"] = [_json_float(score) for score in scores]
    return report


class _RunResult:
    """Accumulate one run across the scan and processing stages.

    The command-line entry point owns the instance because the two stages are
    separate calls that each hold only part of the picture, and because a run
    interrupted midway must still write what it had.
    """

    def __init__(self, *, argv: list[str] | None, anchors: str = "summary") -> None:
        if anchors not in ANCHOR_TIERS:
            raise ValueError(f"Unknown anchor tier: {anchors!r}.")
        self._anchors = anchors
        self._started = time.time()
        self._warnings: list[str] = []
        self._collector = _WarningCollector(self._warnings)
        logging.getLogger(_PACKAGE_LOGGER).addHandler(self._collector)
        self._run: dict = {
            "version": __version__,
            "argv": list(argv) if argv is not None else None,
            "started_at": _timestamp(self._started),
            "settings": {},
        }
        self._inputs: dict = {"mixture": None, "references": []}
        self._output: dict | None = None
        self._segments: list[dict] = []
        self._chunks: list[dict] = []

    def record_inputs(self, mixture_path: str, reference_path: str) -> None:
        self._inputs = {
            "mixture": _probe_media_info(mixture_path, "mixture"),
            "references": [_probe_media_info(reference_path, "reference")],
        }

    def record_settings(self, **settings: object) -> None:
        self._run["settings"].update(settings)

    def record_segments(self, segments: list[AlignmentSegment]) -> None:
        self._segments = [
            {
                "index": index,
                "media_id": "reference",
                "mixture_start": segment.mixture_start,
                "mixture_end": segment.mixture_end,
                # Both ends use their own offset: the whole point of the
                # trajectory is that the offset drifts across the segment.
                "reference_start": segment.mixture_start
                - segment.offset_at(segment.mixture_start),
                "reference_end": segment.mixture_end
                - segment.offset_at(segment.mixture_end),
                "offset_sec": segment.offset_sec,
                "offset_slope": segment.offset_slope,
                "speed": 1.0 - segment.offset_slope,
                "median_score": segment.median_score,
                "trajectory_deviation_sec": (
                    float(np.max(np.abs(_linear_residuals(segment))))
                    if segment.anchor_times
                    else 0.0
                ),
                "anchors": _anchor_report(segment, self._anchors),
            }
            for index, segment in enumerate(segments)
        ]

    def record_output(self, **fields: object) -> None:
        self._output = dict(fields)

    def record_chunk(self, chunk: dict) -> None:
        self._chunks.append(chunk)

    def _summary(self) -> dict:
        counts = {
            "total": len(self._chunks),
            "cancelled": 0,
            "low_confidence": 0,
            "unmatched": 0,
        }
        seconds = {"cancelled": 0.0, "low_confidence": 0.0, "unmatched": 0.0}
        by_mode = {
            "cancelled": "cancelled",
            "low-confidence": "low_confidence",
            "unmatched": "unmatched",
        }
        spans: list[tuple[float, float]] = []
        for chunk in self._chunks:
            key = by_mode[chunk["mode"]]
            counts[key] += 1
            seconds[key] += chunk["end_sec"] - chunk["start_sec"]
            if chunk["mode"] == "low-confidence":
                spans.append((chunk["start_sec"], chunk["end_sec"]))
        total_sec = sum(seconds.values())
        seconds["total"] = total_sec
        seconds["matched_coverage"] = (
            seconds["cancelled"] / total_sec if total_sec > 0.0 else 0.0
        )
        reductions = np.array(
            [
                chunk["control_reduction_db"]
                for chunk in self._chunks
                if chunk.get("control_reduction_db") is not None
            ],
            dtype=np.float64,
        )
        scores = np.array(
            [
                chunk["alignment_score"]
                for chunk in self._chunks
                if chunk.get("alignment_score") is not None
            ],
            dtype=np.float64,
        )
        return {
            "chunks": counts,
            "seconds": seconds,
            "control_reduction_db": _stats(reductions, ("p05", "median", "p95")),
            "alignment_score": _stats(scores, ("min", "median")),
            "passthrough_spans": [
                [start, end] for start, end in _merge_passthrough_spans(spans)
            ],
            "warnings": list(self._warnings),
        }

    def dump(self, path: Path, status: str) -> None:
        """Write the document atomically, including after an interrupted run."""
        logging.getLogger(_PACKAGE_LOGGER).removeHandler(self._collector)
        finished = time.time()
        elapsed = finished - self._started
        self._run["finished_at"] = _timestamp(finished)
        self._run["elapsed_sec"] = elapsed
        written = (self._output or {}).get("duration_sec") or 0.0
        self._run["realtime_factor"] = elapsed / written if written > 0.0 else None
        payload = {
            "schema": SCHEMA,
            "status": status,
            "run": self._run,
            "inputs": self._inputs,
            "output": self._output,
            "segments": self._segments,
            "chunks": self._chunks,
            "summary": self._summary(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name, suffix=".part"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
