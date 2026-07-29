"""Search high B_full retention floors on the real middle clip."""

from __future__ import annotations

import json

import explore_dual_model as dual
import explore_iterative as iterative
import explore_two_pass as base
import export_dual_real as real
import test_exact_pipeline as exact
from audio_overlap_removal.media import _decode_stereo

SR = 48_000
base.SR = SR
iterative.SR = SR
exact.SR = SR
dual.SR = SR


def main() -> None:
    mixture = _decode_stereo(
        str(base.ASSETS / "miyako.webm"), 2200.192, 12.0, SR
    )
    reference = _decode_stereo(
        str(base.ASSETS / "sr.webm"), 2118.659, 12.0, SR
    )
    frames = min(len(mixture), len(reference))
    mixture = mixture[:frames]
    reference = reference[:frames]
    current = base.production_output(mixture, reference)
    current_related = real.related_rms(current, reference)
    report = {
        "current": {
            "related_rms": current_related,
            "output_rms": real.rms(current),
        }
    }
    for floor in (0.50, 0.75, 0.90, 0.95, 0.98):
        output, diagnostics = dual.complete_output(
            mixture,
            reference,
            floor=floor,
            coherence_floor=0.30,
            dominance_floor=0.30,
        )
        related = real.related_rms(output, reference)
        report[str(floor)] = {
            **diagnostics,
            "related_rms": related,
            "related_ratio": related / (current_related + 1e-12),
            "difference_rms_ratio": real.rms(output - current)
            / (real.rms(current) + 1e-12),
        }
    print(json.dumps(report, indent=2))
    (real.ROOT / "dual-floor-search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
