"""Validate high-retention dual-model candidates on synthetic truth."""

from __future__ import annotations

import json

import explore_dual_model as dual
import explore_iterative as iterative
import explore_two_pass as base
import test_exact_pipeline as exact

SR = 16_000
base.SR = SR
iterative.SR = SR
exact.SR = SR
dual.SR = SR


def main() -> None:
    reference, scenarios = dual.build_extended_scenarios()
    results = {}
    for name, (media, foreground) in scenarios.items():
        mixture = media + foreground
        target = foreground.mean(axis=1)
        media_mid = media.mean(axis=1)
        current = base.production_output(mixture, reference)
        scenario = {
            "current": iterative.safe_metrics(
                current, target, media_mid
            )
        }
        for floor in (0.90, 0.95):
            output, diagnostics = dual.complete_output(
                mixture,
                reference,
                floor=floor,
                coherence_floor=0.30,
                dominance_floor=0.30,
            )
            scenario[str(floor)] = {
                **iterative.safe_metrics(output, target, media_mid),
                **diagnostics,
            }
        results[name] = scenario
        print(
            f"{name:20} "
            + " ".join(
                f"{key}: {scenario[key]['suppression_db']:6.2f}/"
                f"{scenario[key]['foreground_db']:5.2f}"
                for key in ("current", "0.9", "0.95")
            )
        )
    (dual.ROOT / "dual-high-floor-results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
