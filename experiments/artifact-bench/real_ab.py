"""Run the real miyako/sr material over a fixed segment file and summarise.

There is no ground truth here; what the run reports is the canceller's own
evidence (Side residual, measured reduction, alignment) plus a byte comparison
against a previous label's output, which certifies "no change" for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = Path(__file__).with_name("real-segments-375-700.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default=str(Path(__file__).with_name("real-out")))
    parser.add_argument("--start", type=float, default=380.0)
    parser.add_argument("--end", type=float, default=620.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.label}.flac"
    subprocess.run(
        [
            sys.executable, "-m", "audio_overlap_removal",
            str(ROOT / "extra-asset" / "miyako.webm"),
            str(ROOT / "extra-asset" / "sr.webm"),
            str(output),
            "--segments", str(SEGMENTS),
            "--output-start", str(args.start), "--output-end", str(args.end),
            "--workers", str(args.workers), "--strength", str(args.strength),
            "--quiet", "--log-level", "debug",
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads((out_dir / f"{args.label}-result.json").read_text(encoding="utf-8"))
    chunks = [c for c in result["chunks"] if c["mode"] == "cancelled"]

    def stat(key: str) -> str:
        values = np.array([c[key] for c in chunks if c.get(key) is not None])
        return f"{np.median(values):6.3f} [{values.min():6.3f}, {values.max():6.3f}]"

    digest = hashlib.sha256(output.read_bytes()).hexdigest()[:12]
    print(f"{args.label}: {len(chunks)} cancelled chunks, sha256 {digest}")
    for key in (
        "alignment_score", "side_residual_ratio", "control_reduction_db",
        "cleanup_output_ratio", "gain_median", "gain_p95",
    ):
        print(f"  {key:24s} {stat(key)}")
    band_keys = sorted(k for k in chunks[0] if k.startswith("side_residual_ratio_"))
    for key in band_keys:
        print(f"  {key:24s} {stat(key)}")


if __name__ == "__main__":
    main()
