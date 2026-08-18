#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate
from driftforge.generator import generate_sample
from driftforge.splits import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and audit manifest records without materializing PNGs.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_manifest(args.manifest)[args.start:args.start + args.limit]
    rows = []
    for index, row in enumerate(records, start=1):
        sample = generate_sample(
            int(row["seed"]), row["architecture"], row["profile"], args.search_supersample,
        )
        started = time.perf_counter()
        result = locate(sample.reference, sample.search)
        runtime = time.perf_counter() - started
        error = math.hypot(result.x - sample.gt_x, result.y - sample.gt_y)
        item = {
            "id": row["id"], "seed": row["seed"], "profile": row["profile"],
            "architecture": row["architecture"], "gt_x": sample.gt_x, "gt_y": sample.gt_y,
            "pred_x": result.x, "pred_y": result.y, "error_px": error,
            "score": result.score, "psr": result.psr, "peak_gap": result.peak_gap,
            "accepted": result.accepted, "ambiguity_detected": result.ambiguity_detected,
            "runtime_s": runtime,
            "accepted_incorrect": bool(result.accepted and error > args.tolerance),
        }
        rows.append(item)
        print(
            f"[{index}/{len(records)}] {row['id']} error={error:.2f}px "
            f"accepted={result.accepted} gap={result.peak_gap:.3f}", flush=True,
        )
    errors = np.asarray([row["error_px"] for row in rows])
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    summary = {
        "count": len(rows), "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)), "max_error_px": float(errors.max()),
        "accuracy_at_px": {str(t): float(np.mean(errors <= t)) for t in (1, 2, 5, 10)},
        "coverage": float(np.mean(accepted)),
        "accepted_incorrect_at_tolerance": int(sum(row["accepted_incorrect"] for row in rows)),
        "tolerance_px": args.tolerance,
        "mean_runtime_s": float(np.mean([row["runtime_s"] for row in rows])),
        "search_supersample": args.search_supersample,
    }
    # R8: the packaged v1 artifact carried `accepted_incorrect_at_5px`, a key
    # this script has never written, so the shipped report could not be
    # reproduced from the shipped code. The tolerance-parameterised key is kept
    # (it is the correct one - the threshold is an argument, not a constant)
    # and the schema is now versioned and self-describing so a future mismatch
    # is detectable rather than silent.
    summary["schema_version"] = 2
    report = {
        "schema_version": 2,
        "generated_by": "scripts/audit_manifest.py",
        "command": " ".join(
            ["python", "scripts/audit_manifest.py", str(args.manifest),
             "--start", str(args.start), "--limit", str(args.limit),
             "--search-supersample", str(args.search_supersample),
             "--tolerance", str(args.tolerance), "--output", str(args.output)]
        ),
        "summary": summary,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["accepted_incorrect_at_tolerance"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

