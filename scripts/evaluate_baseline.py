#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the conservative locator on a labelled materialized split.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()
    with (args.dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    if args.limit:
        labels = labels[:args.limit]

    rows = []
    for i, row in enumerate(labels, start=1):
        ref = np.asarray(Image.open(args.dataset / row["reference"]).convert("L"))
        search = np.asarray(Image.open(args.dataset / row["search"]).convert("L"))
        start = time.perf_counter()
        result = locate(ref, search)
        runtime = time.perf_counter() - start
        error = math.hypot(result.x - float(row["x"]), result.y - float(row["y"]))
        item = {
            "id": row["id"], "architecture": row["architecture"], "profile": row["profile"],
            "gt_x": float(row["x"]), "gt_y": float(row["y"]), "pred_x": result.x, "pred_y": result.y,
            "error_px": error, "score": result.score, "psr": result.psr, "peak_gap": result.peak_gap,
            "accepted": result.accepted, "accepted_incorrect": bool(result.accepted and error > args.tolerance),
            "runtime_s": runtime,
        }
        rows.append(item)
        print(f"[{i}/{len(labels)}] {row['id']} err={error:.2f}px accepted={result.accepted}", flush=True)

    errors = np.asarray([row["error_px"] for row in rows])
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    summary = {
        "count": len(rows),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(errors.max()),
        "accuracy_at_px": {str(t): float(np.mean(errors <= t)) for t in (1, 2, 5, 10)},
        "coverage": float(np.mean(accepted)),
        "accepted_incorrect_at_tolerance": int(sum(row["accepted_incorrect"] for row in rows)),
        "tolerance_px": args.tolerance,
        "mean_runtime_s": float(np.mean([row["runtime_s"] for row in rows])),
    }
    report = {"summary": summary, "samples": rows}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

