"""Standardized benchmark runner.

One command per system, one standardized prediction CSV and summary JSON per
run, so every future model is measured the same way as the Phase-2A baseline.

    python -m experiments.benchmark \
      --system driftforge_v1_leakfree \
      --manifest manifests/validation_phase2a_400.jsonl \
      --output experiments/results/classical_v1

This never touches scripts/inference.py: the competition CLI stays exactly as
the organiser requires, and the benchmark imports the locator directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments import metrics as M            # noqa: E402
from experiments.common import (                # noqa: E402
    CATASTROPHIC_PX, RESULTS, SYSTEMS, get_system, load_manifest, render,
)

PRED_COLS = ["sample_id", "seed", "scene_id", "architecture", "profile",
             "true_x", "true_y", "pred_x", "pred_y", "error_px",
             "runtime_ms", "confidence", "accepted", "peak_gap", "psr",
             "selected_scale", "selected_rotation"]


def _run_one(args) -> dict:
    record, system_key, supersample = args
    predict = SYSTEMS[system_key].factory()
    sample = render(record, supersample)
    pred = predict(sample.reference, sample.search)
    err = math.hypot(pred.x - sample.gt_x, pred.y - sample.gt_y)
    d = pred.diagnostics or {}
    return {
        "sample_id": record["id"], "seed": int(record["seed"]),
        "scene_id": record.get("scene_id", f"scene-{record['seed']}"),
        "architecture": record["architecture"], "profile": record["profile"],
        "true_x": sample.gt_x, "true_y": sample.gt_y,
        "pred_x": pred.x, "pred_y": pred.y, "error_px": err,
        "runtime_ms": pred.runtime_ms, "confidence": pred.confidence,
        "accepted": bool(d.get("accepted", True)),
        "peak_gap": d.get("peak_gap", float("nan")),
        "psr": d.get("psr", float("nan")),
        "selected_scale": d.get("scale", float("nan")),
        "selected_rotation": d.get("rotation_deg", float("nan")),
    }


def _reanalyse(args) -> int:
    """Rebuild summary/plots/galleries from a predictions CSV already on disk.

    Re-running a 400-pair sweep just to redraw a figure wastes ~15 minutes and,
    worse, invites a second set of numbers that differ from the reported ones.
    Re-analysis keeps one measurement and many views of it.
    """
    rows = []
    with args.from_predictions.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = dict(r)
            for k in ("true_x", "true_y", "pred_x", "pred_y", "error_px",
                      "runtime_ms", "confidence", "peak_gap", "psr",
                      "selected_scale", "selected_rotation"):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = float("nan")
            if "seed" in row and row["seed"]:
                row["seed"] = int(float(row["seed"]))
            if "accepted" in row:
                row["accepted"] = str(row["accepted"]).lower() in ("true", "1", "yes")
            row.setdefault("scene_id", f"scene-{row.get('seed', '?')}")
            rows.append(row)

    if args.merge_classes and args.merge_classes.exists():
        with args.merge_classes.open(encoding="utf-8") as fh:
            cls = {r["sample_id"]: r for r in csv.DictReader(fh)}
        for row in rows:
            extra = cls.get(row["sample_id"])
            if extra:
                row["failure_class"] = extra.get("failure_class", "")
                for k in ("patch_zncc", "lattice_m", "lattice_n", "lattice_residual_px"):
                    if k in extra and extra[k] not in ("", None):
                        try:
                            row[k] = float(extra[k])
                        except ValueError:
                            pass
            else:
                row.setdefault("failure_class", "ok" if row["error_px"] <= CATASTROPHIC_PX else "")

    rows.sort(key=lambda r: r["sample_id"])
    summary = M.summarize(rows, tolerance_px=args.tolerance)
    summary["system"] = args.system
    summary["system_label"] = SYSTEMS[args.system].label if args.system in SYSTEMS else args.system
    summary["manifest"] = str(args.manifest.name)
    summary["source"] = f"re-analysis of {args.from_predictions.name}"
    summary["environment"] = environment()

    outdir = args.output or (RESULTS / args.system)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    from experiments import plotting as P
    pd_ = outdir / "plots"
    errs = [r["error_px"] for r in rows]
    rts = [r.get("runtime_ms", float("nan")) for r in rows]
    P.error_cdf(errs, pd_, label=summary["system_label"])
    P.error_histogram(errs, pd_)
    P.accuracy_by_profile(summary, pd_)
    P.accuracy_dram_vs_finfet(summary, pd_)
    if np.isfinite(np.asarray(rts, dtype=float)).any():
        P.runtime_distribution(rts, pd_)
        P.runtime_vs_error(rts, errs, pd_)
    classes = {}
    for r in rows:
        fc = r.get("failure_class", "")
        if r["error_px"] > CATASTROPHIC_PX and fc:
            classes[fc] = classes.get(fc, 0) + 1
    if classes:
        P.failure_class_bar(dict(sorted(classes.items(), key=lambda kv: -kv[1])), pd_)

    if args.galleries:
        from experiments import galleries as G
        made = G.build_gallery(rows, outdir / "galleries", str(args.manifest.name),
                               summary["system_label"], args.search_supersample)
        (outdir / "galleries" / "index.json").write_text(
            json.dumps(made, indent=2), encoding="utf-8")

    a = summary["accuracy"]
    print(f"re-analysed {summary['pairs']} pairs from {args.from_predictions.name}")
    print(f"  acc <=5px {a['5']['acc']:.1%}   median {summary['p50_error_px']:.2f} px   "
          f">25px {summary['catastrophic']['count']}")
    print(f"wrote {outdir}")
    return 0


def environment() -> dict:
    import scipy
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__,
            "scipy": scipy.__version__}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True,
                    help=f"one of: {', '.join(sorted(SYSTEMS))}")
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT / "manifests/validation_phase2a_400.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--plots", action="store_true", help="write distribution plots")
    ap.add_argument("--galleries", action="store_true", help="write success/failure figures")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--from-predictions", type=Path,
                    help="re-analyse an existing predictions CSV instead of re-running "
                         "the system (same metrics, plots and galleries, no recompute)")
    ap.add_argument("--merge-classes", type=Path,
                    help="CSV carrying a failure_class column to join on sample_id")
    args = ap.parse_args(argv)

    if args.from_predictions:
        return _reanalyse(args)

    if args.list:
        print(f"{'key':28s} {'phase':6s} {'impl':5s} label")
        for k, s in SYSTEMS.items():
            print(f"{k:28s} {s.phase:6s} {'yes' if s.implemented else 'no ':5s} {s.label}")
        return 0

    spec = get_system(args.system)           # refuses unimplemented systems
    outdir = args.output or (RESULTS / args.system)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.manifest)
    records = records[args.start: args.start + args.limit if args.limit else None]
    print(f"system={args.system}  pairs={len(records)}  "
          f"supersample={args.search_supersample}  workers={args.workers}", flush=True)

    payload = [(r, args.system, args.search_supersample) for r in records]
    rows: list[dict] = []
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(_run_one, payload), 1):
                rows.append(row)
                if i % 25 == 0 or i == len(payload):
                    print(f"  [{i}/{len(payload)}] {time.perf_counter()-t0:.0f}s", flush=True)
    else:
        for i, p in enumerate(payload, 1):
            rows.append(_run_one(p))
            if i % 25 == 0 or i == len(payload):
                print(f"  [{i}/{len(payload)}] {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["sample_id"])

    summary = M.summarize(rows, tolerance_px=args.tolerance)
    summary["system"] = args.system
    summary["system_label"] = spec.label
    summary["manifest"] = str(args.manifest.name)
    summary["search_supersample"] = args.search_supersample
    summary["wall_clock_s"] = wall
    summary["workers"] = args.workers
    summary["environment"] = environment()
    summary["runtime_note"] = (
        f"runtime_ms is measured around predict() inside each worker; with "
        f"workers={args.workers} it includes CPU contention and is an upper "
        f"bound on single-process latency.")

    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")

    if args.plots:
        from experiments import plotting as P
        pd = outdir / "plots"
        errs = [r["error_px"] for r in rows]
        rts = [r["runtime_ms"] for r in rows]
        P.error_cdf(errs, pd, label=spec.label)
        P.error_histogram(errs, pd)
        P.accuracy_by_profile(summary, pd)
        P.accuracy_dram_vs_finfet(summary, pd)
        P.runtime_distribution(rts, pd)
        P.runtime_vs_error(rts, errs, pd)
        print(f"plots -> {pd}")

    if args.galleries:
        from experiments import galleries as G
        made = G.build_gallery(rows, outdir / "galleries", str(args.manifest.name),
                               spec.label, args.search_supersample)
        (outdir / "galleries" / "index.json").write_text(
            json.dumps(made, indent=2), encoding="utf-8")
        print(f"galleries -> {outdir / 'galleries'}")

    a = summary["accuracy"]
    print(f"\n{spec.label}  n={summary['pairs']}")
    print(f"  acc <=1px {a['1']['acc']:.1%}   <=2px {a['2']['acc']:.1%}   "
          f"<=5px {a['5']['acc']:.1%} [{a['5']['ci95_lo']:.1%}-{a['5']['ci95_hi']:.1%}]")
    print(f"  median {summary['p50_error_px']:.2f} px   P95 {summary['p95_error_px']:.1f} px   "
          f"P99 {summary['p99_error_px']:.1f} px")
    print(f"  >25px  {summary['catastrophic']['count']}/{summary['catastrophic']['n']} "
          f"({summary['catastrophic']['rate']:.1%})")
    print(f"  runtime median {summary['median_runtime_ms']:.0f} ms  "
          f"P95 {summary['p95_runtime_ms']:.0f} ms")
    print(f"\nwrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
