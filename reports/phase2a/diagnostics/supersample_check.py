#!/usr/bin/env python3
"""E1b: matched rendering-fidelity comparison, search_supersample 2 vs 4.

Same records, same locator, same seeds - only the Search-side rendering
oversampling differs. If localisation moves materially, the measured baseline
is partly an artifact of render fidelity and Phase 2 must fit on the better
setting.

Usage:
    python reports/phase2a/diagnostics/supersample_check.py --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

CATASTROPHIC_PX = 25.0


def image_stats(img: np.ndarray) -> dict:
    a = img.astype(np.float64) / 255.0
    gy, gx = np.gradient(a)
    grad = np.hypot(gx, gy)
    # Radially-summed high-frequency energy: the part of the spectrum that
    # aliasing corrupts, and where 2x vs 4x should differ if anywhere.
    f = np.fft.fftshift(np.abs(np.fft.fft2(a - a.mean())))
    n = f.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] - n / 2.0
    r = np.hypot(yy, xx)
    total = float(f.sum()) + 1e-9
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "grad_mean": float(grad.mean()),
        "hf_fraction": float(f[r > n * 0.25].sum() / total),
    }


def run_one(row: dict, supersample: int) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    sample = generate_sample(int(row["seed"]), row["architecture"], row["profile"], supersample)
    gen_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    result = locate(sample.reference, sample.search)
    loc_s = time.perf_counter() - t1

    dx, dy = result.x - sample.gt_x, result.y - sample.gt_y
    error = math.hypot(dx, dy)

    acq = sample.search_acquisition
    t = math.radians(acq.rotation_deg)
    px = acq.scale * sample.scene.pitch_x_nm / 10.0
    py = acq.scale * sample.scene.pitch_y_nm / 10.0
    B = np.column_stack([
        np.array([px * math.cos(t), px * math.sin(t)]),
        np.array([-py * math.sin(t), py * math.cos(t)]),
    ])
    mn = np.linalg.solve(B, np.array([dx, dy]))
    res_m, res_n = abs(mn[0] - round(mn[0])), abs(mn[1] - round(mn[1]))
    is_alias = bool(error > CATASTROPHIC_PX and res_m <= 0.25 and res_n <= 0.25)

    stats = image_stats(sample.search)
    return {
        "sample_id": row["id"], "seed": int(row["seed"]),
        "architecture": row["architecture"], "profile": row["profile"],
        "supersample": supersample, "error_px": error,
        "accepted": bool(result.accepted), "psr": result.psr,
        "peak_gap": result.peak_gap, "is_alias": is_alias,
        "catastrophic": bool(error > CATASTROPHIC_PX),
        "gen_s": gen_s, "locate_s": loc_s, "peak_mem_mb": peak / 1e6,
        "pitch_x_px": px, "pitch_y_px": py,
        **{f"img_{k}": v for k, v in stats.items()},
    }


def side(rows: list[dict]) -> dict:
    e = np.array([r["error_px"] for r in rows])
    cat = np.array([r["catastrophic"] for r in rows])
    return {
        "pairs": len(rows),
        "acc_1px": float(np.mean(e <= 1)),
        "acc_2px": float(np.mean(e <= 2)),
        "acc_5px": float(np.mean(e <= 5)),
        "median_error_px": float(np.median(e)),
        "mean_error_px": float(e.mean()),
        "p95_error_px": float(np.percentile(e, 95)),
        "catastrophic_count": int(cat.sum()),
        "catastrophic_rate": float(cat.mean()),
        "alias_count": int(sum(r["is_alias"] for r in rows)),
        "alias_fraction_of_catastrophic": (
            sum(r["is_alias"] for r in rows) / int(cat.sum()) if cat.sum() else None),
        "coverage": float(np.mean([r["accepted"] for r in rows])),
        "mean_gen_s": float(np.mean([r["gen_s"] for r in rows])),
        "mean_locate_s": float(np.mean([r["locate_s"] for r in rows])),
        "mean_peak_mem_mb": float(np.mean([r["peak_mem_mb"] for r in rows])),
        "img_mean": float(np.mean([r["img_mean"] for r in rows])),
        "img_std": float(np.mean([r["img_std"] for r in rows])),
        "img_grad_mean": float(np.mean([r["img_grad_mean"] for r in rows])),
        "img_hf_fraction": float(np.mean([r["img_hf_fraction"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--outdir", type=Path, default=PROJECT / "reports/phase2a")
    args = ap.parse_args()

    records = read_manifest(args.manifest)[: args.limit]
    rows: list[dict] = []
    for ss in (2, 4):
        t0 = time.perf_counter()
        for i, rec in enumerate(records, 1):
            rows.append(run_one(rec, ss))
            if i % 10 == 0 or i == len(records):
                print(f"  ss={ss} [{i}/{len(records)}] {time.perf_counter()-t0:.0f}s", flush=True)

    by = {ss: [r for r in rows if r["supersample"] == ss] for ss in (2, 4)}
    paired = {}
    for ss in (2, 4):
        paired[ss] = {r["sample_id"]: r["error_px"] for r in by[ss]}
    ids = sorted(set(paired[2]) & set(paired[4]))
    d = np.array([paired[4][i] - paired[2][i] for i in ids])
    flips = sum(
        1 for i in ids
        if (paired[2][i] > CATASTROPHIC_PX) != (paired[4][i] > CATASTROPHIC_PX)
    )

    summary = {
        "pairs": len(ids),
        "supersample_2": side(by[2]),
        "supersample_4": side(by[4]),
        "paired_delta_error_px": {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "abs_median": float(np.median(np.abs(d))),
            "max_abs": float(np.abs(d).max()),
            "n_changed_gt1px": int(np.sum(np.abs(d) > 1.0)),
            "n_catastrophic_status_flips": flips,
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "supersample_comparison.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n", encoding="utf-8")
    cols = list(rows[0].keys())
    with (args.outdir / "supersample_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    s2, s4 = summary["supersample_2"], summary["supersample_4"]
    print("\n=== SUPERSAMPLE 2 vs 4 ===")
    print(f"{'metric':28s} {'ss=2':>12s} {'ss=4':>12s}")
    for k in ("acc_5px", "median_error_px", "p95_error_px", "catastrophic_rate",
              "alias_fraction_of_catastrophic", "coverage", "mean_gen_s",
              "mean_locate_s", "mean_peak_mem_mb", "img_std", "img_grad_mean",
              "img_hf_fraction"):
        v2, v4 = s2[k], s4[k]
        f2 = "n/a" if v2 is None else f"{v2:12.4f}"
        f4 = "n/a" if v4 is None else f"{v4:12.4f}"
        print(f"{k:28s} {f2} {f4}")
    print(f"\npaired |delta| median {summary['paired_delta_error_px']['abs_median']:.3f} px, "
          f"max {summary['paired_delta_error_px']['max_abs']:.1f} px, "
          f"changed>1px {summary['paired_delta_error_px']['n_changed_gt1px']}/{len(ids)}, "
          f"catastrophic flips {summary['paired_delta_error_px']['n_catastrophic_status_flips']}")


if __name__ == "__main__":
    main()
