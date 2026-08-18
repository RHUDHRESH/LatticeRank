#!/usr/bin/env python3
"""E1: leak-free baseline of the *unchanged* v1 locator.

Runs `driftforge.baseline.locate` exactly as shipped over N validation pairs
rendered by the Phase-2A (de-correlated) generator, and classifies every
catastrophic failure against the captured lattice.

The locator is not touched. This script only measures it.

Usage:
    python reports/phase2a/diagnostics/run_leakfree_baseline.py --limit 300
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

CATASTROPHIC_PX = 25.0


# ----------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Wilson score interval. Correct near 0 and 1, unlike the normal approx."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_median(values: np.ndarray, n_boot: int = 4000, seed: int = 7) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    meds = np.median(values[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def acc_block(errors: np.ndarray, tol: float) -> dict:
    k = int(np.sum(errors <= tol))
    p, lo, hi = wilson(k, errors.size)
    return {"k": k, "n": int(errors.size), "acc": p, "ci95_lo": lo, "ci95_hi": hi}


# --------------------------------------------------- direct patch equivalence

PATCH = 100  # Reference footprint at Search scale, per the 10x calibration


def _patch(img: np.ndarray, cx: float, cy: float, size: int = PATCH) -> np.ndarray | None:
    h = size // 2
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    if x0 < 0 or y0 < 0 or x0 + size > img.shape[1] or y0 + size > img.shape[0]:
        return None
    return img[y0:y0 + size, x0:x0 + size].astype(np.float32)


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt(float(a.ravel() @ a.ravel()) * float(b.ravel() @ b.ravel())))
    return float(a.ravel() @ b.ravel() / d) if d > 1e-9 else float("nan")


def patch_equivalence(search: np.ndarray, true_xy, pred_xy, rng_seed: int) -> dict:
    """Do the true and predicted locations actually look the same?

    This is the assumption-free version of the alias test. The lattice-residual
    test is weak here (see alias_null_control.py: the non-affine warp budget is
    comparable to half a lattice cell), so instead of asking "is the offset an
    integer number of periods?" we ask the question that actually matters for
    architecture: *is the thing the locator picked visually indistinguishable
    from the thing it should have picked?*

      high similarity -> a genuine repeat. No local matcher can separate them;
                         this needs global context or the official centre rule.
      low  similarity -> the locator chose something that does not even look
                         like the target. Better evidence CAN fix this.

    A per-image null (random in-bounds locations) calibrates what "high" means
    for this particular texture, so the threshold is not a guess.
    """
    tp = _patch(search, *true_xy)
    pp = _patch(search, *pred_xy)
    if tp is None or pp is None:
        return {"patch_zncc": float("nan"), "patch_null_p95": float("nan"),
                "patch_null_median": float("nan"), "patch_excess": float("nan")}
    obs = _zncc(tp, pp)
    rng = np.random.default_rng(rng_seed)
    h = PATCH // 2
    null = []
    for _ in range(60):
        rx = float(rng.integers(h, search.shape[1] - h))
        ry = float(rng.integers(h, search.shape[0] - h))
        q = _patch(search, rx, ry)
        if q is not None:
            v = _zncc(tp, q)
            if math.isfinite(v):
                null.append(v)
    if not null:
        return {"patch_zncc": obs, "patch_null_p95": float("nan"),
                "patch_null_median": float("nan"), "patch_excess": float("nan")}
    null = np.asarray(null)
    p95 = float(np.percentile(null, 95))
    return {"patch_zncc": obs, "patch_null_p95": p95,
            "patch_null_median": float(np.median(null)),
            "patch_excess": float(obs - p95)}


# ------------------------------------------------------------------ one pair


def run_one(row: dict, supersample: int) -> dict:
    sample = generate_sample(int(row["seed"]), row["architecture"], row["profile"], supersample)
    started = time.perf_counter()
    result = locate(sample.reference, sample.search)
    runtime_ms = (time.perf_counter() - started) * 1000.0

    dx = result.x - sample.gt_x
    dy = result.y - sample.gt_y
    error = math.hypot(dx, dy)

    # Lattice basis of the *captured* Search image, per Phase-1 lattice_fit.py:
    # the naive dx/pitch test is biased by the Search-side scale and rotation.
    acq = sample.search_acquisition
    t = math.radians(acq.rotation_deg)
    px = acq.scale * sample.scene.pitch_x_nm / 10.0
    py = acq.scale * sample.scene.pitch_y_nm / 10.0
    a = np.array([px * math.cos(t), px * math.sin(t)])
    b = np.array([-py * math.sin(t), py * math.cos(t)])
    B = np.column_stack([a, b])
    mn = np.linalg.solve(B, np.array([dx, dy]))
    m_i, n_i = round(float(mn[0])), round(float(mn[1]))
    res_m, res_n = abs(float(mn[0]) - m_i), abs(float(mn[1]) - n_i)
    residual_px = float(np.linalg.norm(B @ np.array([float(mn[0]) - m_i, float(mn[1]) - n_i])))

    equiv = patch_equivalence(
        sample.search, (sample.gt_x, sample.gt_y), (result.x, result.y),
        rng_seed=int(row["seed"]) * 7 + 5,
    )

    return {
        "sample_id": row["id"],
        "seed": int(row["seed"]),
        **equiv,
        "architecture": row["architecture"],
        "profile": row["profile"],
        "true_x": sample.gt_x,
        "true_y": sample.gt_y,
        "pred_x": result.x,
        "pred_y": result.y,
        "error_px": error,
        "accepted": bool(result.accepted),
        "runtime_ms": runtime_ms,
        "peak_gap": result.peak_gap,
        "psr": result.psr,
        "selected_scale": result.scale,
        "selected_rotation": result.rotation_deg,
        "score": result.score,
        "ambiguity_detected": bool(result.ambiguity_detected),
        # Lattice diagnostics (meaningful for catastrophic rows).
        "dx": dx,
        "dy": dy,
        "estimated_pitch_x": px,
        "estimated_pitch_y": py,
        "lattice_m": m_i,
        "lattice_n": n_i,
        "lattice_residual_m": res_m,
        "lattice_residual_n": res_n,
        "lattice_residual_px": residual_px,
        "parity_preserving": bool((m_i + n_i) % 2 == 0),
        "true_scale": acq.scale,
        "true_rotation": acq.rotation_deg,
    }


def classify_lattice(row: dict) -> str:
    """Legacy lattice-arithmetic classifier, retained for comparison only.

    KNOWN WEAK. alias_null_control.py shows the non-affine warp budget
    (raster shear + row jitter + radial distortion, none of which are in the
    affine lattice basis) is comparable to half a lattice cell, so residual
    size barely separates real aliases from chance. Reported so the headline
    number can be shown NOT to depend on this instrument.
    """
    if row["lattice_residual_m"] <= 0.25 and row["lattice_residual_n"] <= 0.25:
        return "periodic_alias"
    if (abs(row["selected_scale"] - row["true_scale"]) > 0.035
            or abs(row["selected_rotation"] - row["true_rotation"]) > 2.0):
        return "wrong_transform"
    if row["psr"] < 4.0 or row["score"] < 0.25:
        return "low_snr"
    if row["profile"] == "boundary":
        return "boundary_context"
    return "other"


def classify(row: dict) -> str:
    """Evidence-based failure class for a catastrophic (>25 px) row.

    Primary evidence is direct image similarity between the predicted and true
    locations, calibrated per image against a random-location null. If the
    locator landed on something that genuinely looks like the target, that is
    a periodic alias whatever the lattice arithmetic says; if it landed on
    something that does not, no amount of integer bookkeeping makes it one.

    Ordering is deliberate: the two *mechanical* faults (wrong transform, dead
    signal) are excluded first, because either can produce a high-similarity
    landing for reasons that are not aliasing.
    """
    if (abs(row["selected_scale"] - row["true_scale"]) > 0.035
            or abs(row["selected_rotation"] - row["true_rotation"]) > 2.0):
        return "wrong_transform"
    if row["psr"] < 4.0 or row["score"] < 0.25:
        return "low_snr"

    z, p95 = row.get("patch_zncc"), row.get("patch_null_p95")
    if z is None or not math.isfinite(z) or p95 is None or not math.isfinite(p95):
        return "other"
    # Clearly above what a random location in this same texture achieves, and
    # high in absolute terms: the two sites are visually equivalent repeats.
    if z >= 0.60 and z > p95:
        return "periodic_alias"
    if z > p95:
        return "boundary_context"   # elevated but not equivalent: partial context match
    return "matcher_error"          # picked something unlike the target


# ---------------------------------------------------------------- aggregation


def summarize(rows: list[dict], tolerance: float) -> dict:
    errors = np.array([r["error_px"] for r in rows])
    runtimes = np.array([r["runtime_ms"] for r in rows])
    accepted = np.array([r["accepted"] for r in rows], dtype=bool)
    cat = errors > CATASTROPHIC_PX

    med_lo, med_hi = bootstrap_median(errors)
    acc_cov, cov_lo, cov_hi = wilson(int(accepted.sum()), len(rows))
    acc_incorrect = int(np.sum(accepted & (errors > tolerance)))
    acc_cat = int(np.sum(accepted & cat))

    out = {
        "pairs": len(rows),
        "accuracy": {str(t): acc_block(errors, t) for t in (1, 2, 5, 10)},
        "median_error_px": float(np.median(errors)),
        "median_error_px_ci95": [med_lo, med_hi],
        "mean_error_px": float(errors.mean()),
        "p90_error_px": float(np.percentile(errors, 90)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "p99_error_px": float(np.percentile(errors, 99)),
        "max_error_px": float(errors.max()),
        "catastrophic_gt25px_count": int(cat.sum()),
        "catastrophic_gt25px_rate": float(cat.mean()),
        "accepted_coverage": acc_cov,
        "accepted_coverage_ci95": [cov_lo, cov_hi],
        "accepted_incorrect_at_tolerance": acc_incorrect,
        "accepted_incorrect_gt25px": acc_cat,
        "tolerance_px": tolerance,
        "mean_runtime_ms": float(runtimes.mean()),
        "median_runtime_ms": float(np.median(runtimes)),
        "p95_runtime_ms": float(np.percentile(runtimes, 95)),
    }

    def slice_stats(subset: list[dict]) -> dict:
        if not subset:
            return {"pairs": 0}
        e = np.array([r["error_px"] for r in subset])
        c = e > CATASTROPHIC_PX
        return {
            "pairs": len(subset),
            "acc_5px": acc_block(e, 5.0),
            "acc_1px": acc_block(e, 1.0),
            "median_error_px": float(np.median(e)),
            "p95_error_px": float(np.percentile(e, 95)),
            "catastrophic_count": int(c.sum()),
            "catastrophic_rate": float(c.mean()),
        }

    archs = sorted({r["architecture"] for r in rows})
    profs = sorted({r["profile"] for r in rows})
    out["by_architecture"] = {
        a: slice_stats([r for r in rows if r["architecture"] == a]) for a in archs
    }
    out["by_profile"] = {
        p: slice_stats([r for r in rows if r["profile"] == p]) for p in profs
    }
    out["by_architecture_profile"] = {
        f"{a}|{p}": slice_stats([r for r in rows if r["architecture"] == a and r["profile"] == p])
        for a in archs for p in profs
    }

    cat_rows = [r for r in rows if r["error_px"] > CATASTROPHIC_PX]
    classes = [r["failure_class"] for r in cat_rows]
    n_cat = len(cat_rows)
    n_alias = classes.count("periodic_alias")
    p_alias, alias_lo, alias_hi = wilson(n_alias, n_cat) if n_cat else (0.0, 0.0, 0.0)
    out["failure_classification"] = {
        "catastrophic_total": n_cat,
        "periodic_alias_count": n_alias,
        "periodic_alias_fraction": p_alias,
        "periodic_alias_fraction_ci95": [alias_lo, alias_hi],
        "wrong_transform_count": classes.count("wrong_transform"),
        "low_snr_count": classes.count("low_snr"),
        "boundary_context_count": classes.count("boundary_context"),
        "matcher_error_count": classes.count("matcher_error"),
        "other_count": classes.count("other"),
        "parity_preserving_among_aliases": int(
            sum(1 for r in cat_rows if r["failure_class"] == "periodic_alias" and r["parity_preserving"])
        ),
        "median_lattice_residual_px_among_aliases": (
            float(np.median([r["lattice_residual_px"] for r in cat_rows
                             if r["failure_class"] == "periodic_alias"]))
            if n_alias else None
        ),
    }

    # --- independent corroboration, not part of the classifier -------------
    # Contacts sit on a checkerboard, so only translates with (m+n) even
    # preserve them. Parity is discrete: chance landings give 50/50. An excess
    # is evidence of genuine lattice structure that does not depend on any
    # residual threshold, and so is immune to the weakness of the lattice test.
    if n_cat:
        k_par = int(sum(1 for r in cat_rows if r["parity_preserving"]))
        p_par = k_par / n_cat
        z_par = (p_par - 0.5) / math.sqrt(0.25 / n_cat)
        out["failure_classification"]["parity_even_count"] = k_par
        out["failure_classification"]["parity_even_fraction"] = p_par
        out["failure_classification"]["parity_even_z_vs_50pct"] = z_par

    # --- legacy instrument, for transparency -------------------------------
    legacy = [r["failure_class_lattice"] for r in cat_rows]
    n_leg = legacy.count("periodic_alias")
    p_leg, leg_lo, leg_hi = wilson(n_leg, n_cat) if n_cat else (0.0, 0.0, 0.0)
    out["failure_classification_lattice_legacy"] = {
        "periodic_alias_count": n_leg,
        "periodic_alias_fraction": p_leg,
        "periodic_alias_fraction_ci95": [leg_lo, leg_hi],
        "note": "weak instrument; see reports/phase2a/alias_null_control.json",
    }

    if cat_rows:
        pz = np.array([r["patch_zncc"] for r in cat_rows], dtype=float)
        pe = np.array([r["patch_excess"] for r in cat_rows], dtype=float)
        good = np.isfinite(pz)
        out["patch_equivalence"] = {
            "n": int(good.sum()),
            "median_patch_zncc": float(np.median(pz[good])) if good.any() else None,
            "median_excess_over_null_p95": float(np.median(pe[np.isfinite(pe)]))
            if np.isfinite(pe).any() else None,
            "frac_above_null_p95": float(np.mean(pe[np.isfinite(pe)] > 0))
            if np.isfinite(pe).any() else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default="leakfree_baseline")
    ap.add_argument("--outdir", type=Path, default=PROJECT / "reports/phase2a")
    args = ap.parse_args()

    records = read_manifest(args.manifest)[args.start:args.start + args.limit]
    print(f"running {len(records)} pairs, supersample={args.search_supersample}, "
          f"workers={args.workers}", flush=True)

    rows: list[dict] = []
    wall0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, r, args.search_supersample) for r in records]
            for i, fut in enumerate(futures, 1):
                rows.append(fut.result())
                if i % 20 == 0 or i == len(futures):
                    print(f"  [{i}/{len(futures)}] elapsed={time.perf_counter()-wall0:.0f}s", flush=True)
    else:
        for i, r in enumerate(records, 1):
            rows.append(run_one(r, args.search_supersample))
            if i % 10 == 0 or i == len(records):
                print(f"  [{i}/{len(records)}] elapsed={time.perf_counter()-wall0:.0f}s", flush=True)
    wall = time.perf_counter() - wall0

    rows.sort(key=lambda r: r["sample_id"])
    for r in rows:
        cat = r["error_px"] > CATASTROPHIC_PX
        r["failure_class"] = classify(r) if cat else "ok"
        r["failure_class_lattice"] = classify_lattice(r) if cat else "ok"

    summary = summarize(rows, args.tolerance)
    summary["wall_clock_s"] = wall
    summary["workers"] = args.workers
    summary["search_supersample"] = args.search_supersample
    summary["manifest"] = str(args.manifest.name)
    summary["manifest_slice"] = [args.start, args.start + len(records)]
    summary["runtime_note"] = (
        "runtime_ms is measured inside each worker around locate() only. With "
        f"workers={args.workers} these values include CPU contention and are an "
        "upper bound on single-process latency."
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"{args.tag}.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n", encoding="utf-8")

    pred_cols = ["sample_id", "seed", "architecture", "profile", "true_x", "true_y",
                 "pred_x", "pred_y", "error_px", "accepted", "runtime_ms", "peak_gap",
                 "psr", "selected_scale", "selected_rotation"]
    with (args.outdir / f"{args.tag}_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=pred_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    cat_cols = pred_cols + ["dx", "dy", "estimated_pitch_x", "estimated_pitch_y",
                            "lattice_m", "lattice_n", "lattice_residual_px",
                            "parity_preserving", "patch_zncc", "patch_null_p95",
                            "patch_excess", "failure_class", "failure_class_lattice"]
    cat_rows = [r for r in rows if r["error_px"] > CATASTROPHIC_PX]
    with (args.outdir / f"{args.tag}_catastrophic.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cat_rows)

    a5 = summary["accuracy"]["5"]
    fc = summary["failure_classification"]
    print("\n=== LEAK-FREE BASELINE ===")
    print(f"pairs                {summary['pairs']}")
    print(f"acc <=1px            {summary['accuracy']['1']['acc']*100:.1f}%")
    print(f"acc <=2px            {summary['accuracy']['2']['acc']*100:.1f}%")
    print(f"acc <=5px            {a5['acc']*100:.1f}%  "
          f"(95% CI {a5['ci95_lo']*100:.1f}-{a5['ci95_hi']*100:.1f})")
    print(f"acc <=10px           {summary['accuracy']['10']['acc']*100:.1f}%")
    print(f"median / P95 / P99   {summary['median_error_px']:.2f} / "
          f"{summary['p95_error_px']:.1f} / {summary['p99_error_px']:.1f} px")
    print(f"catastrophic >25px   {summary['catastrophic_gt25px_count']} "
          f"({summary['catastrophic_gt25px_rate']*100:.1f}%)")
    print(f"alias fraction       {fc['periodic_alias_fraction']*100:.1f}% "
          f"(CI {fc['periodic_alias_fraction_ci95'][0]*100:.1f}-"
          f"{fc['periodic_alias_fraction_ci95'][1]*100:.1f})  "
          f"[{fc['periodic_alias_count']}/{fc['catastrophic_total']}]")
    print(f"coverage             {summary['accepted_coverage']*100:.1f}%  "
          f"accepted-incorrect@{args.tolerance:g}px={summary['accepted_incorrect_at_tolerance']}")
    print(f"wall clock           {wall:.0f}s")


if __name__ == "__main__":
    main()
