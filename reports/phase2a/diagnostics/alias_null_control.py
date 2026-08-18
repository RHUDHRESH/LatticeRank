#!/usr/bin/env python3
"""Is 'lands on a lattice translate' actually evidence, or is it automatic?

The lattice periods here are 3-17 Search px. A point drawn at random is
already within half a cell diagonal of *some* lattice site, so a small
residual is not self-evidently meaningful. Before any alias fraction is
reported, the observed residuals must be compared against a null model.

Null: keep each row's real lattice basis, but replace the observed
displacement with one that carries no alias information. Two nulls are used
because they fail in different ways:

  * `uniform`  - displacement drawn uniformly over the Search field. Tests
                 "would any wrong answer look like an alias?"
  * `shuffled` - displacements permuted across rows, so the magnitude
                 distribution of real failures is preserved exactly and only
                 the pairing with the lattice is destroyed. This is the
                 stricter control.

Also reported: the non-affine warp budget. The lattice basis models only the
Search affine (scale + rotation). The generator additionally applies raster
shear, per-row jitter and radial distortion, none of which are in the basis,
so a true alias cannot have zero residual - it must sit within roughly that
budget. That is what sets an honest residual threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402


def residual_px(dx: float, dy: float, px: float, py: float, rot_deg: float) -> float:
    t = math.radians(rot_deg)
    a = np.array([px * math.cos(t), px * math.sin(t)])
    b = np.array([-py * math.sin(t), py * math.cos(t)])
    B = np.column_stack([a, b])
    mn = np.linalg.solve(B, np.array([dx, dy]))
    frac = mn - np.round(mn)
    return float(np.linalg.norm(B @ frac))


def warp_budget(rows: list[dict], manifest: Path, supersample: int) -> dict:
    """Peak non-affine displacement the lattice basis cannot represent.

    Measured from the actual AcquisitionSpec of a sample of the failing rows:
    radial distortion at the image corner, plus shear, plus 2-sigma row jitter.
    """
    recs = {r["id"]: r for r in read_manifest(manifest)}
    out = []
    for row in rows[:40]:
        rec = recs[row["sample_id"]]
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], supersample)
        acq = s.search_acquisition
        # radial: factor = 1 + k*r2, r2 = 2 at the corner, lever arm ~499.5 px
        radial = abs(acq.radial_k) * 2.0 * 499.5
        shear = abs(acq.scan_shear_px)
        jitter = 2.0 * acq.scan_jitter_px
        out.append(radial + shear + jitter)
    arr = np.asarray(out)
    return {"n_sampled": int(arr.size), "median_px": float(np.median(arr)),
            "p95_px": float(np.percentile(arr, 95)), "max_px": float(arr.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catastrophic", type=Path,
                    default=PROJECT / "reports/phase2a/leakfree_baseline_catastrophic.csv")
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "reports/phase2a/alias_null_control.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.catastrophic.open(encoding="utf-8")))
    px = np.array([float(r["estimated_pitch_x"]) for r in rows])
    py = np.array([float(r["estimated_pitch_y"]) for r in rows])
    rot = np.array([float(r["selected_rotation"]) for r in rows])
    dx = np.array([float(r["dx"]) for r in rows])
    dy = np.array([float(r["dy"]) for r in rows])

    observed = np.array([residual_px(dx[i], dy[i], px[i], py[i], rot[i]) for i in range(len(rows))])

    rng = np.random.default_rng(11)
    null_u, null_s = [], []
    for _ in range(args.n_boot):
        # uniform: any position in the 1000x1000 field, same basis
        udx = rng.uniform(-999, 999, size=len(rows))
        udy = rng.uniform(-999, 999, size=len(rows))
        null_u.append([residual_px(udx[i], udy[i], px[i], py[i], rot[i]) for i in range(len(rows))])
        # shuffled: real displacement magnitudes, wrong lattice pairing
        perm = rng.permutation(len(rows))
        null_s.append([residual_px(dx[perm[i]], dy[perm[i]], px[i], py[i], rot[i])
                       for i in range(len(rows))])
    null_u = np.asarray(null_u).ravel()
    null_s = np.asarray(null_s).ravel()

    # Half the cell diagonal: the largest possible distance to a lattice site.
    half_diag = 0.5 * np.hypot(px, py)

    def block(a: np.ndarray) -> dict:
        return {"median": float(np.median(a)), "p05": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max()),
                "frac_under_1px": float(np.mean(a < 1.0)),
                "frac_under_3px": float(np.mean(a < 3.0)),
                "frac_under_6px": float(np.mean(a < 6.0))}

    budget = warp_budget(rows, args.manifest, args.supersample)

    # Effect size: how far the observed distribution sits below the null.
    z_u = (np.median(null_u) - np.median(observed)) / (np.std(null_u) + 1e-12)
    z_s = (np.median(null_s) - np.median(observed)) / (np.std(null_s) + 1e-12)
    # Permutation p-value against the shuffled null on the median statistic.
    boot_meds = np.median(np.asarray(null_s).reshape(args.n_boot, -1), axis=1)
    p_perm = float((np.sum(boot_meds <= np.median(observed)) + 1) / (args.n_boot + 1))

    out = {
        "n_catastrophic": len(rows),
        "observed_residual_px": block(observed),
        "null_uniform_residual_px": block(null_u),
        "null_shuffled_residual_px": block(null_s),
        "half_cell_diagonal_px": {"median": float(np.median(half_diag)),
                                  "max": float(half_diag.max())},
        "non_affine_warp_budget_px": budget,
        "effect_size_sigma_vs_uniform": float(z_u),
        "effect_size_sigma_vs_shuffled": float(z_s),
        "permutation_p_value_median": p_perm,
        "interpretation": (
            "If observed residuals are not clearly below BOTH nulls, 'lands on a "
            "lattice translate' is not evidence and the alias classification must "
            "be discarded."
        ),
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"catastrophic rows            {len(rows)}")
    print(f"half cell diagonal (median)  {np.median(half_diag):6.2f} px  "
          f"<- max possible distance to a lattice site")
    print(f"non-affine warp budget       median {budget['median_px']:.2f}  "
          f"p95 {budget['p95_px']:.2f}  max {budget['max_px']:.2f} px")
    print()
    print(f"{'distribution':22s} {'median':>8s} {'p95':>8s} {'max':>8s} "
          f"{'<1px':>7s} {'<3px':>7s} {'<6px':>7s}")
    for name, a in (("OBSERVED", observed), ("null: uniform", null_u), ("null: shuffled", null_s)):
        b = block(a)
        print(f"{name:22s} {b['median']:8.2f} {b['p95']:8.2f} {b['max']:8.2f} "
              f"{b['frac_under_1px']:7.1%} {b['frac_under_3px']:7.1%} {b['frac_under_6px']:7.1%}")
    print()
    print(f"effect size vs uniform null   {z_u:6.2f} sigma")
    print(f"effect size vs shuffled null  {z_s:6.2f} sigma")
    print(f"permutation p (median)        {p_perm:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
