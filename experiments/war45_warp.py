"""Does the non-affine warp budget explain the lattice-expansion failure?

Lattice expansion underperformed plain score-ranking (72.5% vs 81.2%) despite
emitting 12,686 candidates. Hypothesis: the generator applies raster shear,
per-row jitter and radial distortion AFTER the affine transform. Phase 2A
measured that budget at 6.52 px median - larger than the 5 px tolerance. If so,
translating a seed by an exact lattice vector lands NEAR the truth but outside
tolerance, and no amount of lattice expansion can fix it.

Discriminating prediction: the misses should cluster just outside 5 px (a
near-miss band), not be scattered at random distances.
"""
from __future__ import annotations

import json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample  # noqa: E402
from driftforge.lattice import estimate_lattice  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402
from experiments.candidate_eval import response_and_seeds  # noqa: E402


def one(rec):
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
    gx, gy = s.gt_x, s.gt_y
    best, seeds = response_and_seeds(s.reference, s.search)
    _, resp, hw, hh, _, _ = best
    lat = estimate_lattice(s.search)
    trans = lat.translations(max_m=5, max_n=5)
    H, W = s.search.shape
    dmin = float("inf")
    for _v, x, y, _sc, _r in seeds[:25]:
        for dx, dy in [(0.0, 0.0)] + [tuple(t) for t in trans]:
            X, Y = x + dx, y + dy
            if 0 <= X < W and 0 <= Y < H:
                d = math.hypot(X - gx, Y - gy)
                if d < dmin:
                    dmin = d
    acq = s.search_acquisition
    warp = abs(acq.radial_k) * 2.0 * 499.5 + abs(acq.scan_shear_px) + 2.0 * acq.scan_jitter_px
    return {"id": rec["id"], "dmin": dmin, "warp": warp, "arch": rec["architecture"]}


def main():
    recs = read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")[200:280]
    with ProcessPoolExecutor(max_workers=6) as p:
        rows = list(p.map(one, recs))
    d = np.array([r["dmin"] for r in rows])
    w = np.array([r["warp"] for r in rows])
    miss = d > 5.0
    near = (d > 5.0) & (d <= 25.0)
    out = {
        "pairs": len(rows),
        "median_nearest_dist_px": float(np.median(d)),
        "p75": float(np.percentile(d, 75)), "p90": float(np.percentile(d, 90)),
        "recall5": float(np.mean(~miss)),
        "misses": int(miss.sum()),
        "miss_median_dist_px": float(np.median(d[miss])) if miss.any() else None,
        "miss_p25_dist_px": float(np.percentile(d[miss], 25)) if miss.any() else None,
        "near_miss_5_to_25px": int(near.sum()),
        "near_miss_share_of_misses": float(near.sum() / max(miss.sum(), 1)),
        "warp_median_px": float(np.median(w)), "warp_p90_px": float(np.percentile(w, 90)),
        "corr_dist_warp": float(np.corrcoef(d, w)[0, 1]),
    }
    (PROJECT / "experiments/war45/warp_limit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"pairs {out['pairs']}")
    print(f"nearest lattice-expanded candidate -> truth:")
    print(f"  median {out['median_nearest_dist_px']:.2f} px  p75 {out['p75']:.2f}  p90 {out['p90']:.2f}")
    print(f"  recall@5 {out['recall5']:.1%}   misses {out['misses']}")
    if out["miss_median_dist_px"]:
        print(f"  MISSES: median {out['miss_median_dist_px']:.2f} px  p25 {out['miss_p25_dist_px']:.2f}")
    print(f"  near-misses 5-25 px: {out['near_miss_5_to_25px']} "
          f"({out['near_miss_share_of_misses']:.0%} of all misses)")
    print(f"warp budget: median {out['warp_median_px']:.2f} px  p90 {out['warp_p90_px']:.2f}")
    print(f"corr(nearest-dist, warp) = {out['corr_dist_warp']:+.3f}")
    for a in ("dram", "finfet"):
        dd = np.array([r["dmin"] for r in rows if r["arch"] == a])
        print(f"  {a}: median {np.median(dd):.2f} px  recall@5 {np.mean(dd <= 5):.1%}")


if __name__ == "__main__":
    main()
