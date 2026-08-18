"""Why is candidate recall only 65%? Is it selection, or is the evidence absent?

Three questions, answered per pair:

  Q1 Is the true site a local maximum of the correlation surface at all?
     If not, no NMS setting and no lattice expansion can ever propose it -
     the failure is upstream, in the evidence.
  Q2 If it is, where does it rank among all local maxima?
     A poor rank is a selection problem (fixable by reranking).
  Q3 Does lattice expansion from ALL seeds (a strict superset of C0) recover it?
     C1 in the first run anchored on 6 seeds while C0 used 25, so it was not a
     superset and the comparison was confounded. C2 fixes that.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample  # noqa: E402
from driftforge.lattice import estimate_lattice  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402
from experiments.candidate_eval import response_and_seeds, score_at  # noqa: E402


def run_one(args):
    rec, ss = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], ss)
    gx, gy = s.gt_x, s.gt_y
    best, seeds = response_and_seeds(s.reference, s.search)
    _, resp, hw, hh, _, _ = best

    # --- Q1/Q2: the response surface itself --------------------------------
    v_true = score_at(resp, hw, hh, gx, gy)
    v_max = float(resp.max())
    # all local maxima of the best surface (5x5 neighbourhood, not the 30px NMS)
    loc = resp == ndimage.maximum_filter(resp, size=5, mode="nearest")
    ys, xs = np.nonzero(loc)
    vals = resp[ys, xs]
    cx = xs + hw
    cy = ys + hh
    d = np.hypot(cx - gx, cy - gy)
    near = d <= 5.0
    is_local_max = bool(near.any())
    if is_local_max:
        v_near = float(vals[near].max())
        rank = int((vals > v_near).sum()) + 1          # 1 = global max
    else:
        v_near, rank = float("nan"), -1
    dist_nearest_localmax = float(d.min()) if d.size else float("inf")

    # --- Q3: C0 vs C2 (all seeds + lattice expansion) -----------------------
    def hit(cands, tol=5.0):
        if not cands:
            return False
        return bool(min(math.hypot(x - gx, y - gy) for x, y in cands) <= tol)

    c0 = [(x, y) for _v, x, y, _sc, _r in seeds[:25]]
    lat = estimate_lattice(s.search)
    trans = lat.translations(max_m=5, max_n=5)
    H, W = s.search.shape
    seen, c2 = set(), []
    for _v, x, y, _sc, _r in seeds[:25]:
        for dx, dy in [(0.0, 0.0)] + [tuple(t) for t in trans]:
            X, Y = x + dx, y + dy
            k = (int(round(X)), int(round(Y)))
            if k not in seen and 0 <= X < W and 0 <= Y < H:
                seen.add(k); c2.append((X, Y))

    return {
        "sample_id": rec["id"], "architecture": rec["architecture"],
        "profile": rec["profile"],
        "v_true": v_true, "v_max": v_max,
        "score_deficit": v_max - v_true,
        "true_is_local_max": is_local_max,
        "true_rank_among_local_maxima": rank,
        "n_local_maxima": int(vals.size),
        "dist_to_nearest_local_max": dist_nearest_localmax,
        "C0_hit5": hit(c0), "C0_n": len(c0),
        "C2_hit5": hit(c2), "C2_n": len(c2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT / "manifests/validation_phase2a_400.jsonl")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=PROJECT / "experiments/war45")
    a = ap.parse_args()
    recs = read_manifest(a.manifest)[a.start:a.start + a.limit]
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(run_one, [(x, 2) for x in recs]), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(recs):
                print(f"  [{i}/{len(recs)}] {time.perf_counter()-t0:.0f}s", flush=True)

    n = len(rows)
    lm = sum(r["true_is_local_max"] for r in rows)
    ranks = [r["true_rank_among_local_maxima"] for r in rows if r["true_rank_among_local_maxima"] > 0]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "diag_evidence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = {
        "pairs": n,
        "true_is_local_max_rate": lm / n,
        "C0_recall5": sum(r["C0_hit5"] for r in rows) / n,
        "C2_recall5": sum(r["C2_hit5"] for r in rows) / n,
        "C2_median_candidates": float(np.median([r["C2_n"] for r in rows])),
        "rank_median": float(np.median(ranks)) if ranks else None,
        "rank_p90": float(np.percentile(ranks, 90)) if ranks else None,
        "rank_top1_rate": (sum(1 for r in ranks if r == 1) / n) if ranks else 0.0,
        "rank_top10_rate": (sum(1 for r in ranks if r <= 10) / n) if ranks else 0.0,
        "rank_top50_rate": (sum(1 for r in ranks if r <= 50) / n) if ranks else 0.0,
        "median_local_maxima": float(np.median([r["n_local_maxima"] for r in rows])),
        "median_score_deficit": float(np.median([r["score_deficit"] for r in rows])),
        "median_dist_to_nearest_local_max": float(np.median([r["dist_to_nearest_local_max"] for r in rows])),
    }
    (a.out / "diag_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== EVIDENCE DIAGNOSIS ===")
    print(f"pairs                                {n}")
    print(f"true site IS a local max (<=5px)     {summary['true_is_local_max_rate']:.1%}   <- CEILING on any proposer")
    print(f"  median local maxima per surface    {summary['median_local_maxima']:.0f}")
    print(f"  median dist truth->nearest localmax {summary['median_dist_to_nearest_local_max']:.2f} px")
    print(f"true site rank among local maxima:   median {summary['rank_median']}  p90 {summary['rank_p90']}")
    print(f"  rank==1 (global max)               {summary['rank_top1_rate']:.1%}")
    print(f"  rank<=10                           {summary['rank_top10_rate']:.1%}")
    print(f"  rank<=50                           {summary['rank_top50_rate']:.1%}")
    print(f"median score deficit (max - true)    {summary['median_score_deficit']:.4f}")
    print(f"C0 recall@5                          {summary['C0_recall5']:.1%}")
    print(f"C2 recall@5 (all seeds + lattice)    {summary['C2_recall5']:.1%}  "
          f"(median {summary['C2_median_candidates']:.0f} candidates)")


if __name__ == "__main__":
    main()
