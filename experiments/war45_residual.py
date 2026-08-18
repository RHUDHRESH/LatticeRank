"""war45: periodic-cancellation residual matching, full-pool benchmark.

One pass over the SAME 80 disjoint validation pairs, delta=0.15 adaptive
candidates (the measured 92.5%-oracle harvest), no pruning. Nine transparent
formulations - {intensity, 0.5int+0.5grad, 0.45int+0.45grad+0.10midband} x
{no mask, top-50% uniqueness, top-30% uniqueness} - evaluated on identical
candidates, scene-z-normalised. No ML, no tuning loop.
"""
from __future__ import annotations

import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.residual import ResidualMatcher  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

OUT = PROJECT / "experiments/war45"
DELTA = 0.15
POS_TOL = 5.0
CENTRE = 499.5

FORMS = {}
for mk in ("m0", "m50", "m30"):
    FORMS[f"A_{mk}"] = ((f"res_int_{mk}", 1.0),)
    FORMS[f"B_{mk}"] = ((f"res_int_{mk}", 0.5), (f"res_grad_{mk}", 0.5))
    FORMS[f"C_{mk}"] = ((f"res_int_{mk}", 0.45), (f"res_grad_{mk}", 0.45), ("midband", 0.10))


def _job(rec):
    t0 = time.perf_counter()
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
    cm = response_maps(s.reference, s.search)
    cands = harvest(cm, delta=DELTA)
    cands.sort(key=lambda c: -max(v for k, v in c.items() if k in CHANNELS
                                  and isinstance(v, float) and math.isfinite(v)))
    cands = cands[:8000]
    rm = ResidualMatcher(s.reference, s.search)
    rows = []
    for c in cands:
        r = {"x": c["x"], "y": c["y"], "midband": c["midband"],
             "error_px": math.hypot(c["x"] - s.gt_x, c["y"] - s.gt_y)}
        r.update(rm.score(c["x"], c["y"]))
        rows.append(r)
    return {"sample_id": rec["id"], "architecture": rec["architecture"],
            "profile": rec["profile"], "lattice_ok": rm.lattice_ok,
            "rows": rows, "wall_s": time.perf_counter() - t0}


def _z(v: np.ndarray) -> np.ndarray:
    m = np.isfinite(v)
    if m.sum() < max(3, 0.5 * v.size):        # lattice failed -> unusable channel
        return None
    mu, sd = v[m].mean(), v[m].std()
    z = (v - mu) / (sd if sd > 1e-9 else 1.0)
    z[~m] = -9.0
    return z


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    recs = read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")[200:200 + n]
    t0 = time.perf_counter()
    scenes = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, sc in enumerate(pool.map(_job, recs), 1):
            scenes.append(sc)
            if i % 20 == 0 or i == len(recs):
                print(f"  [{i}/{len(recs)}]", flush=True)
    wall = time.perf_counter() - t0

    results = {"n_pairs": len(scenes), "wall_s": wall,
               "per_pair_s": float(np.median([s["wall_s"] for s in scenes])),
               "lattice_ok_rate": float(np.mean([s["lattice_ok"] for s in scenes])),
               "oracle_recall5": float(np.mean(
                   [any(r["error_px"] <= POS_TOL for r in s["rows"]) for s in scenes])),
               "median_candidates": float(np.median([len(s["rows"]) for s in scenes]))}
    print(f"\noracle {results['oracle_recall5']:.1%}  "
          f"median cands {results['median_candidates']:.0f}  "
          f"lattice ok {results['lattice_ok_rate']:.1%}  wall {wall:.0f}s")

    for name, terms in FORMS.items():
        errs, top = [], {1: 0, 5: 0, 10: 0}
        arch_n, arch_hit = {}, {}
        for s in scenes:
            rows = s["rows"]
            if not rows:
                errs.append(9e9); continue
            mid = _z(np.array([r["midband"] if math.isfinite(r["midband"]) else np.nan
                               for r in rows], dtype=float))
            score = np.zeros(len(rows))
            usable = True
            for key, wgt in terms:
                v = np.array([r[key] for r in rows], dtype=float) if key != "midband" \
                    else np.array([r["midband"] for r in rows], dtype=float)
                z = _z(v)
                if z is None:
                    usable = False; break
                score = score + wgt * z
            if not usable:                     # graceful: fall back to midband
                score = mid if mid is not None else np.zeros(len(rows))
            e = np.array([r["error_px"] for r in rows])
            o = np.argsort(-score)
            errs.append(float(e[o[0]]))
            for K in top:
                top[K] += bool(np.any(e[o[:K]] <= POS_TOL))
            a = s["architecture"]
            arch_n[a] = arch_n.get(a, 0) + 1
            arch_hit[a] = arch_hit.get(a, 0) + int(e[o[0]] <= POS_TOL)
        e = np.array(errs)
        nn = len(scenes)
        res = {"top1": top[1] / nn, "top5": top[5] / nn, "top10": top[10] / nn,
               "acc_5px": float(np.mean(e <= 5)), "acc_2px": float(np.mean(e <= 2)),
               "acc_1px": float(np.mean(e <= 1)),
               "median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)),
               "cat_rate_gt25": float(np.mean(e > 25)),
               **{f"{k}_top1": arch_hit.get(k, 0) / v for k, v in arch_n.items()}}
        results[name] = res
        print(f"{name:6} top1 {res['top1']:.1%} top5 {res['top5']:.1%} "
              f"top10 {res['top10']:.1%} | >25 {res['cat_rate_gt25']:.1%} "
              f"median {res['median_px']:.1f} | dram {res.get('dram_top1', 0):.1%} "
              f"finfet {res.get('finfet_top1', 0):.1%}")
    (OUT / "residual_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'residual_eval.json'}")


if __name__ == "__main__":
    main()
