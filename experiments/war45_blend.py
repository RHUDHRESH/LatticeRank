"""war45: single complementarity test - R2 ranker + residual evidence.

Residual matching failed standalone (best top-1 32.5% < 45% kill line). This
is the one remaining measured question: does periodic-cancellation evidence
fix scenes the R2 ranker gets wrong? Fixed z-blends, identical delta=0.10
validation candidates, no sweep. If no blend beats R2, R2 ships alone.
"""
from __future__ import annotations

import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from war45_struct import by_scene, load_csv, OUT  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.residual import ResidualMatcher  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

POS_TOL = 5.0
CENTRE = 499.5


def _job(args):
    rec, pts = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
    rm = ResidualMatcher(s.reference, s.search)
    return [rm.score(x, y)["res_int_m50"] for x, y in pts]


def main():
    import joblib
    va = load_csv(OUT / "val_struct.csv")
    groups = by_scene(va)
    man = {r["id"]: r for r in read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")}

    bundle = joblib.load(PROJECT / "driftforge/models/hgb_r2.joblib")
    m, feats = bundle["model"], bundle["features"]

    jobs = [(man[sid], [(r["x"], r["y"]) for r in rows]) for sid, rows in groups.items()]
    print(f"residual scores for {len(jobs)} scenes...")
    t0 = time.perf_counter()
    res_scores = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for (sid, _), sc in zip(sorted(groups.items()), pool.map(_job, jobs)):
            res_scores[sid] = np.array(sc, dtype=float)
    print(f"  done {time.perf_counter() - t0:.0f}s")

    def z(v):
        mfin = np.isfinite(v)
        if mfin.sum() < max(3, 0.5 * v.size):
            return None
        mu, sd = v[mfin].mean(), v[mfin].std()
        out = (v - mu) / (sd if sd > 1e-9 else 1.0)
        out[~mfin] = -9.0
        return out

    variants = {"R2_alone": None, "R2+1.0res": 1.0, "R2+0.5res": 0.5}
    results = {}
    for name, wres in variants.items():
        errs, top = [], {1: 0, 5: 0, 10: 0}
        arch_n, arch_hit, sel_errs = {}, {}, []
        arch_sel = {}
        preds = []
        for sid, rows in groups.items():
            X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows]),
                              nan=0.0, posinf=9.0, neginf=-9.0)
            p = m.predict_proba(X)[:, 1]
            score = z(p.astype(float))
            if wres is not None:
                rz = z(res_scores[sid])
                if rz is not None:
                    score = score + wres * rz
            e = np.array([r["error_px"] for r in rows])
            o = np.argsort(-score)
            errs.append(float(e[o[0]]))
            for K in top:
                top[K] += bool(np.any(e[o[:K]] <= POS_TOL))
            smax = score.max()
            eq = np.nonzero(score >= smax - 0.25)[0]   # z-scale margin
            best = min(eq, key=lambda i: (rows[i]["x"] - CENTRE) ** 2
                       + (rows[i]["y"] - CENTRE) ** 2)
            sel_errs.append(float(e[best]))
            a = rows[0]["architecture"]
            arch_n[a] = arch_n.get(a, 0) + 1
            arch_hit[a] = arch_hit.get(a, 0) + int(e[o[0]] <= POS_TOL)
            arch_sel[a] = arch_sel.get(a, 0) + int(e[best] <= POS_TOL)
            preds.append({"sample_id": sid, "architecture": a,
                          "profile": rows[0]["profile"], "eq_size": int(len(eq)),
                          "pred_x": rows[int(best)]["x"], "pred_y": rows[int(best)]["y"],
                          "gt_x": rows[0]["gt_x"], "gt_y": rows[0]["gt_y"],
                          "error_px": float(e[best]), "top1_error_px": float(e[o[0]])})
        e = np.array(errs)
        se = np.array(sel_errs)
        nn = len(groups)
        results[name] = {
            "top1": top[1] / nn, "top5": top[5] / nn, "top10": top[10] / nn,
            "acc_1px": float(np.mean(e <= 1)), "acc_2px": float(np.mean(e <= 2)),
            "acc_5px": float(np.mean(e <= 5)), "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95)), "p99_px": float(np.percentile(e, 99)),
            "cat_rate_gt25": float(np.mean(e > 25)),
            "sel_acc_1px": float(np.mean(se <= 1)), "sel_acc_2px": float(np.mean(se <= 2)),
            "sel_acc_5px": float(np.mean(se <= 5)), "sel_median_px": float(np.median(se)),
            "sel_p95_px": float(np.percentile(se, 95)),
            "sel_p99_px": float(np.percentile(se, 99)),
            "sel_cat_rate_gt25": float(np.mean(se > 25)),
            **{f"{k}_top1": arch_hit.get(k, 0) / v for k, v in arch_n.items()},
            **{f"{k}_sel5": arch_sel.get(k, 0) / v for k, v in arch_n.items()},
        }
        if name == "R2+1.0res":
            import csv as _csv
            with (OUT / "preds_final.csv").open("w", newline="", encoding="utf-8") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
                w.writeheader(); w.writerows(preds)
        print(f"{name:10} top1 {results[name]['top1']:.1%} top5 {results[name]['top5']:.1%} "
              f"top10 {results[name]['top10']:.1%} >25 {results[name]['cat_rate_gt25']:.1%} "
              f"dram {results[name].get('dram_top1', 0):.1%} "
              f"finfet {results[name].get('finfet_top1', 0):.1%}")
    (OUT / "blend_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
