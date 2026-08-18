"""war45 P0: structural-identity descriptor benchmark + R0/R1/R2 ablation.

Stage 1 (`val`): build the delta=0.10 adaptive candidate set on the SAME 80
disjoint validation pairs used by every war45 comparison, attach structural
descriptor features, and rank by the transparent fixed-weight structural score
alone - no ML. This is the go/no-go gate: descriptor top-10 < 65% means the
hand-crafted family is insufficient and metric learning is next.

Stage 2 (`train`): same table for the 60 training scenes (keep_gt guard on).

Stage 3 (`ablate`): one controlled comparison on identical candidate sets -
    R0  old peak-strength/scene-relative features (the 37.5% family)
    R1  structural features only
    R2  old + structural
using the existing balanced hard-negative protocol and the existing HGB
configuration. Only the feature family varies.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from war45_ranker import FEATURES, POS_TOL, collect, hard_negatives  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402
from driftforge.structural_descriptor import STRUCT_FEATURES, struct_score  # noqa: E402

OUT = PROJECT / "experiments/war45"
DELTA = 0.10
CENTRE = 499.5

STR_COLS = ("scene_id", "sample_id", "architecture", "profile")


def by_scene(rows):
    g = {}
    for r in rows:
        g.setdefault(r["sample_id"], []).append(r)
    return dict(sorted(g.items()))


def save_csv(rows, path: Path):
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows, {len(cols)} cols)")


def load_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (v if k in STR_COLS else float(v)) for k, v in r.items()})
    return rows


def rank_metrics(groups, scores, label):
    """Top-K recall + argmax-selection error stats, overall and per-arch."""
    sids = list(groups)
    topk = {}
    for K in (1, 5, 10, 30):
        hits = sum(
            any(groups[sid][j]["error_px"] <= POS_TOL for j in np.argsort(-scores[sid])[:K])
            for sid in sids)
        topk[f"top{K}"] = hits / max(len(sids), 1)
    errs, arch_top1, arch_top5, arch_n, arch_le5 = [], {}, {}, {}, {}
    for sid in sids:
        rows_, sc = groups[sid], scores[sid]
        o = np.argsort(-sc)
        e = rows_[int(o[0])]["error_px"]
        errs.append(e)
        a = rows_[0]["architecture"]
        arch_n[a] = arch_n.get(a, 0) + 1
        arch_top1[a] = arch_top1.get(a, 0) + (rows_[int(o[0])]["error_px"] <= POS_TOL)
        arch_top5[a] = arch_top5.get(a, 0) + any(rows_[int(j)]["error_px"] <= POS_TOL for j in o[:5])
        arch_le5[a] = arch_le5.get(a, 0) + (e <= 5)
    e = np.array(errs)
    out = {
        "label": label, "n_pairs": len(sids), **topk,
        "acc_1px": float(np.mean(e <= 1)), "acc_2px": float(np.mean(e <= 2)),
        "acc_5px": float(np.mean(e <= 5)), "median_px": float(np.median(e)),
        "p95_px": float(np.percentile(e, 95)), "p99_px": float(np.percentile(e, 99)),
        "cat_rate_gt25": float(np.mean(e > 25)),
        "oracle_recall5": float(np.mean([
            any(r["error_px"] <= POS_TOL for r in groups[sid]) for sid in sids])),
        "median_candidates": float(np.median([len(groups[s]) for s in sids])),
    }
    for a in sorted(arch_n):
        out[f"{a}_top1"] = arch_top1[a] / arch_n[a]
        out[f"{a}_top5"] = arch_top5[a] / arch_n[a]
        out[f"{a}_le5"] = arch_le5[a] / arch_n[a]
    return out


def cmd_val(a):
    t0 = time.perf_counter()
    recs = read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")[200:200 + a.val]
    print(f"VAL: {len(recs)} pairs, delta={DELTA}, struct=True")
    va = collect(recs, a.workers, max_cand=a.val_cand, keep_gt=False,
                 delta=DELTA, struct=True)
    build_s = time.perf_counter() - t0
    save_csv(va, OUT / "val_struct.csv")
    groups = by_scene(va)
    scores = {sid: struct_score(rows_) for sid, rows_ in groups.items()}
    res = rank_metrics(groups, scores, "descriptor_only")
    res["build_wall_s"] = build_s
    res["per_pair_s"] = build_s / max(len(recs), 1)
    (OUT / "struct_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    v = ("GO: >=80 excellent" if res["top10"] >= 0.80 else
         "GO: 65-80 combine with HGB" if res["top10"] >= 0.65 else
         "STOP: <65 -> metric learning")
    print(f"\ndescriptor top10={res['top10']:.1%}  -> {v}")


def cmd_train(a):
    recs = read_manifest(PROJECT / "manifests/train.jsonl")[a.start:a.start + a.train]
    print(f"TRAIN: {len(recs)} scenes [{a.start}:{a.start + a.train}], "
          f"delta={DELTA}, struct=True, keep_gt=True")
    tr = collect(recs, a.workers, max_cand=a.max_cand, keep_gt=True,
                 delta=DELTA, struct=True)
    suffix = f"_{a.start}" if a.start else ""
    save_csv(tr, OUT / f"train_struct{suffix}.csv")


def cmd_ablate(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = []
    for fp in sorted(OUT.glob("train_struct*.csv")):
        part = load_csv(fp)
        print(f"loaded {fp.name}: {len(part)} rows")
        tr.extend(part)
    va = load_csv(OUT / "val_struct.csv")

    tr_g = by_scene(tr)
    tr_bal = []
    for k, (sid, rows_) in enumerate(tr_g.items()):
        tr_bal.extend(hard_negatives(rows_, n_neg=a.n_neg, seed=k))
    print(f"hard-negative subsampling: {len(tr)} -> {len(tr_bal)} train rows "
          f"({sum(r['error_px'] <= POS_TOL for r in tr_bal)} pos)")

    groups = by_scene(va)
    variants = {"R0_old": FEATURES, "R1_struct": STRUCT_FEATURES,
                "R2_old_struct": FEATURES + STRUCT_FEATURES}
    results = {}
    for name, feats in variants.items():
        X = lambda rows_: np.nan_to_num(
            np.array([[r[f] for f in feats] for r in rows_], dtype=np.float64),
            nan=0.0, posinf=9.0, neginf=-9.0)
        y = np.array([r["error_px"] <= POS_TOL for r in tr_bal], dtype=int)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           l2_regularization=1.0, random_state=0,
                                           class_weight="balanced")
        t = time.perf_counter()
        m.fit(X(tr_bal), y)
        fit_s = time.perf_counter() - t
        scores, sel_errs, sizes, preds = {}, [], [], []
        for sid, rows_ in groups.items():
            p = m.predict_proba(X(rows_))[:, 1]
            scores[sid] = p
            pmax = p.max()
            eq = [i for i in range(len(rows_)) if p[i] >= pmax - 0.05]
            sizes.append(len(eq))
            best = min(eq, key=lambda i: (rows_[i]["x"] - CENTRE) ** 2
                       + (rows_[i]["y"] - CENTRE) ** 2)
            sel_errs.append(rows_[best]["error_px"])
            preds.append({"sample_id": sid, "architecture": rows_[0]["architecture"],
                          "profile": rows_[0]["profile"], "n_cand": len(rows_),
                          "eq_size": len(eq),
                          "pred_x": rows_[best]["x"], "pred_y": rows_[best]["y"],
                          "gt_x": rows_[best]["gt_x"], "gt_y": rows_[best]["gt_y"],
                          "error_px": rows_[best]["error_px"],
                          "top1_error_px": rows_[int(np.argmax(p))]["error_px"]})
        save_csv(preds, OUT / f"preds_{name}.csv")
        res = rank_metrics(groups, scores, name)
        e = np.array(sel_errs)
        res.update({
            "fit_s": fit_s, "n_features": len(feats),
            "sel_acc_1px": float(np.mean(e <= 1)), "sel_acc_2px": float(np.mean(e <= 2)),
            "sel_acc_5px": float(np.mean(e <= 5)), "sel_median_px": float(np.median(e)),
            "sel_p95_px": float(np.percentile(e, 95)),
            "sel_p99_px": float(np.percentile(e, 99)),
            "sel_cat_rate_gt25": float(np.mean(e > 25)),
            "median_eq_set": float(np.median(sizes)),
        })
        results[name] = res
        print(f"\n{name}: top1 {res['top1']:.1%} top5 {res['top5']:.1%} "
              f"top10 {res['top10']:.1%} | sel<=5 {res['sel_acc_5px']:.1%} "
              f"median {res['sel_median_px']:.1f}px P95 {res['sel_p95_px']:.0f} "
              f">25 {res['sel_cat_rate_gt25']:.1%} | dram_top1 "
              f"{res.get('dram_top1', float('nan')):.1%} finfet_top1 "
              f"{res.get('finfet_top1', float('nan')):.1%}")
    (OUT / "struct_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'struct_ablation.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["val", "train", "ablate"])
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-cand", type=int, default=8000)
    ap.add_argument("--max-cand", type=int, default=800)
    ap.add_argument("--n-neg", type=int, default=65)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"val": cmd_val, "train": cmd_train, "ablate": cmd_ablate}[a.cmd](a)


if __name__ == "__main__":
    main()
