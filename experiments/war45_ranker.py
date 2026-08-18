"""Candidate-level meta-ranker: the actual ML.

Pipeline under test:
    3 response maps -> union of local maxima -> adaptive harvest (S >= Smax-d)
    -> candidate feature table -> RF / HGB probability
    -> evidential equivalence set -> nearest-Search-centre within it -> (x,y)

Lattice information enters as FEATURES ONLY (phase residual, period-normalised
displacement, symmetry class) - never as a candidate generator, which the
war-45 measurement showed makes recall worse.

Distance-to-centre is NOT a feature. The official nearest-centre rule is a
tie-break applied after the ranker has declared candidates equivalent; feeding
it to the model would let it learn "central = correct" instead of "looks like
the reference".

Train and validation scenes are disjoint splits (train.jsonl vs
validation_phase2a_400.jsonl), so no scene appears in both.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.lattice import estimate_lattice  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

DELTA = 0.15
MAX_CAND = 800          # bound feature cost; p95 harvest is ~7k
POS_TOL = 5.0

#: Scene-RELATIVE features dominate. An absolute `raw=0.71` is meaningless
#: without knowing what the other candidates in THIS image scored; a candidate
#: 3 sigma above its own scene's mean is informative regardless of scale.
#: Absolute integer ranks are also scene-size dependent (rank 50 of 800 != rank
#: 50 of 3000), so percentiles replace them.
FEATURES = ["raw", "midband", "directionality", "votes",
            "chan_spread", "chan_min",
            "raw_pct", "mid_pct", "dir_pct",
            "raw_delta", "mid_delta", "dir_delta",
            "raw_z", "mid_z", "dir_z",
            "agreement_count", "channel_best_count",
            "lat_phase_res", "lat_conf", "lat_agree",
            "disp_periods_x", "disp_periods_y", "parity_even",
            "local_entropy", "noise_est", "n_cand_scene"]


def _entropy(patch: np.ndarray) -> float:
    h, _ = np.histogram(patch, bins=24, range=(0, 255))
    p = h.astype(float) / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build(rec, ss=2, max_cand=MAX_CAND, delta=DELTA, keep_gt=False):
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], ss)
    gx, gy = s.gt_x, s.gt_y
    cm = response_maps(s.reference, s.search)
    cands = harvest(cm, delta=delta)
    if not cands:
        return []
    # bound cost: keep the strongest by best-channel score
    cands.sort(key=lambda c: -max(v for k, v in c.items() if k in CHANNELS
                                  and isinstance(v, float) and math.isfinite(v)))
    if len(cands) > max_cand:
        head = cands[:max_cand]
        if keep_gt:
            # TRAIN ONLY: the cost cap sorts by the very score that fails, so it
            # can discard the true site and silently delete the positive class.
            # Ground truth is legitimate for building training data; it is never
            # consulted for validation (keep_gt=False there).
            head = head + [c for c in cands[max_cand:]
                           if math.hypot(c["x"] - gx, c["y"] - gy) <= POS_TOL]
        cands = head

    lat = estimate_lattice(s.search)
    B = lat.basis
    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        Binv = np.zeros((2, 2))
    top = cands[0]
    N = len(cands)
    maxes, pct, zsc, order = {}, {}, {}, {}
    for c in CHANNELS:
        v = np.array([k[c] if math.isfinite(k[c]) else -9.0 for k in cands])
        maxes[c] = float(v.max())
        r = np.argsort(np.argsort(-v)) + 1          # 1 = best in this scene
        order[c] = r
        pct[c] = r / max(N, 1)                      # scene-size invariant
        sd = float(v.std())
        zsc[c] = (v - float(v.mean())) / (sd if sd > 1e-9 else 1.0)
    # how many channels rank this candidate in their own top-10, and how many
    # channels consider it their single best - cross-channel agreement is the
    # signal a single fused score throws away
    top10 = {c: set(np.argsort(-np.array([k[c] if math.isfinite(k[c]) else -9.0
                                          for k in cands]))[:10]) for c in CHANNELS}
    bestidx = {c: int(np.argmax([k[c] if math.isfinite(k[c]) else -9.0 for k in cands]))
               for c in CHANNELS}

    a = s.search.astype(np.float32)
    lap = (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4 * a[1:-1, 1:-1])
    noise = float(np.median(np.abs(lap - np.median(lap))) * 1.4826 / math.sqrt(20.0))

    rows = []
    for i, c in enumerate(cands):
        vals = [c[k] for k in CHANNELS if math.isfinite(c[k])]
        d = np.array([c["x"] - top["x"], c["y"] - top["y"]])
        mn = Binv @ d
        frac = mn - np.round(mn)
        x0, y0 = int(np.clip(c["x"] - 16, 0, 968)), int(np.clip(c["y"] - 16, 0, 968))
        rows.append({
            "scene_id": rec.get("scene_id", f"scene-{rec['seed']}"),
            "sample_id": rec["id"], "architecture": rec["architecture"],
            "profile": rec["profile"], "x": c["x"], "y": c["y"],
            "error_px": math.hypot(c["x"] - gx, c["y"] - gy),
            "raw": c["raw"], "midband": c["midband"], "directionality": c["directionality"],
            "votes": float(c["votes"]),
            "chan_spread": float(max(vals) - min(vals)) if vals else 0.0,
            "chan_min": float(min(vals)) if vals else 0.0,
            "raw_pct": float(pct["raw"][i]), "mid_pct": float(pct["midband"][i]),
            "dir_pct": float(pct["directionality"][i]),
            "raw_delta": float(maxes["raw"] - c["raw"]) if math.isfinite(c["raw"]) else 9.0,
            "mid_delta": float(maxes["midband"] - c["midband"]) if math.isfinite(c["midband"]) else 9.0,
            "dir_delta": float(maxes["directionality"] - c["directionality"]) if math.isfinite(c["directionality"]) else 9.0,
            "raw_z": float(zsc["raw"][i]), "mid_z": float(zsc["midband"][i]),
            "dir_z": float(zsc["directionality"][i]),
            "agreement_count": float(sum(i in top10[ch] for ch in CHANNELS)),
            "channel_best_count": float(sum(bestidx[ch] == i for ch in CHANNELS)),
            "n_cand_scene": float(N),
            "lat_phase_res": float(np.linalg.norm(B @ frac)),
            "lat_conf": float(lat.confidence), "lat_agree": float(lat.agreement),
            "disp_periods_x": float(mn[0]), "disp_periods_y": float(mn[1]),
            "parity_even": float(int(round(mn[0]) + round(mn[1])) % 2 == 0),
            "local_entropy": _entropy(s.search[y0:y0 + 32, x0:x0 + 32]),
            "noise_est": noise,
            "gt_x": gx, "gt_y": gy,
        })
    return rows


def hard_negatives(rows, n_neg=60, seed=0):
    """Per-scene hard-negative subsampling. TRAINING ONLY.

    At harvest breadth the positive rate is ~0.1% (227 positives in 205k rows),
    so a classifier trivially reaches 99.9% accuracy by always predicting
    negative and learns nothing. Keeping every positive and only the hardest,
    most diverse negatives turns this back into a ranking problem the model can
    actually learn.

    Diversity matters as much as hardness: sampling only the top-scoring
    negatives teaches the model one confusion mode. We draw from each channel's
    top, from the negatives whose scores sit closest to the positive (the true
    confusions), and from distinct spatial regions (which is where periodic
    aliases live - they are by construction far away and high-scoring).
    """
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r["error_px"] <= POS_TOL]
    neg = [r for r in rows if r["error_px"] > POS_TOL]
    if not neg:
        return pos
    keep = set()
    def take(order, k):
        for i in list(order)[:k]:
            keep.add(int(i))
    # explicit quotas: 15 per channel, 10 score-confusions, 10 remote aliases
    for ch in ("raw", "midband", "directionality"):
        v = np.array([r[ch] if math.isfinite(r[ch]) else -9.0 for r in neg])
        take(np.argsort(-v), 15)
    if pos:
        pv = max((p["raw"] for p in pos if math.isfinite(p["raw"])), default=0.0)
        d = np.abs(np.array([r["raw"] if math.isfinite(r["raw"]) else -9.0 for r in neg]) - pv)
        take(np.argsort(d), 10)
        px_, py_ = pos[0]["x"], pos[0]["y"]
        far = [i for i, r in enumerate(neg) if math.hypot(r["x"] - px_, r["y"] - py_) > 100.0]
        if far:
            fv = np.array([neg[i]["raw"] if math.isfinite(neg[i]["raw"]) else -9.0 for i in far])
            take([far[j] for j in np.argsort(-fv)], 10)
    remaining = [i for i in range(len(neg)) if i not in keep]
    if remaining and len(keep) < n_neg:
        for i in rng.choice(remaining, size=min(n_neg - len(keep), len(remaining)), replace=False):
            keep.add(int(i))
    return pos + [neg[i] for i in sorted(keep)]


def _job(a):
    try:
        return build(a[0], a[1], max_cand=a[2], keep_gt=a[3])
    except Exception:
        return []


def collect(recs, workers, ss=2, max_cand=MAX_CAND, keep_gt=False):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(_job, [(x, ss, max_cand, keep_gt) for x in recs]), 1):
            out.extend(r)
            if i % 20 == 0 or i == len(recs):
                print(f"  [{i}/{len(recs)}] {len(out)} candidates", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-cand", type=int, default=MAX_CAND)
    ap.add_argument("--n-neg", type=int, default=65)
    ap.add_argument("--val-cand", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=PROJECT / "experiments/war45")
    a = ap.parse_args()
    t0 = time.perf_counter()

    tr_recs = read_manifest(PROJECT / "manifests/train.jsonl")[:a.train]
    va_recs = read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")[200:200 + a.val]
    print(f"train scenes {len(tr_recs)} | val scenes {len(va_recs)} (disjoint splits)")
    print("building TRAIN candidates...", flush=True)
    tr = collect(tr_recs, a.workers, max_cand=a.max_cand, keep_gt=True)
    print("building VAL candidates...", flush=True)
    va = collect(va_recs, a.workers, max_cand=a.val_cand, keep_gt=False)

    def XY(rows):
        X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=9.0, neginf=-9.0)
        y = np.array([r["error_px"] <= POS_TOL for r in rows], dtype=int)
        return X, y

    # subsample TRAIN only; validation must keep every candidate or the
    # measured accuracy would not reflect inference conditions
    by_scene = {}
    for r in tr:
        by_scene.setdefault(r["sample_id"], []).append(r)
    tr_bal = []
    for k, (sid, rows_) in enumerate(sorted(by_scene.items())):
        tr_bal.extend(hard_negatives(rows_, n_neg=a.n_neg, seed=k))
    print(f"hard-negative subsampling: {len(tr)} -> {len(tr_bal)} train rows")
    tr = tr_bal

    Xtr, ytr = XY(tr)
    Xva, yva = XY(va)
    print(f"\ntrain rows {len(ytr)} ({ytr.sum()} pos)  val rows {len(yva)} ({yva.sum()} pos)")

    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    models = {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=4, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=0),
        "HGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=0,
                                              class_weight="balanced"),
    }
    results = {}
    for name, m in models.items():
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        p = m.predict_proba(Xva)[:, 1]
        # per-sample selection: equivalence set by probability margin, then
        # nearest-Search-centre within it (the official rule, applied last)
        scene_idx = {}
        for i, r in enumerate(va):
            scene_idx.setdefault(r["sample_id"], []).append(i)
        sids = sorted(scene_idx)
        errs, sizes = [], []
        for sid in sids:
            idx = scene_idx[sid]
            pr = p[idx]
            pmax = pr.max()
            eq = [i for i, j in enumerate(idx) if pr[i] >= pmax - 0.05]
            sizes.append(len(eq))
            cx = cy = 499.5
            best = min(eq, key=lambda i: (va[idx[i]]["x"] - cx) ** 2 + (va[idx[i]]["y"] - cy) ** 2)
            errs.append(va[idx[best]]["error_px"])
        e = np.array(errs)
        topk = {}
        for K in (1, 5, 10, 30):
            hits = 0
            for sid in sids:
                idx = scene_idx[sid]
                pr = p[idx]
                o = np.argsort(-pr)[:K]
                if any(va[idx[j]]["error_px"] <= POS_TOL for j in o):
                    hits += 1
            topk[f"top{K}_recall"] = hits / max(len(sids), 1)
        results[name] = {
            "fit_s": fit_s, "n_val_pairs": len(e),
            "acc_1px": float(np.mean(e <= 1)), "acc_2px": float(np.mean(e <= 2)),
            "acc_5px": float(np.mean(e <= 5)), "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95)), "p99_px": float(np.percentile(e, 99)),
            "cat_gt25": int((e > 25).sum()), "cat_rate": float(np.mean(e > 25)),
            "median_eq_set": float(np.median(sizes)), **topk,
        }
        print(f"\n{name}: fit {fit_s:.1f}s  acc<=1 {results[name]['acc_1px']:.1%} "
              f"<=2 {results[name]['acc_2px']:.1%} <=5 {results[name]['acc_5px']:.1%}  "
              f"median {results[name]['median_px']:.2f}px  P95 {results[name]['p95_px']:.0f}  "
              f">25px {results[name]['cat_rate']:.1%}")
        print(f"     rank recall: top1 {topk['top1_recall']:.1%}  top5 {topk['top5_recall']:.1%}  "
              f"top10 {topk['top10_recall']:.1%}  top30 {topk['top30_recall']:.1%}")

    # feature importance from RF
    imp = sorted(zip(FEATURES, models["RF"].feature_importances_), key=lambda t: -t[1])
    results["rf_importance"] = [{"feature": f, "importance": float(v)} for f, v in imp]
    print("\ntop features (RF):", ", ".join(f"{f}={v:.3f}" for f, v in imp[:8]))
    results["oracle_recall5"] = float(np.mean([
        any(va[i]["error_px"] <= 5 for i in range(len(va)) if va[i]["sample_id"] == sid)
        for sid in {r["sample_id"] for r in va}]))
    print(f"candidate-set oracle recall@5 (ceiling): {results['oracle_recall5']:.1%}")
    results["wall_s"] = time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ranker_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out/'ranker_results.json'}  total {results['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
