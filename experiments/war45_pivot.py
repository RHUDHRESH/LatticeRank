"""Decisive test of the pivot: does multi-channel + adaptive harvesting beat
single-channel top-K on true-cell recall, and at what candidate cost?

Also reports, per channel, whether the true site is a local maximum and where
it ranks - because if the new channels rank it better, that is the mechanism
the whole pivot depends on.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, local_maxima, response_maps  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

DELTAS = (0.02, 0.05, 0.10, 0.15, 0.25)


def one(args):
    rec, ss = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], ss)
    gx, gy = s.gt_x, s.gt_y
    cm = response_maps(s.reference, s.search)
    out = {"sample_id": rec["id"], "architecture": rec["architecture"],
           "profile": rec["profile"]}

    # per-channel rank of the true site
    for ch in CHANNELS:
        xs, ys, v = local_maxima(cm.maps[ch], 5)
        d = np.hypot(xs + cm.half_w - gx, ys + cm.half_h - gy)
        near = d <= 5.0
        out[f"{ch}_is_localmax"] = bool(near.any())
        if near.any():
            vn = float(v[near].max())
            out[f"{ch}_rank"] = int((v > vn).sum()) + 1
            out[f"{ch}_deficit"] = float(v.max() - vn)
        else:
            out[f"{ch}_rank"] = -1
            out[f"{ch}_deficit"] = float("nan")

    # adaptive harvest at several deltas
    for dl in DELTAS:
        cands = harvest(cm, delta=dl)
        if cands:
            d = np.array([math.hypot(c["x"] - gx, c["y"] - gy) for c in cands])
            out[f"d{dl}_hit5"] = bool((d <= 5.0).any())
            out[f"d{dl}_hit2"] = bool((d <= 2.0).any())
            out[f"d{dl}_hit1"] = bool((d <= 1.0).any())
        else:
            out[f"d{dl}_hit5"] = out[f"d{dl}_hit2"] = out[f"d{dl}_hit1"] = False
        out[f"d{dl}_n"] = len(cands)
        out[f"d{dl}_multivote"] = sum(1 for c in cands if c["votes"] >= 2)
    return out


def wil(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT / "manifests/validation_phase2a_400.jsonl")
    ap.add_argument("--out", type=Path, default=PROJECT / "experiments/war45")
    a = ap.parse_args()

    recs = read_manifest(a.manifest)[a.start:a.start + a.limit]
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(one, [(x, 2) for x in recs]), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(recs):
                print(f"  [{i}/{len(recs)}] {time.perf_counter()-t0:.0f}s", flush=True)
    n = len(rows)
    summ = {"pairs": n, "wall_s": time.perf_counter() - t0, "channels": {}, "adaptive": {}}

    print("\n=== PER-CHANNEL: is the true site findable, and how well ranked? ===")
    print(f"{'channel':16s} {'is localmax':>12s} {'rank<=10':>9s} {'rank<=50':>9s} {'med rank':>9s} {'med deficit':>12s}")
    for ch in CHANNELS:
        lm = sum(r[f"{ch}_is_localmax"] for r in rows)
        rk = [r[f"{ch}_rank"] for r in rows if r[f"{ch}_rank"] > 0]
        df = [r[f"{ch}_deficit"] for r in rows if np.isfinite(r[f"{ch}_deficit"])]
        blk = {"is_localmax": lm / n,
               "rank_le10": sum(1 for x in rk if x <= 10) / n,
               "rank_le50": sum(1 for x in rk if x <= 50) / n,
               "median_rank": float(np.median(rk)) if rk else None,
               "median_deficit": float(np.median(df)) if df else None}
        summ["channels"][ch] = blk
        print(f"{ch:16s} {blk['is_localmax']:>11.1%} {blk['rank_le10']:>9.1%} "
              f"{blk['rank_le50']:>9.1%} {blk['median_rank']:>9.0f} {blk['median_deficit']:>12.4f}")

    print("\n=== ADAPTIVE HARVEST: S_i >= S_max - delta, union of 3 channels ===")
    print(f"{'delta':>6s} {'recall@1':>10s} {'recall@2':>10s} {'recall@5':>18s} {'med #':>8s} {'p95 #':>8s} {'multi-vote':>11s}")
    for dl in DELTAS:
        k5 = sum(r[f"d{dl}_hit5"] for r in rows)
        p5, lo, hi = wil(k5, n)
        k2 = sum(r[f"d{dl}_hit2"] for r in rows)
        k1 = sum(r[f"d{dl}_hit1"] for r in rows)
        nn = np.array([r[f"d{dl}_n"] for r in rows])
        mv = np.array([r[f"d{dl}_multivote"] for r in rows])
        summ["adaptive"][str(dl)] = {
            "recall1": k1 / n, "recall2": k2 / n,
            "recall5": p5, "ci95": [lo, hi],
            "median_candidates": float(np.median(nn)),
            "p95_candidates": float(np.percentile(nn, 95)),
            "median_multivote": float(np.median(mv))}
        print(f"{dl:>6.2f} {k1/n:>10.1%} {k2/n:>10.1%} "
              f"{p5:>9.1%}[{lo:.0%}-{hi:.0%}] {np.median(nn):>8.0f} "
              f"{np.percentile(nn,95):>8.0f} {np.median(mv):>11.0f}")

    for dl in DELTAS:
        for arch in ("dram", "finfet"):
            sub = [r for r in rows if r["architecture"] == arch]
            k = sum(r[f"d{dl}_hit5"] for r in sub)
            summ["adaptive"][str(dl)][f"recall5_{arch}"] = k / max(len(sub), 1)

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "pivot_recall.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (a.out / "pivot_recall.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    best = max(DELTAS, key=lambda d: summ["adaptive"][str(d)]["recall5"])
    b = summ["adaptive"][str(best)]
    print(f"\nbest delta={best}: recall@5 {b['recall5']:.1%}  "
          f"DRAM {b['recall5_dram']:.1%}  FinFET {b['recall5_finfet']:.1%}  "
          f"median {b['median_candidates']:.0f} candidates")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
