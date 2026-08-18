"""war45 FINAL SWING: lattice-phase equivalence classes + official centre rule.

Every appearance-based ranker (HGB 40%, struct descriptor, Siamese, periodic
residual) stalls at the same wall: the true candidate exists but visually
near-identical periodic copies outscore it. This experiment changes the
inference unit: stop ranking candidates, rank lattice-PHASE classes, then let
the official nearest-centre rule pick within the winning class.

All from the cached delta=0.10 validation table - no image recomputation.

Stage 1 diagnostics (gate before any scoring):
  D1 phase-class oracle: does any candidate share GT's wrapped lattice phase?
  D2 true ceiling: if the correct class were chosen and nearest-centre applied
     inside it, how often is the result <=5 px from GT?

Stage 2 (only if D2 is high): P1 class top-3 mean, P2 phase voting, P3 50/50.

Phase geometry: the per-scene basis B is recovered from the cached rows by
least squares (disp_periods = Binv @ (xy - top_xy) is linear and exact), GT is
mapped through the same Binv, and phases live on the torus embedded as
(cos 2pi p, sin 2pi p) per axis. A parity-aware variant embeds at half
frequency too, so opposite checkerboard sub-lattice classes never merge.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from war45_struct import by_scene, load_csv, OUT  # noqa: E402

POS_TOL = 5.0
CENTRE = 499.5
PHASE_TOL = 0.18          # wrap-distance (periods) for "same phase"


def wrap(p):
    return ((np.asarray(p) + 0.5) % 1.0) - 0.5


def recover_binv(rows):
    """disp_periods = Binv @ (xy - top_xy) is linear -> least squares."""
    top = rows[0]
    d = np.array([[r["x"] - top["x"], r["y"] - top["y"]] for r in rows])
    mn = np.array([[r["disp_periods_x"], r["disp_periods_y"]] for r in rows])
    if len(rows) < 3 or np.abs(d).max() < 1e-6:
        return None
    sol, res, rank, _ = np.linalg.lstsq(d, mn, rcond=None)
    if rank < 2:
        return None
    Binv = sol.T
    pred = d @ Binv.T
    if np.abs(pred - mn).max() > 1e-3:
        return None
    return Binv


def embed(px, py, parity: bool):
    cols = [np.cos(2 * np.pi * px), np.sin(2 * np.pi * px),
            np.cos(2 * np.pi * py), np.sin(2 * np.pi * py)]
    if parity:
        cols += [np.cos(np.pi * px), np.sin(np.pi * px),
                 np.cos(np.pi * py), np.sin(np.pi * py)]
    return np.column_stack(cols)


def phase_dist(pa, pb):
    """max-axis wrapped distance in periods."""
    return np.maximum(np.abs(wrap(pa[..., 0] - pb[..., 0])),
                      np.abs(wrap(pa[..., 1] - pb[..., 1])))


def clusters_of(phases, parity_flags, parity_aware: bool, tol=PHASE_TOL):
    """Greedy leader clustering on the torus (deterministic, no sklearn dep)."""
    n = len(phases)
    labels = -np.ones(n, dtype=int)
    leaders = []
    for i in range(n):
        assigned = False
        for li, (lp, lparity) in enumerate(leaders):
            if parity_aware and parity_flags[i] != lparity:
                continue
            if (abs(float(wrap(phases[i][0] - lp[0]))) <= tol
                    and abs(float(wrap(phases[i][1] - lp[1]))) <= tol):
                labels[i] = li
                assigned = True
                break
        if not assigned:
            leaders.append((phases[i], parity_flags[i]))
            labels[i] = len(leaders) - 1
    return labels


def main():
    va = load_csv(OUT / "val_struct.csv")
    groups = by_scene(va)
    print(f"{len(groups)} scenes from cached val_struct.csv")

    diag = {"n": 0, "binv_fail": 0}
    for parity_aware in (False, True):
        key = "parity" if parity_aware else "fund"
        d1 = d2 = 0
        arch_d2, arch_n = {}, {}
        prof_d2, prof_n = {}, {}
        class_sizes = []
        for sid, rows in groups.items():
            Binv = recover_binv(rows)
            a = rows[0]["architecture"]
            pr = rows[0]["profile"]
            arch_n[a] = arch_n.get(a, 0) + 1
            prof_n[pr] = prof_n.get(pr, 0) + 1
            if Binv is None:
                diag["binv_fail"] += 1 if not parity_aware else 0
                continue
            top = rows[0]
            gt = np.array([rows[0]["gt_x"] - top["x"], rows[0]["gt_y"] - top["y"]])
            mn_gt = Binv @ gt
            pg = wrap(mn_gt)
            par_gt = int(round(mn_gt[0]) + round(mn_gt[1])) % 2
            mn = np.array([[r["disp_periods_x"], r["disp_periods_y"]] for r in rows])
            pc = wrap(mn)
            par = (np.round(mn[:, 0]) + np.round(mn[:, 1])).astype(int) % 2
            same = (np.abs(wrap(pc[:, 0] - pg[0])) <= PHASE_TOL) \
                 & (np.abs(wrap(pc[:, 1] - pg[1])) <= PHASE_TOL)
            if parity_aware:
                same &= (par == par_gt)
            if same.any():
                d1 += 1
                idx = np.nonzero(same)[0]
                cbest = min(idx, key=lambda i: (rows[i]["x"] - CENTRE) ** 2
                            + (rows[i]["y"] - CENTRE) ** 2)
                hit = rows[cbest]["error_px"] <= POS_TOL
                d2 += hit
                arch_d2[a] = arch_d2.get(a, 0) + int(hit)
                prof_d2[pr] = prof_d2.get(pr, 0) + int(hit)
                class_sizes.append(int(same.sum()))
        nn = len(groups)
        diag[key] = {
            "D1_phase_oracle": d1 / nn,
            "D2_ceiling_le5": d2 / nn,
            "median_true_class_size": float(np.median(class_sizes)) if class_sizes else 0,
            "by_arch_D2": {k: arch_d2.get(k, 0) / v for k, v in arch_n.items()},
            "by_profile_D2": {k: f"{prof_d2.get(k, 0)}/{v}" for k, v in prof_n.items()},
        }
        print(f"[{key:6}] D1 phase-oracle {d1 / nn:.1%}   "
              f"D2 ceiling(correct class + centre rule) {d2 / nn:.1%}   "
              f"median true-class size {diag[key]['median_true_class_size']:.0f}")
        print(f"         D2 by arch: {diag[key]['by_arch_D2']}")
        print(f"         D2 by profile: {diag[key]['by_profile_D2']}")

    diag["n"] = len(groups)

    # ---- Stage 2: class scoring (measured regardless; gate interpretation) --
    for parity_aware in (False, True):
        key = "parity" if parity_aware else "fund"
        res = {}
        for pname in ("P1_top3", "P2_vote", "P3_5050"):
            hits, arch_hit, arch_n2, errs = 0, {}, {}, []
            for sid, rows in groups.items():
                a = rows[0]["architecture"]
                arch_n2[a] = arch_n2.get(a, 0) + 1
                Binv = recover_binv(rows)
                if Binv is None:
                    errs.append(9e9); continue
                mn = np.array([[r["disp_periods_x"], r["disp_periods_y"]] for r in rows])
                pc = wrap(mn)
                par = (np.round(mn[:, 0]) + np.round(mn[:, 1])).astype(int) % 2
                labels = clusters_of(pc, par, parity_aware)
                zs = {ch: np.array([r[f"{c}_z"] for r in rows])
                      for ch, c in (("raw", "raw"), ("mid", "mid"), ("dir", "dir"))}
                ncl = labels.max() + 1
                s1 = np.zeros(ncl)
                s2 = np.zeros(ncl)
                for li in range(ncl):
                    m = labels == li
                    for ch in zs:
                        v = np.sort(zs[ch][m])[::-1][:3]
                        s1[li] += v.mean() / 3.0
                # voting: top-100 per channel vote with positive z weight
                for ch in zs:
                    o = np.argsort(-zs[ch])[:100]
                    for i in o:
                        wgt = max(float(zs[ch][i]), 0.0)
                        s2[labels[i]] += wgt
                def norm(v):
                    sd = v.std()
                    return (v - v.mean()) / (sd if sd > 1e-9 else 1.0)
                score = {"P1_top3": s1, "P2_vote": s2,
                         "P3_5050": 0.5 * norm(s1) + 0.5 * norm(s2)}[pname]
                win = int(np.argmax(score))
                members = np.nonzero(labels == win)[0]
                best = min(members, key=lambda i: (rows[i]["x"] - CENTRE) ** 2
                           + (rows[i]["y"] - CENTRE) ** 2)
                e = rows[best]["error_px"]
                errs.append(float(e))
                hits += int(e <= POS_TOL)
                arch_hit[a] = arch_hit.get(a, 0) + int(e <= POS_TOL)
            e = np.array(errs)
            nn = len(groups)
            res[pname] = {
                "acc_5px": hits / nn,
                "median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)),
                "cat_gt25": float(np.mean(e > 25)),
                **{f"{k}_le5": arch_hit.get(k, 0) / v for k, v in arch_n2.items()},
            }
            print(f"[{key:6}] {pname}: <=5px {hits / nn:.1%}  median {np.median(e):.1f}  "
                  f">25 {np.mean(e > 25):.1%}  "
                  + "  ".join(f"{k} {arch_hit.get(k, 0) / v:.1%}" for k, v in arch_n2.items()))
        diag[f"scoring_{key}"] = res

    (OUT / "phase_eval.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'phase_eval.json'}")


if __name__ == "__main__":
    main()
