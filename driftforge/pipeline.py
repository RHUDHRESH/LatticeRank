"""DriftForge v2 inference pipeline (war45 architecture).

    3 evidence channels -> adaptive candidate union (S >= Smax - delta)
    -> scene-relative + structural-identity features
    -> HGB probability (trained on disjoint scenes, hard negatives)
    -> periodic-cancellation residual evidence (z-blend, measured +1.2pt top-1,
       FinFET +2.4pt, catastrophic -1.2pt; tiny Siamese was measured and CUT)
    -> evidential equivalence set -> official nearest-Search-centre tie rule
    -> (x, y)

`compute_candidate_rows` is the SINGLE code path for candidate features: the
training tables (experiments/war45_ranker.py) and this inference module both
call it, so a feature can never drift between fit and predict.

Distance-to-centre is never a feature (Phase-1 R5); the centre enters only as
the official tie-break inside the probability-equivalence set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .channels import CHANNELS, harvest, response_maps
from .lattice import estimate_lattice

MODELS = Path(__file__).resolve().parent / "models"
DELTA = 0.10
MAX_CAND = 8000
EQ_MARGIN = 0.25      # z-score equivalence margin (war45 blend validation)
RES_WEIGHT = 1.0      # residual-evidence blend weight (war45 blend validation)


def _entropy(patch: np.ndarray) -> float:
    h, _ = np.histogram(patch, bins=24, range=(0, 255))
    p = h.astype(float) / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_candidate_rows(reference: np.ndarray, search: np.ndarray,
                           delta: float = DELTA, max_cand: int = MAX_CAND,
                           keep_xy: tuple[float, float] | None = None,
                           keep_tol: float = 5.0, struct: bool = True) -> list[dict]:
    """Adaptive harvest + full feature table for one (reference, search) pair.

    `keep_xy` is a TRAIN-ONLY guard: the cost cap sorts by the very score that
    fails, so it can silently discard the true site; training-table builds pass
    the ground truth here to preserve the positive class. Inference never does.
    """
    cm = response_maps(reference, search)
    cands = harvest(cm, delta=delta)
    if not cands:
        return []
    cands.sort(key=lambda c: -max(v for k, v in c.items() if k in CHANNELS
                                  and isinstance(v, float) and math.isfinite(v)))
    if len(cands) > max_cand:
        head = cands[:max_cand]
        if keep_xy is not None:
            head = head + [c for c in cands[max_cand:]
                           if math.hypot(c["x"] - keep_xy[0], c["y"] - keep_xy[1]) <= keep_tol]
        cands = head

    lat = estimate_lattice(search)
    B = lat.basis
    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        Binv = np.zeros((2, 2))
    top = cands[0]
    N = len(cands)
    maxes, pct, zsc = {}, {}, {}
    for c in CHANNELS:
        v = np.array([k[c] if math.isfinite(k[c]) else -9.0 for k in cands])
        maxes[c] = float(v.max())
        r = np.argsort(np.argsort(-v)) + 1
        pct[c] = r / max(N, 1)
        sd = float(v.std())
        zsc[c] = (v - float(v.mean())) / (sd if sd > 1e-9 else 1.0)
    top10 = {c: set(np.argsort(-np.array([k[c] if math.isfinite(k[c]) else -9.0
                                          for k in cands]))[:10]) for c in CHANNELS}
    bestidx = {c: int(np.argmax([k[c] if math.isfinite(k[c]) else -9.0 for k in cands]))
               for c in CHANNELS}

    a = search.astype(np.float32)
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
            "x": c["x"], "y": c["y"],
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
            "local_entropy": _entropy(search[y0:y0 + 32, x0:x0 + 32]),
            "noise_est": noise,
        })
    if struct and rows:
        from .structural_descriptor import build_context, describe
        ctx = build_context(reference, search)
        for r in rows:
            r.update(describe(ctx, r["x"], r["y"]))
    return rows


@dataclass
class LocateV2Result:
    x: float
    y: float
    probability: float
    eq_set_size: int
    n_candidates: int
    used_residual: bool
    diagnostics: dict = field(default_factory=dict)


def load_ranker():
    import joblib
    return joblib.load(MODELS / "hgb_r2.joblib")


def locate_v2(reference: np.ndarray, search: np.ndarray,
              use_residual: bool = True) -> LocateV2Result:
    """Full war45 pipeline. Always returns a coordinate."""
    rows = compute_candidate_rows(reference, search)
    h, w = search.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    if not rows:                                    # never abstain
        return LocateV2Result(cx, cy, 0.0, 0, 0, False,
                              {"note": "no candidates; centre fallback"})

    bundle = load_ranker()
    m, feats = bundle["model"], bundle["features"]
    X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows], dtype=np.float64),
                      nan=0.0, posinf=9.0, neginf=-9.0)
    p = m.predict_proba(X)[:, 1]

    def _z(v):
        fin = np.isfinite(v)
        if fin.sum() < max(3, 0.5 * v.size):
            return None
        mu, sd = v[fin].mean(), v[fin].std()
        out = (v - mu) / (sd if sd > 1e-9 else 1.0)
        out[~fin] = -9.0
        return out

    score = _z(p.astype(float))
    if score is None:
        score = p.astype(float)
    used_residual = False
    if use_residual:
        from .residual import ResidualMatcher
        rm = ResidualMatcher(reference, search)
        rz = _z(np.array([rm.score(r["x"], r["y"])["res_int_m50"] for r in rows]))
        if rz is not None:
            score = score + RES_WEIGHT * rz
            used_residual = True

    smax = float(score.max())
    eq = [i for i in range(len(rows)) if score[i] >= smax - EQ_MARGIN]
    best = min(eq, key=lambda i: (rows[i]["x"] - cx) ** 2 + (rows[i]["y"] - cy) ** 2)
    return LocateV2Result(
        x=float(rows[best]["x"]), y=float(rows[best]["y"]),
        probability=float(p[best]), eq_set_size=len(eq),
        n_candidates=len(rows), used_residual=used_residual,
        diagnostics={"score_max": smax, "prob_max": float(p.max())})
