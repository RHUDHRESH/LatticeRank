"""war45: LER (line-edge roughness) fingerprint - 10-case hard-alias gate.

Hypothesis (from the public HUNT competitor): persistent geometric edge
wobble along lines is a location fingerprint that survives magnification
reduction, while generic residual matching drowns it in noise.

Method per patch (search scale):
  For the dominant vertical grating (pitch_x): every ROW r gives one complex
  lattice-frequency sample  z_r = sum_c I[r,c] * exp(-2*pi*i*c/pitch_x).
  angle(z_r) is the subpixel line position of that row, |z_r| its grating
  strength. Centre the angles on the weighted circular mean, convert to px
  displacement, remove a linear trend -> the row-wise LER sequence.
  Same along columns for pitch_y. Score = grating-strength-weighted mean of
  the Pearson correlations of the two sequences (template vs patch), with a
  +-2 row/col alignment search (applied identically to true and wrong).

Test: the 10 hardest measured failures (final blend chose a wrong copy
>25 px away). One question: does LER rank TRUE above that WRONG copy?
Gate: strong immediate discrimination or the idea is dropped. No tuning.
"""
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import _robust_contrast, _template_from_reference  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.lattice import estimate_lattice  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

OUT = PROJECT / "experiments/war45"


def ler_sequence(patch: np.ndarray, pitch: float, axis: int):
    """Row-wise (axis=0) or column-wise (axis=1) subpixel line displacement.

    Returns (displacement px, weight) arrays after circular centring and
    linear detrend. axis=0 scans rows against the vertical grating pitch_x.
    """
    a = patch.astype(np.float64)
    if axis == 1:
        a = a.T
    n = a.shape[1]
    c = np.arange(n)
    ph = np.exp(-2j * np.pi * c / pitch)
    z = (a * ph[None, :]).sum(axis=1)               # one complex sample per row
    w = np.abs(z)
    if w.sum() < 1e-9:
        return None, None
    mean_dir = (z / np.maximum(w, 1e-12) * w).sum()
    theta = np.angle(z * np.conj(mean_dir))          # centred on circular mean
    d = theta * pitch / (2 * np.pi)                  # px displacement per row
    r = np.arange(len(d), dtype=float)
    W = w / w.sum()
    # weighted linear detrend
    rm, dm = (W * r).sum(), (W * d).sum()
    cov = (W * (r - rm) * (d - dm)).sum()
    var = (W * (r - rm) ** 2).sum()
    slope = cov / var if var > 1e-9 else 0.0
    return d - dm - slope * (r - rm), w


def wcorr(a, b, wa, wb):
    w = wa * wb
    if w.sum() < 1e-9:
        return 0.0
    w = w / w.sum()
    am, bm = (w * a).sum(), (w * b).sum()
    ac, bc = a - am, b - bm
    den = math.sqrt((w * ac * ac).sum() * (w * bc * bc).sum())
    return float((w * ac * bc).sum() / den) if den > 1e-12 else 0.0


def ler_score(tpl: np.ndarray, patch: np.ndarray, pitch_x: float, pitch_y: float,
              shift_r: int = 2) -> float:
    """Alignment-searched, grating-weighted LER correlation of two patches."""
    total, wsum = 0.0, 0.0
    for pitch, axis in ((pitch_x, 0), (pitch_y, 1)):
        if not (2.0 <= pitch <= 64.0):
            continue
        dt, wt = ler_sequence(tpl, pitch, axis)
        dp, wp = ler_sequence(patch, pitch, axis)
        if dt is None or dp is None:
            continue
        n = min(len(dt), len(dp))
        dt, wt, dp, wp = dt[:n], wt[:n], dp[:n], wp[:n]
        best = -2.0
        for s in range(-shift_r, shift_r + 1):
            if s >= 0:
                cc = wcorr(dt[s:n], dp[:n - s], wt[s:n], wp[:n - s]) if n - s > 8 else -2
            else:
                cc = wcorr(dt[:n + s], dp[-s:n], wt[:n + s], wp[-s:n]) if n + s > 8 else -2
            best = max(best, cc)
        strength = float(np.median(wt) * np.median(wp))
        total += strength * best
        wsum += strength
    return total / wsum if wsum > 1e-12 else 0.0


def crop(a, cx, cy, h, w):
    x0 = int(np.clip(round(cx - w / 2), 0, a.shape[1] - w))
    y0 = int(np.clip(round(cy - h / 2), 0, a.shape[0] - h))
    return a[y0:y0 + h, x0:x0 + w]


def main():
    preds = list(csv.DictReader(open(OUT / "preds_final.csv", encoding="utf-8")))
    man = {r["id"]: r for r in read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")}
    # hardest measured failures: the blend's own chosen wrong copies
    fails = [p for p in preds if float(p["top1_error_px"]) > 25.0][:10]
    print(f"{len(fails)} hard-alias cases (final blend argmax wrong by >25 px)\n")
    rows, margins = [], []
    for p in fails:
        rec = man[p["sample_id"]]
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        sf = _robust_contrast(s.search)
        tpl = _template_from_reference(s.reference, 1.0, 0.0)
        lat = estimate_lattice(s.search)
        th, tw = tpl.shape
        true_p = crop(sf, s.gt_x, s.gt_y, th, tw)
        wrong_p = crop(sf, float(p["pred_x"]), float(p["pred_y"]), th, tw)
        st = ler_score(tpl, true_p, lat.pitch_x, lat.pitch_y)
        sw = ler_score(tpl, wrong_p, lat.pitch_x, lat.pitch_y)
        win = st > sw
        margins.append(st - sw)
        rows.append({"sample_id": p["sample_id"], "architecture": rec["architecture"],
                     "profile": rec["profile"], "err_px": float(p["top1_error_px"]),
                     "ler_true": st, "ler_wrong": sw, "margin": st - sw, "win": bool(win)})
        print(f"  {p['sample_id']} {rec['architecture']:>6} {rec['profile']:>9} "
              f"err {float(p['top1_error_px']):6.1f}  LER(true) {st:+.3f}  "
              f"LER(wrong) {sw:+.3f}  {'WIN' if win else 'LOSS'}")
    m = np.array(margins)
    wins = int(sum(r["win"] for r in rows))
    dprime = float(m.mean() / m.std()) if m.std() > 1e-9 else 0.0
    summary = {"n": len(rows), "wins": wins, "mean_margin": float(m.mean()),
               "std_margin": float(m.std()), "dprime_like": dprime, "cases": rows}
    (OUT / "ler_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwins {wins}/{len(rows)}   mean margin {m.mean():+.3f}   "
          f"margin/std {dprime:+.2f}")
    print("VERDICT:", "STRONG - pursue" if wins >= 8 and m.mean() > 0.1 else
          "WEAK - drop, no tuning" if wins <= 6 else "MARGINAL - drop per gate")


if __name__ == "__main__":
    main()
