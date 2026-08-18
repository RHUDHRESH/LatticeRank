"""war45: latent-ambiguity audit of catastrophic periodic-alias failures.

Question (decision gate before metric learning): when the ranker picks a wrong
periodic copy >25 px away, is the wrong site DISTINGUISHABLE from the true
site in the noise-free latent world? If yes, a learned representation can
recover it. If the two sites are pixel-identical in the latent render, the GT
label is arbitrary up to the official nearest-centre rule and NO model can fix
it - the equivalence handling must.

Method per failure (preds_R2_old_struct.csv, error > 25 px):
  latent = render_scene(spec, 0, 0, WORLD_FOV, 1000)   # no SEM noise, no warp
  true site in latent coords  = nominal_gt (pre-warp)
  wrong site in latent coords = pred - (gt - nominal_gt) (undo mean warp)
  distinguishability = 1 - best_tiny_shift_ZNCC(latent_true, latent_wrong)
with a +-4 px alignment search absorbing residual local warp.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from driftforge.config import WORLD_FOV_NM, OUTPUT_SIZE  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.scene import render_scene  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

OUT = PROJECT / "experiments/war45"
HALF = 50   # ~100x100 patch, the Reference footprint at Search scale
R = 4       # alignment radius absorbing local (non-mean) warp


def _zn(a):
    a = a.astype(np.float64)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def best_ncc(a: np.ndarray, big: np.ndarray) -> float:
    """max ZNCC of `a` against every +-R shift inside `big`."""
    an = _zn(a).ravel()
    n = an.size
    h, w = a.shape
    best = -9.0
    for dy in range(big.shape[0] - h + 1):
        for dx in range(big.shape[1] - w + 1):
            b = big[dy:dy + h, dx:dx + w]
            best = max(best, float(an @ _zn(b).ravel()) / n)
    return best


def crop(a, cx, cy, half=HALF, pad=0):
    x0 = int(np.clip(round(cx - half - pad), 0, a.shape[1] - 2 * (half + pad)))
    y0 = int(np.clip(round(cy - half - pad), 0, a.shape[0] - 2 * (half + pad)))
    return a[y0:y0 + 2 * (half + pad), x0:x0 + 2 * (half + pad)]


def main():
    import csv
    preds = list(csv.DictReader(open(OUT / "preds_R2_old_struct.csv", encoding="utf-8")))
    man = {r["id"]: r for r in read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")}
    fails = [p for p in preds if float(p["error_px"]) > 25.0]
    print(f"{len(fails)} catastrophic failures; auditing up to 30")
    rows = []
    for p in fails[:30]:
        rec = man[p["sample_id"]]
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        latent = render_scene(s.scene, 0.0, 0.0, WORLD_FOV_NM, OUTPUT_SIZE)
        latent = np.asarray(latent, dtype=np.float64)
        wx = float(p["pred_x"]) - (s.gt_x - s.nominal_gt_x)
        wy = float(p["pred_y"]) - (s.gt_y - s.nominal_gt_y)
        a = crop(latent, s.nominal_gt_x, s.nominal_gt_y)                 # true site
        b = crop(latent, wx, wy, pad=R)                                  # wrong site
        ncc = best_ncc(a, b)
        # control: true site against itself (sanity, should be ~1)
        rows.append({
            "sample_id": p["sample_id"], "architecture": rec["architecture"],
            "profile": rec["profile"], "error_px": float(p["error_px"]),
            "latent_ncc_true_vs_wrong": ncc,
            "distinguishability": 1.0 - ncc,
        })
        print(f"  {p['sample_id']} {rec['architecture']:>6} {rec['profile']:>9} "
              f"err {float(p['error_px']):6.1f}px  latentNCC {ncc:+.4f}")
    ncc = np.array([r["latent_ncc_true_vs_wrong"] for r in rows])
    summary = {
        "n_audited": len(rows),
        "median_latent_ncc": float(np.median(ncc)),
        "frac_ncc_ge_098": float(np.mean(ncc >= 0.98)),
        "frac_ncc_ge_095": float(np.mean(ncc >= 0.95)),
        "frac_ncc_le_090": float(np.mean(ncc <= 0.90)),
        "by_profile": {},
        "by_arch": {},
        "cases": rows,
    }
    for key, field in (("by_profile", "profile"), ("by_arch", "architecture")):
        for v in sorted({r[field] for r in rows}):
            sel = np.array([r["latent_ncc_true_vs_wrong"] for r in rows if r[field] == v])
            summary[key][v] = {"n": int(sel.size), "median_ncc": float(np.median(sel)),
                               "frac_ge_095": float(np.mean(sel >= 0.95))}
    (OUT / "latent_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    verdict = ("GENERATOR/EQUIVALENCE ISSUE: most failed aliases are latent-identical"
               if summary["frac_ncc_ge_095"] > 0.6 else
               "LEARNABLE: most failed aliases are latent-distinguishable -> Siamese justified")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
