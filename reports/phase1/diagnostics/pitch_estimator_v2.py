#!/usr/bin/env python3
"""Can the device lattice pitch be recovered from the Search image alone?

The v1 prototype in autopsy.py failed: run on the mid-band (sigma 3 - sigma 15)
channel it locked onto the ~31 px routing period (route_x has period 310 nm)
instead of the 2.8-17 px device lattice.

v2 fixes three things:
  * operate on a HIGH-pass residual, not the mid-band;
  * restrict the period search to the physically possible device range
    (2.5-30 px at 10 nm/px covers both DriftForge and every sponsor preset);
  * score candidate periods by harmonic-summed spectral energy, so the
    fundamental wins even when the 2nd harmonic is individually stronger.
Validated against the known scene pitch (ground truth is used ONLY to score
the estimator, never inside it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import _robust_contrast  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

PMIN, PMAX = 2.5, 30.0


def estimate(search: np.ndarray) -> tuple[float, float, float, float]:
    img = _robust_contrast(search).astype(np.float64)
    # High-pass: remove everything coarser than ~2x the largest device pitch,
    # which is what killed the v1 estimator.
    hp = img - ndimage.gaussian_filter(img, sigma=12.0, mode="reflect")
    h, w = hp.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    P = np.abs(np.fft.fft2(hp * win)) ** 2

    def axis(profile: np.ndarray, length: int) -> tuple[float, float]:
        half = profile[: length // 2]
        periods = np.arange(PMIN, PMAX, 0.05)
        best_p, best_s, scores = np.nan, -1.0, []
        for p in periods:
            s = 0.0
            for harm in (1, 2, 3):
                k = harm * length / p
                if k >= len(half) - 1:
                    break
                k0 = int(np.floor(k))
                frac = k - k0
                val = half[k0] * (1 - frac) + half[k0 + 1] * frac
                s += val / harm          # weight the fundamental most
            scores.append(s)
            if s > best_s:
                best_s, best_p = s, p
        scores = np.asarray(scores)
        conf = float(best_s / (np.median(scores) + 1e-30))
        return float(best_p), conf

    px, cx = axis(P.sum(axis=0), w)   # horizontal frequency -> x pitch
    py, cy = axis(P.sum(axis=1), h)   # vertical   frequency -> y pitch
    return px, py, cx, cy


def main() -> None:
    recs = read_manifest(PROJECT / "manifests/validation.jsonl")[:20]
    rows, okx, oky = [], 0, 0
    for i, rec in enumerate(recs, 1):
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        acq = s.search_acquisition
        tx = acq.scale * s.scene.pitch_x_nm / 10.0
        ty = acq.scale * s.scene.pitch_y_nm / 10.0
        ex, ey, cx, cy = estimate(s.search)

        def verdict(e, t):
            if not np.isfinite(e):
                return "nan", 9.9
            r = e / t
            # accept the fundamental or a small-integer harmonic/subharmonic
            for m in (1, 2, 3):
                if abs(r - m) < 0.06 * m:
                    return (f"x{m}" if m > 1 else "exact"), abs(r - m)
                if abs(r - 1.0 / m) < 0.06:
                    return f"/{m}", abs(r - 1.0 / m)
            return "WRONG", abs(r - 1)

        vx, _ = verdict(ex, tx)
        vy, _ = verdict(ey, ty)
        okx += vx != "WRONG"
        oky += vy != "WRONG"
        rows.append({"id": rec["id"], "arch": rec["architecture"], "profile": rec["profile"],
                     "true_x": round(tx, 3), "est_x": round(ex, 3), "verdict_x": vx, "conf_x": round(cx, 2),
                     "true_y": round(ty, 3), "est_y": round(ey, 3), "verdict_y": vy, "conf_y": round(cy, 2)})
        print(f"[{i:2d}] {rec['id']} {rec['architecture']:6s} "
              f"x: true={tx:6.2f} est={ex:6.2f} {vx:6s} (conf {cx:5.1f}) | "
              f"y: true={ty:6.2f} est={ey:6.2f} {vy:6s} (conf {cy:5.1f})", flush=True)

    print(f"\nx-pitch recovered (fundamental or small harmonic): {okx}/20")
    print(f"y-pitch recovered (fundamental or small harmonic): {oky}/20")
    out = PROJECT / "reports/phase1/diagnostics/pitch_estimator_v2.json"
    out.write_text(json.dumps({"x_ok": okx, "y_ok": oky, "n": len(rows), "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
