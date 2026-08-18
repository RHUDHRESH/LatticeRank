#!/usr/bin/env python3
"""Rigorous lattice-alias test (read-only diagnostic).

The naive test dx/pitch_nominal is biased: the Search capture applies its own
scale (0.992-1.008, wider for `ood`) and rotation (+/-0.35 deg, x2.2 for
`ood`). Over a displacement of tens of lattice periods those accumulate into
several tenths of a period of apparent residual.

Here we build the ACTUAL lattice basis of the captured Search image,

    a = s * (pitch_x/10) * ( cos t, sin t)      # x-direction repeat
    b = s * (pitch_y/10) * (-sin t, cos t)      # y-direction repeat

solve (dx, dy) = m*a + n*b for real (m, n), and report the distance from
(m, n) to the nearest integer pair. A true periodic alias lands on integers.

DRAM/FinFET contacts sit on a checkerboard (parity = (ix+iy) even in
scene.py), so the contact sub-lattice only repeats when (m+n) is even; we
report that separately.

Usage:
    python reports/phase1/diagnostics/lattice_fit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--csv", type=Path, default=PROJECT / "reports/phase1/failure_autopsy.csv")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--out", type=Path, default=PROJECT / "reports/phase1/diagnostics/lattice_fit.json")
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8") as fh:
        preds = {r["sample_id"]: r for r in csv.DictReader(fh)}

    records = read_manifest(args.manifest)[: args.limit]
    rows = []
    for i, rec in enumerate(records, 1):
        p = preds[rec["id"]]
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], args.supersample)
        dx = float(p["pred_x"]) - float(p["gt_x"])
        dy = float(p["pred_y"]) - float(p["gt_y"])
        err = float(p["error_px"])

        acq = s.search_acquisition
        t = math.radians(acq.rotation_deg)
        sc = acq.scale
        px = sc * s.scene.pitch_x_nm / 10.0
        py = sc * s.scene.pitch_y_nm / 10.0
        a = np.array([px * math.cos(t), px * math.sin(t)])
        b = np.array([-py * math.sin(t), py * math.cos(t)])
        B = np.column_stack([a, b])
        mn = np.linalg.solve(B, np.array([dx, dy]))
        m_r, n_r = float(mn[0]), float(mn[1])
        m_i, n_i = round(m_r), round(n_r)
        res_m, res_n = abs(m_r - m_i), abs(n_r - n_i)
        # Residual expressed back in pixels: how far the prediction is from the
        # nearest exact lattice translate of the ground truth.
        resid_px = float(np.linalg.norm(B @ np.array([m_r - m_i, n_r - n_i])))

        rows.append({
            "sample_id": rec["id"], "profile": rec["profile"], "architecture": rec["architecture"],
            "error_px": err, "dx": round(dx, 3), "dy": round(dy, 3),
            "search_rotation_deg": round(acq.rotation_deg, 4), "search_scale": round(sc, 6),
            "pitch_x_px": round(px, 4), "pitch_y_px": round(py, 4),
            "m": round(m_r, 4), "n": round(n_r, 4),
            "m_int": m_i, "n_int": n_i,
            "residual_m": round(res_m, 4), "residual_n": round(res_n, 4),
            "residual_px": round(resid_px, 4),
            "parity_even": bool((m_i + n_i) % 2 == 0),
            "is_lattice_alias": bool(err > 5.0 and res_m <= 0.25 and res_n <= 0.25),
        })
        print(f"[{i}/{len(records)}] {rec['id']:20s} err={err:8.2f}  "
              f"(m,n)=({m_r:+8.3f},{n_r:+8.3f})  resid=({res_m:.3f},{res_n:.3f}) "
              f"= {resid_px:6.2f}px  parity_even={rows[-1]['parity_even']}", flush=True)

    fails = [r for r in rows if r["error_px"] > 5.0]
    alias = [r for r in fails if r["is_lattice_alias"]]
    summary = {
        "n": len(rows), "n_fail_gt5px": len(fails),
        "n_fail_gt25px": len([r for r in fails if r["error_px"] > 25]),
        "n_lattice_alias": len(alias),
        "alias_fraction_of_failures": (len(alias) / len(fails)) if fails else None,
        "max_residual_px_among_aliases": (max(r["residual_px"] for r in alias) if alias else None),
        "median_residual_px_among_aliases": (
            float(np.median([r["residual_px"] for r in alias])) if alias else None),
    }
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n=== LATTICE-ALIAS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
