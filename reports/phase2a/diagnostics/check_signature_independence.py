#!/usr/bin/env python3
"""E0 proof: the structural signature's position is independent of the target.

Read-only. Nothing here modifies the generator; it calls
``driftforge.scene.signature_placement`` - the same function the renderer uses -
so the numbers describe the shipped code path rather than a reimplementation of
it.

Three things are measured:

1. **Distribution.** Across >=1000 scene records, where does the signature land
   relative to the target, and how often is it inside the Reference FOV?
2. **Correlation.** Pearson r between signature and target coordinates. A
   target-correlated placement drives these towards 1.
3. **Null / interventional check.** The strongest evidence available: hold the
   scene fixed, move the target, and show the signature does not move. This is
   a direct causal test, not an observational one - a spurious near-zero
   correlation cannot survive it.

Usage:
    python reports/phase2a/diagnostics/check_signature_independence.py
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.config import REFERENCE_FOV_NM, WORLD_FOV_NM  # noqa: E402
from driftforge.scene import SIGNATURE_RADIUS_NM, make_scene_spec, signature_placement  # noqa: E402

PROFILES = ("standard", "hard", "boundary", "ambiguous")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def expected_inside_fraction() -> dict:
    """Geometric probability that a uniform world draw lands in the Reference.

    The signature centre is uniform on a square of side
    ``WORLD_FOV_NM - 2 * max_radius``; the Reference is a 1000 nm window. The
    target itself is held >= 520 nm from the world edge, so the Reference is
    always fully interior and no edge correction is needed.
    """
    margin = SIGNATURE_RADIUS_NM[1]
    span = WORLD_FOV_NM - 2.0 * margin
    centre_in = (REFERENCE_FOV_NM**2) / (span**2)
    # "Any overlap" is the same calculation dilated by a mean radius.
    mean_r = 0.5 * (SIGNATURE_RADIUS_NM[0] + SIGNATURE_RADIUS_NM[1])
    overlap_in = ((REFERENCE_FOV_NM + 2.0 * mean_r) ** 2) / (span**2)
    return {
        "draw_span_nm": span,
        "centre_inside_reference": centre_in,
        "any_overlap_with_reference": overlap_in,
    }


def collect(n_scenes: int, rows_out: list | None = None) -> dict:
    rows = []
    for i in range(n_scenes):
        architecture = "dram" if i % 2 == 0 else "finfet"
        profile = PROFILES[i % len(PROFILES)]
        # Mirrors generator.generate_sample's scene seeding.
        seed = 900_000 + i
        spec = make_scene_spec(seed * 17 + 3, architecture, profile)
        sx, sy, radius, _ = signature_placement(spec)
        dx = sx - spec.target_cx_nm
        dy = sy - spec.target_cy_nm
        half = REFERENCE_FOV_NM / 2.0
        rows.append(
            {
                "seed": seed,
                "architecture": architecture,
                "profile": profile,
                # The ambiguous profile has defect_density == 0, so the
                # signature is never rendered even though a position exists.
                "rendered": bool(spec.defect_density > 0),
                "sig_x": sx,
                "sig_y": sy,
                "radius": radius,
                "target_x": spec.target_cx_nm,
                "target_y": spec.target_cy_nm,
                "dx": dx,
                "dy": dy,
                "distance": math.hypot(dx, dy),
                "centre_inside_reference": bool(abs(dx) <= half and abs(dy) <= half),
                "overlaps_reference": bool(
                    abs(dx) <= half + radius and abs(dy) <= half + radius
                ),
            }
        )

    rendered = [r for r in rows if r["rendered"]]
    dx = np.array([r["dx"] for r in rendered])
    dy = np.array([r["dy"] for r in rendered])
    dist = np.array([r["distance"] for r in rendered])
    sx = np.array([r["sig_x"] for r in rendered])
    sy = np.array([r["sig_y"] for r in rendered])
    tx = np.array([r["target_x"] for r in rendered])
    ty = np.array([r["target_y"] for r in rendered])

    inside = int(sum(r["centre_inside_reference"] for r in rendered))
    overlap = int(sum(r["overlaps_reference"] for r in rendered))
    if rows_out is not None:
        rows_out.extend(rows)
    return {
        "n_scenes": len(rows),
        "signature_present_count": len(rendered),
        "signature_absent_count": len(rows) - len(rendered),
        "signature_inside_reference_count": inside,
        "signature_inside_reference_fraction": inside / max(len(rendered), 1),
        "signature_overlaps_reference_count": overlap,
        "signature_overlaps_reference_fraction": overlap / max(len(rendered), 1),
        "expected_geometric": expected_inside_fraction(),
        "signature_target_dx_nm_mean": float(dx.mean()),
        "signature_target_dy_nm_mean": float(dy.mean()),
        "signature_target_dx_nm_sd": float(dx.std()),
        "signature_target_dy_nm_sd": float(dy.std()),
        "signature_target_distance_nm_median": float(np.median(dist)),
        "signature_target_distance_nm_p05": float(np.percentile(dist, 5)),
        "signature_target_distance_nm_p95": float(np.percentile(dist, 95)),
        "signature_target_distance_nm_max": float(dist.max()),
        "corr_signature_x_target_x": _corr(sx, tx),
        "corr_signature_y_target_y": _corr(sy, ty),
        "corr_signature_x_target_y": _corr(sx, ty),
        "corr_signature_y_target_x": _corr(sy, tx),
        "by_profile": {
            prof: {
                "n": sum(1 for r in rows if r["profile"] == prof),
                "rendered": sum(1 for r in rows if r["profile"] == prof and r["rendered"]),
                "inside_reference": sum(
                    1 for r in rows if r["profile"] == prof and r["rendered"]
                    and r["centre_inside_reference"]
                ),
            }
            for prof in PROFILES
        },
        "by_architecture": {
            arch: {
                "n": sum(1 for r in rows if r["architecture"] == arch),
                "inside_reference": sum(
                    1 for r in rows if r["architecture"] == arch and r["rendered"]
                    and r["centre_inside_reference"]
                ),
            }
            for arch in ("dram", "finfet")
        },
    }


def null_check(n: int = 200) -> dict:
    """Move the target, hold everything else. The signature must not move.

    Observational independence can arise by luck; this cannot. If any variant
    shifts the signature by even a float ULP, placement still reads the target.
    """
    max_shift = 0.0
    n_moved = 0
    for i in range(n):
        architecture = "dram" if i % 2 == 0 else "finfet"
        spec = make_scene_spec((900_000 + i) * 17 + 3, architecture, PROFILES[i % len(PROFILES)])
        base = signature_placement(spec)[:3]
        for tx, ty in ((520.0, 520.0), (9480.0, 9480.0), (5000.0, 1200.0), (1200.0, 8800.0)):
            moved = dataclasses.replace(spec, target_cx_nm=tx, target_cy_nm=ty)
            got = signature_placement(moved)[:3]
            shift = math.dist(base[:2], got[:2])
            max_shift = max(max_shift, shift)
            if shift > 0.0 or abs(base[2] - got[2]) > 0.0:
                n_moved += 1
    return {
        "n_scenes_tested": n,
        "n_target_positions_per_scene": 4,
        "n_signature_moved": n_moved,
        "max_signature_shift_nm": max_shift,
        "passed": n_moved == 0 and max_shift == 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=1200)
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "reports/phase2a/signature_independence.json")
    args = ap.parse_args()

    per_scene: list[dict] = []
    stats = collect(args.n_scenes, per_scene)
    stats["null_check"] = null_check()

    # Per-scene evidence table. The aggregate statistics are the argument, but
    # the raw rows are what let someone else check it.
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["seed", "architecture", "profile", "signature_present",
                "signature_x_nm", "signature_y_nm", "signature_radius_nm",
                "target_x_nm", "target_y_nm", "dx_nm", "dy_nm",
                "distance_to_target_nm", "inside_reference", "overlaps_reference"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in per_scene:
            w.writerow({
                "seed": r["seed"], "architecture": r["architecture"],
                "profile": r["profile"], "signature_present": r["rendered"],
                "signature_x_nm": round(r["sig_x"], 3),
                "signature_y_nm": round(r["sig_y"], 3),
                "signature_radius_nm": round(r["radius"], 3),
                "target_x_nm": round(r["target_x"], 3),
                "target_y_nm": round(r["target_y"], 3),
                "dx_nm": round(r["dx"], 3), "dy_nm": round(r["dy"], 3),
                "distance_to_target_nm": round(r["distance"], 3),
                "inside_reference": r["centre_inside_reference"],
                "overlaps_reference": r["overlaps_reference"],
            })
    print(f"wrote {csv_path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    exp = stats["expected_geometric"]["centre_inside_reference"]
    print("=== SIGNATURE INDEPENDENCE ===")
    print(f"n_scenes                          {stats['n_scenes']}")
    print(f"signature_present_count           {stats['signature_present_count']}")
    print(f"inside_reference (centre)         {stats['signature_inside_reference_count']}"
          f"  = {stats['signature_inside_reference_fraction']*100:.2f}%")
    print(f"  expected from geometry alone    {exp*100:.2f}%")
    print(f"inside_reference (any overlap)    {stats['signature_overlaps_reference_fraction']*100:.2f}%"
          f"  expected {stats['expected_geometric']['any_overlap_with_reference']*100:.2f}%")
    print(f"dx mean / dy mean (nm)            {stats['signature_target_dx_nm_mean']:+.1f}"
          f" / {stats['signature_target_dy_nm_mean']:+.1f}")
    print(f"distance median (nm)              {stats['signature_target_distance_nm_median']:.1f}")
    print(f"distance p05 / p95 (nm)           {stats['signature_target_distance_nm_p05']:.1f}"
          f" / {stats['signature_target_distance_nm_p95']:.1f}")
    print(f"corr(sig_x, target_x)             {stats['corr_signature_x_target_x']:+.4f}")
    print(f"corr(sig_y, target_y)             {stats['corr_signature_y_target_y']:+.4f}")
    print(f"null check passed                 {stats['null_check']['passed']}"
          f"  (max shift {stats['null_check']['max_signature_shift_nm']} nm)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
