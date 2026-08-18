#!/usr/bin/env python3
"""Phase-1 H2 (generator shortcut) + H4 (channel ablation) diagnostic.

Read-only: the target-correlated process signature is removed by a RUNTIME
monkeypatch inside this disposable script. `driftforge/scene.py` is never
edited, and the manifests/test split are untouched.

For each pair we evaluate three correlation channels from ONE pass of NCC
maps (raw-only, midband-only, and the shipped 0.25/0.75 fusion), under two
scene conditions (signature present = shipped, signature removed).

Usage:
    python reports/phase1/diagnostics/ablation.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import driftforge.scene as scene_mod  # noqa: E402
from driftforge.baseline import Candidate, _ncc_valid, _peaks, _robust_contrast, _template_from_reference  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

ROTATIONS = (-3.0, -1.5, 0.0, 1.5, 3.0)
SCALES = (0.94, 0.97, 1.0, 1.03, 1.06)
CHANNELS = {"raw": (1.0, 0.0), "midband": (0.0, 1.0), "fusion": (0.25, 0.75)}


def _midband(img: np.ndarray) -> np.ndarray:
    return (
        ndimage.gaussian_filter(img, sigma=3.0, mode="reflect")
        - ndimage.gaussian_filter(img, sigma=15.0, mode="reflect")
    ).astype(np.float32)


def _select(all_candidates: list[Candidate], search_shape, tie_delta=0.015):
    """Faithful re-implementation of the shipped selection rule."""
    all_candidates.sort(key=lambda it: it.score, reverse=True)
    best = all_candidates[0]
    cx = (search_shape[1] - 1.0) / 2.0
    cy = (search_shape[0] - 1.0) / 2.0
    sep = 0.35 * 100.0
    rivals = [it for it in all_candidates
              if (it.x - best.x) ** 2 + (it.y - best.y) ** 2 >= sep ** 2]
    peak_gap = float(best.score - (rivals[0].score if rivals else -1.0))
    tied = [it for it in all_candidates if it.score >= best.score - tie_delta]
    chosen = min(tied, key=lambda it: (it.x - cx) ** 2 + (it.y - cy) ** 2)
    return chosen, peak_gap


def locate_channels(reference: np.ndarray, search: np.ndarray) -> dict:
    """One pass of templates/NCC maps, three channel weightings."""
    sf = _robust_contrast(search)
    ss = _midband(sf)
    buckets: dict[str, list[Candidate]] = {k: [] for k in CHANNELS}
    for scale in SCALES:
        for rot in ROTATIONS:
            tmpl = _template_from_reference(reference, scale, rot)
            if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
                continue
            ts = _midband(tmpl)
            r_raw = _ncc_valid(sf, tmpl)
            r_mid = _ncc_valid(ss, ts)
            hw = (tmpl.shape[1] - 1.0) / 2.0
            hh = (tmpl.shape[0] - 1.0) / 2.0
            for name, (wa, wb) in CHANNELS.items():
                resp = (wa * r_raw + wb * r_mid).astype(np.float32)
                for val, left, top in _peaks(resp, tmpl.shape, 12):
                    buckets[name].append(Candidate(left + hw, top + hh, val, rot, scale))
    out = {}
    for name, cands in buckets.items():
        chosen, gap = _select(cands, search.shape)
        out[name] = {"x": chosen.x, "y": chosen.y, "score": chosen.score, "peak_gap": gap,
                     "scale": chosen.scale, "rotation": chosen.rotation_deg}
    return out


# ---------------------------------------------------------------- signature
_ORIGINAL = scene_mod._apply_structural_defects


def _defects_without_signature(img, spec, xs, ys):
    """Shipped random-defect code, with the target-correlated block removed.

    This mirrors driftforge/scene.py::_apply_structural_defects exactly up to
    (but excluding) the deterministic 'shared process signature' that is
    placed at target_cx_nm/target_cy_nm +/- 260 nm.
    """
    rng = np.random.default_rng(spec.seed * 65_537 + 19)
    n = int(round(10 * spec.defect_density))
    x = xs[None, :]
    y = ys[:, None]
    xmin, xmax = float(xs[0]), float(xs[-1])
    ymin, ymax = float(ys[0]), float(ys[-1])
    for _ in range(n):
        cx = float(rng.uniform(0, scene_mod.WORLD_FOV_NM))
        cy = float(rng.uniform(0, scene_mod.WORLD_FOV_NM))
        radius = float(rng.uniform(18, 85))
        if cx + radius < xmin or cx - radius > xmax or cy + radius < ymin or cy - radius > ymax:
            continue
        kind = int(rng.integers(0, 4))
        if kind == 0:
            mask = ((x - cx) / radius) ** 2 + ((y - cy) / (radius * 0.7)) ** 2 <= 1
            img[mask] *= float(rng.uniform(0.15, 0.45))
        elif kind == 1:
            rr = ((x - cx) ** 2 + (y - cy) ** 2) / max(radius ** 2, 1)
            img[:] = np.maximum(img, (0.25 + 0.72 * np.exp(-2.4 * rr)).astype(np.float32))
        elif kind == 2:
            w = radius * 0.45
            mask = (np.abs(x - cx) < radius) & (np.abs(y - cy) < w)
            img[mask] = np.maximum(img[mask], 0.78)
        else:
            slope = float(rng.uniform(-0.6, 0.6))
            dist = np.abs((y - cy) - slope * (x - cx)) / math.sqrt(1 + slope ** 2)
            local = (np.abs(x - cx) < radius * 2.8) & (dist < max(radius * 0.12, 3.0))
            img[local] = np.maximum(img[local], 0.82)
    # NOTE: target-correlated signature intentionally omitted here.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--out", type=Path, default=PROJECT / "reports/phase1/diagnostics/ablation.json")
    args = ap.parse_args()

    records = read_manifest(args.manifest)[: args.limit]
    rows = []
    for i, rec in enumerate(records, 1):
        row = {"id": rec["id"], "profile": rec["profile"], "architecture": rec["architecture"]}
        for cond, patch in (("signature", _ORIGINAL), ("no_signature", _defects_without_signature)):
            scene_mod._apply_structural_defects = patch
            # generator imports render_scene from scene at call time, so the
            # patch is picked up; re-import generate_sample fresh each pass.
            from driftforge.generator import generate_sample
            s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], args.supersample)
            res = locate_channels(s.reference, s.search)
            for ch, r in res.items():
                err = math.hypot(r["x"] - s.gt_x, r["y"] - s.gt_y)
                row[f"{cond}__{ch}__error_px"] = round(err, 4)
                row[f"{cond}__{ch}__peak_gap"] = round(r["peak_gap"], 6)
            row["gt_x"], row["gt_y"] = round(s.gt_x, 3), round(s.gt_y, 3)
        scene_mod._apply_structural_defects = _ORIGINAL
        rows.append(row)
        print(f"[{i}/{len(records)}] {rec['id']} ({rec['profile']}) "
              f"sig/fusion={row['signature__fusion__error_px']:9.2f}  "
              f"nosig/fusion={row['no_signature__fusion__error_px']:9.2f}  "
              f"sig/raw={row['signature__raw__error_px']:9.2f}  "
              f"sig/mid={row['signature__midband__error_px']:9.2f}", flush=True)

    def summarize(key):
        e = np.array([r[key] for r in rows])
        return {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
                "max": float(e.max()), "acc_5px": float(np.mean(e <= 5)),
                "acc_25px": float(np.mean(e <= 25)), "n_gt_25px": int((e > 25).sum())}

    summary = {f"{c}__{ch}": summarize(f"{c}__{ch}__error_px")
               for c in ("signature", "no_signature") for ch in CHANNELS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (error px) ===")
    print(f"{'condition':28s} {'median':>9s} {'p95':>10s} {'max':>10s} {'acc<=5px':>9s} {'>25px':>6s}")
    for k, v in summary.items():
        print(f"{k:28s} {v['median']:9.2f} {v['p95']:10.2f} {v['max']:10.2f} "
              f"{v['acc_5px']:9.2%} {v['n_gt_25px']:6d}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
