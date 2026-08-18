#!/usr/bin/env python3
"""Phase-1 failure autopsy (read-only diagnostic).

Regenerates the frozen validation records, re-runs the SHIPPED locator, and
records lattice/candidate evidence for every pair. Nothing here mutates the
generator, the baseline, the manifests or the locked test split.

Usage:
    python reports/phase1/diagnostics/autopsy.py --manifest manifests/validation.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import (  # noqa: E402  (read-only import of shipped code)
    _ncc_valid,
    _peaks,
    _robust_contrast,
    _template_from_reference,
    locate,
)
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

SEARCH_NM_PER_PX = 10.0


def midband(img: np.ndarray) -> np.ndarray:
    """Same mid-frequency structural channel the shipped baseline uses."""
    return (
        ndimage.gaussian_filter(img, sigma=3.0, mode="reflect")
        - ndimage.gaussian_filter(img, sigma=15.0, mode="reflect")
    ).astype(np.float32)


def estimate_pitch(search: np.ndarray) -> tuple[float, float, float, float]:
    """Estimate dominant lattice pitch (px) from the Search image alone.

    Uses the power spectrum of the mid-band channel: this is exactly the
    information a v2 lattice estimator would have at inference time (no
    ground-truth scene parameters are consulted).
    Returns (pitch_x_px, pitch_y_px, conf_x, conf_y).
    """
    f = midband(_robust_contrast(search))
    f = f - f.mean()
    win = np.hanning(f.shape[0])[:, None] * np.hanning(f.shape[1])[None, :]
    power = np.abs(np.fft.rfft2(f * win)) ** 2
    n = f.shape[0]

    def axis_pitch(profile: np.ndarray, length: int) -> tuple[float, float]:
        # Ignore DC/very-low frequencies (period > 60 px) and periods < 3 px.
        kmin = max(2, int(round(length / 60.0)))
        kmax = max(kmin + 1, int(round(length / 3.0)))
        seg = profile[kmin:kmax].copy()
        if seg.size < 3 or not np.isfinite(seg).any() or seg.max() <= 0:
            return float("nan"), 0.0
        k = int(np.argmax(seg))
        peak = float(seg[k])
        med = float(np.median(seg)) + 1e-12
        # Parabolic interpolation on the spectral peak for sub-bin pitch.
        kk = k + kmin
        if 0 < k < seg.size - 1:
            a, b, c = seg[k - 1], seg[k], seg[k + 1]
            denom = (a - 2 * b + c)
            if abs(denom) > 1e-20:
                kk = kk + 0.5 * (a - c) / denom
        if kk <= 0:
            return float("nan"), 0.0
        return float(length / kk), float(peak / med)

    # rows of `power` index vertical frequency, columns horizontal frequency
    prof_x = power.sum(axis=0)                    # horizontal frequency -> x pitch
    full_v = np.abs(np.fft.fft2(f * win)) ** 2
    prof_y = full_v[: n // 2].sum(axis=1)         # vertical frequency -> y pitch
    px, cx = axis_pitch(prof_x, f.shape[1])
    py, cy = axis_pitch(prof_y, f.shape[0])
    return px, py, cx, cy


def fused_response(reference, search_feature, search_structure, scale, rotation):
    template = _template_from_reference(reference, scale, rotation)
    t_struct = midband(template)
    resp = (
        0.25 * _ncc_valid(search_feature, template)
        + 0.75 * _ncc_valid(search_structure, t_struct)
    ).astype(np.float32)
    return resp, template


def classify(err, dx, dy, dxp, dyp, gap, best, sel_scale, sel_rot, prof, pitch_ok):
    if err <= 5.0:
        return "correct", "within tolerance"
    frac_x = abs(dxp - round(dxp)) if np.isfinite(dxp) else 9.0
    frac_y = abs(dyp - round(dyp)) if np.isfinite(dyp) else 9.0
    near_lattice = pitch_ok and frac_x <= 0.30 and frac_y <= 0.30
    big = math.hypot(dx, dy) > 25.0
    if near_lattice and big:
        return "periodic_alias", (
            f"displacement is {dxp:+.2f} x-pitch, {dyp:+.2f} y-pitch "
            f"(fractional residual {frac_x:.2f}/{frac_y:.2f})"
        )
    if big and gap < 0.02:
        return "remote_rival_low_gap", f"remote jump with peak_gap={gap:.4f} (<0.02)"
    if big:
        return "remote_non_lattice", (
            f"remote jump not on the estimated lattice "
            f"(dx={dxp:.2f}, dy={dyp:.2f} pitch units)"
        )
    return "local_refinement_error", f"local miss of {err:.2f}px at scale={sel_scale}, rot={sel_rot}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT / "manifests/validation.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--outdir", type=Path, default=PROJECT / "reports/phase1")
    args = ap.parse_args()

    records = read_manifest(args.manifest)[args.start:args.start + args.limit]
    figdir = args.outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rec in enumerate(records, 1):
        sample = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], args.supersample)
        res = locate(sample.reference, sample.search)
        err = math.hypot(res.x - sample.gt_x, res.y - sample.gt_y)
        dx, dy = res.x - sample.gt_x, res.y - sample.gt_y

        est_px, est_py, conf_x, conf_y = estimate_pitch(sample.search)
        # True structural pitch in Search pixels (for cross-checking the estimator).
        true_px = sample.scene.pitch_x_nm / SEARCH_NM_PER_PX
        true_py = sample.scene.pitch_y_nm / SEARCH_NM_PER_PX
        # Use the true pitch for alias arithmetic; the FFT estimate is reported
        # alongside so the v2 estimator can be validated against it.
        dxp = dx / true_px if true_px > 0 else float("nan")
        dyp = dy / true_py if true_py > 0 else float("nan")

        # Remote rival relative to the CHOSEN point (>=35px away, matching the
        # separation the shipped locator uses).
        chosen = np.array([res.x, res.y])
        remote = [c for c in res.candidates if math.hypot(c.x - chosen[0], c.y - chosen[1]) >= 35.0]
        second_remote = remote[0].score if remote else float("nan")
        best_score = res.candidates[0].score

        # Distance from ground truth to the best candidate that IS near the GT,
        # to tell "GT was never a candidate" from "GT was a candidate but lost".
        gt_near = [c for c in res.candidates if math.hypot(c.x - sample.gt_x, c.y - sample.gt_y) <= 5.0]
        gt_candidate_score = max((c.score for c in gt_near), default=float("nan"))

        pitch_ok = bool(np.isfinite(true_px) and np.isfinite(true_py) and true_px > 1 and true_py > 1)
        fclass, evidence = classify(
            err, dx, dy, dxp, dyp, res.peak_gap, best_score,
            res.scale, res.rotation_deg, rec["profile"], pitch_ok,
        )

        rows.append({
            "sample_id": rec["id"],
            "seed": rec["seed"],
            "architecture": rec["architecture"],
            "profile": rec["profile"],
            "gt_x": round(sample.gt_x, 4), "gt_y": round(sample.gt_y, 4),
            "pred_x": round(res.x, 4), "pred_y": round(res.y, 4),
            "error_px": round(err, 4),
            "dx": round(dx, 4), "dy": round(dy, 4),
            "selected_scale": res.scale, "selected_rotation": res.rotation_deg,
            "best_score": round(best_score, 6),
            "second_remote_score": (round(second_remote, 6) if np.isfinite(second_remote) else ""),
            "peak_gap": round(res.peak_gap, 6),
            "psr": round(res.psr, 4),
            "estimated_pitch_x": (round(est_px, 4) if np.isfinite(est_px) else ""),
            "estimated_pitch_y": (round(est_py, 4) if np.isfinite(est_py) else ""),
            "true_pitch_x_px": round(true_px, 4), "true_pitch_y_px": round(true_py, 4),
            "pitch_conf_x": round(conf_x, 3), "pitch_conf_y": round(conf_y, 3),
            "dx_in_pitch_units": (round(dxp, 4) if np.isfinite(dxp) else ""),
            "dy_in_pitch_units": (round(dyp, 4) if np.isfinite(dyp) else ""),
            "candidate_count": len(res.candidates),
            "gt_candidate_score": (round(gt_candidate_score, 6) if np.isfinite(gt_candidate_score) else ""),
            "accepted": res.accepted,
            "ambiguity_detected": res.ambiguity_detected,
            "failure_class": fclass,
            "evidence": evidence,
        })
        print(f"[{i}/{len(records)}] {rec['id']} err={err:8.2f} class={fclass} "
              f"dxp={dxp:+.2f} dyp={dyp:+.2f} gap={res.peak_gap:.4f}", flush=True)

        # Visual sheet for every failure above tolerance.
        if err > 5.0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                sf = _robust_contrast(sample.search)
                ss = midband(sf)
                resp, tmpl = fused_response(sample.reference, sf, ss, res.scale, res.rotation_deg)
                hw = (tmpl.shape[1] - 1) / 2.0
                hh = (tmpl.shape[0] - 1) / 2.0

                fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
                ax[0].imshow(sample.reference, cmap="gray")
                ax[0].set_title(f"Reference 1nm/px\n{rec['id']} {rec['architecture']}/{rec['profile']}")
                ax[1].imshow(sample.search, cmap="gray")
                ax[1].plot(sample.gt_x, sample.gt_y, "o", mfc="none", mec="lime", ms=18, mew=2.5, label="ground truth")
                ax[1].plot(res.x, res.y, "x", color="red", ms=18, mew=2.5, label="prediction")
                for c in res.candidates[:12]:
                    ax[1].plot(c.x, c.y, ".", color="yellow", ms=5, alpha=0.8)
                ax[1].legend(loc="upper right", fontsize=8)
                ax[1].set_title(f"Search 10nm/px — err={err:.1f}px\n"
                                f"d=({dx:+.1f},{dy:+.1f})px = ({dxp:+.2f},{dyp:+.2f}) pitch")
                im = ax[2].imshow(resp, cmap="viridis")
                ax[2].plot(sample.gt_x - hw, sample.gt_y - hh, "o", mfc="none", mec="lime", ms=16, mew=2.2)
                ax[2].plot(res.x - hw, res.y - hh, "x", color="red", ms=16, mew=2.2)
                ax[2].set_title(f"fused NCC surface @ scale={res.scale}, rot={res.rotation_deg}\n"
                                f"gap={res.peak_gap:.4f} psr={res.psr:.1f} class={fclass}")
                fig.colorbar(im, ax=ax[2], fraction=0.046)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                fig.tight_layout()
                fig.savefig(figdir / f"fail_{rec['id']}_{fclass}.png", dpi=110)
                plt.close(fig)
            except Exception as exc:  # pragma: no cover - figures are best-effort
                print(f"    (figure failed: {exc})")

    out_csv = args.outdir / "failure_autopsy.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (args.outdir / "diagnostics" / "autopsy_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
