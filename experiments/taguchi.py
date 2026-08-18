"""Taguchi L27 robustness infrastructure.

Purpose: measure how localization degrades under *physical* acquisition
conditions, not to tune hyperparameters. Learning rate and batch size never
appear here.

Design: L27(3^13) built over GF(3) from three generators, with 8 factors
assigned to orthogonal columns. Orthogonality is verified numerically at build
time rather than trusted from a textbook table.

Confounding, stated plainly: L27 carries 26 degrees of freedom. Eight
three-level factors consume 16, leaving 10. Two-factor interactions among the
assigned factors are therefore aliased with unassigned columns and fall into
the residual. This is a main-effects screening design; it will rank which
physical stresses matter, and it cannot resolve interactions between them.

    python -m experiments.taguchi --emit-design
    python -m experiments.taguchi --design experiments/taguchi_L27_design.csv \
        --pairs-per-run 30 --repeats 3 --output manifests/taguchi/
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

#: Seed base for Taguchi scenes. Disjoint from train (100k), validation (300k),
#: test (500k), stress (700k) and validation_phase2a (900k).
TAGUCHI_SEED_BASE = 1_200_000

FACTORS = ["A_dose", "B_psf", "C_rotation", "D_scale", "E_raster",
           "F_charging", "G_edge", "H_structure"]

#: Assigned L27 columns. All 13 columns are pairwise orthogonal, so any choice
#: gives clean main effects; these are spread across the generator and
#: interaction groups so no two factors share a generator pair.
ASSIGNED_COLUMNS = [1, 2, 5, 9, 10, 11, 12, 13]


# ------------------------------------------------------------ the L27 array


def l27_array() -> np.ndarray:
    """Full L27(3^13) orthogonal array, columns 1..13, levels in {0,1,2}."""
    rows = []
    for a, b, c in itertools.product(range(3), repeat=3):
        col = {
            1: a, 2: b, 3: (a + b) % 3, 4: (a + 2 * b) % 3, 5: c,
            6: (a + c) % 3, 7: (a + 2 * c) % 3, 8: (b + c) % 3,
            9: (a + b + c) % 3, 10: (a + 2 * b + c) % 3, 11: (b + 2 * c) % 3,
            12: (a + b + 2 * c) % 3, 13: (a + 2 * b + 2 * c) % 3,
        }
        rows.append([col[i] for i in range(1, 14)])
    return np.asarray(rows, dtype=int)


def verify_orthogonality(arr: np.ndarray, columns: list[int]) -> dict:
    """Every pair of assigned columns must show all 9 level combinations equally."""
    idx = [c - 1 for c in columns]
    problems = []
    for i, j in itertools.combinations(idx, 2):
        counts = np.zeros((3, 3), dtype=int)
        for r in range(arr.shape[0]):
            counts[arr[r, i], arr[r, j]] += 1
        if not np.all(counts == counts[0, 0]):
            problems.append({"cols": [i + 1, j + 1], "counts": counts.tolist()})
    balanced = all(
        np.all(np.bincount(arr[:, i], minlength=3) == arr.shape[0] // 3) for i in idx)
    return {"runs": int(arr.shape[0]), "assigned_columns": columns,
            "pairwise_balanced": not problems, "level_balanced": bool(balanced),
            "violations": problems}


# ------------------------------------------------------- factor level tables

#: Every level maps to a real generator parameter and stays inside (or at) the
#: documented range in driftforge/sem.py and scene.py. Rationale is recorded
#: per factor in experiments/configs/taguchi_L27.yaml.
LEVELS: dict[str, dict] = {
    "A_dose": {
        "mechanism": "Poisson shot noise / electron dose on the Search capture",
        "parameter": "AcquisitionSpec.dose (search branch)",
        "levels": [600.0, 200.0, 60.0],
        "units": "electrons (proxy)",
        "note": "generator draws 70-700 (140-700 non-hard); level 3 sits at the "
                "noisy end. Sponsor 'severe' is 25 - see DEFERRED_FINDINGS.",
    },
    "B_psf": {
        "mechanism": "beam spot size / defocus (Gaussian PSF)",
        "parameter": "AcquisitionSpec.psf_sigma_x_nm, psf_sigma_y_nm",
        "levels": [2.5, 5.0, 8.5],
        "units": "nm",
        "note": "generator range 2.0-7.0 (x) and 2.0-8.5 (y); level 3 is the y-max.",
    },
    "C_rotation": {
        "mechanism": "relative stage/scan rotation between Reference and Search",
        "parameter": "AcquisitionSpec.rotation_deg (reference branch)",
        "levels": [0.0, 1.2, 2.5],
        "units": "degrees",
        "note": "reference +/-2.2 and search +/-0.35 give ~2.55 deg relative; "
                "level 3 is the realistic worst case and sits inside the "
                "locator's +/-3 deg transform bank.",
    },
    "D_scale": {
        "mechanism": "magnification-ratio error between the two captures",
        "parameter": "AcquisitionSpec.scale (reference branch)",
        "levels": [1.000, 1.015, 1.035],
        "units": "ratio",
        "note": "reference range 0.965-1.035; level 3 is the edge of the "
                "locator's +/-6% scale bank.",
    },
    "E_raster": {
        "mechanism": "raster drift: progressive shear plus per-row jitter",
        "parameter": "AcquisitionSpec.scan_shear_px, scan_jitter_px (search)",
        "levels": [[0.5, 0.15], [2.2, 0.75], [4.5, 1.70]],
        "units": "px (shear, jitter sigma)",
        "note": "generator: shear +/-2.2 (+/-4.5 hard), jitter 0.05-1.7. "
                "Literature (PMC6541586) measures 0.5-5 px scan-line shifts, so "
                "even level 3 is conservative.",
    },
    "F_charging": {
        "mechanism": "local specimen charging: smooth field plus streaks",
        "parameter": "AcquisitionSpec.charging_strength, charging_streaks",
        "levels": [[0.0, 0], [0.10, 2], [0.20, 5]],
        "units": "normalised intensity, count",
        "note": "generator 0-0.10 (0-0.20 hard), streaks 0-2 (0-5 hard).",
    },
    "G_edge": {
        "mechanism": "SEM edge brightening (secondary-electron edge yield)",
        "parameter": "AcquisitionSpec.edge_strength",
        "levels": [0.14, 0.28, 0.42],
        "units": "normalised intensity",
        "note": "generator range 0.14-0.42; spans it exactly. Officially "
                "required by the problem statement and absent from the "
                "sponsor generator.",
    },
    "H_structure": {
        "mechanism": "fabrication roughness and structural defect density",
        "parameter": "SceneSpec.roughness_nm, defect_density",
        "levels": [[0.4, 0.4], [1.2, 0.9], [3.0, 2.0]],
        "units": "nm, defects per 10 um world (scaled)",
        "note": "generator roughness 0.4-2.0 (x1.5 hard -> 3.0), density "
                "0.4-1.4 (x1.4 -> 1.96).",
    },
}


def build_design() -> list[dict]:
    arr = l27_array()
    rows = []
    for r in range(arr.shape[0]):
        row = {"run": r + 1}
        for f, col in zip(FACTORS, ASSIGNED_COLUMNS):
            lv = int(arr[r, col - 1])
            row[f] = lv + 1                       # report levels as 1/2/3
            row[f"{f}_value"] = json.dumps(LEVELS[f]["levels"][lv]) \
                if isinstance(LEVELS[f]["levels"][lv], list) else LEVELS[f]["levels"][lv]
        rows.append(row)
    return rows


def condition_overrides(design_row: dict) -> dict:
    """Generator parameter overrides for one L27 condition.

    Returns the concrete AcquisitionSpec / SceneSpec field values this run
    requires. NOTE: `generate_sample()` does not yet accept spec overrides, so
    applying these needs a small generator hook. That hook is Phase-2E work and
    is recorded in reports/phase2a/DEFERRED_FINDINGS.md - it is deliberately
    not bolted on here, because Phase 2A must not change the generator beyond
    the E0 leak fix.
    """
    def val(f):
        lv = design_row[f] - 1
        return LEVELS[f]["levels"][lv]

    shear, jitter = val("E_raster")
    charge, streaks = val("F_charging")
    rough, density = val("H_structure")
    return {
        "search_acquisition": {
            "dose": val("A_dose"),
            "psf_sigma_x_nm": val("B_psf"),
            "psf_sigma_y_nm": val("B_psf"),
            "scan_shear_px": shear,
            "scan_jitter_px": jitter,
            "charging_strength": charge,
            "charging_streaks": streaks,
            "edge_strength": val("G_edge"),
        },
        "reference_acquisition": {
            "rotation_deg": val("C_rotation"),
            "scale": val("D_scale"),
            "psf_sigma_x_nm": val("B_psf"),
            "psf_sigma_y_nm": val("B_psf"),
            "edge_strength": val("G_edge"),
        },
        "scene": {"roughness_nm": rough, "defect_density": density},
    }


# ------------------------------------------------------- batch manifests


def build_batch_manifests(design: list[dict], pairs_per_run: int, repeats: int,
                          outdir: Path) -> dict:
    """One manifest per (run, repeat). Architecture-balanced, globally disjoint."""
    outdir.mkdir(parents=True, exist_ok=True)
    if pairs_per_run % 2:
        raise SystemExit("pairs-per-run must be even so DRAM/FinFET balance exactly")

    seed = TAGUCHI_SEED_BASE
    index, all_seeds = [], set()
    for row in design:
        for rep in range(1, repeats + 1):
            recs = []
            for i in range(pairs_per_run):
                arch = "dram" if i < pairs_per_run // 2 else "finfet"
                # Profiles cycle so every condition sees the same profile mix;
                # `ambiguous` is included because periodic ambiguity is the
                # phenomenon under study.
                profile = ("standard", "hard", "boundary", "ambiguous")[i % 4]
                recs.append({
                    "id": f"taguchi-r{row['run']:02d}-rep{rep}-{i:03d}",
                    "split": "taguchi", "run": row["run"], "repeat": rep,
                    "seed": seed, "scene_id": f"scene-{seed}",
                    "architecture": arch, "profile": profile,
                })
                all_seeds.add(seed)
                seed += 1
            path = outdir / f"run{row['run']:02d}_rep{rep}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
            index.append({"run": row["run"], "repeat": rep, "manifest": str(path),
                          "pairs": len(recs),
                          "levels": {f: row[f] for f in FACTORS}})

    # Disjointness against every production split.
    from driftforge.splits import read_manifest
    clash = {}
    for split in ("train", "validation", "test", "stress", "validation_phase2a_400"):
        p = PROJECT / f"manifests/{split}.jsonl"
        if p.exists():
            other = {int(r["seed"]) for r in read_manifest(p)}
            clash[split] = len(all_seeds & other)
    meta = {"runs": len(design), "repeats": repeats, "pairs_per_run": pairs_per_run,
            "total_evaluations": len(design) * repeats * pairs_per_run,
            "seed_range": [TAGUCHI_SEED_BASE, seed - 1],
            "seed_overlap_with_splits": clash,
            "disjoint": all(v == 0 for v in clash.values()),
            "index": index}
    (outdir / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ------------------------------------------------------------- analysis


def level_summary(design: list[dict], results: dict[int, list[dict]]) -> list[dict]:
    """Per-factor, per-level response table. `results` maps run -> prediction rows."""
    from experiments.metrics import sn_smaller_is_better
    out = []
    for f in FACTORS:
        for lv in (1, 2, 3):
            runs = [d["run"] for d in design if d[f] == lv]
            rows = [r for run in runs for r in results.get(run, [])]
            if not rows:
                continue
            e = np.array([r["error_px"] for r in rows], dtype=float)
            rt = np.array([r.get("runtime_ms", np.nan) for r in rows], dtype=float)
            out.append({
                "factor": f, "level": lv, "n": int(e.size),
                "mean_error": float(e.mean()), "median_error": float(np.median(e)),
                "p95_error": float(np.percentile(e, 95)),
                "accuracy_5px": float(np.mean(e <= 5.0)),
                "catastrophic_rate": float(np.mean(e > 25.0)),
                "sn_ratio_error": sn_smaller_is_better(e),
                "runtime_ms": float(np.nanmean(rt)) if np.isfinite(rt).any() else float("nan"),
            })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, default=PROJECT / "experiments/taguchi_L27_design.csv")
    ap.add_argument("--emit-design", action="store_true")
    ap.add_argument("--pairs-per-run", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=Path, default=PROJECT / "manifests/taguchi")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    arr = l27_array()
    check = verify_orthogonality(arr, ASSIGNED_COLUMNS)
    print("L27 orthogonality:", json.dumps(
        {k: v for k, v in check.items() if k != "violations"}, indent=2))
    if not (check["pairwise_balanced"] and check["level_balanced"]):
        print("VIOLATIONS:", json.dumps(check["violations"], indent=2))
        return 2
    if args.verify_only:
        return 0

    design = build_design()
    if args.emit_design or not args.design.exists():
        cols = ["run"] + [c for f in FACTORS for c in (f, f"{f}_value")]
        args.design.parent.mkdir(parents=True, exist_ok=True)
        with args.design.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(design)
        print(f"wrote {args.design} ({len(design)} runs)")
        if args.emit_design:
            return 0

    meta = build_batch_manifests(design, args.pairs_per_run, args.repeats, args.output)
    print(f"\nbatch manifests -> {args.output}")
    print(f"  {meta['runs']} runs x {meta['repeats']} repeats x "
          f"{meta['pairs_per_run']} pairs = {meta['total_evaluations']} evaluations")
    print(f"  seeds {meta['seed_range'][0]}-{meta['seed_range'][1]}  "
          f"disjoint={meta['disjoint']}  overlap={meta['seed_overlap_with_splits']}")
    print("\nNOT EXECUTED. Running the campaign needs the generator override hook "
          "described in reports/phase2a/DEFERRED_FINDINGS.md, and model weights "
          "must be frozen first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
