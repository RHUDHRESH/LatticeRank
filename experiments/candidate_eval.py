"""C0 (NMS top-k) vs C1 (lattice expansion) true-cell candidate recall.

The single number that caps every downstream model: if the proposal set does
not contain the true site, no reranker - classical or learned - can recover it.

Both strategies share ONE correlation pass per pair, so the comparison is
strictly about candidate selection and the runtimes are directly comparable.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from driftforge.baseline import _ncc_valid, _peaks, _robust_contrast, _template_from_reference  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.lattice import estimate_lattice  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

TOL = (1.0, 2.0, 5.0)


def _midband(a):
    return (ndimage.gaussian_filter(a, 3.0, mode="reflect")
            - ndimage.gaussian_filter(a, 15.0, mode="reflect")).astype(np.float32)


def response_and_seeds(reference, search, scales=(0.97, 1.0, 1.03),
                       rotations=(-1.5, 0.0, 1.5), per_variant=12):
    """One correlation pass. Returns the best fused map, its offset, and seeds.

    The transform bank is trimmed from the shipped 5x5 to 3x3: Phase-1 profiling
    showed 79.6% of runtime is NCC calls, and this measurement is about
    candidate selection, not transform search.
    """
    sf = _robust_contrast(search)
    ss = _midband(sf)
    best = None
    seeds = []
    for sc in scales:
        for rot in rotations:
            t = _template_from_reference(reference, sc, rot)
            if t.shape[0] >= search.shape[0] or t.shape[1] >= search.shape[1]:
                continue
            resp = (0.25 * _ncc_valid(sf, t) + 0.75 * _ncc_valid(ss, _midband(t))).astype(np.float32)
            hw, hh = (t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0
            pk = _peaks(resp, t.shape, per_variant)
            for v, left, top in pk:
                seeds.append((v, left + hw, top + hh, sc, rot))
            if best is None or (pk and pk[0][0] > best[0]):
                best = (pk[0][0] if pk else -1.0, resp, hw, hh, sc, rot)
    seeds.sort(key=lambda s: -s[0])
    return best, seeds


def score_at(resp, hw, hh, x, y):
    c, r = int(round(x - hw)), int(round(y - hh))
    if 0 <= r < resp.shape[0] and 0 <= c < resp.shape[1]:
        return float(resp[r, c])
    return float("nan")


def propose(reference, search, sample_gt, lat=None):
    """Return (C0 candidates, C1 candidates, lattice, timings)."""
    t0 = time.perf_counter()
    best, seeds = response_and_seeds(reference, search)
    t_resp = time.perf_counter() - t0
    _, resp, hw, hh, _, _ = best

    # ---- C0: what the shipped locator keeps (NMS top-k, no lattice) --------
    c0 = [(x, y, v, sc, rot, 0, 0) for v, x, y, sc, rot in seeds[:25]]

    # ---- C1: seeds + explicit lattice translations -------------------------
    t1 = time.perf_counter()
    if lat is None:
        lat = estimate_lattice(search)
    trans = lat.translations(max_m=5, max_n=5)
    H, W = search.shape
    seen = set()
    c1 = []

    def add(x, y, sc, rot, m, n):
        key = (int(round(x)), int(round(y)))
        if key in seen or not (0 <= x < W and 0 <= y < H):
            return
        seen.add(key)
        c1.append((float(x), float(y), score_at(resp, hw, hh, x, y), sc, rot, m, n))

    B = lat.basis
    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        Binv = None
    for v, x, y, sc, rot in seeds[:6]:            # strong seeds only
        add(x, y, sc, rot, 0, 0)
        for d in trans:
            mn = Binv @ d if Binv is not None else np.array([0.0, 0.0])
            add(x + d[0], y + d[1], sc, rot, int(round(mn[0])), int(round(mn[1])))
    t_lat = time.perf_counter() - t1
    return c0, c1, lat, (t_resp, t_lat)


def recall_row(cands, gt):
    if not cands:
        return {f"hit_{t:g}": False for t in TOL} | {"n": 0, "best_dist": float("inf")}
    d = np.array([math.hypot(c[0] - gt[0], c[1] - gt[1]) for c in cands])
    out = {f"hit_{t:g}": bool((d <= t).any()) for t in TOL}
    out["n"] = len(cands)
    out["best_dist"] = float(d.min())
    return out


def run_one(args):
    rec, ss = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], ss)
    gt = (s.gt_x, s.gt_y)
    c0, c1, lat, (t_resp, t_lat) = propose(s.reference, s.search, gt)
    true_px = s.scene.pitch_x_nm / 10.0 * s.search_acquisition.scale
    true_py = s.scene.pitch_y_nm / 10.0 * s.search_acquisition.scale
    row = {"sample_id": rec["id"], "seed": int(rec["seed"]),
           "architecture": rec["architecture"], "profile": rec["profile"],
           "true_x": gt[0], "true_y": gt[1],
           "lat_pitch_x": lat.pitch_x, "lat_pitch_y": lat.pitch_y,
           "lat_orientation_deg": lat.orientation_deg,
           "lat_conf": lat.confidence, "lat_agreement": lat.agreement,
           "true_pitch_x": true_px, "true_pitch_y": true_py,
           "resp_s": t_resp, "lattice_s": t_lat}
    for tag, c in (("C0", c0), ("C1", c1)):
        for k, v in recall_row(c, gt).items():
            row[f"{tag}_{k}"] = v
    return row, c1, gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT / "manifests/validation_phase2a_400.jsonl")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outdir", type=Path, default=PROJECT / "experiments/war45")
    args = ap.parse_args()

    recs = read_manifest(args.manifest)[args.start:args.start + args.limit]
    print(f"{len(recs)} pairs, workers={args.workers}", flush=True)
    rows, t0 = [], time.perf_counter()
    payload = [(r, args.supersample) for r in recs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (row, _c1, _gt) in enumerate(pool.map(run_one, payload), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(payload):
                print(f"  [{i}/{len(payload)}] {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    def wilson(k, n, z=1.959963985):
        if n == 0:
            return 0.0, 0.0, 0.0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return p, max(0.0, c - h), min(1.0, c + h)

    summary = {"pairs": len(rows), "wall_s": wall, "manifest": args.manifest.name,
               "slice": [args.start, args.start + len(rows)], "strategies": {}}
    for tag in ("C0", "C1"):
        blk = {}
        for t in TOL:
            k = sum(1 for r in rows if r[f"{tag}_hit_{t:g}"])
            p, lo, hi = wilson(k, len(rows))
            blk[f"recall_{t:g}px"] = {"k": k, "n": len(rows), "p": p, "ci95": [lo, hi]}
        n = np.array([r[f"{tag}_n"] for r in rows])
        blk["median_candidates"] = float(np.median(n))
        blk["p95_candidates"] = float(np.percentile(n, 95))
        for arch in ("dram", "finfet"):
            sub = [r for r in rows if r["architecture"] == arch]
            k = sum(1 for r in sub if r[f"{tag}_hit_5"])
            p, lo, hi = wilson(k, len(sub))
            blk[f"recall_5px_{arch}"] = {"k": k, "n": len(sub), "p": p, "ci95": [lo, hi]}
        summary["strategies"][tag] = blk
    summary["mean_response_s"] = float(np.mean([r["resp_s"] for r in rows]))
    summary["mean_lattice_s"] = float(np.mean([r["lattice_s"] for r in rows]))

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "candidate_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (args.outdir / "candidate_recall.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== TRUE-CELL CANDIDATE RECALL ===")
    print(f"{'strat':6s} {'@1px':>16s} {'@2px':>16s} {'@5px':>16s} {'med#':>6s} {'p95#':>6s}")
    for tag in ("C0", "C1"):
        b = summary["strategies"][tag]
        cells = "".join(f"{b[f'recall_{t:g}px']['p']:>9.1%}"
                        f"[{b[f'recall_{t:g}px']['ci95'][0]:.0%}-{b[f'recall_{t:g}px']['ci95'][1]:.0%}]"
                        for t in TOL)
        print(f"{tag:6s}{cells}{b['median_candidates']:>7.0f}{b['p95_candidates']:>7.0f}")
    for tag in ("C0", "C1"):
        b = summary["strategies"][tag]
        print(f"{tag}: DRAM@5 {b['recall_5px_dram']['p']:.1%}  FinFET@5 {b['recall_5px_finfet']['p']:.1%}")
    print(f"\nmean response {summary['mean_response_s']:.2f}s  lattice {summary['mean_lattice_s']:.3f}s "
          f" wall {wall:.0f}s")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
