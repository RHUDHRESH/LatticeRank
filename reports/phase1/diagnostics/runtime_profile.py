#!/usr/bin/env python3
"""H3: runtime cost centres of the shipped 25-variant transform search."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import _ncc_valid, _peaks, _robust_contrast, _template_from_reference, locate  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402

ROT = (-3.0, -1.5, 0.0, 1.5, 3.0)
SCA = (0.94, 0.97, 1.0, 1.03, 1.06)


def midband(img):
    return (ndimage.gaussian_filter(img, sigma=3.0, mode="reflect")
            - ndimage.gaussian_filter(img, sigma=15.0, mode="reflect")).astype(np.float32)


def main() -> None:
    recs = read_manifest(PROJECT / "manifests/validation.jsonl")[:3]
    acc = {"preprocess": 0.0, "template_build": 0.0, "template_midband": 0.0,
           "ncc_raw": 0.0, "ncc_midband": 0.0, "peaks": 0.0, "select_psr": 0.0}
    n_ncc = 0
    totals = []
    for rec in recs:
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        t_all = time.perf_counter()

        t = time.perf_counter()
        sf = _robust_contrast(s.search)
        ss = midband(sf)
        acc["preprocess"] += time.perf_counter() - t

        scores = []
        for scale in SCA:
            for rot in ROT:
                t = time.perf_counter(); tmpl = _template_from_reference(s.reference, scale, rot)
                acc["template_build"] += time.perf_counter() - t
                t = time.perf_counter(); ts = midband(tmpl)
                acc["template_midband"] += time.perf_counter() - t
                t = time.perf_counter(); r1 = _ncc_valid(sf, tmpl)
                acc["ncc_raw"] += time.perf_counter() - t
                t = time.perf_counter(); r2 = _ncc_valid(ss, ts)
                acc["ncc_midband"] += time.perf_counter() - t
                n_ncc += 2
                resp = (0.25 * r1 + 0.75 * r2).astype(np.float32)
                scores.append(resp)
                t = time.perf_counter(); _peaks(resp, tmpl.shape, 12)
                acc["peaks"] += time.perf_counter() - t
        t = time.perf_counter()
        pooled = np.concatenate([m.ravel()[:: max(1, m.size // 60_000)] for m in scores])
        np.median(np.abs(pooled - np.median(pooled)))
        acc["select_psr"] += time.perf_counter() - t
        totals.append(time.perf_counter() - t_all)

    n = len(recs)
    per = {k: v / n for k, v in acc.items()}
    total = sum(per.values())
    # Independent check against the shipped entry point.
    s = generate_sample(int(recs[0]["seed"]), recs[0]["architecture"], recs[0]["profile"], 2)
    t = time.perf_counter(); locate(s.reference, s.search); shipped = time.perf_counter() - t

    print(f"{'stage':20s} {'s/pair':>9s} {'% of total':>11s}")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"{k:20s} {v:9.3f} {100*v/total:10.1f}%")
    print(f"{'TOTAL (instrumented)':20s} {total:9.3f}")
    print(f"{'locate() end-to-end':20s} {shipped:9.3f}")
    print(f"\nNCC calls per pair: {n_ncc // n}  (25 transform variants x 2 channels)")
    print(f"cost per NCC call: {(per['ncc_raw']+per['ncc_midband'])/(n_ncc//n):.4f} s")
    out = {"per_stage_s": per, "total_instrumented_s": total, "locate_end_to_end_s": shipped,
           "ncc_calls_per_pair": n_ncc // n,
           "pct": {k: 100 * v / total for k, v in per.items()}}
    p = PROJECT / "reports/phase1/diagnostics/runtime_profile.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
