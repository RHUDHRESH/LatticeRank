"""war45: equivalence-margin sweep for the shipped selection stage.

The tie rule is spec-required, but the margin that defines "genuinely
equivalent" is a shipped constant. Measured effect on the final blend
selection across margins; as margin -> 0 selection converges to argmax.
"""
from __future__ import annotations

import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from war45_struct import by_scene, load_csv, OUT  # noqa: E402
from war45_blend import _job  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402


def main():
    import joblib
    va = load_csv(OUT / "val_struct.csv")
    groups = by_scene(va)
    man = {r["id"]: r for r in read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")}
    bundle = joblib.load(PROJECT / "driftforge/models/hgb_r2.joblib")
    m, feats = bundle["model"], bundle["features"]
    jobs = [(man[sid], [(r["x"], r["y"]) for r in rows]) for sid, rows in sorted(groups.items())]
    res_scores = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for (sid, _), sc in zip(sorted(groups.items()), pool.map(_job, jobs)):
            res_scores[sid] = np.array(sc, dtype=float)

    def z(v):
        fin = np.isfinite(v)
        if fin.sum() < max(3, 0.5 * v.size):
            return None
        mu, sd = v[fin].mean(), v[fin].std()
        o = (v - mu) / (sd if sd > 1e-9 else 1.0)
        o[~fin] = -9.0
        return o

    out = {}
    for margin in (0.25, 0.10, 0.05, 0.02, 1e-9):
        sel = []
        for sid, rows in groups.items():
            X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rows]),
                              nan=0.0, posinf=9.0, neginf=-9.0)
            score = z(m.predict_proba(X)[:, 1].astype(float))
            rz = z(res_scores[sid])
            if rz is not None:
                score = score + rz
            eq = np.nonzero(score >= score.max() - margin)[0]
            best = min(eq, key=lambda i: (rows[i]["x"] - 499.5) ** 2
                       + (rows[i]["y"] - 499.5) ** 2)
            sel.append(rows[best]["error_px"])
        e = np.array(sel)
        out[str(margin)] = {"sel_acc5": float(np.mean(e <= 5)),
                            "median": float(np.median(e)),
                            "cat25": float(np.mean(e > 25))}
        print(margin, out[str(margin)])
    (OUT / "margin_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
