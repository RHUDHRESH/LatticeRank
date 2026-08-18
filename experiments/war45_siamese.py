"""war45 P2: tiny Siamese metric learner over hard periodic negatives.

Sanctioned by the measured gate (struct_eval.json: descriptor-only top-10 =
56.2% < 65%): the hand-crafted structural family is real signal (R2 top-1
38.8% > R0 32.5% on identical candidates) but not sufficient, so the next step
is a LEARNED patch representation trained specifically to separate true sites
from their periodic aliases.

Discipline:
  - scene-disjoint: training patches come only from train.jsonl scenes;
    every number reported comes from the same 80 untouched validation pairs.
  - hard negatives ONLY: the top-scoring wrong candidates of each scene
    (per-channel tops + structural-score tops) - never easy random sites.
  - tiny model: 4 conv blocks -> 64-d L2-normalised embedding, no pretrained
    weights, no internet dependency, CPU-trainable in minutes.
  - inference follows the war45 funnel: adaptive candidates -> structural
    descriptor prune to top-150 -> Siamese similarity -> ensemble.

Subcommands: data (extract patch tensors), train, eval.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))

from war45_struct import OUT, by_scene, load_csv  # noqa: E402
from driftforge.baseline import _robust_contrast, _template_from_reference  # noqa: E402
from driftforge.generator import generate_sample  # noqa: E402
from driftforge.splits import read_manifest  # noqa: E402
from driftforge.structural_descriptor import struct_score  # noqa: E402

PATCH = 96
POS_TOL = 5.0
MODELS = PROJECT / "driftforge" / "models"


# --------------------------------------------------------------- patch access


def _crop(a: np.ndarray, cx: float, cy: float, size: int = PATCH) -> np.ndarray:
    h, w = a.shape
    x0 = int(np.clip(round(cx - size / 2), 0, w - size))
    y0 = int(np.clip(round(cy - size / 2), 0, h - size))
    return a[y0:y0 + size, x0:x0 + size]


def _to_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def _scene_patches(args):
    """One scene -> (template_u8, [patch_u8], [error_px], [x], [y])."""
    rec, rows, n_pos, n_neg = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
    t = _template_from_reference(s.reference, 1.0, 0.0)
    tc = _crop(t, t.shape[1] / 2, t.shape[0] / 2)
    sf = _robust_contrast(s.search)

    pos = sorted((r for r in rows if r["error_px"] <= POS_TOL),
                 key=lambda r: r["error_px"])[:n_pos]
    neg = [r for r in rows if r["error_px"] > POS_TOL]
    keep: dict[int, dict] = {}
    if neg:
        sc = struct_score(neg)
        for i in np.argsort(-sc)[:12]:
            keep[int(i)] = neg[int(i)]
        for ch in ("raw", "midband", "directionality"):
            v = np.array([r[ch] if math.isfinite(r[ch]) else -9.0 for r in neg])
            for i in np.argsort(-v)[:8]:
                keep[int(i)] = neg[int(i)]
    sel = pos + list(keep.values())[:n_neg]
    patches = [_to_u8(_crop(sf, r["x"], r["y"])) for r in sel]
    return (_to_u8(tc), patches,
            [r["error_px"] for r in sel], [r["x"] for r in sel], [r["y"] for r in sel],
            rec["architecture"])


def cmd_data(a):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.perf_counter()
    man = {r["id"]: r for r in read_manifest(PROJECT / "manifests/train.jsonl")}
    tr = []
    for fp in sorted(OUT.glob("train_struct*.csv")):
        tr.extend(load_csv(fp))
    groups = by_scene(tr)
    jobs = [(man[sid], rows, 3, 30) for sid, rows in groups.items() if sid in man]
    print(f"extracting patches for {len(jobs)} train scenes...")
    P, E, S = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, (tc, patches, errs, _, _, arch) in enumerate(pool.map(_scene_patches, jobs)):
            S.append((tc, len(patches)))
            P.extend(patches); E.extend(errs)
            if (k + 1) % 50 == 0:
                print(f"  [{k + 1}/{len(jobs)}]", flush=True)
    # scene index per patch + template stack
    tmpl = np.stack([tc for tc, n in S])
    scene_idx = np.concatenate([np.full(n, i, dtype=np.int32) for i, (_, n) in enumerate(S)])
    np.savez_compressed(OUT / "siamese_train.npz",
                        templates=tmpl, patches=np.stack(P),
                        errors=np.array(E, dtype=np.float32), scene_idx=scene_idx)
    print(f"train: {len(P)} patches, {len(S)} scenes, "
          f"{int(np.sum(np.array(E) <= POS_TOL))} pos, {time.perf_counter() - t0:.0f}s")

    # validation: template + top-150 struct-pruned candidates per scene
    manv = {r["id"]: r for r in read_manifest(PROJECT / "manifests/validation_phase2a_400.jsonl")}
    va = load_csv(OUT / "val_struct.csv")
    vgroups = by_scene(va)

    def _val_job(args):
        rec, rows = args
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        t = _template_from_reference(s.reference, 1.0, 0.0)
        tc = _crop(t, t.shape[1] / 2, t.shape[0] / 2)
        sf = _robust_contrast(s.search)
        sc = struct_score(rows)
        order = np.argsort(-sc)[:150]
        sel = [rows[int(i)] for i in order]
        return (_to_u8(tc), [_to_u8(_crop(sf, r["x"], r["y"])) for r in sel],
                [r["error_px"] for r in sel], [r["x"] for r in sel],
                [r["y"] for r in sel], rec["architecture"],
                [float(sc[int(i)]) for i in order])

    jobs = [(manv[sid], rows) for sid, rows in vgroups.items()]
    Tv, Pv, Ev, Xv, Yv, Av, SCv, sidx = [], [], [], [], [], [], [], []
    print(f"extracting patches for {len(jobs)} val scenes...")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, (tc, patches, errs, xs, ys, arch, scs) in enumerate(pool.map(_val_job, jobs)):
            Tv.append(tc); Av.append(arch)
            Pv.extend(patches); Ev.extend(errs); Xv.extend(xs); Yv.extend(ys)
            SCv.extend(scs)
            sidx.append(len(patches))
    np.savez_compressed(OUT / "siamese_val.npz",
                        templates=np.stack(Tv), patches=np.stack(Pv),
                        errors=np.array(Ev, dtype=np.float32),
                        xs=np.array(Xv, dtype=np.float32), ys=np.array(Yv, dtype=np.float32),
                        struct_scores=np.array(SCv, dtype=np.float32),
                        counts=np.array(sidx, dtype=np.int32),
                        archs=np.array(Av))
    print(f"val: {len(Pv)} patches over {len(Tv)} scenes")


# --------------------------------------------------------------------- model


def make_model():
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.a = nn.Conv2d(c, c, 3, padding=1)
            self.b = nn.Conv2d(c, c, 3, padding=1)
            self.r = nn.ReLU()

        def forward(self, x):
            return self.r(x + self.b(self.r(self.a(x))))

    class Encoder(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),   # 48
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 24
                Res(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 12
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))

        def forward(self, x):
            import torch.nn.functional as tF
            return tF.normalize(self.net(x), dim=1)

    return Encoder()


def _prep(u8):
    import torch
    x = torch.from_numpy(u8.astype(np.float32) / 255.0).unsqueeze(1)
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)


def cmd_train(a):
    import torch
    import torch.nn.functional as tF
    torch.manual_seed(0)
    d = np.load(OUT / "siamese_train.npz")
    tmpl, patches, errors, sidx = d["templates"], d["patches"], d["errors"], d["scene_idx"]
    n_scene = tmpl.shape[0]
    scenes = []
    for s in range(n_scene):
        m = np.nonzero(sidx == s)[0]
        pos = m[errors[m] <= POS_TOL]
        neg = m[errors[m] > POS_TOL]
        if len(pos) and len(neg) >= 5:
            scenes.append((s, pos, neg))
    print(f"{len(scenes)} usable scenes (with >=1 pos and >=5 hard negs)")

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tau = 0.07
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        rng.shuffle(scenes)
        tot, nb = 0.0, 0
        for s, pos, neg in scenes:
            p = int(rng.choice(pos))
            nn_ = rng.choice(neg, size=min(len(neg), 25), replace=False)
            batch = np.concatenate([tmpl[s:s + 1], patches[p:p + 1], patches[nn_]])
            z = model(_prep(batch))
            anchor, rest = z[0:1], z[1:]
            logits = (anchor @ rest.T) / tau
            loss = tF.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"epoch {epoch + 1}/{a.epochs}  loss {tot / max(nb, 1):.4f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "siamese.pt")
    print(f"saved {MODELS / 'siamese.pt'}")


def cmd_eval(a):
    import torch
    d = np.load(OUT / "siamese_val.npz")
    tmpl, patches, errors = d["templates"], d["patches"], d["errors"]
    xs, ys, ssc, counts, archs = d["xs"], d["ys"], d["struct_scores"], d["counts"], d["archs"]
    model = make_model()
    model.load_state_dict(torch.load(MODELS / "siamese.pt", weights_only=True))
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        zt = model(_prep(tmpl))
        zp = []
        for i in range(0, len(patches), 512):
            zp.append(model(_prep(patches[i:i + 512])))
        zp = torch.cat(zp)
    infer_s = time.perf_counter() - t0

    off = np.concatenate([[0], np.cumsum(counts)])
    res = {}
    for mode in ("siamese", "ensemble"):
        errs_sel, topk_hits = [], {1: 0, 5: 0, 10: 0}
        arch_top1 = {}
        arch_n = {}
        for s in range(len(counts)):
            lo, hi = off[s], off[s + 1]
            sim = (zt[s:s + 1] @ zp[lo:hi].T).numpy().ravel()
            st = ssc[lo:hi]
            zsim = (sim - sim.mean()) / (sim.std() + 1e-9)
            zst = (st - st.mean()) / (st.std() + 1e-9)
            score = zsim if mode == "siamese" else zsim + zst
            e = errors[lo:hi]
            o = np.argsort(-score)
            errs_sel.append(float(e[o[0]]))
            for K in topk_hits:
                topk_hits[K] += bool(np.any(e[o[:K]] <= POS_TOL))
            arch = str(archs[s])
            arch_n[arch] = arch_n.get(arch, 0) + 1
            arch_top1[arch] = arch_top1.get(arch, 0) + int(e[o[0]] <= POS_TOL)
        e = np.array(errs_sel)
        n = len(counts)
        res[mode] = {
            "top1": topk_hits[1] / n, "top5": topk_hits[5] / n, "top10": topk_hits[10] / n,
            "acc_1px": float(np.mean(e <= 1)), "acc_2px": float(np.mean(e <= 2)),
            "acc_5px": float(np.mean(e <= 5)), "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95)), "cat_rate_gt25": float(np.mean(e > 25)),
            **{f"{k}_top1": arch_top1.get(k, 0) / v for k, v in arch_n.items()},
        }
        print(mode, json.dumps(res[mode], indent=2))
    res["oracle_top150_recall5"] = float(np.mean(
        [bool(np.any(errors[off[s]:off[s + 1]] <= POS_TOL)) for s in range(len(counts))]))
    res["infer_s_total"] = infer_s
    print(f"top-150 pruned oracle: {res['oracle_top150_recall5']:.1%}  "
          f"siamese inference {infer_s:.1f}s / 80 pairs")
    (OUT / "siamese_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
