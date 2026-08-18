"""Recall vs candidate budget. The evidence diagnosis showed the true site is a
local maximum 97.5% of the time but ranks median-3 of ~9900 by score, with a
long tail (p90 rank 5380). So the question is not "is it there" but "how many
candidates must we keep to contain it", and whether lattice structure lets us
keep fewer."""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from driftforge.generator import generate_sample
from driftforge.lattice import estimate_lattice
from driftforge.splits import read_manifest
from experiments.candidate_eval import response_and_seeds

KS = (25, 50, 100, 200, 500, 1000, 2000)

def run_one(args):
    rec, ss = args
    s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], ss)
    gx, gy = s.gt_x, s.gt_y
    best, seeds = response_and_seeds(s.reference, s.search)
    _, resp, hw, hh, _, _ = best
    out = {"sample_id": rec["id"], "architecture": rec["architecture"], "profile": rec["profile"]}
    for fp in (3, 5, 9):
        loc = resp == ndimage.maximum_filter(resp, size=fp, mode="nearest")
        ys, xs = np.nonzero(loc)
        v = resp[ys, xs]
        order = np.argsort(v)[::-1]
        cx = xs[order] + hw; cy = ys[order] + hh
        d = np.hypot(cx - gx, cy - gy)
        first = int(np.argmax(d <= 5.0)) if (d <= 5.0).any() else -1
        for K in KS:
            out[f"nms{fp}_top{K}"] = bool(first >= 0 and first < K)
        out[f"nms{fp}_total"] = int(v.size)
        out[f"nms{fp}_rank"] = first + 1 if first >= 0 else -1
    # lattice-expanded top-50 (small budget + structure)
    lat = estimate_lattice(s.search)
    trans = lat.translations(max_m=4, max_n=4)
    loc = resp == ndimage.maximum_filter(resp, size=5, mode="nearest")
    ys, xs = np.nonzero(loc); v = resp[ys, xs]; o = np.argsort(v)[::-1][:50]
    H, W = s.search.shape; seen=set(); hit=False; cnt=0
    for i in o:
        bx, by = xs[i]+hw, ys[i]+hh
        for dx, dy in [(0.0,0.0)]+[tuple(t) for t in trans]:
            X, Y = bx+dx, by+dy
            k=(int(round(X)),int(round(Y)))
            if k in seen or not(0<=X<W and 0<=Y<H): continue
            seen.add(k); cnt+=1
            if math.hypot(X-gx,Y-gy)<=5.0: hit=True
    out["lat50_hit"]=hit; out["lat50_n"]=cnt
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=80); ap.add_argument("--start",type=int,default=200)
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--manifest",type=Path,default=PROJECT/"manifests/validation_phase2a_400.jsonl")
    ap.add_argument("--out",type=Path,default=PROJECT/"experiments/war45")
    a=ap.parse_args()
    recs=read_manifest(a.manifest)[a.start:a.start+a.limit]
    rows=[];t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(run_one,[(x,2) for x in recs]),1):
            rows.append(r)
            if i%20==0 or i==len(recs): print(f"  [{i}/{len(recs)}] {time.perf_counter()-t0:.0f}s",flush=True)
    n=len(rows)
    def wil(k,nn,z=1.959963985):
        if nn==0: return 0,0,0
        p=k/nn; d=1+z*z/nn; c=(p+z*z/(2*nn))/d
        h=z*math.sqrt(p*(1-p)/nn+z*z/(4*nn*nn))/d
        return p,max(0,c-h),min(1,c+h)
    summ={"pairs":n,"grid":{}}
    print("\n=== RECALL@5px vs CANDIDATE BUDGET ===")
    print(f"{'NMS':>4s} {'total':>7s} " + "".join(f"{('top'+str(K)):>9s}" for K in KS))
    for fp in (3,5,9):
        tot=float(np.median([r[f"nms{fp}_total"] for r in rows]))
        cells=""
        for K in KS:
            k=sum(r[f"nms{fp}_top{K}"] for r in rows); p,lo,hi=wil(k,n)
            summ["grid"][f"nms{fp}_top{K}"]={"p":p,"ci95":[lo,hi],"k":k,"n":n}
            cells+=f"{p:>9.1%}"
        print(f"{fp:>4d} {tot:>7.0f} {cells}")
    k=sum(r["lat50_hit"] for r in rows); p,lo,hi=wil(k,n)
    summ["lattice_top50"]={"p":p,"ci95":[lo,hi],"median_n":float(np.median([r["lat50_n"] for r in rows]))}
    print(f"\nlattice-expanded top-50 seeds: recall@5 {p:.1%} [{lo:.0%}-{hi:.0%}], median {summ['lattice_top50']['median_n']:.0f} candidates")
    for fp in (3,5,9):
        rk=[r[f"nms{fp}_rank"] for r in rows if r[f"nms{fp}_rank"]>0]
        summ["grid"][f"nms{fp}_rank_median"]=float(np.median(rk)) if rk else None
        summ["grid"][f"nms{fp}_found_rate"]=len(rk)/n
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/"recall_sweep.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (a.out/"recall_sweep.json").write_text(json.dumps(summ,indent=2),encoding="utf-8")
    print("\nwrote",a.out)
if __name__=="__main__": main()
