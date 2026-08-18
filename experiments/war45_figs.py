import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; W=P/"experiments/war45"; F=W/"figures"; F.mkdir(parents=True,exist_ok=True)
sw=json.load(open(W/"recall_sweep.json")); dg=json.load(open(W/"diag_evidence.json")); wl=json.load(open(W/"warp_limit.json"))
KS=[25,50,100,200,500,1000,2000]

# --- Fig 1: recall vs candidate budget, with the evidence ceiling ----------
fig,ax=plt.subplots(figsize=(7.6,4.8))
for fp,mk in ((3,'o'),(5,'s'),(9,'^')):
    y=[sw["grid"][f"nms{fp}_top{K}"]["p"] for K in KS]
    lo=[sw["grid"][f"nms{fp}_top{K}"]["ci95"][0] for K in KS]
    hi=[sw["grid"][f"nms{fp}_top{K}"]["ci95"][1] for K in KS]
    ax.plot(KS,y,marker=mk,lw=1.8,label=f"top-K local maxima (NMS {fp}px)")
    ax.fill_between(KS,lo,hi,alpha=0.12)
ceil=dg["true_is_local_max_rate"]
ax.axhline(ceil,color="tab:green",ls="--",lw=2,label=f"evidence ceiling {ceil:.1%} (true site IS a local max)")
ax.axhline(0.99,color="tab:red",ls=":",lw=2,label="99% gate required before ML")
ax.scatter([sw["lattice_top50"]["median_n"]],[sw["lattice_top50"]["p"]],s=150,marker="*",
           color="tab:purple",zorder=5,label=f"lattice expansion ({sw['lattice_top50']['median_n']:.0f} cand) = {sw['lattice_top50']['p']:.1%}")
ax.set_xscale("log"); ax.set_xlabel("candidates kept per pair (log)"); ax.set_ylabel("true-cell recall @5 px")
ax.set_ylim(0,1.02); ax.grid(alpha=0.3,which="both"); ax.legend(fontsize=8,loc="lower right")
ax.set_title("Candidate recall saturates far below the evidence ceiling\nn=80 leak-free pairs, 95% Wilson bands")
fig.tight_layout(); fig.savefig(F/"candidate_recall_comparison.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# --- Fig 2: where the true site ranks -------------------------------------
fig,ax=plt.subplots(figsize=(6.8,4.4))
names=["rank 1\n(global max)","rank <=10","rank <=50","is a local max\n(any rank)"]
vals=[dg["rank_top1_rate"],dg["rank_top10_rate"],dg["rank_top50_rate"],dg["true_is_local_max_rate"]]
b=ax.bar(range(4),vals,color=["tab:red","tab:orange","tab:blue","tab:green"],edgecolor="black",lw=.6)
for i,v in enumerate(vals): ax.text(i,v+.02,f"{v:.1%}",ha="center",fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=9); ax.set_ylim(0,1.1)
ax.set_ylabel("fraction of pairs"); ax.grid(alpha=.3,axis="y")
ax.set_title(f"The true site is almost always present, but poorly ranked\nmedian rank {dg['rank_median']:.0f} of ~{dg['median_local_maxima']:.0f} local maxima; median score deficit {dg['median_score_deficit']:.3f} NCC")
fig.tight_layout(); fig.savefig(F/"true_site_rank.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# --- Fig 3: lattice-expansion miss distances (falsification) ---------------
import csv
fig,ax=plt.subplots(figsize=(6.8,4.4))
ax.axvspan(0,5,color="tab:green",alpha=.15,label="within 5 px tolerance")
ax.axvspan(5,25,color="tab:orange",alpha=.15,label="near-miss band (warp hypothesis)")
ax.axvline(wl["miss_median_dist_px"],color="tab:red",lw=2,ls="--",label=f"median miss {wl['miss_median_dist_px']:.0f} px")
ax.axvline(wl["warp_median_px"],color="tab:blue",lw=2,ls=":",label=f"warp budget {wl['warp_median_px']:.1f} px")
ax.set_xscale("log"); ax.set_xlim(0.1,1000)
ax.set_xlabel("distance from truth to nearest lattice-expanded candidate (px, log)")
ax.set_yticks([]); ax.legend(fontsize=9,loc="upper left")
ax.set_title(f"Warp hypothesis falsified: only {wl['near_miss_share_of_misses']:.0%} of misses are near-misses\ncorr(distance, warp budget) = {wl['corr_dist_warp']:+.3f}")
fig.tight_layout(); fig.savefig(F/"lattice_falsification.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("wrote 3 figures to",F)
