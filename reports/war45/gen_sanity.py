import sys, json, math
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from driftforge.generator import generate_sample
from driftforge.scene import render_scene, make_scene_spec
OUT=Path("reports/war45/generator_gallery"); OUT.mkdir(parents=True,exist_ok=True)
CASES=[("dram","standard","DRAM CLEAN",950001),("dram","hard","DRAM HARD",950003),
       ("finfet","standard","FINFET CLEAN",950002),("finfet","hard","FINFET HARD",950004)]
meta=[]
for arch,prof,label,seed in CASES:
    s=generate_sample(seed,arch,prof,2)
    sc=s.scene
    # clean latent master: analytic render, no acquisition at all
    master=render_scene(sc, sc.target_cx_nm-500, sc.target_cy_nm-500, 1000.0, 500)
    h=50; x0=int(np.clip(round(s.gt_x)-h,0,900)); y0=int(np.clip(round(s.gt_y)-h,0,900))
    crop=s.search[y0:y0+100,x0:x0+100]
    fig,ax=plt.subplots(1,4,figsize=(19,5))
    ax[0].imshow(master,cmap="gray",interpolation="nearest")
    ax[0].set_title(f"LATENT MASTER (no acquisition)\n{label} | pitch {sc.pitch_x_nm:.0f}x{sc.pitch_y_nm:.0f} nm",fontsize=10)
    ax[1].imshow(s.reference,cmap="gray"); ax[1].set_title("Reference 1000x1000 @1nm/px",fontsize=10)
    ax[2].imshow(s.search,cmap="gray")
    ax[2].add_patch(plt.Rectangle((s.gt_x-50,s.gt_y-50),100,100,fill=False,ec="lime",lw=2))
    ax[2].plot(s.gt_x,s.gt_y,"o",mfc="none",mec="lime",ms=12,mew=2)
    ax[2].set_title("Search 1000x1000 @10nm/px + GT",fontsize=10)
    ax[3].imshow(crop,cmap="gray",interpolation="nearest"); ax[3].set_title("true target crop (100x100)",fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{label}  seed={seed} profile={prof}  preset={sc.preset}",fontsize=13)
    fig.tight_layout(); p=OUT/f"{arch}_{'clean' if prof=='standard' else 'hard'}.png"
    fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    meta.append(dict(label=label,seed=seed,architecture=arch,profile=prof,preset=sc.preset,
        pitch_x_nm=round(sc.pitch_x_nm,2),pitch_y_nm=round(sc.pitch_y_nm,2),
        width_x_nm=round(sc.width_x_nm,2),width_y_nm=round(sc.width_y_nm,2),
        contact_nm=round(sc.contact_x_nm,2),gt=(round(s.gt_x,2),round(s.gt_y,2)),
        ref_dose=round(s.reference_acquisition.dose,1),search_dose=round(s.search_acquisition.dose,1),
        search_psf_nm=round(s.search_acquisition.psf_sigma_x_nm,2),
        rel_rot_deg=round(s.reference_acquisition.rotation_deg-s.search_acquisition.rotation_deg,3),
        rel_scale=round(s.reference_acquisition.scale/s.search_acquisition.scale,5),
        figure=str(p)))
    print("wrote",p,flush=True)
(OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
print("DONE")
