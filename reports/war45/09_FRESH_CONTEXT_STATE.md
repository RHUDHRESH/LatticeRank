# 09 — Fresh-context state recovery (2026-08-18)

- branch: `ultracode/war45`
- HEAD: `e773844c2cf1257f270fcabd9d2b2a27dac16876` (up to date with origin)
- dirty files: none (working tree clean at recovery time)
- latest completed experiment: war45 pivot + balanced HGB ranker
  (`experiments/war45/ranker_results.json`: HGB top1 37.5%, top10 51.2%,
  oracle recall@5 92.5% at delta=0.15; pivot recall in `pivot_recall.json`)
- next action: **NONE — campaign complete and packaged (2026-08-18).**
  The structural descriptor, R0/R1/R2 ablation, Siamese, residual, phase-voting
  and blend experiments all ran to completion on the unchanged 80-pair disjoint
  validation split. Final shipped system: R2 HGB + 1.0x z-scored residual blend,
  top-1/<=5px 41.25% (`experiments/war45/blend_eval.json`, `preds_final.csv`).
  Inference CLI verified end-to-end (`python scripts/inference.py ref.png
  search.png` -> "x.xx,y.yy"). Full engineering report with the final benchmark
  table, all measured-and-cut branches, and the artifact inventory:
  `reports/war45/10_FINAL_SYSTEM.md`. Only remaining housekeeping: commit and
  push the final artifacts on `ultracode/war45` (working tree currently carries
  the untracked final JSONs/models/figures).
