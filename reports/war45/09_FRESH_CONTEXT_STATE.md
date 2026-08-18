# 09 — Fresh-context state recovery (2026-08-18)

- branch: `ultracode/war45`
- HEAD: `e773844c2cf1257f270fcabd9d2b2a27dac16876` (up to date with origin)
- dirty files: none (working tree clean at recovery time)
- latest completed experiment: war45 pivot + balanced HGB ranker
  (`experiments/war45/ranker_results.json`: HGB top1 37.5%, top10 51.2%,
  oracle recall@5 92.5% at delta=0.15; pivot recall in `pivot_recall.json`)
- next action: P0 structural identity descriptor
  (`driftforge/structural_descriptor.py`) + descriptor-only rank benchmark at
  delta=0.10, then R0/R1/R2 descriptor+HGB ablation on the unchanged 80-pair
  disjoint validation split.
