# DriftForge

DriftForge is an original, deterministic synthetic-data system **and** a measured
localization pipeline for SEMICON India 2026 Track 2 / Applied Materials
**Drift-Sense**. This is a cross-scale localization problem, not image
classification.

## Problem

Given a 1000×1000 high-magnification **Reference** image (1 nm/px, 1,000 nm
field of view) and a 1000×1000 **Search** image at ~10× lower magnification
(10 nm/px, 10,000 nm field of view), output the coordinate `(x, y)` — in Search
pixels — of the centre of the Reference field inside the Search field. The
official tie rule: among genuinely equivalent matches, choose the one nearest
the Search centre.

Each generated sample contains:

- `reference`: 1000×1000 grayscale, 1 nm/px;
- `search`: 1000×1000 grayscale, 10 nm/px;
- label `(x, y)`: centre of the reference field in Search pixels after
  Search-side geometry and raster distortion;
- full provenance: structural parameters, acquisition parameters, scene ID and seed.

## Leak history — read before citing any number

The historical **v1.0.0 "80% ≤5 px" result is superseded** and must not be
cited as localization performance. A Phase-1 audit found a target-correlated
landmark in the generator: a unique high-contrast defect was placed at the
target centre ±260 nm, landing inside the Reference field in 100% of
non-ambiguous samples — the generator marked its own answer. Phase 2A removed
the leak (defects now placed uniformly in world coordinates, verified by an
interventional test in `tests/test_phase2a_leakfree.py`). The same locator on
leak-free data scores **31.0% ≤5 px**. The full evidence trail is preserved in
[RESULTS.md](RESULTS.md) and [reports/phase2a/](reports/phase2a/).

Every number below is measured on leak-free data. Unless stated otherwise,
war45 numbers come from the same **80-pair scene-disjoint validation split**
(39 DRAM / 41 FinFET) with δ=0.10 adaptive candidate harvesting, reproducible
from the committed JSONs in [experiments/war45/](experiments/war45/).

## Final system architecture (war45)

The pipeline is the survivor of a sequence of measured experiments; everything
that failed its gate was cut and is documented below.

1. **3-channel adaptive candidate harvesting.** NCC maps on three channels —
   raw intensity, mid-band structure, and local directionality — with an
   adaptive union of each channel's local maxima within δ of its own best peak.
   Measured recall@5 px of the true site in the candidate pool
   (`pivot_recall.json`): the old single proposer C0 gave **65.0%**; the
   3-channel union gives **92.5%** at δ=0.15 (median 546 candidates; DRAM
   97.4%, FinFET 87.8%) and **90.0%** at δ=0.10 (median 120 candidates).
   Diagnosis justifying this stage (`diag_evidence.json`): the true site is a
   local max within 5 px in **97.5%** of pairs, at median rank 3 of ~9,896
   maxima, with a median NCC deficit of only 0.0206 to the top peak — the
   evidence exists; selection, not detection, is the problem.
2. **Per-candidate features.** 26 scene-relative peak-strength features
   (per-channel score, margin-to-best, z-score, percentile, cross-channel
   agreement, lattice-phase residual, local entropy, …) concatenated with a
   structural identity descriptor (52 features: tiny-shift ZNCC, blockwise NCC
   grids, directionality blocks, row/column profiles, local spectra, phase
   coherence, multi-scale summaries).
3. **Balanced HistGradientBoosting ranker** trained on 300 scene-disjoint
   training scenes (shipped as `driftforge/models/hgb_r2.joblib`). Controlled
   ablation on identical candidates (`struct_ablation.json`): peak-strength
   features alone top-1 37.5%; structural descriptor alone 26.2%; combined
   (R2) **40.0%** top-1, 57.5% top-10. Training on 5× more scenes moved R2
   only 38.8% → 40.0% — the model is data-saturated; the ceiling is
   representational.
4. **Periodic-cancellation residual blend.** Final score =
   z(HGB probability) + 1.0 × z(residual score `res_int_m50`), z-scored per
   scene. The residual channel subtracts the scene's dominant periodic
   component and scores what remains — weak standalone (see cuts) but a
   complementary signal in the blend.
5. **Equivalence set + official nearest-centre tie rule.** Candidates within a
   z-margin of 0.05 of the best blended score form an equivalence set; within
   it, the candidate nearest the Search centre is selected. The margin was
   measured (`margin_sweep.json`): anywhere in 0.02–0.10 the tie layer selects
   at the full 41.25% ≤5 px (identical to argmax); the initially tried 0.25 was
   too wide (37.5%) and declared non-equivalent candidates equivalent.
6. Output `(x, y)` with optional `--json` diagnostics.

Runtime: ~5 s/pair, single process, no network, no GPU.

## Measured results

All rows argmax (top-1), ≤5 px accuracy. B0 was measured on the 400-pair
leak-free Phase-2A split; the remaining rows on the 80-pair war45 split.

| System | ≤5 px | >25 px | Median (px) | P95 (px) | DRAM ≤5 px | FinFET ≤5 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 classical v1 locator (leak-free) | 31.0% | 69.0% | 248.0 | 659.0 | 38.0% | 24.0% |
| HGB on peak-strength features (`ranker_results.json`) | 37.5% | 62.5% | 245.7 | 760.4 | — | — |
| R2: HGB, peak-strength + structural (`struct_ablation.json`) | 40.0% | 60.0% | 180.5 | 743.3 | 48.7% | 31.7% |
| **Final: R2 + 1.0×residual blend (`blend_eval.json`)** | **41.25%** | **58.75%** | **178.6** | **688.0** | **48.7%** | **34.1%** |

Final system, additional measured detail (`blend_eval.json`, `R2+1.0res`):
≤1 px 33.75%, ≤2 px 40.0%, top-5 53.75%, top-10 60.0%, P99 904.9 px. With the
shipped equivalence-set + nearest-centre tie layer (z-margin 0.05): 41.25%
≤5 px, identical to argmax (`margin_sweep.json`). The candidate
pool's oracle recall (92.5%) versus top-10 (60.0%) quantifies how much is left
on the table by ranking rather than harvesting.

## Measured-and-cut branches

Each branch below was implemented, measured on the same split, and cut on its
numbers. The JSONs are committed.

- **Lattice expansion as candidate generator** — 72.5% recall at ~6,573
  candidates vs 90.0% at ~120 for the 3-channel union; strictly dominated,
  cut. (Warp hypothesis also falsified: error–warp correlation +0.002.)
- **Structural descriptor as standalone ranker** (`struct_eval.json`) — top-1
  40.0%, top-5 52.5%, top-10 56.2%, below its 65% promotion gate; demoted to
  a feature block inside the HGB ranker.
- **Tiny Siamese metric learner** (`siamese_eval.json`) — 60-scene
  hard-negative InfoNCE on struct-pruned top-150 pools (pruned oracle 70%):
  siamese-only top-1 5.0%; ensembled with R2, 33.75% — worse than R2 alone
  (40.0%); cut.
- **Periodic-cancellation residual as standalone matcher**
  (`residual_eval.json`) — best formulation `A_m50` top-1 32.5% on the δ=0.15
  pool (oracle 92.5%) against a 45% kill line; cut as standalone, retained
  only as a blend channel (worth +1.25 points of top-1 in the blend).
- **Lattice-phase equivalence classes + centre-rule voting**
  (`phase_eval.json`) — the phase-class oracle is 98.8% (D1), but the ceiling
  of "correct class + official centre rule" is only 15–17.5% (D2) and
  end-to-end scoring reaches 7.5–12.5%; cut. The true class has a median of
  28–59 members spanning the whole image, ground truth is uniformly placed, so
  the centre rule cannot recover it — and the latent audit shows the copies
  are distinguishable, so collapsing them discards real evidence.

The latent-ambiguity audit (`latent_audit.json`) is why the ranking problem is
considered winnable at all: across 30 catastrophic failures, the median
noise-free latent NCC between the true site and the chosen wrong site is
0.859; only 6.7% are ≥0.95 and 76.7% are ≤0.90. The aliases are physically
distinguishable — the generator is not the bottleneck, the representation is.

## Why this generator is hard to fool

1. One analytical world-coordinate layout is rendered at both physical scales. The Reference is never resized and pasted into Search.
2. Reference and Search use independent SEM acquisition RNG streams: PSF, edge brightening, dose/Poisson noise, detector noise, charging, hot pixels and contrast are not shared.
3. The label is obtained by warping a target mask through the exact Search geometry—rotation, scale, translation, raster shear/jitter and radial distortion—then taking its centroid.
4. Structural defects are placed uniformly in world coordinates, independent of the target — so a defect may fall inside the Reference by chance, outside it, or nowhere near it. (Before Phase 2A one defect class was positioned relative to the target and leaked the answer; see [reports/phase2a/01_E0_IMPLEMENTATION.md](reports/phase2a/01_E0_IMPLEMENTATION.md).) A separate `ambiguous` profile is deliberately wallpaper-like and obeys the official nearest-Search-centre rule.
5. Train/validation/test/stress scene IDs and seed ranges do not overlap.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Generate data

The included manifests define 5,100 reproducible pairs: 3,000 train, 600 validation, 1,000 private test and 500 stress. They are small by design; materializing all PNGs would require several gigabytes.

```bash
# Materialize the first 30 training pairs.
python scripts/generate_dataset.py \
  --manifest manifests/train.jsonl \
  --limit 30 \
  --output-dir data/train

# Or create a custom balanced DRAM + FinFET set.
python scripts/generate_dataset.py \
  --count 30 \
  --architecture both \
  --profile hard \
  --output-dir data/custom

# Build an unlabeled public test hand-off while retaining labels elsewhere.
python scripts/generate_dataset.py \
  --manifest manifests/test.jsonl \
  --limit 30 \
  --hide-labels \
  --output-dir data/public_test
```

Use `--search-supersample 2` (default) for production. `1` is a faster smoke test; `4` is a higher-quality but memory-heavier audit mode.

## Inference CLI

```bash
python scripts/inference.py reference.png search.png
# prints the predicted coordinate as: x.xx,y.yy

python scripts/inference.py --json reference.png search.png
# full diagnostics: candidates, scores, equivalence set, tie-rule decision
```

Verified end-to-end. Loads `driftforge/models/hgb_r2.joblib` automatically,
runs the full pipeline (harvest → features → HGB → residual blend →
equivalence set → nearest-centre tie rule), needs no network, ~5 s/pair.

Dataset tooling: `scripts/validate_dataset.py data/train` checks a
materialized set; `scripts/audit_manifest.py` runs a release-gate audit.

## Reproducing the benchmark numbers

All war45 numbers regenerate from committed code + manifests; each script
writes its JSON into `experiments/war45/`.

```bash
python experiments/war45_blend.py      # final shipped numbers (blend_eval.json)
python experiments/war45_struct.py     # struct descriptor val/train + R0/R1/R2 ablation
python experiments/war45_residual.py   # residual standalone formulations
python experiments/war45_phase.py      # phase equivalence-class post-mortem
python experiments/war45_siamese.py    # siamese train + eval
python experiments/war45_pivot.py      # 3-channel harvesting recall
```

The generic harness (`python -m experiments.benchmark --list`) remains
available for the classical baseline and visual galleries — see
[experiments/README.md](experiments/README.md).

Stop-rule compliance: the private test split was never touched; train and
validation are scene-disjoint; distance-to-Search-centre was never used as a
ranker feature (the centre rule exists only in the final spec-compliance
layer); every number in this README traces to a committed JSON and git SHA.

## Limitations

- **The 58.75% catastrophic rate (>25 px) is THE open problem.** The system
  either lands on the right lattice site or hundreds of pixels away; the
  latent audit shows most of these failures are distinguishable in principle,
  so this is a representation/ranking gap, not an information-theoretic wall.
- **Heavy tail.** Median 178.6 px is dominated by the catastrophic mass;
  P95 ≈ 688 px, P99 ≈ 905 px.
- **Architecture gap.** FinFET (34.1% top-1) remains markedly harder than
  DRAM (48.7%); its fin pitch reaches 2.8 Search px.
- **Synthetic-only validation.** All numbers are on DriftForge-generated
  data (80-pair main split); no real SEM imagery was available. The
  tie-rule/selective layer scores 37.5% ≤5 px and is not a safety net.
- **Ranking, not harvesting, is the binding constraint.** Oracle recall of the
  candidate pool is 92.5%; the ranker converts that into 41.25% top-1.

## ML loader

`driftforge.ml.OnTheFlyPairDataset` lazily renders immutable manifest entries and returns full Reference/Search arrays, a 100×100 reference at Search scale, coordinate targets and a stride-4 Gaussian heatmap. Set `as_torch=True` after installing PyTorch.

See [TRAINING.md](TRAINING.md), [SOURCES.md](SOURCES.md) and [RESULTS.md](RESULTS.md).
