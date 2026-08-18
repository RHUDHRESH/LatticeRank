# 10 — War-45 final system report

**Campaign:** Drift-Sense war45 — SEMICON India Hackathon 2026, Track 2 (Applied Materials).
**Problem:** localize a 1000x1000 high-magnification Reference inside a 1000x1000 ~10x
lower-magnification Search image; output `(x,y)`. Official tie rule: among genuinely
equivalent matches, choose the one nearest the Search centre.
**Evaluation:** every number below is measured on the same 80-pair scene-disjoint,
leak-free validation split (`validation_phase2a`, slice 200–280), delta=0.10 adaptive
candidates unless stated otherwise. All numbers are reproducible from the committed
JSONs in `experiments/war45/` plus the git SHA.

> Historical note: the v1.0.0 "80% ≤5 px" figure is **superseded** — it was inflated by
> a target-correlated generator leak and is mentioned here only as superseded history.
> Nothing in this report descends from it.

---

## 1. Executive verdict

The shipped system is a two-channel blend: the R2 gradient-boosted ranker
(peak-strength + structural-identity features) plus a periodic-cancellation residual
channel, combined as `z(HGB prob) + 1.0 * z(residual res_int_m50)` over the delta=0.10
adaptive-union candidate pool. Source: `experiments/war45/blend_eval.json` (`R2+1.0res`),
predictions in `preds_final.csv` / `preds_final_argmax.csv`.

| Metric (80-pair leak-free split) | Final system |
|---|---|
| top-1 / ≤5 px | **41.25%** |
| ≤2 px | 40.0% |
| ≤1 px | 33.75% |
| top-5 recall | 53.75% |
| top-10 recall | 60.0% |
| median error | 178.6 px |
| P95 / P99 error | 688.0 px / 904.9 px |
| catastrophic (>25 px) | 58.75% |
| DRAM top-1 | 48.7% |
| FinFET top-1 | 34.1% |
| Candidate-pool oracle (recall@5) | 90.0% (median 120 candidates) |
| Runtime | ~5 s/pair, single process, no network |

With the equivalence-set + official nearest-centre tie rule applied (z-margin 0.25),
the selected point scores **37.5% ≤5 px** (`sel_acc_5px`). The tie-rule layer is kept
for spec compliance; the argmax numbers above are the headline. Against the leak-free
classical baseline B0 (≤5 px ~31%), the campaign moved top-1 accuracy from ~31% to
41.25% and reduced the catastrophic rate from ~69% to 58.75%, with every intermediate
step measured and every dead end explicitly killed by a number.

Inference CLI (verified end-to-end):

```
python scripts/inference.py reference.png search.png
# prints "x.xx,y.yy"; --json for diagnostics; loads driftforge/models/hgb_r2.joblib automatically
```

---

## 2. The five measured bets

Each bet had a pre-registered gate or kill line, was measured on the unchanged 80-pair
split, and was kept or cut on the number alone.

### Bet 1 — HGB ranker on peak-strength features (ceiling probe)

- Source: `ranker_results.json`, `struct_ablation.json`.
- Balanced HGB on 26 peak-strength features: **top-1 37.5%, top-10 51.2%**, against a
  candidate-pool oracle of 92.5%. RF was equivalent (top-1 37.5%).
- 5x more training data (60 → 300 scenes) moved the richer R2 model only
  38.8% → 40.0% top-1: **data saturation**, so the gap to oracle is a
  **representation ceiling**, not a data or capacity problem.
- **Decision: KEEP** as the ranking backbone; stop investing in more training data.

### Bet 2 — Structural identity descriptor (52 features)

- Source: `struct_eval.json`, `struct_ablation.json`. Descriptor: tiny-shift ZNCC,
  blockwise NCC grids, directionality blocks, row/col profiles, local spectra,
  phase coherence, multi-scale features. ~3.5 s/pair for full candidate features.
- Descriptor-only ranking: **top-1 40.0%, top-5 52.5%, top-10 56.2%** — below the 65%
  gate for a standalone ranker.
- Controlled ablation (300 train scenes, identical candidates): R0 old features top-1
  37.5%; R1 struct-only **26.2%** (worse alone); R2 old+struct **40.0% top-1,
  57.5% top-10, 62.5% >25 px**.
- **Decision: CUT as standalone (missed gate), KEEP as a feature block inside R2**
  (+2.5 pt top-1, +6.2 pt top-10 over R0).

### Bet 3 — Tiny Siamese metric learner

- Source: `siamese_eval.json`. 60-scene hard-negative InfoNCE, evaluated on the
  struct-pruned top-150 pool (pruned oracle 70%).
- Siamese-only: **top-1 5.0%**. Ensembled with R2: **33.75%** — strictly worse than
  R2 alone (40.0%).
- **Decision: CUT.** At this data scale the learned embedding destroys information the
  hand-built features already capture.

### Bet 4 — Periodic-cancellation residual matching

- Source: `residual_eval.json`. Nine formulations (A/B/C x masking m0/m30/m50) on the
  delta=0.15 full pool (oracle 92.5%).
- Best standalone: **A_m50 top-1 32.5%** against a 45% kill line — **CUT as standalone**.
- But blended as a second z-scored channel with R2 (`blend_eval.json`): top-1
  40.0% → **41.25%**, top-10 57.5% → 60.0%, >25 px 60.0% → 58.75%, P95 743 → 688 px.
- **Decision: CUT standalone, KEEP as blend channel** — the one bet that produced the
  final measurable gain (+1.25 pt top-1).

### Bet 5 — Lattice-phase equivalence-class voting + centre rule

- Source: `phase_eval.json`. Hypothesis: collapse candidates into lattice-phase
  equivalence classes, pick the class, let the official nearest-centre rule pick the
  member.
- Phase-oracle D1 (true class identifiable): **98.8%** — the physics works. But the
  D2 ceiling (correct class + centre rule) is only **15–17.5%**, and end-to-end
  scoring lands at **7.5–12.5% ≤5 px**.
- Cause, measured: the true class has median 28–59 members spanning the whole image;
  ground truth is uniform over the image, so the centre rule cannot recover it — and
  the latent audit (Bet 1's companion) shows the copies are physically distinguishable,
  so collapsing them discards real evidence.
- **Decision: CUT.**

---

## 3. How the winning system was actually found

The campaign's value is the causal chain, each link measured before the next was built.

1. **Failure analysis before therapy.** B0 (leak-free classical locator) sat at ~31%
   ≤5 px, ~69% >25 px, P95 ~659 px. Diagnosis (`diag_evidence.json`): the true site is a
   local maximum of the correlation surface within ≤5 px in **97.5%** of pairs, at
   median rank **3** of ~9,896 maxima, losing to the winner by a median of only
   **0.0206 NCC**. The answer was already in the surface; the bottleneck was
   discrimination among periodic twins, not candidate generation.

2. **Falsifications that saved the budget.** Lattice expansion as a candidate generator:
   72.5% recall at ~6,573 candidates — worse than cheaper alternatives, killed
   (`recall_sweep.json`, `warp_limit.json`). Warp-limited near-miss hypothesis:
   correlation of miss distance with warp budget **+0.002**, killed.

3. **Multi-channel harvesting pivot** (`pivot_recall.json`). Replacing the single
   proposer C0 (recall@5 **65.0%**) with a 3-channel adaptive-union harvester
   (raw / midband / directionality): **92.5%** recall@5 at delta=0.15 (median 546
   candidates; DRAM 97.4%, FinFET 87.8%) and **90.0%** at delta=0.10 (median 120
   candidates). This single pivot converted the problem from "the answer is missing"
   to "the answer is present, rank it".

4. **The representation wall.** HGB on peak-strength features: top-1 37.5% against a
   92.5% oracle. The structural descriptor lifted R2 to 40.0%; 5x training data moved
   it 38.8 → 40.0 (saturation). The latent-ambiguity audit (`latent_audit.json`, 30
   catastrophic failures) showed the wall is not fundamental: median latent NCC between
   the true site and the chosen wrong site is **0.859**, only **6.7%** of confusions are
   ≥0.95, and **76.7%** are ≤0.90 — the aliases ARE physically distinguishable, so the
   generator/equivalence structure is NOT the bottleneck. The features are.

5. **Residual complementarity.** The only channel that added information on top of R2
   was periodic-cancellation residual matching — too weak standalone (32.5%), but
   orthogonal enough that a 1.0-weight z-blend bought the final **+1.25 pt** to
   **41.25%** top-1, plus improvements in top-10, P95 and catastrophic rate.

The two most expensive-looking ideas (Siamese, phase voting) were measured and cut in
hours; the two cheapest measured ideas (harvesting union, residual blend) produced the
entire end-to-end gain. Every cut is documented with its kill number.

---

## 4. Final benchmark table

All rows on the 80-pair scene-disjoint leak-free split; delta=0.10 adaptive candidates
unless noted. `n/m` = not measured. top-1 equals ≤5 px accuracy for all argmax rows.

| ID | System | top-1 / ≤5 px | ≤2 px | ≤1 px | top-5 | top-10 | median px | P95 px | >25 px | DRAM top-1 | FinFET top-1 | Source |
|----|--------|--------------|-------|-------|-------|--------|-----------|--------|--------|------------|--------------|--------|
| B0 | Classical v1 locator (leak-free)¹ | ~31% | n/m | n/m | n/m | n/m | n/m | ~659 | ~69% | n/m | n/m | Phase 2A benchmark |
| B1 | HGB, peak-strength features (R0) | 37.5% | 37.5% | 32.5% | 47.5% | 51.25% | 192.7 | 747.3 | 62.5% | 48.7% | 26.8% | `struct_ablation.json`² |
| B2 | Structural descriptor only (no learner) | 40.0% | 40.0% | 35.0% | 52.5% | 56.25% | 128.3 | 727.0 | 60.0% | 51.3% | 29.3% | `struct_eval.json` |
| B3 | HGB, old+struct features (R2) | 40.0% | 40.0% | 33.75% | 50.0% | 57.5% | 180.5 | 743.3 | 60.0% | 48.7% | 31.7% | `struct_ablation.json` |
| B4 | Siamese ensemble — **measured and cut** | 33.75% | 33.75% | 27.5% | 51.25% | 55.0% | 221.9 | 690.1 | 66.25% | 38.5% | 29.3% | `siamese_eval.json` |
| — | Residual standalone A_m50 — **measured and cut**³ | 32.5% | 32.5% | 30.0% | 50.0% | 53.75% | 222.8 | 807.2 | 67.5% | 41.0% | 24.4% | `residual_eval.json` |
| **B5** | **Shipped blend R2 + 1.0·residual** | **41.25%** | **40.0%** | **33.75%** | **53.75%** | **60.0%** | **178.6** | **688.0** | **58.75%** | **48.7%** | **34.1%** | `blend_eval.json` |

¹ B0 measured on the Phase 2A leak-free 400-pair benchmark, before candidate harvesting existed.
² Same feature set as `ranker_results.json` HGB (top-1 37.5%, top-10 51.25%, median 245.7 px, P95 760.4 px); the B1 row uses the 300-scene retrain so B1/B3 are a controlled pair on identical candidates.
³ Residual standalone evaluated on the delta=0.15 pool (oracle 92.5%, median 546 candidates).

With the equivalence-set + nearest-centre tie-rule layer (z-margin 0.25) on B5:
sel ≤5 px 37.5%, sel median 216.0 px, sel P95 669.3 px, sel >25 px 62.5%. The
lattice-phase class-voting variant (D2 ceiling 15–17.5%, end-to-end 7.5–12.5% ≤5 px,
`phase_eval.json`) is excluded from the table as it never reached a full-column
benchmark before being cut.

---

## 5. Remaining failure mode and what comes next

**What still fails.** 58.75% of pairs remain >25 px wrong. These are not near-misses:
the median error (178.6 px) is a lattice-scale jump onto a periodic twin. The evidence
chain says exactly where the loss is:

- The candidate pool contains the truth 90% of the time (recall@5 at delta=0.10), so
  harvesting is not the ceiling.
- The latent audit says the chosen wrong sites are physically distinguishable from the
  true site (median latent NCC 0.859; only 6.7% ≥0.95), so the generator's equivalence
  structure is not the ceiling either.
- Therefore the residual gap — roughly 41% top-1 vs 90% oracle — is **appearance-level
  aliasing**: at ~10x magnification loss plus degradation, our features cannot extract
  the sub-0.02-NCC evidence that separates a true site from its periodic twins, even
  though that evidence exists in the latent scene.
- FinFET remains systematically harder than DRAM (34.1% vs 48.7% top-1), consistent
  with its weaker harvesting recall (87.8% vs 97.4% at delta=0.15) and denser alias
  structure.

**Next steps, in priority order (given more time):**

1. **Better residual estimation at reference scale.** The residual channel is currently
   computed against a coarse periodic model at search scale; estimating and cancelling
   the periodic component at the reference's native resolution before downscaling
   should sharpen exactly the sub-0.02-NCC discriminative signal the diagnosis
   identified. It is the only channel that provably added information (B3 → B5), and it
   is the least developed.
2. **Learned features at scale.** The Siamese cut was a data-scale verdict, not a
   concept verdict: 60 scenes cannot train a metric that beats 77 engineered features.
   With thousands of generated scenes and the correct magnification ratio baked into
   training, a learned embedding is the natural way past the hand-crafted
   representation wall the ablation exposed.
3. **Multi-FOV capture strategy.** Single-FOV localization is information-limited by
   aliasing. If the acquisition side can take two or three reference captures at
   offset positions (or two magnifications), the alias sets of the captures intersect
   at the true site — turning the dominant failure mode into a triangulation problem
   rather than a discrimination problem.

---

## 6. Reproduction and artifact inventory

### Stop-rule compliance

- Private test set untouched throughout the campaign.
- Train/validation scene-disjoint; the 80-pair validation split never changed.
- Distance-to-centre was never used as a ranker feature (the centre rule is applied
  only as the official tie-break layer, post-ranking).
- All numbers reproducible from the committed JSONs + git SHA.

### Reproduction commands

```bash
# Final shipped numbers (blend_eval.json, preds_final*.csv)
python experiments/war45_blend.py

# Structural descriptor: validation benchmark, training table, R0/R1/R2 ablation
python experiments/war45_struct.py val
python experiments/war45_struct.py train
python experiments/war45_struct.py ablate

# Cut branches (kill numbers)
python experiments/war45_residual.py     # residual_eval.json
python experiments/war45_phase.py       # phase_eval.json
python experiments/war45_siamese.py     # siamese_eval.json

# Earlier campaign stages
python experiments/war45_pivot.py        # pivot_recall.json (harvesting pivot)
python experiments/war45_ranker.py       # ranker_results.json
python experiments/war45_diag.py         # diag_evidence.json
python experiments/war45_sweep.py        # recall_sweep.json
python experiments/war45_warp.py         # warp_limit.json
python experiments/war45_latent_audit.py # latent_audit.json
python experiments/candidate_eval.py     # candidate_recall.json

# End-to-end inference (no network; loads driftforge/models/hgb_r2.joblib)
python scripts/inference.py reference.png search.png        # prints "x.xx,y.yy"
python scripts/inference.py reference.png search.png --json # diagnostics
```

### Artifact inventory — `experiments/war45/`

JSON results:

| File | Contents |
|---|---|
| `blend_eval.json` | Final system: R2 alone vs R2+0.5res vs R2+1.0res (shipped) |
| `candidate_recall.json` | C0/C1 proposer recall (C0 recall@5 65.0%) |
| `diag_evidence.json` | Evidence diagnosis: 97.5% local-max rate, rank 3/~9,896, 0.0206 NCC deficit |
| `latent_audit.json` | 30-failure latent-ambiguity audit (median latent NCC 0.859) |
| `phase_eval.json` | Lattice-phase class voting: D1 98.8%, D2 15–17.5%, end-to-end 7.5–12.5% (cut) |
| `pivot_recall.json` | 3-channel adaptive-union harvesting sweep (delta 0.02–0.25) |
| `ranker_results.json` | RF/HGB on peak-strength features + RF feature importances |
| `recall_sweep.json` | NMS radius x budget recall grid; lattice-expansion falsification (72.5%) |
| `residual_eval.json` | 9 residual formulations standalone (best A_m50 32.5%, cut) |
| `siamese_eval.json` | Siamese-only 5.0% / ensemble 33.75% (cut); pruned oracle 70% |
| `struct_ablation.json` | Controlled R0/R1/R2 ablation, 300 train scenes, identical candidates |
| `struct_eval.json` | Descriptor-only ranking benchmark (~3.5 s/pair) |
| `warp_limit.json` | Warp-hypothesis falsification (corr +0.002) |

CSV tables:

| File | Contents |
|---|---|
| `candidate_predictions.csv` | Per-pair proposer predictions (C0/C1 stage) |
| `diag_evidence.csv` | Per-pair evidence diagnosis rows |
| `pivot_recall.csv` | Per-pair harvesting recall detail |
| `preds_R0_old.csv`, `preds_R1_struct.csv`, `preds_R2_old_struct.csv` | Per-pair ablation predictions |
| `preds_final.csv` | Final blend per-pair predictions (tie-rule selected + top-1 error) |
| `preds_final_argmax.csv` | Final blend argmax-only per-pair errors |
| `recall_sweep.csv` | Recall-sweep grid detail |
| `val_struct.csv`, `train_struct.csv`, `train_struct_60.csv` | Full candidate feature tables (val / 300-scene train / 60-scene train) |

Supporting artifacts: `struct_run.log`, `struct_run2.log`, `siamese_run.log`,
`siamese_run2.log`, `blend_run2.log` (run logs); `siamese_train.npz`,
`siamese_val.npz` (patch tensors); `figures/` (9 PNGs: candidate-recall ablation,
ranker top-k recall, final error CDF, catastrophic-rate ablation, structural-descriptor
visual, DRAM-vs-FinFET, recall comparison, lattice falsification, true-site rank).
Shipped models: `driftforge/models/hgb_r2.joblib` (ranker), `driftforge/models/siamese.pt`
(retained for the record; not used at inference).
