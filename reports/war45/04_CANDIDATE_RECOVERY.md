# 04 — Candidate recovery: **STOP GATE FAILED**, and the frozen architecture is falsified

**Gate:** true-cell candidate recall @5 px ≥ 98% (stretch 99%); **STOP ML if <95%**.
**Result: 65.0% – 81.2% depending on budget. STOP. No ML was trained.**

Data: `experiments/war45/{candidate_recall.json, diag_evidence.json, recall_sweep.json, warp_limit.json}`
Figures: `experiments/war45/figures/{candidate_recall_comparison, true_site_rank, lattice_falsification}.png`
n = 80 leak-free pairs, `validation_phase2a_400[200:280]`, unseen by any prior decision.

---

## 1. Headline

| Strategy | recall@1 px | recall@2 px | recall@5 px | median candidates |
|---|---:|---:|---:|---:|
| **C0** NMS top-25 (shipped behaviour) | 62.5% | 65.0% | **65.0%** [54–75] | 25 |
| **C1** lattice expansion, 6 seeds | 56.2% | 58.8% | **58.8%** [48–69] | 2,134 |
| **C2** lattice expansion, all 25 seeds | — | — | **72.5%** | 6,573 |
| **C3** top-2000 local maxima by score | — | — | **81.2%** | 2,000 |
| *(evidence ceiling — see §2)* | | | *97.5%* | ~9,900 |

DRAM 74.4% vs FinFET 56.1% at C0 — the FinFET gap seen in Phase 2A persists and
is larger here.

**Lattice expansion loses to plain score ranking.** C3 keeps 2,000 candidates and
reaches 81.2%; C2 keeps 6,573 and reaches 72.5%. Spending 3× the candidate
budget on lattice structure bought −8.7 points.

## 2. The evidence is present — this is a ranking failure, not a detection failure

The decisive diagnostic. For each pair we asked whether the true site is a local
maximum of the correlation surface at all, and if so, where it ranks:

| Question | Answer |
|---|---:|
| True site is a local maximum (5 px footprint, within 5 px) | **97.5%** |
| Median distance truth → nearest local maximum | **0.49 px** |
| True site is the **global** maximum | **38.8%** |
| True site ranks ≤10 | 62.5% |
| True site ranks ≤50 | 70.0% |
| Median rank among ~9,900 local maxima | **3** |
| 90th-percentile rank | **5,380** |
| Median score deficit (global max − true site) | **0.0206 NCC** |

Read together these say something specific and useful:

- **The correlation surface does peak on the right answer, to within half a
  pixel, 97.5% of the time.** The upstream evidence is not the problem, and
  the 97.5% figure is the true ceiling for any proposer built on this surface.
- **But the score cannot tell that peak from its periodic twins.** The winning
  peak beats the true peak by 0.02 NCC — noise-level. The rank distribution is
  violently bimodal: 38.8% at rank 1, yet p90 at rank 5,380.

That bimodality is the periodic-alias phenomenon measured directly, at the
level of the ranking rather than the final prediction.

## 3. Why lattice expansion failed — and the hypothesis it falsified

Phase 2A measured the generator's non-affine warp budget (raster shear + row
jitter + radial distortion, none of them in the affine lattice basis) at
**6.52 px median**, larger than the 5 px tolerance. The natural hypothesis was
that lattice translation lands *near* the truth but just outside tolerance.

**Falsified.** Testing the discriminating prediction — misses should cluster in
a 5–25 px near-miss band:

| Prediction if warp-limited | Measured |
|---|---|
| Misses cluster just outside 5 px | Miss distance median **79.45 px**, p25 **45.97 px** |
| Most misses in the 5–25 px band | Only **3 of 22 (14%)** |
| Distance correlates with warp budget | **corr = +0.002** |

The misses are not near-misses. When lattice expansion fails, the true site is
~80 px away — roughly 10–25 lattice periods, well outside the ±5-period window
searched, meaning **none of the 25 seed peaks is anywhere near the correct
region**. Widening the window would help only by brute force, and at ±25
periods the candidate count explodes past 100k.

So the lattice machinery is not broken — it is being asked to repair the wrong
stage. It can only translate you around a seed you already have; it cannot
conjure a seed in a region the correlation ranking never surfaced.

## 4. What this means for the frozen architecture

The Phase-1 freeze rested on a specific causal claim (H1): *NMS with a ~30 px
footprint destroys legitimate periodic hypotheses; restoring them via
lattice-aware expansion recovers the true cell.* The footprint arithmetic was
correct — 4.4 to 67 lattice sites do collapse into one NMS cell. **The
therapeutic claim is now measured and is wrong.**

- Shrinking NMS to 3 px changes recall by **0.0 points** (67.5% at top-25 for
  footprints 3, 5 and 9 alike).
- Lattice expansion at 3× the budget *reduces* recall.
- Only raw budget helps, and it saturates at 81.2% — still 16 points below the
  97.5% ceiling.

`MEASURED RESULT` — **candidate generation is not the bottleneck. Candidate
*discrimination* is.** The true site is available and cheap to find; what is
missing is evidence that separates it from ~9,900 near-identical rivals, when
the current fused ZNCC separates them by 0.02.

This reorders the v2 architecture. Components that add **independent evidence**
(directionality / polarity-invariant orientation, boundary-context scoring,
learned patch similarity) move ahead of components that add **candidate
bookkeeping** (lattice expansion, NMS tuning). The lattice estimator remains
useful — as a *feature* (is this candidate a lattice translate of the top peak?)
rather than as a *generator*.

## 5. Gate decision

```
recall@5 px = 65.0% (shipped)  ...  81.2% (best measured strategy)
gate        = 98%,  STOP threshold = 95%
```

**STOP. No Siamese, no meta-ranker, no ensemble was trained**, exactly as the
war-45 script directs: a reranker cannot recover a site the proposer never
emits, and at 81.2% recall every downstream model is capped 14+ points below
the ≥95% accuracy target regardless of how good it is.

Training ML here would have produced a model whose ceiling was set by a
solvable upstream defect — the single most expensive mistake available in this
project.

## 6. Recommended next actions, in priority order

1. **Rank on better evidence, not more candidates.** Add the polarity-invariant
   directionality channel and a boundary/mat-context score and re-measure §2's
   rank distribution. The target metric is "true site rank ≤50", currently 70%.
2. **Fix the transform search.** 38.8% global-max rate with a 0.02 NCC deficit
   suggests the fused score is close but mis-ordered; a correct per-pair
   scale/rotation estimate (rather than a 3×3 or 5×5 bank) may sharpen the true
   peak more than any candidate strategy.
3. **Re-run this exact measurement after each change.** It is 90 s for 80 pairs
   and it is the number that governs everything downstream.
4. **Only then** revisit ML. The candidate dataset export is pointless until
   recall clears 95%.
