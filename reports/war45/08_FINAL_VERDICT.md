# 08 — War-45 final verdict

**STOPPED AT GATE §7 (candidate recall).** Recall @5 px = 65.0% shipped, 81.2%
best measured, against a 98% gate and a 95% STOP threshold. No ML was trained.

Elapsed compute: ~22 min of the ~45 min budget. **The remainder was deliberately
not spent**, because every downstream stage (§8 meta-ranker, §9 Siamese, §11
ablation, §13 L9 robustness) is capped by a number that failed its gate, and
measuring capped systems would have produced results that look like progress
and are not.

---

## Gate results

| # | Gate | Result |
|---|---|---|
| §2 | DRAM clean credible | ✅ **PASS** — orthogonal word/bit-lines, checkerboard contacts, 49×78 nm, Manhattan, mat boundary visible |
| §2 | FinFET clean credible | ✅ **PASS** — parallel fins at 28.5 nm, gate crossings at 61.4 nm, periphery strip |
| §3 | Target leak removed | ✅ **PASS** (Phase 2A) — interventional test, 0.0 nm signature movement across 800 relocations |
| §4 | Leak-free benchmark | ✅ **PASS** (Phase 2A) — 400 pairs, 31.0% ≤5 px |
| §5 | Failure taxonomy | ✅ **PASS** (Phase 2A) — 70.7% visually-equivalent repeats, parity z = +7.10 |
| §6 | Lattice estimator | ✅ **PASS** — `driftforge/lattice.py`, 136 ms/image, exact on DRAM, returns the physically correct 2× on checkerboard FinFET |
| **§7** | **Candidate recall ≥98%** | ❌ **FAIL — 65.0% / 81.2%** |
| §8–13 | ML, ablation, L9 | ⛔ **NOT RUN** — gated behind §7 |

## The finding that matters

The frozen v2 architecture assumed candidate *generation* was the bottleneck:
NMS destroys periodic hypotheses, so restore them with lattice expansion. The
footprint arithmetic behind that was right, but the therapy is wrong.

**Measured:** the true site is a local maximum of the correlation surface in
**97.5%** of pairs, sitting **0.49 px** from a peak, ranking **median 3** among
~9,900 local maxima — and yet only **38.8%** of the time is it the global
maximum, with the winner beating it by **0.02 NCC**.

Meanwhile shrinking NMS from 9 px to 3 px changes recall by **zero**, and
lattice expansion at 3× the candidate budget *loses* 8.7 points.

So the answer is already in the candidate set and already nearly top-ranked.
What is missing is evidence that separates it from its periodic twins. **The
bottleneck is discrimination, not generation.** That reverses the build order:
independent evidence channels (directionality, boundary context, learned
similarity) come before candidate bookkeeping, and the lattice estimator
becomes a *feature* rather than a *generator*.

A secondary hypothesis was also killed: the misses are not warp-limited
near-misses (median miss 79 px, corr with warp budget +0.002), so widening the
tolerance or modelling the non-affine warps would not have helped either.

## What was built and kept

- `driftforge/lattice.py` — production lattice estimator (spectral + autocorrelation
  consensus, harmonic-summed scoring, orientation, calibrated confidence,
  parity-aware translations). Validated; useful as a feature source.
- `experiments/candidate_eval.py`, `war45_diag.py`, `war45_sweep.py`, `war45_warp.py`
  — the recall harness. 90 s for 80 pairs; re-runnable after every change.
- `reports/war45/generator_gallery/` — the four canonical panels, each showing
  the latent master alongside the degraded captures.
- 3 figures: recall-vs-budget with the evidence ceiling, true-site rank
  breakdown, and the warp falsification.

## Submission readiness: **NO-GO** (for a claimed system)

The honest position today: leak-free localization is **31.0% ≤5 px**, and the
path to improving it is now measured rather than assumed. That is a real asset
for the write-up — the deck can show *why* naive matching fails on periodic
layouts, with direct evidence rather than speculation — but it is not a
competitive result yet.

The next 45-minute block should go to item 1 in `04_CANDIDATE_RECOVERY.md` §6:
add the directionality channel and re-measure the rank distribution. That is
the cheapest test of the newly-identified bottleneck.
