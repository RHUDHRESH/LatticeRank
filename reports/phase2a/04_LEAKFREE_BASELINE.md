# 04 — E1: the leak-free baseline

**System:** `driftforge.baseline.locate`, **unchanged** from v1.0.0 (SHA-256 pinned
by `tests/test_phase2a_leakfree.py::LocatorFrozenTests`).
**Data:** `manifests/validation_phase2a_400.jsonl` — 400 pairs, a new split, disjoint
from every existing manifest (`03_SPLIT_AND_MANIFEST_AUDIT.md`).
**Generator:** Phase-2A de-correlated signature (E0).

Raw data: [`leakfree_predictions.csv`](leakfree_predictions.csv),
[`leakfree_baseline.json`](leakfree_baseline.json).

---

## 1. Headline

> **The published v1 figure was 80% accuracy at 5 px.
> The leak-free figure is 31.0%.**

| Metric | Value | 95% CI |
|---|---:|---|
| Pairs | 400 | |
| Accuracy ≤ 1 px | 28.2% (113/400) | 24.1% – 32.9% |
| Accuracy ≤ 2 px | 31.0% (124/400) | 26.7% – 35.7% |
| **Accuracy ≤ 5 px** | **31.0%** (124/400) | **26.7% – 35.7%** |
| Accuracy ≤ 10 px | 31.0% (124/400) | 26.7% – 35.7% |
| Median error | 248.01 px | 216.6 – 281.1 |
| Mean error | 248.5 px | |
| P90 | 551.4 px | |
| P95 | 659.0 px | |
| P99 | 833.5 px | |
| Max | 914.0 px | |
| Catastrophic (>25 px) | **276 (69.0%)** | |
| Accepted coverage | 24.0% | 20.1% – 28.4% |
| Accepted-incorrect @5 px | **26** | |
| Accepted-incorrect >25 px | **26** | |
| Mean runtime | 12,440 ms/pair | contended, see §5 |
| Median runtime | 10,644 ms/pair | contended |
| P95 runtime | 21,258 ms/pair | contended |

This is the ugly number the brief asked for, published without qualification.

## 2. The error distribution is bimodal, and that is the whole story

Accuracy is **identical at 2 px, 5 px and 10 px** (all 31.0%). There is no
population of "nearly right" predictions. A pair is either localised to within
~2 px or it is hundreds of pixels away — median error among failures is in the
low hundreds. Widening the tolerance would buy nothing.

The consequence for Phase 2: **sub-pixel refinement is worthless until cell
selection is fixed.** Every accuracy point available lives in the catastrophic
tail, not in precision.

## 3. Architecture breakdown — FinFET is significantly harder

| Architecture | n | Acc ≤5 px | 95% CI | Median | P95 | Catastrophic |
|---|---:|---:|---|---:|---:|---:|
| **DRAM** | 200 | **38.0%** | 31.6% – 44.9% | 216.5 px | 633.7 px | 62.0% |
| **FinFET** | 200 | **24.0%** | 18.6% – 30.4% | 278.3 px | 668.6 px | 76.0% |

The intervals **do not overlap** (DRAM lower bound 31.6% vs FinFET upper bound
30.4%), so this is a real difference, not sampling noise.

Mechanism: FinFET presets are finer. `finfet_dense` fin pitch reaches 28 nm =
**2.8 Search px**, below the 4.0 px minimum of any sponsor FinFET preset. At
that pitch the lattice is barely resolved and near-identical repeats are packed
denser, so more of them fall inside any given search neighbourhood. Both
families are in the hidden test, so this gap must be closed, not averaged away.

## 4. Profile breakdown — boundary context is the strongest genuine cue

| Profile | n | Acc ≤5 px | Median | Catastrophic |
|---|---:|---:|---:|---:|
| `boundary` | 60 | **45.0%** | 111.5 px | 55.0% |
| `standard` | 200 | 34.5% | 216.8 px | 65.5% |
| `hard` | 100 | 28.0% | 274.9 px | 72.0% |
| `ambiguous` | 40 | **0.0%** | 412.4 px | **100.0%** |

Two results matter:

1. **`boundary` is the *best* profile (45.0%)**, despite being designed to be
   harder. Crops that straddle a mat/periphery seam carry unique large-scale
   context, and that context is what survives when the planted landmark is
   removed. This corroborates the Phase-1 observation that 4 of the 6 pairs
   surviving signature removal were `boundary`. It is direct evidence for
   weighting mat/boundary context in v2 — and for the sponsor's own
   `boundary_bias = 0.35` default being the right instinct.

2. **`ambiguous` is 0.0% accurate with a 100% catastrophic rate.** This profile
   is an exact wallpaper with `defect_density = 0`, so nothing distinguishes
   the repeats. The locator cannot be blamed: several positions are *equally
   correct matches*, and only the official nearest-Search-centre rule selects
   among them. The current implementation applies that rule to an NMS-decimated
   candidate list (Phase-1 H1), so it fails all 40. **These 40 pairs are a
   tie-rule bug, not a matching bug**, and they alone account for 14.5% of all
   catastrophic failures.

## 5. Runtime — measured under contention, not comparable to Phase 1

Mean 12,440 ms/pair looks 3× worse than the Phase-1 figure of 3,687 ms. It is
not a regression: E1 ran **6 worker processes on 8 cores**, and `runtime_ms` is
wall time inside each worker, so it absorbs CPU contention. Total wall clock was
1,113 s for 400 pairs — about 2.8 s/pair of real throughput.

Treat these numbers as an **upper bound on latency**, not as single-process
latency. The Phase-1 sequential profile (3.69 s/pair, 79.6% in 50 NCC calls)
remains the reference for the runtime budget. No conclusion in this phase
depends on the runtime column.

## 6. Confidence gating collapsed, and it took the safety property with it

| | v1 (leaked, 20 pairs) | Phase 2A (leak-free, 400 pairs) |
|---|---:|---:|
| Coverage | 75% | **24.0%** |
| Accepted-incorrect @5 px | **0** | **26** |

The abstention gate was calibrated on leaked data. On honest data it both
rejects three-quarters of all pairs *and* still accepts 26 wrong answers, 26 of
which are catastrophic. The v1 "zero accepted-incorrect" property **does not
survive** and must not be quoted.

The thresholds (`score ≥ 0.20`, `psr ≥ 3.0`, `peak_gap ≥ 0.020`) are absolute
constants fitted to a contaminated distribution. Phase 2D must recalibrate them
against a calibrated probability, per the equivalence-set design.

## 7. What this does and does not say

- It **does** say v1's classical locator, absent a generator leak, solves under
  a third of pairs, and that the published 80% was substantially an artifact.
- It **does not** say the problem is unsolvable at 31%. 100% of `ambiguous`
  failures are a tie-rule defect; `boundary` already reaches 45% on genuine
  context; and 29.3% of catastrophic failures land on sites that do *not* look
  like the target (`05_FAILURE_CLASSIFICATION.md`) and are therefore
  addressable by better evidence.
- It **does not** measure `ood`: that profile is absent from the validation mix
  (`DEFERRED_FINDINGS.md` D8).

## 8. Reproduce

```bash
python reports/phase2a/diagnostics/run_leakfree_baseline.py \
  --manifest manifests/validation_phase2a_400.jsonl --limit 400 --workers 6

python -m experiments.benchmark --system driftforge_v1_leakfree \
  --from-predictions reports/phase2a/leakfree_predictions.csv \
  --merge-classes reports/phase2a/catastrophic_failures.csv \
  --manifest manifests/validation_phase2a_400.jsonl \
  --output experiments/results/classical_v1 --galleries
```
