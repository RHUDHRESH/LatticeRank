# Verification results

This file records frozen release-gate results produced from the included
manifests and code. It is updated only from commands that can be reproduced.

The evidence trail is kept deliberately: the superseded v1.0.0 result is
retained below, marked, so the correction is auditable rather than silently
rewritten.

## Generator guarantees checked by tests

- deterministic output for a given manifest record;
- 1000×1000 uint8 Reference/Search pairs;
- independent capture pipelines and noise streams;
- transform-aware floating-point `(x, y)` labels;
- exact scene-ID disjointness across all splits;
- both DRAM and FinFET in every split;
- centre-rule behavior for the explicit ambiguity profile;
- framework-neutral model arrays and heatmap labels;
- **structural-defect placement independent of the target coordinate**
  (Phase 2A; interventional test in `tests/test_phase2a_leakfree.py`).

---

# Leak-free Phase-2A baseline — current

**Authoritative for all performance claims.**

```bash
python reports/phase2a/diagnostics/run_leakfree_baseline.py \n  --manifest manifests/validation_phase2a_400.jsonl \n  --limit 400 --search-supersample 2 --workers 6
```

400 pairs from `validation_phase2a_400`, a **newly created split, disjoint in
both scene ID and seed from train / validation / test / stress** (200 DRAM /
200 FinFET; 200 standard, 100 hard, 60 boundary, 40 ambiguous). Locator
byte-identical to v1.0.0, verified by SHA-256 in the test suite.

| Metric | Result | 95% CI |
|---|---:|---|
| Pairs | 400 | — |
| Accuracy ≤1 px | 28.2% | 24.1 – 32.9 |
| Accuracy ≤2 px | 31.0% | 26.7 – 35.7 |
| **Accuracy ≤5 px** | **31.0%** | **26.7 – 35.7** |
| Accuracy ≤10 px | 31.0% | 26.7 – 35.7 |
| Median error | 248.01 px | 216.6 – 281.1 |
| P90 / P95 / P99 | 551.4 / 659.0 / 833.5 px | — |
| Max error | 914.0 px | — |
| Catastrophic >25 px | 276 (69.0%) | — |
| Selective coverage | 24.0% | 20.1 – 28.4 |
| **Accepted-incorrect @5 px** | **26** | — |

By architecture: DRAM **38.0%** (CI 31.6–44.9), FinFET **24.0%** (CI 18.6–30.4).
The intervals do not overlap: FinFET is significantly harder, its fin pitch
reaching 2.8 Search px.

By profile: boundary **45.0%** (median 111.5 px) — the *best* profile, because
mat/periphery seams carry genuine large-scale context — then standard 34.5%,
hard 28.0%, ambiguous **0.0%** (100% catastrophic; a tie-rule defect, not a
matching one).

Accuracy is identical at 2, 5 and 10 px: the locator is either on the correct
lattice cell or hundreds of pixels away. A discrete selection failure, not a
precision problem.

## Catastrophic failure classification

276 catastrophic failures. The Phase-1 lattice-residual instrument proved
**unreliable at scale** — the non-affine warp budget (4.96 px median) is
comparable to half a lattice cell (5.59 px), so residual size barely separates
real aliases from chance (effect size 0.4 σ vs a permutation null). It was
replaced with a direct test: does the predicted location *look like* the true
location, calibrated per image against random locations?

| Measurement | Alias fraction | vs 0.80 gate |
|---|---:|---|
| **Visual equivalence (primary)** | **70.7%** (CI 65.0–75.7) | **FAIL** |
| Above per-image null p95 (loose) | 83.7% | passes on a weak bar |
| Legacy lattice arithmetic | 73.9% | FAIL (unreliable instrument) |
| Mechanism-first bucketing | 52.2% | FAIL |

Independent corroboration that does not depend on any threshold: **71.4% of all
catastrophic failures preserve contact-checkerboard parity (z = +7.10 vs the
50% null)**. Lattice structure is real and pervasive — including among the
failures that are *not* visually equivalent (71.8% parity-even), which is
direct evidence that ~29% of catastrophic failures are context mismatches that
better evidence can fix.

**This is below the 80% alias dominance the Phase-2 architecture was frozen
against**, so Phase 2B is gated pending review — see
[reports/phase2a/05_FAILURE_CLASSIFICATION.md](reports/phase2a/05_FAILURE_CLASSIFICATION.md)
and [reports/phase2a/07_PHASE2A_VERDICT.md](reports/phase2a/07_PHASE2A_VERDICT.md).

Full report: [reports/phase2a/04_LEAKFREE_BASELINE.md](reports/phase2a/04_LEAKFREE_BASELINE.md)

---

# Historical v1.0.0 locator release gate

> ## SUPERSEDED FOR PERFORMANCE CLAIMS — generator leakage discovered
>
> The numbers below were produced by a generator later proven to contain a
> target-correlated landmark. `driftforge/scene.py` placed a unique
> dark-core/bright-rim defect at `target_c ± 260 nm` with radius ≤135 nm.
> Because 260 + 135 < 500 nm (half the Reference FOV), that defect lay inside
> the Reference in **100%** of non-ambiguous samples, was larger than every
> random defect, and sat in the mid-frequency band carrying 75% of the
> locator's fused score. The generator marked its own answer.
>
> A Phase-1 ablation removing only that block dropped accuracy from 80% to 30%
> and raised catastrophic failures from 4 to 14 on the same 20 pairs. Phase 2A
> implemented the fix for real and reproduced those numbers, including an
> identical P95 (752.2 px) and max (895.9 px).
>
> **Do not cite the table below as evidence of localisation performance.** It
> is retained as the evidence trail for the correction.

Command-equivalent to:

```bash
python scripts/audit_manifest.py manifests/validation.jsonl \
  --limit 20 \
  --search-supersample 2 \
  --tolerance 5 \
  --output reports/validation_release_gate.json
```

| Metric | Result |
|---|---:|
| Pairs | 20 |
| Median error | 0.600 px |
| Accuracy ≤1 px | 70% |
| Accuracy ≤2 px | 80% |
| Accuracy ≤5 px | 80% |
| Selective coverage | 75% |
| Accepted-incorrect at 5 px | 0 |
| Mean locator runtime | 4.97 s/pair |

Five pairs were not auto-accepted: two exact-wallpaper ambiguity cases, two
remote periodic jumps and one correct but low-gap match.

The "zero accepted-incorrect" figure did not survive scale. On 400 leak-free
pairs the same gate admits **26** incorrect predictions, all of them >25 px
errors. Selective prediction is not currently a safety net.

The packaged artifact `reports/validation_release_gate.json` is retained
unmodified. It carries a key (`accepted_incorrect_at_5px`) that the shipped
`scripts/audit_manifest.py` has never written — the R8 reproducibility defect.
That script now emits a versioned, self-describing schema (`schema_version: 2`)
so a future mismatch is detectable; the historical artifact was deliberately
**not** regenerated, because overwriting it with leak-free numbers under the
same filename would destroy the audit trail.

---

## What changed between the two results

Only the generator, and only one block of it: structural-defect placement moved
from target-relative to world-uniform coordinates. The locator
(`driftforge/baseline.py`) is byte-identical, pinned by a checksum test. Fusion
weights, `tie_delta`, NMS, scale and rotation candidates, PSR and acceptance
thresholds are unchanged. Runtime differences between the two tables reflect
different machines and process counts, not algorithmic change.
