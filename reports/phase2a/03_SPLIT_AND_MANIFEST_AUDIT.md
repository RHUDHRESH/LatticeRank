# 03 — Split and manifest audit

**Question.** Is the Phase-2A benchmark measured on data that is genuinely
separate from everything else, and is the locked private test still untouched?

Machine-readable evidence: [`split_audit.json`](split_audit.json).

---

## 1. What was created

`manifests/validation_phase2a_400.jsonl` — **400 pairs**, a new split, not a
re-slice of the existing validation manifest.

An earlier iteration of this phase measured the baseline on
`validation.jsonl[:300]`. That is a *slice of an existing split*, not an
extension, and it reuses scenes that the v1 release gate and the Phase-1
autopsy already touched. It was replaced with a genuinely disjoint split so
that no Phase-2A number is measured on a scene any earlier decision saw.

### Composition

| Property | Value |
|---|---|
| Pairs | 400 |
| DRAM / FinFET | **200 / 200** (exact) |
| Profiles | standard 200, hard 100, boundary 60, ambiguous 40 |
| Unique scenes | 400 (one scene per pair) |
| Seed range | 900,000 – 900,399 |
| Profile mix | mirrors `validation`, so results are comparable to the v1 gate |

The mix deliberately matches `validation` rather than being re-balanced: the
purpose is to measure the same distribution the v1 gate claimed 80% on, at a
sample size that can actually support a claim.

## 2. Additive changes to the split machinery

Two additive edits, no existing value altered:

- `driftforge/config.py` — added a `validation_phase2a` entry to `PROFILE_MIX`.
- `driftforge/splits.py` — added `"validation_phase2a": 900_000` to `SPLIT_SEED_BASE`.

No existing seed base, profile mix, or manifest file was modified. The four
original manifests are byte-identical to the v1.0.0 release (verified in §5).

## 3. Disjointness — measured, not asserted

`assert_disjoint()` was run across all five splits and passed. Beyond that,
every pair of splits was checked directly for both scene-ID and seed overlap:

| Split | Seeds | Range |
|---|---:|---|
| train | 3,000 | 100,000 – 102,999 |
| validation | 600 | 300,000 – 300,599 |
| test *(locked)* | 1,000 | 500,000 – 500,999 |
| stress | 500 | 700,000 – 700,499 |
| **validation_phase2a** | **400** | **900,000 – 900,399** |

| Check | Result |
|---|---|
| Max pairwise scene-ID overlap | **0** |
| Max pairwise seed overlap | **0** |
| `all_disjoint` | **true** |
| Scene-spec seed collisions (`seed × 17 + 3`) | **0** |

The last row matters and is easy to miss. Scene parameters are drawn from
`default_rng(seed * 17 + 3)`, so two splits could in principle hold different
manifest seeds that map to the *same scene realisation*. That mapping was
checked across all 5,500 records: **zero collisions**.

## 4. Taguchi scenes

`manifests/taguchi/` (27 runs × 3 repeats = 81 manifests, 2,430 pairs) uses
seed base **1,200,000**, verified disjoint from all five splits above:

```
seed_overlap_with_splits = {train: 0, validation: 0, test: 0,
                            stress: 0, validation_phase2a_400: 0}
```

## 5. The locked private test is untouched

`manifests/test.jsonl` was not read for any measurement, not regenerated, and
not modified. It is byte-identical to the v1.0.0 release, and no Phase-2A
threshold, classifier or architecture decision was fitted on it.

Its 1,000 pairs remain reserved for a single one-shot release gate.

## 6. Verdict

**PASS.** Split integrity is clean. The Phase-2A baseline is measured on 400
scenes that no previous phase, threshold or decision has seen.
