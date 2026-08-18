# 07 — Phase-2A verdict

**Overall: PASS on the leak work, FAIL on the alias-dominance gate.
Phase 2B is STOP pending human review.**

---

## 1. Gate-by-gate

| # | Gate | Result | Evidence |
|---|---|---|---|
| 1 | Leak removed | ✅ **PASS** | signature placement seeded only by scene identity; interventional test moves the target to 4 positions × 200 scenes and the signature never moves (max shift **0.0 nm**) |
| 2 | No replacement leakage | ✅ **PASS** | exhaustive target-dependency audit over `driftforge/`; 12 call sites classified, 1 confirmed leak (fixed), 0 new |
| 3 | Deterministic rendering preserved | ✅ **PASS** | repeat-render equality tests; E0 preserves v1 draw order and count |
| 4 | ≥200 leak-free validation pairs | ✅ **PASS** | **400** pairs on a new disjoint split |
| 5 | Split integrity clean | ✅ **PASS** | 0 scene-ID overlap, 0 seed overlap, 0 scene-spec (`seed×17+3`) collisions across all 5 splits |
| 6 | Baseline locator unchanged | ✅ **PASS** | `baseline.py` SHA-256 pinned in the test suite |
| 7 | **Periodic aliases ≥80% of catastrophic** | ❌ **FAIL** | **70.7%** (CI 65.0–75.7); three of four measurement methods fail, the fourth passes only on a weak criterion |
| 8 | Benchmark harness works | ✅ **PASS** | `experiments/` runs end to end; registry refuses unimplemented systems |
| 9 | Visual galleries generated | ✅ **PASS** | 9 traceable slots incl. DRAM/FinFET successes and each failure class |
| 10 | Taguchi design infrastructure validated | ✅ **PASS** | L27 orthogonality verified numerically; 2,430-evaluation batch built, seeds disjoint |
| 11 | Rendering fidelity checked (E1b) | ✅ **PASS** | two 50-pair runs disagree on the *sign* of the ss2-vs-ss4 delta (+4.0 / −4.0 pp): noise. `ss=2` retained; `ss=4` costs 3.9× memory |
| 12 | Locked private test untouched | ✅ **PASS** | `manifests/test.jsonl` never read, never regenerated, byte-identical |

**Eleven of twelve pass. Gate 7 fails, and the brief makes that a stop condition.**

## 2. Why the gate failed, in one paragraph

Not because aliasing turned out to be unimportant — it is still the largest
single failure class, and the parity test (71.4% contact-preserving, z = +7.10)
proves lattice structure pervades essentially every catastrophic failure. The
gate failed because **~29% of catastrophic failures land on sites that do not
look like the target at all**. Those are context mismatches: the fine lattice
aligns, the surrounding mat/boundary/defect content does not. The frozen
architecture treated catastrophic failure as one phenomenon; it is two, in
roughly a 71/29 split, and they need different fixes.

A secondary finding is that the Phase-1 measurement was over-confident. The
lattice-residual instrument that produced "4/4 aliases" cannot separate real
aliases from chance here, because the generator's non-affine warps (6.52 px
median) are *larger* than half a lattice cell (5.25 px). Observed residuals sit
only 0.44–0.52 σ below a permutation null. That was n=4 on a weak instrument;
this is n=276 on a direct one.

## 3. Recommended amendments to the architecture freeze

The freeze should be **amended, not rewritten**. Every frozen component still
attacks one of the two populations. What changes:

1. **Add an explicit non-alias workstream (~29% of catastrophic failures).**
   Target the `boundary_context` and `matcher_error` classes with mid-band
   context, the directionality channel and ensemble ranking. This work was
   implicitly bundled into "better evidence"; it now needs its own measurable
   target, because ~81 of 276 failures will not be touched by lattice
   candidates.

2. **Promote the tie rule to a first-class deliverable.** The 40 `ambiguous`
   pairs are 100% catastrophic, 0% accurate, and 14.5% of all catastrophic
   failures. They are an exact wallpaper: several positions are equally correct
   and only the official nearest-centre rule chooses. This is a **tie-rule
   defect, not a matching defect** — the cheapest identified win in the phase,
   and it does not need a model.

3. **Change the Phase-2B success criterion.** "True-cell recall ≥99% in the
   candidate set" is necessary but no longer sufficient: for the 29%, the true
   cell may already be proposed and simply out-ranked. Add a ranking-quality
   criterion alongside recall.

4. **Close the FinFET gap explicitly.** DRAM 38.0% vs FinFET 24.0%, with
   non-overlapping intervals. Both families are in the hidden test.

5. **Treat prediction *stability* as a first-class metric.** The supersample
   check found 23 of 50 pairs move by >1 px and 4 flip catastrophically when
   only the render oversampling changes. Cell choice on hard pairs is decided
   by noise-level input differences, not evidence. The sponsor's
   `generate_sample_family()` design (one scene, several acquisitions, one
   ground truth) measures exactly this and should be adopted.

6. **Recalibrate confidence, and stop quoting the v1 safety property.**
   Coverage collapsed 75% → 24.0% and accepted-incorrect went 0 → 26. The
   absolute thresholds were fitted to contaminated data.

7. **Meta-ranker is now a multi-model ensemble** (`DEFERRED_FINDINGS.md` D10,
   `experiments/configs/ensemble_plan.yaml`) rather than a single
   HistGradientBoosting, with a leave-one-out ablation so each member must earn
   its place.

## 4. What is NOT recommended

- **Do not discard the lattice work.** 70.7% is still the dominant class and
  the parity evidence is overwhelming.
- **Do not weaken the 0.80 threshold to make it pass.** It was set in advance;
  the honest response is to report the failure and amend the plan.
- **Do not start Phase 2C (Siamese training).** It was already gated behind
  Phase-2B candidate recall, and that gate has not been attempted.
- **Do not touch the locked test split.**

## 5. Decision requested

Phase 2B is **STOP** until a human confirms one of:

- **(a) Amend and proceed** — adopt §3 and continue to Phase 2B with the added
  non-alias workstream and revised success criteria. *This is the recommended
  option;* the evidence supports amendment rather than redesign.
- **(b) Reopen fully** — treat 70.7% as insufficient justification for a
  lattice-centred architecture and re-run the Phase-1 decision matrix against
  the two-population finding.
- **(c) Re-measure first** — extend to the 500-pair `stress` split (which also
  covers `ood`, absent from validation) before deciding.

## 6. One-line summary

The generator no longer marks its own answer; the honest baseline is **31.0%**
rather than 80%; catastrophic failure is **two phenomena in a 71/29 split**, not
one; and the architecture needs an amendment, not a rewrite.
