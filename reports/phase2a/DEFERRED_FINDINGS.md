# Deferred findings — observed in Phase 2A, deliberately not fixed

Phase 2A's scope is E0 (remove the target-correlated leak) and E1 (measure the
true baseline). Everything below was seen while doing that and left alone,
because fixing it would either change the measured baseline or belongs to a
later experiment. Nothing here blocked E0 or E1.

---

## D1 — `test_classical_baseline_smoke` encoded leaked behaviour *(changed, recorded here for transparency)*

**Observed.** `tests/test_generator.py::test_classical_baseline_smoke` asserted
`error < 2.0 px` for seed 300000. With the leak removed the same seed lands
**425 px** away, so the test failed.

**Why this is not a Phase-2A regression.** The assertion was only ever
satisfiable because the generator planted a landmark next to the answer. It
was measuring the leak, not the locator.

**Action taken.** The test now asserts what a smoke test legitimately can — the
locator runs end to end and emits a finite, in-bounds coordinate with a
non-empty candidate set. The accuracy claim it used to make is now measured
properly on 300 leak-free pairs in `04_LEAKFREE_BASELINE.md`. The docstring
records the old threshold and why it went. This is the one behavioural test
change in Phase 2A and it is disclosed rather than silently relaxed.

## D2 — R13: quasi-periodic roughness may be manufacturing its own aliases

**Observed.** `scene.py:_line_distance` derives line-edge roughness from
`_hash_wave(idx, ...)` — a deterministic *sinusoid in line index*, not a random
draw per line. Two lattice sites separated by an integer number of periods can
therefore receive correlated roughness, which is exactly the condition that
lets an exact alias survive.

**Why deferred.** This is Phase-2 experiment **E4**, and the brief explicitly
classes it as a diagnostic rather than a fix. More importantly, changing it
would alter the alias statistics that E1 exists to measure — the two changes
must not land in the same commit or neither is interpretable.

**Risk if left.** Part of DriftForge's measured ambiguity may be self-inflicted
rather than physical. This does **not** invalidate the architecture: the
organiser states the ambiguity is real in *their* data regardless of ours.

## D3 — R10: train/inference preprocessing mismatch

**Observed.** `ml.py:24` decimates the reference with `zoom(0.1,
prefilter=False)` and **no anti-alias filter**; `baseline.py:55` applies
`gaussian_filter(σ=4.0)` first. A model trained through `ml.py` would see a
different reference representation than the classical path.

**Why deferred.** Phase-2A must not touch the locator, and no training happens
in this phase. Belongs to **E9**, and must be unified into one shared function
*before* any training starts.

## D4 — R11: submission output format

**Observed.** `scripts/inference.py:31` prints `(x, y)`; the sponsor convention
is a bare `x,y`. `--strict` exits with code 2 on abstention, which would be
read as a crash by a harness that expects a coordinate for every pair.

**Why deferred.** Release packaging is **E10**, not Phase 2A, and changing
inference output has no bearing on the leak-free baseline.

**Risk if left.** Critical at submission time. It should be fixed well before
release, not discovered there.

## D5 — R14: noise ceiling below the sponsor's hardest level

**Observed.** DriftForge's Search dose floor is 70 against the sponsor's
`severe` 25; hot-pixel rate 7e-4 against ~1.2e-2 (≈17×); multiplicative speckle
is absent entirely.

**Why deferred.** Belongs to **E8**. Adding noise now would change the baseline
being measured.

**Risk if left.** The leak-free baseline in `04` is measured on data that is
*easier* than the hidden test in at least this respect, so it should be read as
an optimistic bound on noise robustness.

## D6 — The packaged v1 release-gate artifact is retained with a known schema mismatch

**Observed.** R8. `reports/validation_release_gate.json` carries
`accepted_incorrect_at_5px`; the shipped `scripts/audit_manifest.py` has never
written that key — it writes `accepted_incorrect_at_tolerance` plus
`tolerance_px`. The packaged artifact could not be reproduced from the
packaged code.

**Action taken (partial).** `audit_manifest.py` now emits a versioned,
self-describing report (`schema_version: 2`, plus the exact command), so a
future mismatch is detectable rather than silent.

**Why the old artifact was left in place.** It is the evidence for the
superseded v1 claim. Regenerating it under Phase-2A code would overwrite a
historical record with leak-free numbers under a filename that `RESULTS.md`
describes as the v1 gate — destroying the audit trail the brief requires be
preserved. It stays as-is, and `RESULTS.md` now marks it superseded.

## D7 — Runtime figures are contention-inflated

**Observed.** E1 ran with 6 worker processes on 8 cores. `runtime_ms` is
measured inside each worker around `locate()` only, but includes CPU
contention.

**Why deferred.** Not a code issue. Recorded so the runtime column is not read
as single-process latency. `04_LEAKFREE_BASELINE.md` reports a clean
sequential measurement alongside it, and neither is comparable across machines.

## D8 — `ood` profile is absent from the validation split

**Observed.** `PROFILE_MIX["validation"]` contains no `ood`, so the E1 slice
covers `standard`/`hard`/`boundary`/`ambiguous` only. `ood` lives in `test`
(10%) and `stress` (30%).

**Why deferred.** Changing the profile mix means regenerating a manifest, which
the brief forbids. The `stress` split is the correct place to measure `ood` and
is available to a later phase without touching the locked test set.

**Risk if left.** No leak-free `ood` number exists yet. Any claim about
out-of-distribution behaviour is currently unsupported.

## D9 — The Taguchi campaign needs a generator override hook that does not exist

**Observed.** `experiments/taguchi.py::condition_overrides()` produces the
concrete `AcquisitionSpec` / `SceneSpec` field values each L27 condition
requires, but `generate_sample()` samples acquisition internally from
`sample_acquisition(rng, profile, reference)` and accepts no spec overrides.
There is currently no supported way to *render* an L27 condition.

**Why deferred.** Phase 2A must not change the generator beyond the E0 leak
fix. Adding an override path now would alter the code that produced the E1
baseline in the same commit that reports it, and neither number would be
interpretable afterwards.

**What is needed in Phase 2E.** An additive, optional parameter on
`generate_sample()` — e.g. `acquisition_overrides` / `scene_overrides` applied
via `dataclasses.replace` after sampling — so the default path is
bit-identical to today. The design, factor mapping, level table, batch
manifests and statistics are all complete and validated; only the render hook
is missing.

**Risk if left.** None to Phase 2A. It is a hard blocker for Phase 2E and is
the reason the L27 campaign is reported as *designed and validated, not
executed*.

## D10 — Meta-ranker widened from one model to a multi-model ensemble

**Observed / decided.** The Phase-1 freeze specified a single
HistGradientBoosting meta-ranker. That is now a stacked ensemble of several
learners (`experiments/configs/ensemble_plan.yaml`): HGB, RandomForest,
ExtraTrees, a calibrated linear model, and an optional small MLP, combined by
scene-grouped out-of-fold stacking.

**Why this is evidence-driven rather than decorative.** The Phase-1 leak-free
ablation measured expert *disagreement* on 14/20 pairs, with the oracle
best-of-two beating any fixed fusion by 15 points. Disagreement is the
exploitable resource, and a single learner imposes one inductive bias on a
problem where candidates fail for different reasons. Trees capture the sharp
`lattice_residual` / `peak_gap` thresholds; a calibrated linear member
extrapolates more safely to acquisition conditions outside the training
envelope, which matters because the hidden test is explicitly noisier than
training data.

**Cost.** Negligible: ranking runs over <=50 candidates per pair, so several
CPU models cost microseconds against a locator budget in the hundreds of
milliseconds.

**Guard.** `ensemble_plan.yaml` requires a leave-one-out ablation, so every
member must demonstrably earn its place rather than being included for
variety. If members do not decorrelate in practice, the ensemble collapses to
its best member and that is the honest result.
