# 09 — Phase-2 Experiment Plan

Ordered. Each experiment states its question, method, expected output, and an explicit
**stop/go rule**. Experiments E0–E2 are blocking: nothing downstream is interpretable
until they land.

**Standing rules**
- The 1,000-pair `test.jsonl` manifest stays locked. No architecture or threshold decision
  is made on it. One-shot use at release only.
- Never regenerate a manifest or change a seed base.
- Every accuracy number reported after E0 must state whether the signature was present.
- Report emitted-coordinate accuracy as the headline; coverage is a secondary diagnostic
  (R15).

---

## E0 — Neutralise the target-correlated signature *(blocking, do first)*

**Question.** What is DriftForge's true, leak-free performance, and what should the
signature become?

**Method.** Three options, to be decided by human review before implementation:

| Option | Description | Pro | Con |
|---|---|---|---|
| **0a — delete** | Remove the signature block (`scene.py:205-219`) entirely | Simplest; guarantees no leak | Loses a legitimate "sometimes there *is* a unique landmark" case |
| **0b — de-correlate** (recommended) | Keep the defect kind, but place it **uniformly in the world** like every other random defect, with no reference to `target_c` | Keeps defect diversity; removes the leak; matches the sponsor's uncorrelated model | Slightly more code than 0a |
| **0c — probabilistic** | Place near the target only with probability p≈0.15, uniformly otherwise | Models "landmarks exist but are not guaranteed" | Leaves a residual, harder-to-audit leak; needs a leak-magnitude budget |

**Recommendation: 0b.** It preserves everything defensible about the current defect model
and removes the only part with no source support (`06` §2.14).

**Expected output.** `reports/phase2/E0_leakfree_baseline.json`; regenerated release gate
on ≥200 validation pairs; `RESULTS.md` and `README.md` corrected.

**Stop/go.** GO once leak-free numbers are published and the 80% claim is withdrawn from
all documentation. **This is a hard gate on E9 (learned reranker)** — training on leaked
data would bake the shortcut into the model.

**Predicted result** (`HYPOTHESIS`, from `05` §4.2): acc@5px lands in the 30–45% band.
If it lands materially above 50%, the signature was not the dominant factor and this
audit's H2 conclusion must be revisited.

## E1 — Expand the evidence base

**Question.** Are the 20-pair conclusions stable?

**Method.** Re-run the gate on ≥200 validation pairs, leak-free, with the shipped locator.
Fix R8 (regenerate the packaged artifact with the shipped script; add `tolerance_px`).

**Expected output.** Stable median / P95 / max / acc@{1,2,5,10} / >25 px count, with
confidence intervals; failure-class histogram.

**Stop/go.** GO if the alias fraction of catastrophic failures stays ≥80%. **STOP and
re-open the autopsy** if a materially different failure class appears at scale — the whole
architecture is predicated on aliasing being the dominant mode.

### E1b — Rendering-fidelity check
Regenerate a 50-pair subset at `--search-supersample 4` and compare. If locator error moves
materially, `supersample=4` becomes the default for all fitting/validation data (`06` §3.1).

## E2 — Lattice estimator, hardened

**Question.** Can pitch and lattice basis be recovered reliably enough to drive candidate
generation on data we have not tuned on?

**Method.** Promote `diagnostics/pitch_estimator_v2.py` to a real module. Extend from
axis-aligned pitch to a full 2D basis (recover lattice **orientation** too, not just x/y
periods). Validate against known scene pitch on ≥200 leak-free pairs **and** on
sponsor-generated pairs (E8). Calibrate the confidence statistic into a usable gate.

**Expected output.** Recovery rate for fundamental-or-small-harmonic; ROC of the confidence
statistic vs recovery; per-architecture breakdown.

**Stop/go.** GO if recovery ≥90% at high confidence **and** the confidence statistic
separates successes from failures (Phase-1 prototype: 18/20 and 19/20, with all 3 failures
at confidence 8.7–18.4 vs a 30–2255 success range — promising but n=20).
**If confidence is not discriminative, lattice expansion must be run unconditionally**
rather than gated, which raises candidate count and runtime — re-cost before proceeding.

## E3 — Transform estimator replaces the brute-force bank

**Question.** Can log-polar/phase estimation replace 25 variants at equal accuracy and
≤1 s/pair?

**Method.** Implement Reddy–Chatterji rotation/scale estimation. Compare against the
shipped 5×5 bank on identical pairs: accuracy, and recovered rotation/scale vs the known
`AcquisitionSpec` values. Measure runtime.

**Expected output.** Accuracy delta, transform-estimate error distribution, s/pair.
**Target:** ≤6 NCC calls, ≈0.55 s/pair (`04` §5 budget arithmetic).

**Stop/go.** GO if accuracy is within noise of the 25-variant bank. If it degrades, fall
back to a **coarse-to-fine** bank (1 coarse pass + local refinement, ~4 NCC calls) rather
than reinstating brute force.

## E4 — Is the aliasing physical or self-inflicted?

**Question.** Do the exact integer aliases survive genuinely stochastic line-edge roughness?

**Method.** Replace the quasi-periodic `_hash_wave` roughness (`scene.py:104-111`) with
per-line pseudo-random values seeded by (scene seed, line index) — same determinism,
genuinely aperiodic. Re-run the lattice fit and compare residual distributions.

**Expected output.** Alias residual distribution before/after.

**Stop/go.** This is a **diagnostic, not a fix.** If residuals grow, part of DriftForge's
ambiguity was self-inflicted (R13) and the generator should be corrected — but the
architecture does **not** change, because the organiser has stated the ambiguity is real in
*their* data regardless of ours.

## E5 — Candidate-set integrity (repairs H1)

**Question.** Does lattice expansion restore the hypotheses NMS destroys?

**Method.** Keep NMS as a seed-finder. For each seed peak, generate translates on the
**centred-rectangular** lattice `(2·pitch_x, 0)`, `(pitch_x, pitch_y)` (`08` Expert C).
Measure: (a) does the candidate set now contain the true cell, and how often;
(b) candidate-count and runtime impact.

**Expected output.** Recall of the true cell in the candidate set vs the shipped baseline;
candidate-count distribution.

**Stop/go.** GO if true-cell recall reaches ≥99%. **A reranker cannot fix what is not in
the candidate set** — this recall number is the ceiling on everything downstream, so it
must be measured before E6/E7.

## E6 — Channel ablation, leak-free and at scale

**Question.** Which evidence channels actually carry non-leaked signal?

**Method.** Re-run the `diagnostics/ablation.py` design on ≥200 leak-free pairs, adding the
directionality channel. Report per-channel accuracy, per-channel disagreement, and
oracle-best-of-N.

**Expected output.** Channel value table; measured headroom for a ranker
(Phase-1 leak-free: fused 30% vs oracle-of-2 45%).

**Stop/go.** GO to E7 if oracle-best-of-N exceeds the best single channel by ≥5 points.
If not, the ranker has no headroom and should be cut in favour of the single best channel.

## E7 — Candidate meta-ranker + equivalence set

**Question.** Can a candidate-level ranker plus a calibrated equivalence set and the
official centre rule beat the shipped selector?

**Method.** Features per candidate: `zncc_score`, `midband_score`, `directionality_score`,
`lattice_residual`, `lattice_confidence`, `psr`, `best_second_gap`, `expert_vote_count`,
`scale_consistency`, `rotation_consistency`, `noise_estimate`, `blur_estimate`,
`local_entropy`, boundary/context score.
**Explicitly excluded: distance-to-centre (R5).**
Train HistGradientBoosting on train-split candidates; calibrate on validation; build the
equivalence set by probability margin; apply nearest-to-centre within it.

**Expected output.** acc@{1,2,5} , >25 px count, equivalence-set size distribution,
true-cell containment rate, feature importances.

**Stop/go.** GO if >25 px failures fall by ≥50% versus the leak-free baseline with no
accuracy regression. Inspect feature importances: **if any centre-correlated proxy
dominates, stop and re-audit the feature set.**

## E8 — Cross-generator sanity test *(the domain-shift alarm)*

**Question.** How does DriftForge perform on the organiser's own data, untuned?

**Method.** Install OpenCV in the audit environment only. Generate ≥100 pairs with the
sponsor's `generate_dataset.py` across all 12 presets and the 4 documented noise levels.
Run: (a) sponsor ZNCC baseline, (b) DriftForge v1 locator, (c) v2 — with **no tuning**.
Also run the reverse direction (sponsor baseline on DriftForge pairs).

**Caveat to state in the report:** input distributions are **not** identical — the sponsor
has no rotation, no scale and no edge brightening, and its labels are uncorrected for its
own shear/barrel warps (`02` §2.3), giving it 1–4 px of label noise. Any comparison must
say so; do not claim a fair head-to-head.

**Expected output.** 2×3 accuracy table with per-preset breakdown, especially the coarse
presets (`dram_legacy`, `finfet_45nm`) that DriftForge's priors do not cover (R3).

**Stop/go.** **STOP and re-open the architecture** if v2 does not beat the sponsor ZNCC
baseline on sponsor data. That is the minimum bar; failing it means the design is tuned to
DriftForge's quirks rather than to the problem.

## E9 — Learned reranker *(EXPERIMENTAL; gated on E0 and R10)*

**Question.** Does a small Siamese reranker beat the classical meta-ranker on hard,
visually-similar candidates?

**Preconditions (hard):** E0 landed (no leak); R10 fixed (unified decimation across
`ml.py` and the locator); scene-level split separation verified.

**Method.** Shared-weight encoder over 100×100 patches, embedding dim ≤128, CPU-inference
budget ≤0.3 s for 50 candidates. Hard negatives = integer-lattice translates (including
`m+n` odd, which are *true* negatives) plus top false peaks from classical experts.

**Stop/go.** Ship only if it beats the E7 ranker on **leak-free** validation **and** on
sponsor data (E8), and stays inside the runtime budget. Otherwise **cut it** — a
pure-classical pipeline with no torch dependency is a reproducibility asset (R7).

## E10 — Release engineering

**Method.** Initialise the public GitHub repository (R9). Verify on a **clean machine, clean
clone, fresh venv, no network**: `inference.py ref.png search.png` prints bare `x,y`,
emits a coordinate for every pair, never abstains by default (R11), and reproduces the
gate. Pin dependency versions. Convert `06` §2 into a per-parameter citation table in
`SOURCES.md` (official requirement #14).

**Stop/go.** Blocking for submission.

## E11 — Sub-pixel refinement *(last)*

**Method.** Upsampled-DFT refinement (Guizar-Sicairos) applied **only** to the final
selected cell. Target median ≤0.2 px, from the current 0.52 px quantisation floor.

**Stop/go.** Ship only if the >25 px failure count is unchanged. Never trade cell-selection
accuracy for sub-pixel precision.

---

## Ordering and dependencies

```
E0 (leak) ──┬─> E1 (scale) ──> E1b (fidelity)
            ├─> E2 (lattice) ──> E5 (candidate integrity) ──> E7 (ranker) ──> E11 (subpixel)
            ├─> E3 (transform/runtime) ──────────────────────^
            ├─> E4 (roughness diagnostic)
            └─> E6 (channels) ───────────────────────────────^
                                     E8 (cross-generator) gates release
                                     E9 (learned) optional, gated on E0
                                     E10 (release) blocking
```

**Critical path to a defensible submission:** E0 → E2 → E5 → E7 → E8 → E10.
E3 buys the runtime budget that E5/E7 spend. E9 and E11 are upside, not prerequisites.
