# 08 — Architecture Decision Matrix

Scores are **1–5, higher is better in every column** (so "runtime cost 5" means *cheap*,
"complexity 5" means *simple*, "dependency risk 5" means *low risk*).

Every score traces to one of: an official requirement, sponsor code/statement, a
reproduced DriftForge measurement, a source-code defect, credible primary research, or an
explicit cost-benefit calculation. Where a claim cannot be traced, it is marked
`SPECULATIVE` and excluded from the freeze.

---

## 1. Evidence assessments of the candidate experts

### Expert A — ZNCC / photometric correlation (`baseline_solution/zncc.py`; Lewis)
**Keep, but demote from decider to evidence provider.**
`MEASURED`: raw-NCC alone reaches 45% @5px on leak-free data — the *best* single classical
channel, better than the shipped fusion (30%). It is a strong scorer and a poor selector.
**Retain** for candidate scoring; **remove** its role as global arg-max.
`RECOMMENDATION`: adopt the sponsor's wider scale bracket (±11% vs DriftForge's ±6%) and
test true area-average decimation against the current σ=4.0 pre-blur (`03` D7).

### Expert B — polarity-insensitive edge-directionality (US20160035076A1)
The proposed representation is
`m = √(gx²+gy²)`, `θ = atan2(gy,gx)`, `Dx = m·cos 2θ`, `Dy = m·sin 2θ`.

- **Mathematically sound.** Under a gradient polarity flip `g → −g`, `θ → θ+π`, so
  `cos 2θ` and `sin 2θ` are unchanged and `m` is unchanged. The map is therefore exactly
  invariant to edge-sign polarity — consistent with the sign-invariant orientation concept.
  Equivalently `Dx = (gx²−gy²)/m`, `Dy = 2gxgy/m`: the classical orientation-doubling used
  in structure-tensor analysis.
- **Why it may help here — grounded in this generator, not in general.** DriftForge's edge
  brightening (`sem.py:50-57`) brightens **both** sides of every line, so the intensity
  gradient reverses sign across a feature. Reference and Search draw `edge_strength`
  (0.14–0.42) and `gamma` (0.82–1.22) **independently**. Gradient *orientation* is invariant
  to both; gradient *magnitude* and raw intensity are not. This is a real, measured
  mismatch the current channels do not neutralise.
- **Why it may fail.** On a dense uniform array the orientation field is itself perfectly
  periodic — every fin is vertical — so it contributes **nothing** to disambiguating
  between repeats. Its value is concentrated at mat boundaries, routing strips and
  defects, which is precisely where `05` §4.2 showed the genuine (non-leaked) signal lives.
  It is a *robustness* channel, not an *anti-alias* channel, and must be scored that way.
- **Cost.** 2 Sobel passes + trig per image, then 2 extra NCC channels ≈ **+0.12 s/pair**.
- **Ensemble diversity.** High — it is quadratic in the gradient and contrast-polarity
  invariant, structurally unlike the two intensity-domain channels already present.
- **Patent status.** The formulation is *consistent with the concept* described in
  US20160035076A1. **No equivalence to the patent claims is asserted**, and none should be
  claimed in the submission unless the implementation is checked against the actual claims.

### Expert C — lattice / repetitive-pattern model (Doubek et al., ICPR 2010)
The repetitive-pattern literature's core point is that under a repeated texture the
translation ambiguity is **structural, not a matcher deficiency** — the correct response is
to represent the lattice explicitly and reason over it. `05` §2 confirms this is exactly
DriftForge's failure mode.

Answers to the four Phase-1 questions:
- **Can pitch be estimated robustly?** `MEASURED`: **yes, after fixing the band.** A first
  prototype run on the mid-band channel *failed* — it locked onto the 31 px routing period
  (`route_x` has period 310 nm) instead of the 2.8–17 px device lattice. A corrected
  estimator (high-pass residual, period search restricted to 2.5–30 px, harmonic-summed
  scoring) recovers pitch on **18/20 (x)** and **19/20 (y)** validation pairs. Crucially,
  **all failures self-report low confidence** (8.7, 11.7, 18.4 vs a typical 30–2255), so the
  estimator knows when it is unreliable and the lattice stage can be gated on it.
- **Candidate generation or just another score?** **Candidate generation.** As a score it
  is nearly useless (the lattice is translation-invariant by definition); its value is that
  it tells you *where the other legitimate hypotheses are* — which is exactly what
  NMS destroys (`05` §3, 4.4–67 sites lost per cell).
- **Replicating seed peaks at integer offsets:** yes, and this is the mechanism that
  repairs H1 without touching NMS itself.
- **Lattice confidence near boundaries:** must decay. The estimator's confidence statistic
  already does this behaviourally, and boundary regions are where non-lattice evidence
  should dominate.
- **Different priors for DRAM vs FinFET?** `MEASURED`: **the lattice is not the naive
  rectangular grid.** Contacts sit on a checkerboard (`scene.py:142`, and the sponsor does
  the same), so only translates with `m+n` **even** preserve the pattern — all 4 measured
  aliases satisfy this. The correct alias set is the **centred-rectangular lattice**
  generated by `(2·pitch_x, 0)` and `(pitch_x, pitch_y)`. Independently confirmed by the
  pitch estimator, which returns `2·pitch_x` on 6/20 pairs — the physically correct
  fundamental period of the full pattern. Expanding on the naive grid would waste half the
  reranker's budget on provably-worse odd translates.

### Expert D — phase correlation / log-polar (Reddy & Chatterji 1996)
**Critical question — does periodic texture break it?**
As a *locator*: **yes, decisively.** A periodic image produces a *lattice of
phase-correlation peaks* of near-equal height. This is the same failure as ZNCC, in a
different basis; it would not fix R1.
As a *transform estimator*: **periodicity actively helps.** A periodic pattern has sharp
discrete spectral peaks, which give a strong, well-conditioned signature in the log-polar
magnitude spectrum for recovering rotation and scale.
**Verdict: adopt as a transform estimator only, never as the final locator.** This is also
the fix for R6 — it replaces the 25-variant brute-force bank (79.6% of runtime) with one
estimate plus a narrow local refinement, cutting ~50 NCC calls to ~6.

### Expert E — sub-pixel refinement (Guizar-Sicairos et al. 2008; US20160275672A1)
`MEASURED`: every prediction currently lands on the half-integer grid; median error on
correct pairs is **0.52 px**, i.e. the quantisation floor. Upsampled-DFT refinement should
reach ~0.1 px.
**But the value is small compared to the failure mode**: fixing all 4 catastrophic errors
is worth ~20 accuracy points at 5 px; sub-pixel refinement is worth ~0.4 px on pairs that
already pass. **Sequence it last, and only after cell selection** — refining the wrong cell
to 0.05 px is a 614 px error stated more precisely.

### Expert F — learned Siamese / metric-learning reranker
Role: rerank **≤50 hard candidate patches** (100×100 px at Search scale), not process the
full 1000×1000 pair.
`MEASURED` headroom: leak-free oracle best-of-two-classical = 45% vs fused 30%. Classical
evidence alone appears to cap near ~45%; closing the remaining gap plausibly needs a
learned matcher.
**Risks, both serious:**
- **Shortcut learning.** Trained on today's data it would learn the target signature
  (`05` §4) — the strongest and most predictive feature present. Training **must not begin**
  before E0 lands.
- **Preprocessing mismatch (R10)** — `ml.py` decimates without anti-aliasing while
  `baseline.py` pre-blurs at σ=4.0; a model would learn that artifact.
Hard negatives must include integer-lattice translates (including the `m+n` odd ones, as
*true* negatives) and the top false peaks from the classical experts. Scene-level
train/val/test separation already exists and must be preserved.
**Verdict: EXPERIMENTAL, gated on E0 + R10.**

### Expert G — candidate-level meta-ranker (RF / HistGradientBoosting)
`MEASURED`: leak-free expert disagreement rises to **14/20**, and the oracle best-of-two
beats the fixed 0.25/0.75 fusion by **15 points**. The evidence for arbitrating between
channels *per candidate* is direct and strong; the evidence for any *fixed* weighting is
negative (`05` §6).
`HistGradientBoostingClassifier` is in scikit-learn — one well-known dependency, CPU-cheap
on ≤50 candidates. **XGBoost is not justified** at this scale.
**Feature-set rule (R5):** distance-to-centre must **never** be a ranker feature. The ranker
estimates *"is this candidate a visual match?"*; the official centre rule is applied
afterwards, to the equivalence set.

**Equivalence-set calibration.** Two candidates are equivalent when the ranker cannot
distinguish them beyond noise. Concretely: calibrate ranker scores to probabilities on
validation, then admit candidate `i` to the set if `p_i ≥ p_max − δ`, with `δ` chosen so
that the set contains the true cell at a target rate (e.g. 99%) on validation. Then apply
nearest-to-centre. This replaces the current uncalibrated absolute `tie_delta=0.015`
(`03` D3), which is neither probability-scaled nor noise-aware.

---

## 2. Decision matrix

| Component | Acc. gain | Alias↓ | Domain robust | Runtime | Simplicity | Dep. risk | Explain | Novelty | Evidence | **Σ** | Decision |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| Existing ZNCC / raw channel | 4 | 1 | 4 | 3 | 5 | 5 | 5 | 1 | 5 | **33** | **MUST HAVE** |
| Existing mid-band channel | 2 | 1 | 2 | 4 | 5 | 5 | 4 | 3 | 5 | **31** | **MUST HAVE** (as feature, not weighted decider) |
| Lattice estimation | 3 | 5 | 4 | 5 | 4 | 5 | 5 | 5 | 4 | **40** | **MUST HAVE** |
| Lattice candidate expansion | 5 | 5 | 4 | 4 | 4 | 5 | 5 | 5 | 4 | **41** | **MUST HAVE** |
| Candidate meta-ranker (RF/HGB) | 5 | 4 | 3 | 4 | 3 | 4 | 4 | 5 | 4 | **36** | **MUST HAVE** |
| Uncertainty / equivalence + centre rule | 4 | 4 | 4 | 5 | 3 | 5 | 5 | 5 | 5 | **40** | **MUST HAVE** |
| Phase / log-polar transform estimation | 2 | 1 | 4 | 5 | 3 | 5 | 4 | 4 | 4 | **32** | **SHOULD HAVE** (runtime fix for R6) |
| Edge-directionality channel | 3 | 2 | 5 | 4 | 4 | 5 | 4 | 5 | 3 | **35** | **SHOULD HAVE** |
| Sub-pixel DFT refinement | 2 | 1 | 3 | 5 | 4 | 5 | 5 | 3 | 5 | **33** | **SHOULD HAVE** (sequenced last) |
| Siamese / learned reranker | 4 | 4 | 2 | 2 | 2 | 2 | 2 | 4 | 2 | **24** | **EXPERIMENTAL** |
| Architecture classifier / gating | 1 | 1 | 2 | 4 | 3 | 5 | 3 | 2 | 1 | **22** | **CUT** |

### Score justifications for the non-obvious entries

- **Lattice expansion 41 (highest).** It is the only component that directly repairs the
  measured cause of 100% of catastrophic failures *and* the measured H1 defect, at ~zero
  runtime (candidate coordinates are arithmetic, and scoring ≤50 patches is cheap).
- **Mid-band channel scores low on accuracy (2) yet is MUST HAVE.** Leak-free it is the
  worst single channel (15%) — but it is the channel that carries mat/boundary context,
  and boundary pairs dominate the genuinely-solvable set (`05` §4.2). Its value is as a
  *feature the ranker can weight per-sample*, which is precisely what the fixed 0.75
  weighting gets wrong.
- **Directionality: alias-reduction only 2.** Honest scoring. On a uniform array the
  orientation field is as periodic as the intensity field. It buys robustness to
  independent edge-strength/gamma draws, not disambiguation.
- **Phase correlation: alias-reduction 1.** Explicitly scored down because as a locator it
  reproduces the same failure. It earns its place purely on runtime (5) and transform
  robustness (4).
- **Siamese 24.** Real accuracy potential (4) and real alias-reduction potential (4),
  dragged down by dependency risk (2), explainability (2), runtime (2) and — decisively —
  evidence strength (2): there is currently **no trained model and no leak-free training
  data**, so any performance claim would be speculation.
- **Architecture classifier — CUT.** No evidence supports it. Pitch and orientation
  statistics already feed the ranker directly; an explicit DRAM/FinFET label adds a
  failure mode (misclassification) without adding information the continuous features
  lack. It also risks re-introducing the architecture *input* the official interface
  forbids (`01` §2.1). Marked `SPECULATIVE` — excluded from the freeze.

## 3. Explicitly rejected

| Rejected idea | Why |
|---|---|
| Architecture classifier / gating | No evidence; adds a failure mode; features already available continuously |
| Distance-to-centre as a ranker feature | Directly creates R5; misreads the official tie rule |
| Abstention in the submitted inference path | Contest requires a coordinate for every pair (R11) |
| Phase correlation as the final locator | Reproduces the aliasing failure in a different basis |
| Retuning the fixed 0.25/0.75 fusion weights | `MEASURED`: any fixed weighting is a single point on a curve the data says must be sample-dependent |
| XGBoost | Dependency and latency unjustified at ≤50 candidates; HGB is in scikit-learn |
| Full-image deep heatmap regression | 1000×1000 CPU inference blows the runtime budget; maximal shortcut-learning exposure |
| Removing/softening the periodic ambiguity in the generator | Organiser states it is "a real matching ambiguity rather than necessarily a generator bug" — it must be solved, not designed away |
