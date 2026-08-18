# 06 — Generator Physics / Augmentation Audit

Scope: `driftforge/sem.py`, `driftforge/scene.py`, `driftforge/config.py`, compared against
the organiser's generator and public SEM literature.

**Framing, stated up front.** DriftForge is a 2D procedural proxy, not a Monte-Carlo
electron-transport simulator, and `SOURCES.md` already says so. This audit does not fault
it for that. It asks a narrower question: *are the mechanisms present, correctly ordered,
and randomised widely enough that a matcher tuned on them will survive the hidden test?*

### Primary sources used

| Ref | Source | What it supports |
|---|---|---|
| S1 | Organiser starter, `src/sem_imaging.py`, `src/pipeline.py` (read in full, 2026-08-18) | which effects the sponsor considers in scope, and their magnitudes |
| S2 | *Correction of SEM Imaging Artifacts in a Novel Digital Image Correlation Framework*, PMC6541586 | scan-line shifts **0.5–5 px**; spatial distortion up to **~9.9 px**, increasing **at lower magnification**; drift distortion as time-dependent tension/compression/shear |
| S3 | NIST — *Real-Time Image Composition with Correction of Drift Distortion* | drift "commonly causes blur and distortions"; may render images unusable for sub-nm metrology; motivates row-wise/frame-wise drift modelling |
| S4 | i4c.in official Track 2 statement | edge brightening, blur, rotation, scale, independent noise, search noisier than reference — all mandatory |
| S5 | `SOURCES.md` existing citations (6F² DRAM patent + paper; Berkeley FinFET short course; FinFET electron-tomography study; Lewis, *Fast Normalized Cross-Correlation*) | structural priors and NCC basis |

---

## 1. Capture-chain ordering — verdict: physically correct

```
Reference:  edge_brighten -> PSF -> affine(rot,scale,trans) -> raster shear/jitter
            -> radial -> vignette -> charging -> gain/offset -> gamma -> Poisson
            -> read noise -> hot pixels
Search:     edge_brighten -> PSF -> area_downsample -> [same geometric+photometric chain]
```

`CODE OBSERVATION` — the ordering is right in the ways that matter:
- edge brightening is applied to the **specimen signal before the beam PSF**, which is
  correct: enhanced secondary-electron yield at edges is a property of the sample, and the
  beam then blurs it. (The sponsor omits this stage entirely.)
- PSF is applied **before** decimation (`sem.py:154-156`), so the Search image is
  band-limited before sampling rather than after — the physically correct order and the
  same choice the sponsor makes.
- Poisson shot noise is applied **after** gain/gamma and **before** additive read noise,
  which matches a real detector chain (signal-dependent noise first, electronics noise
  second).

This is a genuine strength and should be preserved unchanged.

## 2. Per-effect audit

Legend: **too weak** · **plausible** · **too aggressive** · **unsupported**

### 2.1 Edge brightening
- **Mechanism.** Elevated secondary-electron escape at edges/sidewalls.
- **Implementation.** `sem.py:50-57`. `img + strength · clip(‖∇(G_σ*I)‖ / P99.5, 0, 1)`,
  Sobel gradient magnitude, additive, clipped to [0,1].
- **Range.** `edge_strength` U(0.14, 0.42); `edge_sigma_nm` U(1.0, 4.5). Scale-aware
  (`sigma_px = sigma_nm / pixel_nm`).
- **Source support.** S4 (mandatory); S1 shows the sponsor **omits** it.
- **Verdict — plausible.** Polarity-symmetric (both sides of an edge brighten), which is
  the physically right behaviour and is *exactly* the property that motivates a
  polarity-insensitive orientation channel in v2 (Expert B).
- **Risk.** Reference and Search draw `edge_strength` **independently** (separate RNG
  streams), so one image can have 0.14 and the other 0.42 — a 3× mismatch on the single
  most informative cue. Defensible as aggressive domain randomisation, but it should be a
  deliberate decision, not an accident of the two-stream design.
- **Phase-2 action.** Keep. Consider a shared latent "material/beam-energy" factor with
  independent jitter around it, rather than two fully independent draws.

### 2.2 PSF / blur
- **Mechanism.** Finite beam spot + astigmatism.
- **Implementation.** `sem.py:60-63`, separable anisotropic Gaussian (σy, σx).
- **Range.** σx U(2.0, 7.0) nm, σy U(2.0, 8.5) nm; ood 11.0 / 13.0 nm. Anisotropy up to
  ~4× at the extremes.
- **Source support.** S1 (sponsor `beam_spot_size_nm=5.0`, `astigmatism_ratio` knob), S4.
- **Verdict — plausible**, well matched to the sponsor's 5 nm default.
- **Risk.** See §3 — the *rendering grid*, not the σ value, is the weak point.

### 2.3 Poisson shot noise / dose
- **Implementation.** `sem.py:134-135`, `poisson(out·dose)/dose`.
- **Range.** Reference dose 900–4200 (log-uniform); Search 70–700 (140–700 non-hard).
- **Source support.** S1 (sponsor: ref 2000, search 25–800), S4 ("search image will have
  more noise than the reference").
- **Verdict — plausible and correctly asymmetric.** DriftForge's search-dose floor (70) sits
  between the sponsor's `high` (60) and `medium` (200) levels.
- **Risk.** The sponsor's `severe` level is **dose 25** — ~3× noisier than anything
  DriftForge generates. Given S4's explicit warning that the test set "will be MORE noisy
  than the training examples", **DriftForge's noise ceiling is too low.**
- **Phase-2 action.** Extend the Search dose floor to ~25 and add a `severe` profile.

### 2.4 Gaussian / read noise
- **Implementation.** `sem.py:136`, additive `N(0, σ)` in normalised units.
- **Range.** Reference σ 0.003–0.016; Search σ 0.010–0.045 → **2.6–11.5 grey levels** on a
  0–255 scale.
- **Source support.** S1 — sponsor `detector_noise_sigma_search` spans **2–12** grey levels.
- **Verdict — plausible**, near-identical to the sponsor's range. Correctly asymmetric.

### 2.5 Raster shear (drift)
- **Implementation.** `sem.py:99-100`, linear row-indexed shear in x.
- **Range.** Search ±2.2 px (±4.5 hard); Reference ±0.25 px.
- **Source support.** S2 (drift distortion as shear-like), S3, S1 (sponsor
  `shear_amplitude_px` 0.5–4.0).
- **Verdict — plausible**, matches the sponsor almost exactly, correctly stronger on the
  slow wide-area Search scan.

### 2.6 Row jitter (vibration)
- **Implementation.** `sem.py:98`, per-row `N(0, σ)` x-shift.
- **Range.** Search σ 0.05–0.75 px (1.7 hard); Reference 0.0–0.15 px.
- **Source support.** S2 measures real scan-line shifts at **0.5–5 px amplitude**;
  S1 sponsor `drift_jitter_px` 0.2–1.8.
- **Verdict — too weak.** The literature's measured range starts around where DriftForge's
  *hard* ceiling ends. `MEASURED-BY-SOURCE` gap: S2's 0.5–5 px vs DriftForge's 0.05–1.7 px.
- **Phase-2 action.** Raise the Search jitter ceiling toward ~3–5 px. This directly stresses
  the matcher in the way real tools do and is cheap to change.

### 2.7 Radial / lens distortion
- **Implementation.** `sem.py:93-96`, `factor = 1 + k·r²` on normalised radius.
- **Range.** `radial_k` U(−0.006, 0.006), ood ±0.018 → corner displacement ≈ **6 px**
  (18 px ood) on a 1000 px frame.
- **Source support.** S2 reports spatial distortion up to **~9.9 px** and explicitly notes
  it **increases at lower magnification**; S1 sponsor `barrel_distortion_k`.
- **Verdict — plausible in magnitude, but physically inconsistent in application.**
- **Risk (real).** `sem.py:37` draws `radial_k` from the **same distribution for Reference
  and Search**. Per S2, the Search image — being the 10× *lower*-magnification capture —
  should carry systematically **more** spatial distortion than the Reference. The sponsor
  gets this right by construction (`barrel_distortion_k * 0.3` for the reference,
  `pipeline.py`). DriftForge does apply a 0.3× analogue for shear/jitter but **not** for
  radial distortion.
- **Phase-2 action.** Scale Reference `radial_k` down (≈0.3×) relative to Search, matching
  both S2 and the sponsor.

### 2.8 Rotation and scale
- **Range.** Reference rot ±2.2° (±3.3° ood), scale 0.965–1.035 (0.93–1.07 ood);
  Search rot ±0.35° (±0.77° ood), scale 0.992–1.008 (0.975–1.025 ood).
  Relative rotation between the pair therefore reaches **~2.55°**, relative scale ~±4.4%.
- **Source support.** S4 explicitly mandates "rotation, and scaling variations".
- **Verdict — plausible, and a clear advantage:** `VERIFIED FACT` the sponsor generator
  implements **neither**. DriftForge is the only one of the two that satisfies this
  official requirement.
- **Risk.** The locator's transform bank is tuned to *this* range (±3°, ±6%) while the
  sponsor's is ±11% in scale and 0° in rotation. See `05` §5.

### 2.9 Charging
- **Implementation.** `sem.py:122-130`. A smooth low-frequency random field plus 0–5
  bright horizontal streak bands.
- **Range.** `charging_strength` 0–0.10 (0–0.20 hard); `charging_streaks` 0–2 (0–5 hard).
- **Source support.** S1 (sponsor `charging_streak_prob`/`intensity`); phenomenological.
  S2 does **not** address charging.
- **Verdict — plausible but phenomenological.** Honestly labelled as such in `SOURCES.md`.
- **Note.** Real charging is material-dependent (dielectric regions charge, metal does
  not). Both generators apply it position-independently. Acceptable simplification;
  flagging it as a known limitation is the right response, not modelling it.

### 2.10 Hot pixels / impulse noise
- **Range.** `hot_pixel_rate` 0–0.00015 (0–0.0007 hard) → **0–700 pixels** per 10⁶.
- **Source support.** S1 — sponsor `salt_pepper_prob` up to **0.012 = 12,000 ppm**.
- **Verdict — too weak by ~17×** relative to the sponsor's `severe` level.
- **Phase-2 action.** Raise the ceiling toward ~0.01. Cheap; directly relevant since
  impulse noise is exactly what breaks naive normalised correlation.

### 2.11 Vignette, gain, offset, gamma
- **Range.** vignette 0–0.20 (0.34 hard); gain 0.82–1.20; offset ±0.08;
  gamma 0.82–1.22 (0.65–1.45 ood).
- **Source support.** S1 (sponsor `vignette_strength`, `gamma`).
- **Verdict — plausible.** Gain/offset are largely neutralised by ZNCC anyway; gamma is the
  one that genuinely matters because it is **non-linear** and therefore *not* removed by
  zero-mean normalisation.
- **Note.** This is a further argument for an orientation/edge-direction channel in v2:
  gradient *direction* is far more gamma-stable than gradient magnitude or raw intensity.

### 2.12 Speckle / multiplicative noise
- **Implementation.** **Absent from DriftForge.**
- **Source support.** S1 — the sponsor implements `speckle_sigma` and uses it at its two
  hardest evaluation levels (0.15, 0.3).
- **Verdict — unsupported gap.** The sponsor considers signal-dependent multiplicative
  noise in scope; DriftForge models only Poisson + additive Gaussian.
- **Phase-2 action.** Add multiplicative speckle. Low cost, closes a known sponsor-visible gap.

### 2.13 Fabrication roughness / line-width variation
- **Implementation.** `scene.py:104-111`. Deterministic hash-waves: per-line position
  jitter, per-line width modulation, and two-component sinusoidal line-edge roughness.
- **Range.** `line_jitter_nm` 0.25–1.8; `width_jitter_fraction` 0.02–0.12;
  `roughness_nm` 0.4–2.0 (×1.5 hard/ood); `roughness_lambda_nm` 90–420.
- **Source support.** S5 (FinFET tomography — real non-ideal morphology); S1 (sponsor
  per-instance linewidth/position jitter + `linewidth_bias_nm` + `corner_rounding_px`).
- **Verdict — plausible in magnitude, but structurally weaker than it looks.**
- **Risk (important for aliasing).** The roughness is built from **sinusoids indexed by
  line number** (`_hash_wave(idx, 1.618…)`), i.e. it is itself **quasi-periodic**, not
  stochastic. Genuinely random line-edge roughness is what makes real arrays
  *distinguishable* between repeats; a periodic roughness function preserves the ambiguity
  and can even create *new* long-period aliases. This plausibly contributes to the exact
  integer-lattice aliases measured in `05` §2.
- **Phase-2 action.** Replace the hash-wave with a per-line pseudo-random draw seeded by
  (scene seed, line index) — same determinism, genuinely aperiodic. **Then re-measure
  whether aliasing is still exact.** This is the cleanest single test of whether
  DriftForge's ambiguity is physical or self-inflicted (E4 in `09`).
- **Missing vs sponsor.** No global CD/etch bias (`linewidth_bias_nm`) and no corner
  rounding. Both are cheap to add and both are cited by the sponsor as real
  litho/etch effects.

### 2.14 Structural defects
- **Implementation.** `scene.py:175-219`. Four random kinds (etched patch, bright particle,
  bridge, scratch), `round(10·density)` = 4–14 across the 10 µm world, radius 18–85 nm —
  **plus a target-correlated signature** (radius 72–135 nm) always inside the Reference FOV.
- **Source support.** Random defect kinds are physically reasonable and richer than the
  sponsor's. The **target-correlated signature has no physical justification and no source
  support.**
- **Verdict — the random defects are plausible; the signature is unsupported and is the
  single most damaging element in the generator.** See `05` §4: removing it drops accuracy
  from 80% to 30%.
- **Missing vs sponsor.** No **pattern collapse / bridging** model
  (`maybe_collapse_gap`, gaps < 10 nm bridge with p=0.7) — a well-motivated real effect
  (capillary collapse of high-aspect-ratio features) that the sponsor documents with a
  worked example. Worth adding.

### 2.15 Array / periphery boundaries
- **Implementation.** `scene.py:156-168`. Mats of period 2200–3600 nm separated by
  220–480 nm strips carrying routing lines (periods 310 nm in x, 370 nm in y).
- **Source support.** S5 (6F² DRAM organisation); S1 (sponsor `mat_size_nm=2600`,
  `strip_width_nm=320`) — near-identical parameters, independently arrived at.
- **Verdict — plausible and well matched to the sponsor.**
- **Risk.** The `boundary` profile is only **12–15%** of train/validation, whereas the
  sponsor's `boundary_bias` default is **0.35**. `05` §4.2 measured that boundary pairs are
  over-represented among the samples that survive signature removal (4 of 6) — i.e. **mat
  boundaries are the strongest genuine, non-leaked localisation cue in the whole
  generator**, and DriftForge under-samples them by ~2.5×.
- **Phase-2 action.** Raise `boundary` to ≈0.30–0.35 in `PROFILE_MIX`. High value, one-line
  change.

## 3. Rendering-fidelity issues (not "effects", but they shape the data)

### 3.1 Search supersampling is marginal at fine pitch
`generator.py:86-87` renders the Search world at `1000·supersample` px. At the default
`supersample=2` that is **5 nm/px**. The PSF (σ 2–7 nm → 0.4–1.4 px at that grid) is applied
there, then a 2× area average.

For `finfet_dense` at a 28 nm fin pitch, 5 nm/px gives only **5.6 samples per period**
before blurring, and the underlying geometry is a hard binary threshold
(`np.where`, `scene.py:140-153`) with **no geometric anti-aliasing**. The result is that
some of the "aliasing" in the Search image is a *rendering* artifact of the sampling grid,
not the physical beam-plus-detector aliasing the sponsor describes.

The sponsor avoids this by rendering the full 10000² canvas at 1 nm/px and blurring there.

`RECOMMENDATION` — quantify the difference by regenerating a subset at `supersample=4`
(2.5 nm/px) and comparing locator error. If results move materially, `supersample=4`
should become the default for any data used to fit or validate v2. Tracked as E1b in `09`.

### 3.2 The Reference is rendered without supersampling
`generator.py:83` renders the Reference at exactly 1 nm/px with the same hard thresholds.
PSF (σ ≥ 2 px there) largely masks it, so this is lower-risk than §3.1 — but the two images
are produced by **different effective anti-aliasing paths**, which is a systematic
Reference/Search mismatch that no real tool would have.

### 3.3 `ml.py` decimates differently from `baseline.py`
Already flagged in `03` §4: `ml.py:24` uses `zoom(0.1, prefilter=False)` with **no**
anti-alias pre-filter, whereas `baseline.py:55` applies σ=4.0 first. A learned reranker
trained on `ml.py` arrays would see a decimation artifact the classical path does not.
Must be unified before any Phase-2 training. (**R10**)

## 4. Summary table

| Effect | Verdict | Hidden-test risk | Phase-2 action |
|---|---|---|---|
| Edge brightening | plausible | Low | keep; consider shared latent |
| PSF / astigmatism | plausible | Low | keep |
| Poisson / dose | plausible | **Medium** | extend floor to ~25 |
| Read noise | plausible | Low | keep |
| Raster shear | plausible | Low | keep |
| Row jitter | **too weak** | **Medium** | raise to ~3–5 px (S2) |
| Radial distortion | plausible, **misapplied** | Medium | make Reference ≈0.3× Search |
| Rotation / scale | plausible | Low | keep — **sponsor has none** |
| Charging | plausible (phenomenological) | Low | keep |
| Hot pixels | **too weak (~17×)** | Medium | raise toward 0.01 |
| Vignette/gain/gamma | plausible | Low | keep |
| Speckle | **absent** | Medium | add |
| Roughness / CD variation | plausible but **quasi-periodic** | **High** | make genuinely aperiodic; add CD bias + corner rounding |
| Random structural defects | plausible | Low | keep |
| **Target-correlated signature** | **unsupported** | **Critical** | see `09` E0 |
| Pattern collapse | **absent** | Low | add (sponsor-documented) |
| Mat/periphery boundaries | plausible | **Medium** | raise share to ~0.35 |
| Structural pitch coverage | **too narrow** | **High** | widen to sponsor range (`02` §2.5) |
| Search supersampling | marginal at fine pitch | Medium | test `supersample=4` |

## 5. Citation compliance (official requirement #14)

S4 requires "at least 2-3 credible public references" per augmentation, noise model and
structural parameter. `SOURCES.md` currently cites **families of effects**, not individual
parameters, and several effects above (speckle, hot-pixel rate, charging magnitude,
row-jitter range) have **no numeric citation at all**.

`RECOMMENDATION` — convert the §2 table into a per-parameter citation table in `SOURCES.md`
with explicit numeric ranges and the source that justifies each. S2's measured
0.5–5 px scan-line shifts and ~9.9 px spatial distortion are exactly the kind of numeric
anchors currently missing, and adopting them also fixes the two "too weak" verdicts.
This is low-effort and directly scored.
