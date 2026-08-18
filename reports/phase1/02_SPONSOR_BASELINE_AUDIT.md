# 02 — Applied Materials / Organiser Starter Repository Audit

**Source:** <https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data>
**Tree:** `/tree/main` — 107 files, fetched and read in full on **2026-08-18**.
**Method:** files downloaded via the HF API and read as source, not summarised from the
Space UI. Local copy retained under the session scratchpad for diff-on-change.

Repository layout at time of audit (matches the state described in the Phase-1 brief —
the "methodology slide deck, layer-stack/ZNCC/sample-family tooling" commit is present):

```
baseline_solution/{__init__,infer,zncc,evaluate}.py  slides/{index.html,make_assets.py,script.js,style.css,assets/*}
src/{__init__,pipeline,presets,sem_imaging,structural_defects}.py
src/patterns/{__init__,dram,finfet,zones}.py         tests/test_pipeline.py
generate_dataset.py  generate_family_dataset.py  visualize_layers.py  visualize_sample.py
app.py (Streamlit)   Dockerfile  requirements.txt   README.md
```

---

## 1. The sponsor baseline, extracted exactly

From `baseline_solution/infer.py` and `baseline_solution/zncc.py`:

| Property | Value | Tag |
|---|---|---|
| Template scaling convention | `template = resize(reference, ref_dim / scale)` — `scale` is a **divisor** | `VERIFIED FACT` |
| Scale candidates | `(9.0, 9.5, 10.0, 10.5, 11.0)` → templates 111²,105²,100²,95²,91² px | `VERIFIED FACT` |
| Equivalent multiplicative scale | 1.111, 1.053, 1.000, 0.952, 0.909 → **±11% around nominal** | `CODE OBSERVATION` |
| Resampling | `cv2.resize(..., INTER_AREA)` — area-average, correctly anti-aliased | `VERIFIED FACT` |
| Rotation searched | **No** — no rotation anywhere in the baseline | `VERIFIED FACT` |
| Matching metric | `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` = ZNCC | `VERIFIED FACT` |
| Global maximum selection | `cv2.minMaxLoc` → single arg-max per scale, best score across scales | `VERIFIED FACT` |
| Multiple peaks retained | **No** — one location per scale, only the winner survives | `VERIFIED FACT` |
| Centre tie handling | **Not implemented** | `VERIFIED FACT` |
| Periodicity modelled | **No** | `VERIFIED FACT` |
| Sub-pixel refinement | **No** — integer `max_loc` + fixed `tw/2, th/2` offset | `VERIFIED FACT` |
| Fallback on failure | `best_center` initialised to the Search image centre | `CODE OBSERVATION` |
| Output format | `print(f"{x:.2f},{y:.2f}")` — bare comma-separated | `VERIFIED FACT` |

### 1.1 Two defects in the sponsor baseline worth naming

- `CODE OBSERVATION` — **The official nearest-to-centre rule is not implemented at all.**
  `infer.py` takes a pure global arg-max. On a wallpaper-periodic tile the sponsor
  baseline cannot satisfy the official multi-match clause even in principle.
- `CODE OBSERVATION` — **Sub-pixel accuracy is structurally capped.** The reported centre
  is `max_loc + tw/2.0`; `max_loc` is integer and `tw` is an integer template width. Best
  achievable precision is ±0.5 px, and the scale quantisation (91–111 px templates in 5
  steps) adds a systematic centre offset of up to ~±0.5 px more.

### 1.2 Sponsor evaluation procedure

From `baseline_solution/evaluate.py`:

- `VERIFIED FACT` — 4 noise levels (`low/medium/high/severe`) defined by
  `dose_search` ∈ {800, 200, 60, 25}, `detector_noise_sigma_search` ∈ {2, 5, 8, 12},
  `shear_amplitude_px` ∈ {0.5, 1.5, 2.5, 4.0}, `drift_jitter_px` ∈ {0.2, 0.5, 1.0, 1.8},
  plus `speckle_sigma` and `salt_pepper_prob` at the two hardest levels.
- `VERIFIED FACT` — architectures cycled: `["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]`.
- `VERIFIED FACT` — correctness criterion is **Euclidean distance ≤ `--tolerance-px`,
  default 5.0 px**. Metrics reported: accuracy@5px and Average Precision from a
  score-thresholded PR sweep.

**Implication.** `5 px` is the organiser's own working tolerance and `accuracy@5px` is the
organiser's own headline metric. DriftForge's choice of `tolerance=5.0` is aligned —
keep it. The AP/PR framing also tells us the organiser regards a **confidence score** as a
legitimate and expected output, which validates retaining a calibrated confidence in v2
(but see §1.1: it must not gate emission of a coordinate).

## 2. Sponsor generator, extracted exactly

### 2.1 Pipeline (`src/pipeline.py`)

```
fine canvas 10000x10000 @ 1 nm/px  (zone-composed mats + strips)
   |-- random 1000x1000 crop ---------> image_reference()  -> Reference (1 nm/px)
   \-- PSF blur -> 10x area-average --> image_search()     -> Search (10 nm/px)
ground truth = (crop_x0/10 + 50, crop_y0/10 + 50)
```

- `VERIFIED FACT` — Reference and Search derive from **one shared rendered bitmap**. The
  Reference is a literal crop of the same pixels the Search is downsampled from.
- `VERIFIED FACT` — `boundary_bias` default **0.35**: 35% of crops are deliberately centred
  on a mat/strip boundary.
- `VERIFIED FACT` — `mat_size_nm=2600`, `strip_width_nm=320` default.
- `VERIFIED FACT` — `generate_sample_family()` fixes one structure + one Reference crop and
  renders 5 Search variants (`clean`, `low_dose`, `heavy_drift`, `speckle_salt_pepper`,
  `charging`) sharing one ground truth.

### 2.2 Three officially-required effects the sponsor generator does NOT implement

| Official requirement | Sponsor generator | Evidence |
|---|---|---|
| "Apply edge-brightening to mimic real SEM" | **absent** | no edge/gradient term anywhere in `src/sem_imaging.py`; repo-wide grep for `edge` returns only vignette/docs/UI hits |
| "rotation ... variations" | **absent** | no rotation/affine/warpAffine in `sem_imaging.py`; only shear + barrel |
| "scaling variations" | **absent** | fixed integer `factor = 10`; no scale jitter |

`CODE OBSERVATION` — this is a **substantive gap between the starter repo and the official
problem statement**, and it is the single most important strategic fact in this audit.

### 2.3 A label-noise defect in the sponsor generator

`CODE OBSERVATION` — the ground truth is computed as `x0/10 + 50` from the **pre-warp crop
origin**, but `image_search()` subsequently applies `apply_raster_drift` (shear up to
4.0 px + per-row jitter up to 1.8 px) and `apply_barrel_distortion`. The label is never
corrected for these warps. The sponsor's own ground truth therefore carries a systematic
error of order **1–4 px** at the harder noise levels — a large fraction of its own 5 px
tolerance.

DriftForge derives its label by warping a target mask through the *exact* Search geometry
and taking the centroid (`generator.py:91-100`, `sem.py:105-112`). **This is a genuine and
demonstrable DriftForge advantage over the sponsor starter.**

### 2.4 Structural presets (`src/presets.py`) — exact values

| DRAM preset | F (nm) | WL pitch | BL pitch | | FinFET preset | fin pitch | gate pitch |
|---|---:|---:|---:|---|---|---:|---:|
| `dram_dense` | 24 | 48 | 72 | | `finfet_7nm` | 40 | 76 |
| `dram_1x` | 32 | 64 | 96 | | `finfet_10nm` | 48 | 90 |
| `dram_compact` | 36 | 72 | 108 | | `finfet_14nm` | 60 | 110 |
| `dram_loose` | 48 | 96 | 144 | | `finfet_22nm` | 80 | 150 |
| `dram_wide` | 60 | 120 | 180 | | `finfet_28nm` | 96 | 180 |
| `dram_legacy` | 80 | 160 | 240 | | `finfet_45nm` | 140 | 260 |

At Search resolution (10 nm/px) this spans **4.8 – 24.0 px** (DRAM) and **4.0 – 26.0 px**
(FinFET) of lattice period.

`VERIFIED FACT` (from the module docstring) — the presets are "intentionally on the
larger/coarser end of what's publicly discussed", chosen for legibility at 1 px = 1 nm.

### 2.5 Sponsor pitch coverage vs DriftForge — a measured domain-shift gap

| Family | Sponsor range (nm) | DriftForge range (nm) | Verdict |
|---|---|---|---|
| DRAM, vertical-line pitch (BL) | 72 – 240 | 48 – 120 | **DriftForge tops out at half the sponsor maximum** |
| DRAM, horizontal-line pitch (WL) | 48 – 160 | 72 – 170 | comparable; DriftForge misses the 48–72 fine end |
| FinFET, fin pitch | 40 – 140 | 28 – 90 | DriftForge misses 90–140 |
| FinFET, gate pitch | 76 – 260 | 48 – 130 | **DriftForge tops out at half the sponsor maximum** |

`MEASURED RESULT` — DriftForge v1's structural prior does **not cover the coarse half** of
the sponsor's own preset family. If the hidden test draws from anything resembling
`dram_legacy` / `finfet_45nm`, DriftForge is extrapolating. This is a concrete,
cheap-to-fix Phase-2 action (widen `PRESETS` in `config.py`), tracked as R3.

### 2.6 Structural defects

`VERIFIED FACT` — the sponsor's only structural defect is **pattern collapse/bridging**
(`src/structural_defects.py::maybe_collapse_gap`): gaps ≥ 10 nm never collapse; gaps below
collapse with p=0.7. It is applied **once to the fine canvas**, so it appears consistently
in both images. There is **no particle, scratch, missing-contact or residue model**, and
critically **no defect whose position is correlated with the Reference crop location**.

This matters enormously for DriftForge — see `05_FAILURE_AUTOPSY.md` H2.

## 3. Key sponsor statements — verified and interpreted

The methodology deck (`slides/index.html`) states, verbatim in substance:

> "DRAM/FinFET patterns repeat every 3–5 px at Search resolution. A naive global
> cross-correlation search can lock onto the **wrong periodic repeat** — a genuine
> matching-ambiguity limitation, not a generator bug."
>
> "Verified directly: the Search image really does contain the true downsampled Reference
> at the stated ground-truth box (pixel-level diff confirms it). The baseline's
> global-argmax search is what fails on highly periodic tiles."
>
> Suggested directions: "Try: local search near a prior · Try: feature descriptors ·
> Try: phase correlation · Try: learned matching"

### Engineering implications for DriftForge — not just quotation

1. **The organiser has certified the failure mode as legitimate, not a bug.** This removes
   the option of "fixing" it by softening the generator. The correct response is an
   algorithmic one. It also means a submission that *demonstrably* solves periodic
   aliasing is attacking the exact weakness the organiser has publicly flagged — the
   highest-scoring available differentiator.

2. **"Not a generator bug" also means the ambiguity is often genuinely irreducible.** When
   a tile is exact wallpaper, several positions are *equally correct* as matches, and the
   official nearest-to-centre rule — not a better matcher — is what determines the right
   answer. This splits the problem in two, and v2 must treat them separately:
   - *reducible* ambiguity: a unique correct cell exists and is discoverable from
     boundaries/defects/context → fix with better evidence and lattice-aware candidates;
   - *irreducible* ambiguity: multiple cells are truly indistinguishable → fix by
     constructing the equivalence set correctly and applying the centre rule to it.
   DriftForge v1 conflates these: it applies the centre rule to an NMS-decimated,
   noise-selected candidate list (see H1), so it gets the irreducible case wrong too.

3. **The four suggested directions are a scoring hint, not a recipe.** "Local search near a
   prior" is inapplicable — the problem statement gives no prior (that is the whole point
   of navigation-error recovery). "Phase correlation" on a periodic texture produces a
   *lattice of peaks*, so it is a transform estimator, not a locator (see `08`). The two
   that survive scrutiny are feature/descriptor evidence and learned matching, and both
   are candidate-level rerankers, not replacements for correlation.

4. **`generate_sample_family()` defines the robustness test the organiser cares about**:
   one Reference must stay findable across a *range* of degraded Search captures. v2
   should adopt this as a first-class internal metric — same scene, N acquisitions,
   measure prediction *variance*, not just mean error.

## 4. Sponsor baseline vs DriftForge v1 — capability table

Legend: ✅ present · ⚠️ partial/flawed · ❌ absent

| Capability | Sponsor starter | DriftForge v1 | Verified advantage | Verified weakness | Hidden-test risk | Verdict |
|---|---|---|---|---|---|---|
| DRAM support | ✅ 6 presets | ✅ 2 presets | sponsor: wider pitch range | DF pitch range ~½ sponsor's | **High** | **change** — widen presets |
| FinFET support | ✅ 6 presets | ✅ 2 presets | sponsor: wider pitch range | DF misses coarse nodes | **High** | **change** |
| Physical world-coordinate scene | ❌ one bitmap, cropped | ✅ analytic nm-space, rendered twice | **DriftForge** | costlier | Low | **keep** |
| Independent captures | ⚠️ shared canvas, separate imaging | ✅ fully independent RNG streams | **DriftForge** | — | Low | **keep** |
| Ref/Search noise independence | ✅ | ✅ | tie | — | Low | keep |
| Edge brightening | ❌ **absent** | ✅ Sobel-magnitude boost | **DriftForge** | phenomenological | Low | **keep** |
| PSF / blur | ✅ + astigmatism ratio | ✅ anisotropic σx,σy | tie | — | Low | keep |
| Raster shear / row jitter | ✅ | ✅ | tie | — | Low | keep |
| Scale variation | ❌ **absent** | ✅ ref 0.965–1.035, search 0.992–1.008 | **DriftForge** | — | Low | **keep** |
| Rotation variation | ❌ **absent** | ✅ ref ±2.2°, search ±0.35° | **DriftForge** | — | Low | **keep** |
| Radial/barrel distortion | ✅ | ✅ | tie | — | Low | keep |
| Boundary / periphery context | ✅ `boundary_bias=0.35` | ⚠️ `boundary` profile 12–15% | sponsor more aggressive | DF under-samples boundaries | **Medium** | **change** — raise weight |
| Structural defects | ⚠️ collapse only | ✅ 4 random kinds **+ target-correlated signature** | DF richer | **signature is a leak** — see H2 | **Critical** | **change** |
| Warp-aware ground truth | ❌ uncorrected crop origin | ✅ warped-mask centroid | **DriftForge** | — | Low | **keep** |
| Deterministic seeds | ✅ | ✅ | tie | — | Low | keep |
| Train/val/test separation | ⚠️ split flag only | ✅ disjoint seed ranges + scene IDs, asserted | **DriftForge** | — | Low | **keep** |
| ZNCC matching | ✅ | ✅ | tie | — | Low | keep |
| Multi-scale | ✅ ±11% | ✅ ±6% | sponsor wider | **DF scale bank narrower than sponsor's** | Medium | **investigate** |
| Multi-rotation | ❌ | ✅ 5 × ±3° | **DriftForge** | 5× runtime cost for own generator only | Medium | **change** — estimate, don't search |
| Multi-channel correlation | ❌ | ✅ raw + mid-band | **DriftForge** | fixed weights, untested | Medium | **investigate** (H4) |
| Multiple hypotheses retained | ❌ single arg-max | ⚠️ up to 300, then NMS-decimated | DriftForge | NMS destroys lattice peers (H1) | **Critical** | **change** |
| Centre-rule behaviour | ❌ none | ⚠️ within `tie_delta` of top score | **DriftForge** | applied to a corrupted candidate set | **High** | **change** |
| Periodicity model | ❌ | ❌ | neither | — | **Critical** | **add** |
| Confidence / abstention | ⚠️ raw score only | ✅ score + PSR + peak-gap | **DriftForge** | must not gate emission | Low | **keep, adapt** |
| Sub-pixel refinement | ❌ integer | ⚠️ implicit via scale bank only | DriftForge marginally | no explicit refinement stage | Medium | **add** |
| Runtime | ~0.1 s/pair (5 ZNCC) | **3.69 s/pair** (50 NCC) | sponsor far faster | 37× slower | Medium | **change** |
| Learned matcher | ❌ | ❌ (loader only) | neither | — | Medium | **investigate** |
| Ensemble / meta-ranking | ❌ | ❌ | neither | — | Low | **investigate** |
| Fresh-machine 2-image CLI | ✅ | ✅ | tie | DF default prints `(x, y)` not `x,y` | Low | **change** (cosmetic) |

## 5. Dependency note

`VERIFIED FACT` — the sponsor stack requires `opencv-python-headless`; DriftForge requires
only `numpy/scipy/Pillow/matplotlib` and **has no OpenCV dependency**. Running the sponsor
baseline for comparison (E5/E8) therefore requires installing OpenCV into the audit
environment. This is a diagnostic-environment change only and must not become a
DriftForge runtime dependency — a pure-SciPy stack is a reproducibility asset on an
unknown grading machine.

## 6. What DriftForge should retain from the sponsor baseline

1. **`INTER_AREA`-equivalent decimation.** DriftForge blurs with σ=4.0 px then uses
   `zoom(order=1)`. σ=4.0 at 1 nm/px is a wider low-pass than a 10× box average and
   discards real mid-band signal. `RECOMMENDATION`: test true area-averaging as the
   template decimator (E-A in `09`).
2. **A wider scale bank.** The sponsor searches ±11%; DriftForge only ±6%. If the hidden
   test's magnification ratio is not exactly 10.000, DriftForge's bank is too narrow.
3. **The bare `x,y` stdout convention.**
4. **`accuracy@5px` + AP as headline metrics.**

## 7. Defensible convergence argument (independent of who judges)

The v2 architecture should be justifiable from four public, verifiable facts, none of
which depend on any individual's preferences:

1. the organiser's own baseline **publicly demonstrates** wrong-periodic-repeat failure;
2. the official test description **pre-announces** a hard periodic case;
3. Applied Materials has **published work** on polarity-insensitive directionality-map
   registration of microscopic images (US20160035076A1) and on sub-pixel design/inspection
   alignment (US20160275672A1) — i.e. the sponsor's own literature points at
   orientation-based evidence plus late sub-pixel refinement;
4. the official scoring weights realistic augmentation, a working inference script and
   reproducibility.

An architecture that (a) models periodicity explicitly, (b) adds an orientation-based
evidence channel, (c) resolves ties by the official centre rule over a properly
constructed equivalence set, and (d) refines to sub-pixel **only after** cell selection,
is defensible on all four counts simultaneously.
