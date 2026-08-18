# 03 — DriftForge v1.0.0 Actual Code Architecture

Confirmed against source, not against `README.md`. Every claim below cites a file and line.
Total package: **1,671 lines** across 8 modules and 7 scripts.

> **Provenance note.** The working directory `C:\Users\SEC\Desktop\LatticeRank` was empty at
> the start of this audit. The authoritative project was recovered from
> `C:\Users\SEC\Downloads\DriftForge_v1.0.0.zip` (11.5 MB, 62 entries) and extracted to
> `LatticeRank\DriftForge\`. No prior working tree, git history or uncommitted changes
> exist. This is not a git repository — **R9: the official requirement is a public GitHub
> repo with reproducible history, and none exists yet.**

---

## 1. End-to-end data path

```
manifest record {id, split, seed, scene_id, architecture, profile}
        |                                    splits.py:42-51
        v
generate_sample(seed, architecture, profile, search_supersample)      generator.py:63
        |
        |-- master rng = default_rng(seed)                            generator.py:72
        |     `-- architecture coin-flip if not given                 generator.py:73
        |
        |-- scene = make_scene_spec(seed*17+3, arch, profile)         generator.py:74 -> scene.py:19
        |     `-- rng = default_rng(seed*17+3)
        |         preset = {dram,finfet}_{dense(p=.65),open}          scene.py:24
        |         pitch/width/contact ~ U(preset ranges)              scene.py:26-27,65-67
        |         target (cx,cy) ~ U(margin, 10000-margin)            scene.py:31-33
        |         profile=='boundary' -> cx snapped to a mat seam     scene.py:39-42
        |         profile=='ambiguous'-> (cx,cy) snapped to the       scene.py:43-49
        |                                phase-equivalent nearest the
        |                                world centre  [centre rule]
        |         defect_density = 0 if ambiguous else U(0.4,1.4)     scene.py:52
        |
        |-- ref_rng    = default_rng(seed*7_919   + 101)              generator.py:76
        |-- search_rng = default_rng(seed*1_000_003 + 313)            generator.py:77
        |     `-- INDEPENDENT streams: satisfies official req. #6
        |
        |-- ref_acq    = sample_acquisition(ref_rng,    profile, reference=True)   sem.py:9
        |-- search_acq = sample_acquisition(search_rng, profile, reference=False)
        |
        |== REFERENCE BRANCH ==============================================
        |   origin = (target_cx - 500, target_cy - 500) nm             generator.py:81-82
        |   physical_ref = render_scene(scene, origin, fov=1000nm, 1000px)  scene.py:115
        |   reference = acquire_reference(physical_ref, 1.0 nm/px, ...)     sem.py:146
        |        edge_brighten -> apply_psf -> geometric_warp -> photometric_capture
        |
        |== SEARCH BRANCH =================================================
        |   super_size = 1000 * supersample            (default 2 -> 2000)
        |   physical_search = render_scene(scene, (0,0), fov=10000nm, super_size)
        |   nominal (x,y) = target_c/10                                generator.py:88-89
        |   mask_super = soft box, 100px*supersample wide              generator.py:44-51
        |   search, target_mask = acquire_search(...)                  sem.py:153
        |        edge_brighten -> apply_psf -> area_downsample(2)
        |        -> geometric_warp(img AND mask, shared row_jitter)    sem.py:158
        |        -> photometric_capture
        |   gt = centroid(warped target_mask)                          generator.py:100
        |
        v
Sample{reference uint8, search uint8, gt_x, gt_y, scene, ref_acq, search_acq}
        |
        v
locate(reference, search)                                             baseline.py:94
        v
LocateResult{x, y, score, peak_gap, psr, rotation, scale, accepted, candidates[:25]}
        v
scripts/inference.py -> "(x, y)"                                      inference.py:31
```

### 1.1 Stage table

| Stage | File/function | Inputs | Outputs | Randomness | Assumption | Physical meaning | Hidden-test risk |
|---|---|---|---|---|---|---|---|
| Scene spec | `scene.make_scene_spec` | seed, arch, profile | `SceneSpec` (22 fields) | `default_rng(seed*17+3)` | one pitch pair per scene; whole world is one preset | die layout parameters | **High** — pitch range too narrow (§2.5 of `02`) |
| Scene render | `scene.render_scene` | spec, origin, fov, px | float32 image | none (analytic) | geometry is a closed-form function of world nm | layout at any magnification | Medium — hard binary thresholds, no geometric AA |
| Defects | `scene._apply_structural_defects` | img, spec | in-place | `default_rng(seed*65537+19)` and `*104729+43` | 10·density random defects **+ 1 target-correlated signature** | particles/scratches/bridges | **Critical** — leak, see H2 |
| Acquisition sample | `sem.sample_acquisition` | rng, profile, is_ref | `AcquisitionSpec` (20 fields) | per-branch rng | ref is a careful capture, search a fast wide scan | tool settings | Medium |
| Edge brighten | `sem.edge_brighten` | img, px_nm, strength, σ | img | none | SEM edge contrast ∝ ‖∇(G_σ*I)‖ | secondary-electron edge effect | Low |
| PSF | `sem.apply_psf` | img, spec | img | none | separable anisotropic Gaussian | beam spot + astigmatism | Low |
| Downsample | `sem.area_downsample` | img, factor | img | none | box average | detector pixel integration | Low |
| Geometric warp | `sem.geometric_warp` | img, spec, rng, mask | img, mask | `rng.normal` row jitter | affine ∘ (radial ∘ shear/jitter) | stage rotation/scale + raster drift | Medium |
| Photometric | `sem.photometric_capture` | img, spec, rng | img | charging field, Poisson, read, hot px | vignette→gain/offset→γ→Poisson→read→hot | detector chain | Medium |
| Label | `generator._mask_centroid` | warped mask | (x, y) | none | centroid of warped soft box = matching-region centre | true target centre | Low |
| Locate | `baseline.locate` | ref, search | `LocateResult` | none (deterministic) | ±3° rotation, ±6% scale bracket the truth | classical matcher | **Critical** — see §3 |

## 2. Split contract (verified)

`splits.py:12-17` — seed bases `train=100_000`, `validation=300_000`, `test=500_000`,
`stress=700_000`; counts 3000/600/1000/500 (`make_manifests.py:15`).

- `VERIFIED FACT` — seed ranges are disjoint (100000–102999, 300000–300599,
  500000–500999, 700000–700499).
- `VERIFIED FACT` — `assert_disjoint()` (`splits.py:77`) enforces scene-ID disjointness at
  manifest-build time; `make_manifests.py:29` calls it.
- `CODE OBSERVATION` — scene RNG is `seed*17+3`, injective over the disjoint seed ranges,
  so no scene-parameter collision across splits. **No leakage found.**
- `VERIFIED FACT` — `manifests/summary.json` and the four `.jsonl` files ship pre-built and
  were **not** regenerated during this audit.

## 3. The locator, exactly as implemented (`baseline.py`)

```python
locate(reference, search,
       rotations=(-3.0, -1.5, 0.0, 1.5, 3.0),      # 5
       scales=(0.94, 0.97, 1.0, 1.03, 1.06),       # 5
       score_threshold=0.20, psr_threshold=3.0,
       tie_delta=0.015, candidates_per_variant=12)
```

**Per variant (25 total):**
1. `_template_from_reference` — `gaussian_filter(σ=4.0)` → `zoom(0.1*scale, order=1)` →
   optional `rotate` → `_robust_contrast` (0.5/99.5 percentile stretch). `baseline.py:50-59`
2. mid-band template = `G(σ=3) − G(σ=15)`. `baseline.py:128-131`
3. fused response = `0.25·NCC(search_raw, tmpl) + 0.75·NCC(search_mid, tmpl_mid)`. `:135-138`
4. `_peaks(response, template.shape, 12)` — `maximum_filter(size=round(0.30·min(tmpl)))`
   ≈ **30 px**, top-12. `:79-91`
5. candidate centre = `(left + (w−1)/2, top + (h−1)/2)`. `:140-143`

**Selection:**
- sort all ≤300 candidates by score; `best` = top. `:147-148`
- `rivals` = candidates ≥ 35 px from `best`; `peak_gap = best.score − rivals[0].score`. `:153-159`
- `tied` = all candidates with `score ≥ best.score − 0.015`; `chosen` = the tied candidate
  nearest the Search centre. `:165-166`
- `psr = (chosen.score − median) / (1.4826·MAD)` over a sub-sampled pool of **all 25**
  response maps. `:170-173`
- `accepted = score ≥ 0.20 AND psr ≥ 3.0 AND peak_gap ≥ 0.020`. `:176-180`

### 3.1 Defects identified by reading the locator

| # | Defect | Location | Consequence |
|---|---|---|---|
| D1 | NMS footprint ≈30 px ≫ lattice pitch (2.8–17 px) | `:82` | 4.4–67 genuine lattice hypotheses collapse to one noise-selected representative — **H1** |
| D2 | Centre rule applied to the NMS-decimated set | `:165-166` | the official tie rule operates on a corrupted, noise-selected hypothesis set |
| D3 | `tie_delta=0.015` is an **absolute** NCC margin | `:165` | not calibrated to per-image score scale or noise; too wide in clean images, too narrow in noisy ones |
| D4 | Scores compared **across** transform variants | `:147` | different scales/rotations produce differently-biased NCC distributions; the arg-max is partly a transform-selection artifact |
| D5 | `psr` mixes 25 heterogeneous response maps | `:170-173` | the "sidelobe" statistic is not a per-surface PSR; comment concedes this |
| D6 | `separation = 0.35*100.0` hard-codes the 100 px template | `:153` | breaks silently if the scale bank or magnification ratio changes |
| D7 | Template built by `gaussian_filter(σ=4.0)` then `zoom(order=1)` | `:55-56` | σ=4.0 px at 1 nm/px is a **wider** low-pass than a true 10× box average; attenuates the mid-band that carries 75% of the fused score |
| D8 | Scale bank ±6% vs sponsor ±11% | `:98` | too narrow if the true magnification ratio is not ≈10.000 |
| D9 | `_peaks` uses `score == maximum_filter(score)` | `:83` | on flat plateaus every plateau pixel is marked a maximum; benign here but fragile |
| D10 | No sub-pixel refinement stage | — | precision is capped by the scale-bank quantisation |

### 3.2 What the locator does *right* (retain in v2)

- Correct **whole-reference decimation** — never crops or pastes the Reference
  (`:51-54`); this is the physically correct interpretation of "shrunk 10×".
- **Two heterogeneous channels** — a genuine ensemble seed (H4 tests whether the fixed
  0.25/0.75 weighting is the right way to use them).
- **Multiple hypotheses retained** at all (`candidates[:25]` returned) — the sponsor
  baseline keeps only one. This is the foundation a v2 reranker needs.
- **A real confidence triple** (`score`, `psr`, `peak_gap`) with an explicit abstention
  decision, and the honesty to return the best coordinate anyway.
- **`x = column, y = row`** convention documented and correct (`:107-109`).
- **Deterministic** — no RNG in the locator; verified bit-identical across runs (`04`).

## 4. ML scaffolding (`ml.py`) — status

`OnTheFlyPairDataset` renders manifest rows lazily and returns `reference`,
`reference_at_search_scale` (100×100), `search`, `target_xy`, `target_xy_normalized`
and a stride-4 Gaussian heatmap (250×250, σ=2). Framework-neutral, optional Torch.

- `CODE OBSERVATION` — **no model, no training loop, and no trained weights exist.**
  `TRAINING.md` describes a recommended Siamese/correlation network; none is implemented.
  Any claim that DriftForge "has ML" is unsupported by the code.
- `CODE OBSERVATION` — `ml.py:24` builds `reference_at_search_scale` with
  `zoom(0.1, order=1, prefilter=False)` and **no anti-aliasing pre-filter at all** — unlike
  `baseline.py:55` which applies σ=4.0 first. The dataset's model input is therefore
  aliased differently from the baseline's template. Any Phase-2 learned reranker trained on
  this array would learn a decimation artifact the classical path does not share.
  **This is a real bug for Phase 2** and is tracked as R10.
