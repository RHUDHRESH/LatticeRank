# 02 — Target-dependency audit

**Question.** Does any code path place *distinctive physical structure* using
the target centre, the Reference crop boundary, the label coordinate, or the
target mask?

**Method.** Exhaustive grep over `driftforge/` for `target_c`, `gt_x`, `gt_y`,
`nominal_`, `ref_origin`, `target_mask`, `centroid`, followed by reading every
hit in context. Scope was held to the placement question; this is not a repeat
of the Phase-1 physics audit.

**Result.** One confirmed leak — the signature, fixed under E0. Every other hit
is legitimate label or rendering geometry. **No second target-correlated
landmark was found**, so no stop condition was triggered.

---

## Classification

Legend: **L** legitimate label/rendering geometry · **H** harmless ·
**S** suspicious · **C** confirmed leakage

| # | Location | Code | Class | Reasoning |
|---|---|---|---|---|
| 1 | `scene.py` `_apply_structural_defects` (v1) | `sx = spec.target_cx_nm + U(-260, 260)` | **C** | The Phase-1 headline finding. A unique high-contrast blob pinned inside the Reference FOV in 100% of non-ambiguous samples. **Fixed under E0.** |
| 2 | `config.py:32-33` | `target_cx_nm`, `target_cy_nm` fields | **L** | The `SceneSpec` must record where the target is. Storage, not placement. |
| 3 | `scene.py:106-107` | `target_cx_nm=cx, target_cy_nm=cy` | **L** | Assigns the target that `make_scene_spec` drew. This *defines* the answer; it does not leak it into pixels. |
| 4 | `scene.py:44-47` (`boundary`) | `cx = clip(kx * mat_period - zone_x + U(-120,120), ...)` | **L** | Chooses **where to look** so the crop intersects a mat seam. It moves the target to the structure, never structure to the target. The seam is a global periodic feature covering the whole world, identical everywhere, so it carries no positional information about this particular target. |
| 5 | `scene.py:48-54` (`ambiguous`) | `_nearest_equivalent_to_center(cx, pitch_x)` | **L** | Implements the official nearest-Search-centre tie rule by snapping the target to a phase-equivalent site. Modifies the label, not the rendered scene. `ambiguous` also renders **no** defects (`defect_density == 0`). |
| 6 | `generator.py:81-83` | `ref_origin = target_c - REFERENCE_FOV/2` | **L** | The Reference **is** the crop centred on the target. This is the problem statement, not leakage. |
| 7 | `generator.py:88-89` | `nominal_x = target_cx_nm / (WORLD_FOV/OUTPUT_SIZE)` | **L** | Nanometre → pixel conversion of the target for label construction. |
| 8 | `generator.py:44-52, 91` | `_target_mask(...)` soft box at the target | **L** | Label-carrier only. See the mask-isolation check below. |
| 9 | `generator.py:54-60, 100` | `_mask_centroid(target_mask)` | **L** | Recovers the label after the Search warp. Reads the mask, writes nothing to the image. |
| 10 | `sem.py:153-159` `acquire_search` | `mask = area_downsample(target_mask_super, ...)`; `out, mask = geometric_warp(out, spec, rng, mask=mask)` | **L** | The mask rides through the identical geometry so the label tracks the distortion. **Verified isolated** — see below. |
| 11 | `ml.py:29-31` | `target_xy`, `target_heatmap_stride4` | **L** | Training labels, exposed deliberately to the model-facing API. Not part of image synthesis. |
| 12 | `scene.py:214` | `spec.target_c*` inside a comment | **H** | Documents the removed leak. No runtime effect. |

## The mask-isolation check

Hit #10 deserved direct verification, because a target mask that reached the
image would be a perfect leak. Reading `sem.py:153-159`:

```python
out  = edge_brighten(physical_super, ...)
out  = apply_psf(out, ...)
out  = area_downsample(out, supersample)
mask = area_downsample(target_mask_super, supersample)
out, mask = geometric_warp(out, spec, rng, mask=mask)
out  = photometric_capture(out, spec, rng)
return out, mask
```

`mask` is warped **alongside** `out` and returned **beside** it. It is never
added to, multiplied into, or otherwise combined with the image, and the final
`photometric_capture` sees only `out`. The mask exists purely so the label can
be re-measured after warping. Confirmed legitimate.

## Why the `boundary` profile is not leakage

Hit #4 is the one that most resembles a leak and is worth stating plainly. The
`boundary` profile positions the target so the Reference crop straddles a
mat/periphery seam. That *is* a target-conditioned decision — but the direction
of causation is the opposite of a leak. The seam is a world-wide periodic
structure (`mat_period_nm` ≈ 2200–3600 nm) that exists identically across the
entire 10 µm field whether or not a target is chosen. Placing the target on one
adds no unique, locally-identifying mark. Phase 1 corroborates this
empirically: 4 of the 6 pairs that survived signature removal were `boundary`
cases, i.e. they were being solved by genuine large-scale context.

The 400-pair leak-free baseline settles it at scale: `boundary` is the
**best-performing profile (45.0% at 5 px, vs 34.5% standard and 28.0% hard)**.
If seam placement leaked the answer, boundary crops would have collapsed along
with the signature; instead they are the only profile that holds up. Mat seams
are legitimate context, and they are the strongest genuine cue the generator
provides.

By contrast, the E0 signature was a *unique* object with no duplicate anywhere
else in the world, guaranteed inside the Reference. That distinction — global
repeating structure versus a unique object at the answer — is the line between
legitimate context and leakage.

## Verdict

- Confirmed leakage: **1** (the signature), fixed under E0.
- Suspicious requiring change: **0**.
- No stop condition triggered; Phase 2A proceeds to E1.

Remaining target references are exactly what a cross-scale localisation
generator must contain: a definition of where the target is, a crop taken
there, and a label measured through the capture geometry. None of them writes
a distinguishing mark into the rendered scene.
