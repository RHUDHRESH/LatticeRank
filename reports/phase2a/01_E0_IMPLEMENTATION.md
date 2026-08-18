# 01 — E0: neutralising the target-correlated signature

**Status:** `PASS`
**Option implemented:** 0b (de-correlate), as directed.
**Branch:** `phase2a-leakfree-baseline`

---

## 1. Old behaviour

`driftforge/scene.py`, at the end of `_apply_structural_defects()`:

```python
signature_rng = np.random.default_rng(spec.seed * 104_729 + 43)
sx = spec.target_cx_nm + float(signature_rng.uniform(-260, 260))
sy = spec.target_cy_nm + float(signature_rng.uniform(-260, 260))
radius = float(signature_rng.uniform(72, 135))
```

The position was *defined* as the target centre plus a bounded offset. Three
properties made this a severe leak rather than a cosmetic one:

1. **It was always visible in the Reference.** `260 + 135 = 395 < 500` nm, half
   the 1000 nm Reference FOV, so the cluster lay fully inside the Reference in
   **100%** of samples with `defect_density > 0` — every profile except
   `ambiguous`.
2. **It was unique.** Radius 72–135 nm exceeded every random defect (18–85 nm),
   and no random kind has its dark-core-plus-bright-rim morphology. Only
   `round(10 · density)` = 4–14 random defects exist across the whole 10 µm
   world, so this blob was effectively a fingerprint.
3. **The locator was tuned to see it.** At Search scale it is a 14–27 px
   high-contrast blob, squarely in the σ=3 − σ=15 mid-band channel that carries
   **75%** of the fused NCC score.

The generator marked its own answer, and the locator weighted the exact
frequency band that saw the mark.

## 2. New behaviour

Placement is now drawn uniformly in **world** coordinates from a stream seeded
only by scene identity:

```python
def signature_placement(spec) -> tuple[float, float, float, np.random.Generator]:
    rng = np.random.default_rng(spec.seed * 104_729 + 43)
    margin = SIGNATURE_RADIUS_NM[1]
    sx = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    sy = float(rng.uniform(margin, WORLD_FOV_NM - margin))
    radius = float(rng.uniform(*SIGNATURE_RADIUS_NM))
    return sx, sy, radius, rng
```

The defect can now fall inside the Reference by chance, outside it, near the
Search boundary, or nowhere near the target — all of which are correct
behaviour for a process defect.

**Margin choice.** `margin = 135 nm`, the maximum signature radius, so the
cluster renders whole wherever it lands. At Search scale that is 13.5 px from
the image edge, which still allows genuine near-boundary cases.

## 3. What was deliberately preserved

Per the Phase-2A brief, the flaw is location leakage, not the existence of a
process defect:

| Property | Status |
|---|---|
| Radius distribution `U(72, 135)` nm | unchanged |
| Dark-core / bright-rim morphology | unchanged |
| Rendering semantics (`missing` core, `rim` annulus) | unchanged |
| Deterministic generation | unchanged |
| Shared latent structure across Reference/Search | unchanged |
| Independent acquisition noise on top | unchanged |
| `ambiguous` profile omits it (`defect_density == 0`) | unchanged |
| RNG stream, seed formula, **draw order and count** | unchanged |

The draw order matters: `uniform(a, b)` consumes one underlying double
regardless of bounds, so `radius` and both brightness draws take the **same
stream values as v1**. Only the two position draws are rescaled. This keeps the
change to the smallest possible surface.

## 4. Why 0b rather than 0a or 0c

Directed by the brief, and independently the right call. **0a (delete)** would
have removed a legitimate defect class and with it the "sometimes a genuine
unique landmark exists" case, which is real on wafers. **0c (probabilistic)**
leaves a residual, budget-dependent leak that is far harder to audit — you must
argue about how much leakage is acceptable rather than demonstrate there is
none. 0b removes the coupling entirely while keeping every defensible physical
property, so the audit reduces to a clean independence claim.

## 5. Files and functions changed

| File | Change |
|---|---|
| `driftforge/scene.py` | Added `SIGNATURE_RADIUS_NM`; added `signature_placement()`; rewrote the placement block in `_apply_structural_defects()` |
| `tests/test_phase2a_leakfree.py` | New — 12 tests (independence, determinism, sharing, splits, locator freeze) |
| `tests/test_generator.py` | `test_classical_baseline_smoke` no longer asserts sub-2 px accuracy (that threshold was leak-dependent); see §7 |
| `scripts/audit_manifest.py` | R8: versioned, self-describing report schema |

**Not modified:** `driftforge/baseline.py` (locator frozen, sha256
`15f591b375c8d9804daada13af8b6539a8931b45cc52439105c606bca72dd0c8`, pinned by
test), `sem.py`, `config.py`, `splits.py`, `ml.py`, and every file under
`manifests/`.

Placement was extracted into `signature_placement()` deliberately: the
independence diagnostic and the tests call the **same function the renderer
calls**, so they cannot drift from the shipped behaviour the way a
reimplementation would.

## 6. Proof the location is independent of the target

Evidence: [`signature_independence.json`](signature_independence.json) (aggregates)
and [`signature_independence.csv`](signature_independence.csv) (one row per scene:
seed, architecture, profile, signature x/y/radius, target x/y, dx/dy, distance,
inside/overlaps Reference).

`reports/phase2a/diagnostics/check_signature_independence.py`, 1,200 scene
records (600 DRAM / 600 FinFET, profiles `standard`/`hard`/`boundary`/
`ambiguous`; 900 render a signature — the 300 `ambiguous` records have
`defect_density == 0` by design).

| Statistic | v1 | Phase 2A | Geometric expectation |
|---|---:|---:|---:|
| Signature centre inside Reference | **100%** | **0.89%** (8/900) | 1.06% |
| Any overlap with Reference | 100% | 1.22% | 1.54% |
| `corr(sig_x, target_x)` | ≈1 by construction | **−0.0443** | 0 |
| `corr(sig_y, target_y)` | ≈1 by construction | **+0.0033** | 0 |
| Median target distance | ≤ 368 nm | **4,827 nm** | — |
| Distance p05 / p95 | — | 1,243 / 8,898 nm | — |

The observed inside-Reference rate tracks the probability predicted by geometry
alone (8 observed against 9.5 expected — well inside Poisson noise), which is
the correct target. The brief is explicit that the goal is not "zero inside
Reference": a defect that *never* landed in the Reference would itself be a
target-dependent rule.

**Interventional null check.** Correlation near zero can happen by luck; this
cannot. For 200 scenes, the target was moved to each of four widely separated
positions with the scene otherwise identical:

```
n_scenes_tested            200
target positions per scene 4      (800 relocations)
n_signature_moved          0
max_signature_shift_nm     0.0
passed                     True
```

The signature does not move when the answer moves. A source-level test also
asserts that `signature_placement`'s body contains no reference to
`target_cx_nm`, `target_cy_nm`, `gt_x`, `gt_y` or `nominal_*`, so a regression
fails loudly rather than silently.

## 7. Corroboration from Phase 1

Phase 1 simulated this change by monkeypatching the signature out
(`diagnostics/ablation.py`) and predicted the result. Phase 2A implements it
for real. On the **same 20 validation records**:

| | median err | acc ≤5 px | catastrophic >25 px |
|---|---:|---:|---:|
| Phase-1 ablation (predicted) | 273.01 px | 30% | 14 |
| **Phase-2A actual** | **255.31 px** | **30%** | **14** |

The independent implementation lands on Phase 1's predicted numbers. That is
strong evidence E0 removed the intended mechanism and only that mechanism.

## 8. Determinism proof

Determinism is preserved. Every determinism test was run twice, and
`test_render_is_deterministic_across_repeats` itself loops twice internally.

```
same Phase-2A code + same manifest record + same seed
  => identical Reference    (np.array_equal, True)
  => identical Search       (np.array_equal, True)
  => identical label        (gt_x, gt_y equal)
  => identical signature    (sx, sy, radius equal)
```

Old and new images **do differ** for non-ambiguous scenes, as expected — the
signature moved. No byte-identity with v1 is claimed, and none should be. The
`ambiguous` profile is unaffected by construction, since it renders no
structural defects at all.
