# 04 — Baseline Reproduction (E1)

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro, 10.0.26200 SP0 (`Windows-11-10.0.26200-SP0`) |
| Python | 3.12.10 (tags/v3.12.10:0cc8128, MSC v.1943 64-bit AMD64) |
| CPU | Intel64 Family 6 Model 191 Stepping 5 (Raptor Lake refresh), `os.cpu_count()` = 8 |
| GPU | none used — the locator is pure NumPy/SciPy CPU |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.1 |
| opencv | **not installed** (not a DriftForge dependency) |
| Working tree | `C:\Users\SEC\Desktop\LatticeRank\DriftForge` (extracted from `DriftForge_v1.0.0.zip`) |

## 2. Command

```bash
python scripts/audit_manifest.py manifests/validation.jsonl \
  --limit 20 --search-supersample 2 --tolerance 5 \
  --output reports/phase1/repro_validation_release_gate.json
```

Wall-clock: ~4 min including on-the-fly rendering of 20 pairs at `supersample=2`.
The packaged `reports/validation_release_gate.json` was **not** overwritten.

## 3. Result — reproduced exactly

| Metric | Packaged | Reproduced | Match |
|---|---:|---:|---|
| Pairs | 20 | 20 | ✅ |
| Median error | 0.6002004103495411 | 0.6002004103495411 | ✅ **bit-identical** |
| P95 error | 559.0049831622375 | 559.0049831622375 | ✅ **bit-identical** |
| Max error | 614.3717142877948 | 614.3717142877948 | ✅ **bit-identical** |
| Accuracy ≤1 px | 0.70 | 0.70 | ✅ |
| Accuracy ≤2 px | 0.80 | 0.80 | ✅ |
| Accuracy ≤5 px | 0.80 | 0.80 | ✅ |
| Accuracy ≤10 px | 0.80 | 0.80 | ✅ |
| Selective coverage | 0.75 | 0.75 | ✅ |
| Accepted-incorrect @5 px | 0 | 0 | ✅ |
| Mean runtime | 4.974 s/pair | **3.854 s/pair** | ❌ machine-dependent |

`MEASURED RESULT` — **all accuracy metrics reproduce bit-identically.** The generator and
locator are deterministic as claimed; no hidden RNG, no platform-dependent numerics
affecting the result. Determinism is a genuine asset and must be preserved in v2.

Runtime differs because it was measured on different hardware. This is expected and is
**not** a reproduction failure. No stop condition is triggered.

## 4. Two discrepancies worth recording

### 4.1 The packaged report was not produced by the shipped script

`CODE OBSERVATION` — `reports/validation_release_gate.json` contains the key
`accepted_incorrect_at_5px` and has no `tolerance_px` field. The shipped
`scripts/audit_manifest.py:61-62` writes `accepted_incorrect_at_tolerance` **and**
`tolerance_px`. The packaged artifact therefore came from an **earlier, unshipped version**
of the script.

Impact: the numbers are confirmed correct, but the released repository cannot regenerate
its own headline artifact byte-for-byte. For a competition judged partly on
reproducibility this is a cheap and worthwhile fix (regenerate the report with the shipped
script, or restore the key name). Tracked as **R8**.

### 4.2 The "0.60 px median" is a quantisation floor, not sub-pixel skill

`MEASURED RESULT` — every prediction returned by `locate()` lies on the half-integer grid:
candidate centres are `left + (w−1)/2` with `left` an integer and `w` an integer template
width (`baseline.py:140-143`). Verified across all 20 pairs.

Median error over the 16 correct pairs is **0.52 px** — consistent with the ~0.4 px RMS you
would expect from rounding a uniformly-distributed true centre to a half-integer grid.
DriftForge v1 has **no sub-pixel refinement**; its precision is set by pixel quantisation
plus the coarseness of the 5-point scale bank.

**Consequence for Phase 2 targets:** a genuine sub-pixel stage should reduce the median
from ~0.52 px to ~0.1–0.2 px. But this is worth far less than eliminating the 61–614 px
failures, and must be sequenced after cell selection (see `10`).

## 5. Runtime profile (E7 / H3)

Measured over 3 validation pairs with per-stage instrumentation
(`reports/phase1/diagnostics/runtime_profile.py`):

| Stage | s/pair | % of total |
|---|---:|---:|
| `_ncc_valid` on mid-band channel (25 calls) | 1.451 | 40.0% |
| `_ncc_valid` on raw channel (25 calls) | 1.439 | 39.6% |
| `_template_from_reference` (25 calls) | 0.390 | 10.7% |
| `_peaks` NMS + top-k (25 calls) | 0.237 | 6.5% |
| Search preprocessing (once) | 0.070 | 1.9% |
| PSR pooling / selection | 0.027 | 0.8% |
| template mid-band (25 calls) | 0.015 | 0.4% |
| **Total instrumented** | **3.630** | |
| `locate()` end-to-end (independent check) | 3.687 | |

`MEASURED RESULT` — **79.6% of runtime is the 50 `_ncc_valid` calls**, at **57.8 ms each**.
The 25-variant transform bank (5 scales × 5 rotations) is the entire cost driver;
everything else is noise.

**Budget arithmetic for the ≤1 s/pair target.** At 57.8 ms per FFT-based NCC call, a 1 s
budget affords ~13 calls after overhead. Options:

| Option | NCC calls | Est. s/pair | Note |
|---|---:|---:|---|
| Current (5 scales × 5 rot × 2 ch) | 50 | 3.69 | baseline |
| Estimate rotation once, search 3 scales × 2 ch | 6 | ~0.55 | needs a transform estimator (Expert D) |
| Estimate rotation+scale, 1 variant × 2 ch, + local refine | 2 | ~0.25 | leaves headroom for lattice + reranker |
| Coarse-to-fine: 1 coarse pass + local refinement | ~4 | ~0.35 | |

`RECOMMENDATION` — the ≤1 s/pair target is **comfortably attainable**, but only by replacing
the brute-force transform bank with an estimator. The freed budget (~3 s) is exactly what
pays for lattice expansion, an orientation channel and a candidate reranker. This is the
cost-benefit basis for the v2 architecture in `10`.

## 6. Reproducibility verdict

✅ **No stop condition triggered.** Metrics reproduce, no manifest/test leakage found
(`03` §2), organiser requirements unchanged in substance. Phase 1 may proceed.
