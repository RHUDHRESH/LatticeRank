# 06 — Rendering-fidelity check (E1b): supersample 2 vs 4

**Recommendation: keep `search_supersample=2` for Phase 2.**
No material systematic shift was found, and 4 costs 6.5× generation time and
3.9× peak memory. The one result worth carrying forward is not the accuracy
delta — it is what the per-pair flips reveal about *how* the locator fails.

Artifacts: `supersample_comparison.json`, `supersample_comparison.csv`
Script: `reports/phase2a/diagnostics/supersample_check.py`
Design: 50 matched records, identical seeds, identical locator, **only** the
Search-side render oversampling differs.

---

## 1. Results

| Metric | ss = 2 | ss = 4 | Δ |
|---|---:|---:|---:|
| Accuracy ≤5 px | 34.0% | 38.0% | +4.0 pp |
| Median error | 218.34 px | 220.08 px | +1.7 px |
| P95 error | 714.52 px | 746.74 px | +32.2 px |
| Catastrophic rate | 66.0% | 60.0% | −6.0 pp |
| Alias fraction of catastrophic | 78.8% | 76.7% | −2.1 pp |
| Coverage | 22.0% | 24.0% | +2.0 pp |

**Image statistics** — near-identical, which is the important negative result:

| Statistic | ss = 2 | ss = 4 | Δ |
|---|---:|---:|---:|
| Intensity σ | 0.1930 | 0.1877 | −2.7% |
| Mean gradient | 0.1033 | 0.1011 | −2.1% |
| High-frequency energy fraction | 0.6458 | 0.6504 | +0.7% |

**Cost**

| | ss = 2 | ss = 4 | Ratio |
|---|---:|---:|---:|
| Mean generation time | 1.58 s | 10.29 s | **6.5×** |
| Mean peak memory | 193 MB | 748 MB | **3.9×** |
| Mean locate time | 4.42 s | 10.50 s | *see §4* |

## 2. Is the accuracy difference real?

Almost certainly not, at this sample size. +4.0 pp on n=50 is **two samples**.
The Wilson intervals are 34.0% (21.9–48.6) and 38.0% (25.5–52.3) — they overlap
across nearly their entire range. The median moved 1.7 px on a distribution
whose median is ~219 px, i.e. 0.8%.

The pitch appearance is unchanged: mean gradient differs by 2.1% and
high-frequency energy fraction by 0.7%. If 2× were aliasing away real lattice
structure, that is where it would show, and it does not.

## 3. What the paired comparison actually shows

The aggregate numbers hide the interesting result. Pairing each record with
itself across the two settings:

| Paired statistic | Value |
|---|---:|
| Median \|Δerror\| | **0.172 px** |
| Max \|Δerror\| | **895.1 px** |
| Records changing >1 px | 17 / 50 |
| **Catastrophic-status flips** | **5 / 50** |

So 33 of 50 records are essentially untouched (sub-pixel), while 5 flip
between "correct cell" and "hundreds of pixels wrong". The distribution of
change is not a small shift applied to everything — it is nothing at all for
most pairs and a total reversal for a few.

That is the same bimodality seen in the main baseline, and it says something
useful: on knife-edge pairs the choice between the true cell and a periodic
rival is decided by margins so thin that a **render-quality change of ~2% in
gradient energy is enough to flip it**. Those pairs are not being solved; they
are being coin-flipped. It is further evidence that the failure is discrete
candidate *selection* under near-tied evidence, not measurement precision.

## 4. Runtime caveat

The ss=4 pass ran while the 300-pair baseline re-run was competing for the same
8 cores. `mean_locate_s` (4.42 → 10.50 s) is therefore **contaminated** and must
not be read as a fidelity effect — `locate()` receives a 1000×1000 image in both
cases and cannot depend on the render oversampling. Generation time (6.5×) and
peak memory (3.9×) are structural and trustworthy: ss=4 renders a 4000×4000
world instead of 2000×2000.

## 5. Recommendation

**Retain `search_supersample=2`.** Justification:

1. No material systematic movement in localisation (median 0.8%, well inside
   noise) or in image statistics (≤2.7% on every measure).
2. The accuracy delta is two samples on n=50 and is not separable from noise.
3. The cost is 6.5× generation and 3.9× memory, which directly limits how many
   pairs future phases can afford — and Phase 1's central lesson was that
   sample size, not render quality, was the binding constraint on conclusions.

**Do not treat this as settled.** A ±4 pp effect is exactly the size this
experiment cannot resolve. If Phase 2 wants a defensible answer, the matched
comparison needs ~300 pairs, which at ss=4 is roughly an hour of generation.
That is worth paying **once**, before any model is fitted, and not before.

Defaults were not changed. `search_supersample=2` remains the production
setting, as shipped.
