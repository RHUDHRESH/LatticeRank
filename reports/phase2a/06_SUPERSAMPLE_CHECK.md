# 06 — Rendering-fidelity check (E1b): supersample 2 vs 4

**Recommendation: keep `search_supersample = 2` for Phase 2.**
Two independent 50-pair runs disagree on the *sign* of the accuracy difference,
which settles it as noise, while `4` costs ~3.9× peak memory in both.

Artifacts: `supersample_comparison.json`, `supersample_comparison.csv`
Script: `reports/phase2a/diagnostics/supersample_check.py`
Design: 50 matched records, identical seeds, identical locator, **only** the
Search-side render oversampling differs.

---

## 1. Two runs, opposite signs

| | Run A (`validation`, 50 pairs) | Run B (`validation_phase2a_400`, 50 pairs) |
|---|---:|---:|
| Acc ≤5 px, ss=2 | 34.0% | 26.0% |
| Acc ≤5 px, ss=4 | 38.0% | 22.0% |
| **Δ (ss4 − ss2)** | **+4.0 pp** | **−4.0 pp** |
| Catastrophic, ss=2 | 66.0% | 74.0% |
| Catastrophic, ss=4 | 60.0% | 78.0% |
| Δ | −6.0 pp | +4.0 pp |

Run A favoured `ss=4`; Run B, on a disjoint manifest, favoured `ss=2` by the
same margin. **±4 pp on n=50 is two samples.** Wilson intervals overlap heavily
in both runs (Run B: 26.0% [15.6–40.0] vs 22.0% [12.5–35.6]).

This is a cleaner negative result than either run alone. A single run showing
+4 pp would have been tempting to read as "higher fidelity helps"; two runs with
opposite signs show it is sampling noise.

## 2. Run B detail (disjoint manifest, authoritative)

| Metric | ss = 2 | ss = 4 | Δ |
|---|---:|---:|---:|
| Accuracy ≤5 px | 26.0% | 22.0% | −4.0 pp |
| Median error | 239.81 px | 239.08 px | −0.7 px |
| P95 error | 726.35 px | 659.80 px | −66.5 px |
| Catastrophic rate | 74.0% | 78.0% | +4.0 pp |
| Alias fraction of catastrophic | 81.1% | 84.6% | +3.5 pp |
| Coverage | 20.0% | 18.0% | −2.0 pp |

**Median error differs by 0.73 px** — the two renderings put the locator in
essentially the same place when it succeeds.

### Image statistics — the important negative result

| Statistic | ss = 2 | ss = 4 | Δ |
|---|---:|---:|---:|
| Intensity σ | 0.1983 | 0.1917 | −3.3% |
| Mean gradient | 0.1117 | 0.1088 | −2.6% |
| High-frequency energy fraction | 0.6528 | 0.6568 | +0.6% |

Doubling the render oversampling changes the Search image's second-order
statistics by under 3.5%. The PSF is applied before decimation in both cases,
so the beam has already band-limited the signal by the time sampling happens —
`ss=2` is not leaving significant aliasing on the table.

### Cost

| | ss = 2 | ss = 4 | Ratio |
|---|---:|---:|---:|
| Mean peak memory | 192 MB | 746 MB | **3.9×** |
| Mean generation time | 2.96 s | 4.26 s | 1.4× *(contended, see §4)* |
| Mean locate time | 8.82 s | 4.01 s | *meaningless, see §4* |

Memory is the reliable cost signal: **3.9×** in both runs (193→748 MB in Run A,
192→746 MB in Run B). At `ss=4` the Search world is a 4000×4000 float32 canvas.

## 3. Per-pair behaviour is what actually matters

Run B paired deltas: **median |Δ| = 0.173 px**, max 338.9 px,
**23/50 pairs moved by >1 px**, and **4 catastrophic flips**.

The distribution is the same bimodality seen everywhere in this phase: when the
locator picks the right cell, render fidelity moves the answer by ~0.2 px; when
it is choosing between near-identical repeats, a sub-pixel change in the input
flips which repeat wins, and the answer moves by hundreds of pixels.

**That is the finding worth carrying forward.** The cell choice is not merely
*wrong* on hard pairs — it is *unstable*, decided by noise-level differences in
the input rather than by evidence. A selection mechanism this sensitive cannot
be fixed by improving the rendering; it has to be replaced by one that reasons
over an explicit candidate set. It also predicts that the sponsor's
`generate_sample_family()` test (one scene, several acquisitions, one ground
truth) would expose high prediction *variance*, and argues for adopting that as
a first-class Phase-2 metric.

## 4. Why the timing columns are not usable

Run B's `ss=2` pass overlapped the tail of another job; its `ss=4` pass ran
alone. That is why `ss=2` shows a *longer* locate time (8.82 s) than `ss=4`
(4.01 s), which is impossible on the merits — both produce identical
1000×1000 images, so `locate()` cannot know which rendering produced them.
Run A had the same contamination in the opposite direction (4.42 s vs 10.50 s).

Both timing columns are contention artifacts and no conclusion here rests on
them. Memory, measured with `tracemalloc` inside the process, is unaffected.

## 5. Recommendation

**Keep `search_supersample = 2` as the default.** No material systematic
accuracy shift across two independent runs, second-order image statistics
within 3.5%, and `ss=4` costs 3.9× memory.

Carry two caveats forward:

1. The `ss=4` path stays available and should be re-checked at n ≥ 300 if a
   Phase-2 result ever looks fidelity-sensitive. Neither run can resolve a
   difference smaller than ~10 pp.
2. This does **not** clear the deeper rendering concern from the Phase-1 physics
   audit (§3.1): at `ss=2` the Search world is rendered at 5 nm/px, so a 28 nm
   FinFET fin pitch gets 5.6 samples per period from hard-thresholded geometry
   with no geometric anti-aliasing. `ss=4` halves that grid but does not change
   the underlying binary-threshold rendering, so it cannot settle the question.
   The proper fix is geometric anti-aliasing in `render_scene`, which is a
   generator change and out of scope for Phase 2A.
