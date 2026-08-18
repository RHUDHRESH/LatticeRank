# 05 — Multi-channel pivot + candidate meta-ranker

**Verdict: the pivot's *first half* works and its *second half* does not.**
Candidate recall 65.0% → **92.5%**. Final accuracy 31.0% → **37.5%**.
Per the pre-agreed decision rule ("still ~35–40% → stop spending time on HGB"),
**tabular ranking is exhausted. A local structural descriptor is required.**

Artifacts: `experiments/war45/{pivot_recall.json, ranker_results.json}`

---

## 1. What the pivot fixed

Three cheap channels (raw ZNCC, mid-band, polarity-invariant directionality),
union of local maxima, adaptive harvest `S_i ≥ S_max − Δ` per channel:

| Δ | recall@1 | recall@2 | recall@5 | median candidates |
|---:|---:|---:|---:|---:|
| 0.02 | 56.2% | 62.5% | 63.7% | 13 |
| 0.05 | 71.2% | 78.8% | 78.8% | 28 |
| 0.10 | 83.8% | 90.0% | 90.0% | 120 |
| **0.15** | 87.5% | 92.5% | **92.5%** [85–97] | 546 |
| 0.25 | 88.8% | 92.5% | 92.5% | 4,372 |

vs the old lattice-expansion route: **65.0%** (NMS top-25) and 72.5%
(lattice, 6,573 candidates). DRAM reaches **97.4%**, FinFET 87.8%.

Per-channel, the true site is a local maximum in **100%** (raw), 90% (mid-band)
and **100%** (directionality) of pairs. Directionality has the best rank≤10
(61.3% vs 57.5% for the other two), so the sign-invariant orientation channel
does add independent information — it earns its place.

**The adaptive rule matters more than the channels.** A fixed top-K is wrong at
both ends; on a wallpaper tile hundreds of sites are genuinely equivalent.

## 2. What the ranker did not fix

Training formulation corrected as directed — all ≤5 px candidates kept as
positives (so no near-identical 2 px neighbour is labelled negative), plus
per-scene diverse hard negatives (15 per channel + 10 score-confusions + 10
remote aliases), scene-relative features (percentile, delta-from-best, z-score,
cross-channel agreement), class weighting, and a GT-safe cost cap so the
positive class can never be silently deleted.

| | before | after |
|---|---:|---:|
| Train rows | 205,304 | **9,128** |
| Positives | 227 (0.11%) | 236 (**2.6%**) |
| Oracle ceiling | 85.0% | **92.5%** |
| HGB accuracy ≤5 px | 36.2% | **37.5%** |

`MEASURED RESULT` — a 20× better class balance and 12 scene-relative features
moved final accuracy by **+1.3 points**.

### The diagnostic that settles it

| Rank recall (HGB) | value |
|---|---:|
| top-1 | 37.5% |
| top-5 | 46.2% |
| top-10 | **51.2%** |
| top-30 | 56.2% |
| *candidate-set oracle* | *92.5%* |

You asked for this specifically: if the ranker got the true candidate into the
top-10 in 85–90% of scenes, a cheap second-stage patch matcher on 10 patches
would finish the job. **It does not — top-10 is 51.2%.** A second stage bolted
on here would inherit a 51.2% ceiling, not a 92.5% one.

The ranker converts only ~40% of its available ceiling, and the shortfall is
*not* at the top-1 tie-break; it is spread across the whole ranking.

## 3. Why — and it is not fixable by more tabular features

Top RF features are all mid-band relative measures (`mid_delta` 0.098,
`mid_z` 0.073, `mid_pct` 0.069, `midband` 0.063, `agreement_count` 0.060).
The model is doing the sensible thing with what it has.

But every feature in the table answers *"how strong is this candidate?"*, and
periodic copies are strong **in identical ways** — that is what makes them
copies. Scene-relative normalisation makes strengths comparable across scenes;
it cannot separate two sites that are equally strong within one scene. The
Phase-2A measurement already showed the winner beats the true site by
**0.02 NCC**, and 70.7% of catastrophic failures are sites that are *visually
equivalent* to the truth.

No aggregate of scalar scores distinguishes them. Only a representation that
compares **local structural content** — which contact is missing, how the mat
boundary falls, the specific roughness realisation — can.

## 4. Honest accounting of the tail

Accuracy ≤5 px improved (31.0% → 37.5%) but the tail did **not**:

| | v1 leak-free (400 pairs) | pivot + HGB (80 pairs) |
|---|---:|---:|
| ≤5 px | 31.0% | **37.5%** |
| median | 248.0 px | 245.7 px |
| P95 | 659 px | 760 px |
| >25 px | 69.0% | 62.5% |

Catastrophic rate improves 6.5 points; P95 worsens. When the equivalence set
contains no correct candidate, the nearest-centre rule picks a central wrong
one, which can be further from truth than the raw arg-max was. This is a real
cost of the tie-rule and is not hidden.

## 5. Recommendation

1. **Stop tuning the tabular ranker.** Two formulation fixes bought 1.3 points.
2. **Build the local structural descriptor** — a small patch comparator over the
   ~120 candidates surviving Δ=0.10 (90% recall, 120 candidates: the cheapest
   operating point). This is the one component the evidence now demands.
3. **Keep**: the 3-channel adaptive harvest (the +27.5 point recall win), the
   directionality channel, scene-relative features, hard-negative sampling.
4. **Do not** re-attempt lattice-driven candidate generation; it is measured to
   hurt. Lattice stays a feature.
