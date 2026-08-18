# DriftForge

DriftForge is an original, deterministic synthetic-data system for SEMICON India 2026 Track 2 / Applied Materials **Drift-Sense**. This is a cross-scale localization problem, not image classification.

Each sample contains:

- `reference`: 1000×1000 grayscale, 1 nm/px, 1,000 nm field of view;
- `search`: 1000×1000 grayscale, 10 nm/px, 10,000 nm field of view;
- label `(x, y)`: centre of the reference field in Search pixels after Search-side geometry and raster distortion;
- full provenance: structural parameters, acquisition parameters, scene ID and seed.

## Why this generator is hard to fool

1. One analytical world-coordinate layout is rendered at both physical scales. The Reference is never resized and pasted into Search.
2. Reference and Search use independent SEM acquisition RNG streams: PSF, edge brightening, dose/Poisson noise, detector noise, charging, hot pixels and contrast are not shared.
3. The label is obtained by warping a target mask through the exact Search geometry—rotation, scale, translation, raster shear/jitter and radial distortion—then taking its centroid.
4. Structural defects are placed uniformly in world coordinates, independent of the target — so a defect may fall inside the Reference by chance, outside it, or nowhere near it. (Before Phase 2A one defect class was positioned relative to the target and leaked the answer; see [reports/phase2a/01_E0_IMPLEMENTATION.md](reports/phase2a/01_E0_IMPLEMENTATION.md).) A separate `ambiguous` profile is deliberately wallpaper-like and obeys the official nearest-Search-centre rule.
5. Train/validation/test/stress scene IDs and seed ranges do not overlap.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Generate data

The included manifests define 5,100 reproducible pairs: 3,000 train, 600 validation, 1,000 private test and 500 stress. They are small by design; materializing all PNGs would require several gigabytes.

```bash
# Materialize the first 30 training pairs.
python scripts/generate_dataset.py \
  --manifest manifests/train.jsonl \
  --limit 30 \
  --output-dir data/train

# Or create a custom balanced DRAM + FinFET set.
python scripts/generate_dataset.py \
  --count 30 \
  --architecture both \
  --profile hard \
  --output-dir data/custom

# Build an unlabeled public test hand-off while retaining labels elsewhere.
python scripts/generate_dataset.py \
  --manifest manifests/test.jsonl \
  --limit 30 \
  --hide-labels \
  --output-dir data/public_test
```

Use `--search-supersample 2` (default) for production. `1` is a faster smoke test; `4` is a higher-quality but memory-heavier audit mode.

## Baseline inference and evaluation

```bash
python scripts/inference.py data/train/reference/train-000000.png data/train/search/train-000000.png
python scripts/inference.py --json data/train/reference/train-000000.png data/train/search/train-000000.png
python scripts/evaluate_baseline.py data/train --output reports/train_baseline.json
python scripts/validate_dataset.py data/train
python scripts/audit_manifest.py manifests/validation.jsonl --limit 20 --output reports/validation_release_gate.json
```

The baseline is multi-scale/multi-rotation NCC with a mid-frequency structural channel, centre-aware tie handling, a peak-gap ambiguity check and an `accepted`/abstention flag. It is an audit baseline, not a promise of zero error on unknown hidden data. `--strict` exits with code 2 when a prediction is unsafe to auto-accept.

### Baseline performance — corrected

The original v1.0.0 20-pair release gate reported 0.60 px median error, 80% accuracy at 5 px, 75% selective coverage and zero accepted-incorrect predictions. **A Phase-1 audit found a target-correlated synthetic landmark in the generator that inflated those numbers**: a unique high-contrast defect was placed at the target centre ±260 nm, which put it inside the Reference field of view in 100% of non-ambiguous samples and effectively marked the answer. Phase 2A removed the leakage by placing that defect uniformly in world coordinates, independent of the target.

Leak-free baseline, same locator unchanged, on **400 pairs from a newly created,
fully disjoint split** (`manifests/validation_phase2a_400.jsonl`):

| Metric | Leak-free (400 pairs) | Contaminated v1 claim (20 pairs) |
|---|---:|---:|
| Accuracy ≤5 px | **31.0%** (95% CI 26.7–35.7) | 80% |
| Median error | **248.0 px** (CI 216.6–281.1) | 0.60 px |
| P95 / P99 | **659.0 / 833.5 px** | — |
| Catastrophic >25 px | **69.0%** | 20% |
| Accepted coverage | 24.0% | 75% |
| Accepted-incorrect @5 px | **26** | 0 |

Selective prediction is **not** a safety net: all 26 accepted-incorrect predictions are
>25 px errors. The "zero accepted-incorrect" figure was an artifact of a 20-pair sample
on leaked data.

Accuracy is identical at 2 px, 5 px and 10 px — the locator is either on the correct
lattice cell or hundreds of pixels away. This is a discrete selection failure, not a
precision problem, so sub-pixel refinement is worth nothing until cell selection is fixed.

By architecture: DRAM **38.0%**, FinFET **24.0%** (non-overlapping intervals — FinFET is
genuinely harder, its pitch reaching 2.8 Search px). By profile: `boundary` **45.0%** —
the *best* profile, because crops straddling a mat/periphery seam carry the genuine
large-scale context that survives when the planted landmark is removed — and `ambiguous`
**0.0%**, which is a tie-rule defect rather than a matching one.

Full metrics, breakdowns and confidence intervals:
[reports/phase2a/04_LEAKFREE_BASELINE.md](reports/phase2a/04_LEAKFREE_BASELINE.md).

## Experiment suite

`experiments/` is the reusable harness for benchmarking, visual evidence and Taguchi
robustness work — see [experiments/README.md](experiments/README.md).

```bash
python -m experiments.benchmark --list
python -m experiments.benchmark --system driftforge_v1_leakfree   --manifest manifests/validation_phase2a_400.jsonl   --output experiments/results/classical_v1 --plots --galleries
```

## ML loader

`driftforge.ml.OnTheFlyPairDataset` lazily renders immutable manifest entries and returns full Reference/Search arrays, a 100×100 reference at Search scale, coordinate targets and a stride-4 Gaussian heatmap. Set `as_torch=True` after installing PyTorch.

See [TRAINING.md](TRAINING.md), [SOURCES.md](SOURCES.md) and [RESULTS.md](RESULTS.md).
