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
4. Ordinary scenes contain shared but fabrication-like structural signatures. A separate `ambiguous` profile is deliberately wallpaper-like and obeys the official nearest-Search-centre rule.
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

The packaged 20-pair mixed-profile gate achieved 0.60 px median error, 80% accuracy at 5 px, 75% selective coverage and **zero accepted-incorrect predictions at 5 px**. The sample is intentionally small; expand the locked private test before making operational claims.

## ML loader

`driftforge.ml.OnTheFlyPairDataset` lazily renders immutable manifest entries and returns full Reference/Search arrays, a 100×100 reference at Search scale, coordinate targets and a stride-4 Gaussian heatmap. Set `as_torch=True` after installing PyTorch.

See [TRAINING.md](TRAINING.md), [SOURCES.md](SOURCES.md) and [RESULTS.md](RESULTS.md).
