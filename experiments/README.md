# LatticeRank experiment suite

Reusable harness for benchmarking, visual evidence and Taguchi robustness
work. Built in Phase 2A so that later phases add *systems*, not new one-off
scripts.

Nothing here changes `scripts/inference.py`. The competition CLI keeps the
interface the organiser requires; the benchmark imports the locator directly.

## Layout

```
experiments/
  common.py      Prediction / CandidateRecord contracts, system registry
  metrics.py     accuracy, percentiles, Wilson + bootstrap CIs, S/N, ANOVA
  plotting.py    distribution, comparison, training-curve and Taguchi figures
  galleries.py   traceable success/failure image evidence
  benchmark.py   the runner: one command per system, one standard output shape
  taguchi.py     L27 design, batch manifests, level/ANOVA analysis
  configs/
    ensemble_plan.yaml   multi-model meta-ranker design (Phase 2D)
    siamese_plan.yaml    metric-learning expert design (Phase 2C)
    taguchi_L27.yaml     8 physical factors x 3 levels, with rationale
  taguchi_L27_design.csv the 27-run orthogonal array
```

## Running a benchmark

```bash
python -m experiments.benchmark --list

python -m experiments.benchmark \
  --system driftforge_v1_leakfree \
  --manifest manifests/validation_phase2a_400.jsonl \
  --output experiments/results/classical_v1 \
  --plots --galleries
```

Outputs, identical in shape for every system:

```
<output>/predictions.csv     one row per pair, standard columns
<output>/summary.json        metrics + CIs + breakdowns + environment
<output>/plots/*.png|pdf     error CDF, histogram, accuracy splits, runtime
<output>/galleries/*.png     traceable success/failure figures
```

## The registry is the honest record of what exists

`experiments/common.py` declares every planned system B0–B9. Only
`driftforge_v1_leakfree` (B1) is implemented. Requesting an unimplemented
system **exits with an error** rather than emitting a placeholder row — a
number that later gets mistaken for a measurement is exactly the failure mode
Phase 1 was written to prevent.

| Key | Label | Phase | Implemented |
|---|---|---|---|
| `sponsor_zncc` | B0 sponsor ZNCC | 2B | no (needs OpenCV) |
| `driftforge_v1_leakfree` | B1 DriftForge v1 leak-free | 2A | **yes** |
| `lattice_candidates` | B2 lattice candidates | 2B | no |
| `lattice_directionality` | B3 + directionality | 2B | no |
| `siamese_random_neg` | B4 Siamese (random negatives) | 2C | no |
| `siamese_hard_neg` | B5 Siamese (hard negatives) | 2C | no |
| `classical_ensemble` | B6 classical ensemble ranker | 2D | no |
| `siamese_plus_ensemble` | B7 + learned similarity | 2D | no |
| `calibrated_ensemble` | B8 + calibration/equivalence | 2D | no |
| `ensemble_plus_subpixel` | B9 + subpixel | 2D | no |

## Adding a system

```python
from experiments.common import Prediction, SystemSpec, register

def _my_system():
    def predict(reference, search) -> Prediction:
        ...
        return Prediction(x=x, y=y, runtime_ms=ms, confidence=c,
                          diagnostics={"accepted": True})
    return predict

register(SystemSpec("my_system", "B? label", "what it does",
                    implemented=True, factory=_my_system, phase="2B"))
```

Nothing else changes: metrics, plots, galleries and comparison all follow.

## Candidate records (the ML data contract)

`CandidateRecord` in `common.py` is the row the Phase-2D ensemble trains on.
Two rules are enforced in code, not by convention:

- **Missing means NaN.** Features from experts that do not exist yet
  (`directionality_score`, `siamese_similarity`, `phase_score`,
  `lattice_confidence`, `expert_vote_count`) stay NaN. A placeholder number
  would later be indistinguishable from a measurement.
- **Distance-to-centre is rejected.** `build_candidate()` raises if any
  centre-distance feature is supplied. The official nearest-centre rule is a
  *tie-break applied after* an equivalence set exists; as a similarity feature
  it becomes the shortcut recorded as Phase-1 R5.

## Training-log contract (Phase 2C)

Training scripts must emit CSV with these columns;
`plotting.training_curves()` consumes them directly:

```
epoch, train_loss, val_loss,
positive_similarity_mean, positive_similarity_std,
hard_negative_similarity_mean, hard_negative_similarity_std,
random_negative_similarity_mean, candidate_rank_accuracy, learning_rate
```

## Taguchi

```bash
python -m experiments.taguchi --verify-only     # check orthogonality
python -m experiments.taguchi --emit-design     # write the 27-run array
python -m experiments.taguchi --pairs-per-run 30 --repeats 3 \
    --output manifests/taguchi/
```

8 physical factors x 3 levels in an L27(3^13), 27 runs x 3 repeats x 30 pairs
= 2,430 evaluations per frozen model. Orthogonality is verified numerically at
build time, not trusted from a table.

**Designed and validated, not executed.** Running it needs a generator
override hook that does not exist yet (`DEFERRED_FINDINGS.md` D9), and model
weights must be frozen first.

## Figure naming

Every gallery figure encodes outcome, architecture and sample id, and carries
a caption panel with sample_id, seed, manifest and system:

```
success_easy_dram_validation_phase2a-000012.png
failure_alias_finfet_validation_phase2a-000107.png
aliascmp_dram_validation_phase2a-000233.png
```

Any figure in a deck must be reproducible from its caption alone.
