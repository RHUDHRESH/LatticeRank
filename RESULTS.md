# Verification results

This file records frozen release-gate results produced by `scripts/evaluate_baseline.py`. It is updated only from commands that can be reproduced from the included manifests and code.

## Generator guarantees checked by tests

- deterministic output for a given manifest record;
- 1000×1000 uint8 Reference/Search pairs;
- independent capture pipelines and noise streams;
- transform-aware floating-point `(x, y)` labels;
- exact scene-ID disjointness across all splits;
- both DRAM and FinFET in every split;
- centre-rule behavior for the explicit ambiguity profile;
- framework-neutral model arrays and heatmap labels.

## Locator release gate

Command-equivalent to:

```bash
python scripts/audit_manifest.py manifests/validation.jsonl \
  --limit 20 \
  --search-supersample 2 \
  --tolerance 5 \
  --output reports/validation_release_gate.json
```

| Metric | Result |
|---|---:|
| Pairs | 20 |
| Median error | 0.600 px |
| Accuracy ≤1 px | 70% |
| Accuracy ≤2 px | 80% |
| Accuracy ≤5 px | 80% |
| Selective coverage | 75% |
| Accepted-incorrect at 5 px | **0** |
| Mean locator runtime | 4.97 s/pair |

Five pairs were not auto-accepted: two exact-wallpaper ambiguity cases, two remote periodic jumps and one correct but low-gap match. The ambiguity pairs expose an important limitation: the classical baseline may not return the correct coordinate when it abstains. It preserves the highest-quality tied hypothesis and flags the case; a contest submission that must always emit a coordinate needs a stronger learned/global-context matcher.

This is a compact engineering gate, not statistical proof of zero hidden-test errors. The frozen 1,000-pair private manifest exists specifically for the team to run before release.
