# Training and validation protocol

## Split contract

| Split | Pairs | Purpose | Labels during development |
|---|---:|---|---|
| Train | 3,000 | Fit model and mine hard negatives | Visible |
| Validation | 600 | Tune transformations, thresholds and checkpoints | Visible |
| Test | 1,000 | One-shot internal release gate | Keep private |
| Stress | 500 | Diagnose ambiguity, OOD acquisition and boundaries | Visible after test lock |

Never regenerate the test manifest while tuning. Never select a checkpoint on test. The four seed ranges and scene IDs are disjoint.

## Recommended model

Use a Siamese/correlation localization network, not a classification CNN:

1. Encode the 1000×1000 Reference with a branch whose total stride accounts for the 10× physical scale difference, or feed `reference_at_search_scale` to a shared-weight encoder.
2. Encode Search fully convolutionally; do not global-pool away position.
3. Cross-correlate Reference features over Search features and predict a heatmap.
4. Train with focal/BCE heatmap loss plus soft-argmax coordinate loss. Weight the coordinate term more heavily after the heatmap begins to converge.
5. Mine hard negatives from remote high-scoring peaks, especially within repeated arrays. Do not randomly declare a periodic patch negative: it may be physically indistinguishable.
6. At inference, refine the top peak with local NCC/phase correlation and report peak-to-sidelobe, remote peak gap and runtime.

## Curriculum

- Epochs 1–3: `standard`, mild acquisition variation.
- Epochs 4–8: add `boundary` and `hard`; randomize noise/blur/contrast independently.
- Epochs 9+: add `ambiguous`; use nearest-centre supervision only for tied candidates.
- Final: fine-tune with OOD acquisition without changing the locked test split.

## Metrics

Report median, p95 and maximum Euclidean coordinate error, accuracy at 1/2/5/10 px, mean/p95 runtime, coverage after confidence gating, and **accepted-incorrect count**. Accuracy alone hides catastrophic periodic-array jumps.

## False-positive policy

No responsible system can guarantee zero false localization on unseen sponsor data. For an operational zero-false-positive target, use selective prediction:

- auto-accept only if the absolute match score, PSR and remote peak gap pass thresholds calibrated on validation;
- otherwise abstain/request another capture;
- freeze thresholds before the private test;
- release only if private-test `accepted_incorrect == 0` at the declared pixel tolerance.

The contest may require a coordinate for every pair. In that mode, still emit the best coordinate but preserve confidence so low-confidence cases are visible rather than silently trusted.

