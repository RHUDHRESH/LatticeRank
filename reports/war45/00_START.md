# 00 — War-45 start state

**Budget:** ~45 min wall-clock compute. **Branch:** `ultracode/war45`.
**Start SHA:** `43b3acd` (branched from `phase2a-leakfree-baseline`).

```
main                      b05a6cd  Phase 1: audit, autopsy, architecture freeze
phase2a-leakfree-baseline 43b3acd  Phase 2A (leak fix + 400-pair leak-free baseline)
ultracode/war45           43b3acd  <- this campaign
```

Working tree clean at branch creation.

## What is already banked — do not re-buy

| War-45 stage | Status | Evidence |
|---|---|---|
| §3 Remove target leak | **DONE (Phase 2A)** | placement seeded only by scene identity; interventional test moves the target to 4 positions × 200 scenes, signature moves **0.0 nm** |
| §3 Leak independence check | **DONE, 1,200 scenes** (war-45 asks 200) | inside-Reference 100% → **0.89%** vs 1.06% predicted by geometry; corr −0.044 / +0.003 |
| §4 Leak-free benchmark | **DONE, 400 pairs** (war-45 asks 120) | 31.0% ≤5 px (CI 26.7–35.7), median 248 px, P95 659, 69% catastrophic |
| §5 Failure taxonomy | **DONE, 276 failures** | 70.7% visually-equivalent repeats; parity-even 71.4%, z = +7.10 |
| Benchmark harness | **DONE** | `experiments/{metrics,benchmark,plotting,galleries}.py` |

So this campaign starts at §2 (generator sanity) and spends its budget on **§6–§7:
lattice estimation and candidate recovery**, which have never been measured.

## Compute reality, stated up front

Measured throughput: **2.78 s/pair** (400 pairs / 1,113 s, 6 workers, 8 cores, no GPU).

The full roadmap in the parent prompt costs ~35–40 hr at that rate (1,200-pair
severity sweep 56 min; 10-model ablation 9.3 hr; Taguchi L27 ×10 models 19 hr).
Against a 45-minute budget that is a ~50× gap, so this campaign deliberately
buys **one thing**: the true-cell candidate recall number that caps every
downstream model. Everything else is reported from already-banked evidence or
explicitly not run.

## Deviation from the war-45 script, and why

- **§4 severity buckets (CLEAN/MEDIUM/HARD) not built.** `generate_sample()`
  has no acquisition-override hook (Phase-2A deferred finding D9), so severity
  levels require a generator change first. The existing `profile` axis
  (`standard` / `hard` / `boundary` / `ambiguous`) is used as the stratifier
  instead. Stated rather than silently substituted.
- **§7 five candidate variants reduced to two (C0, C1)** — as the war-45 script
  itself directs, overriding the earlier "all 5 strategies" answer.
- **§13 L9 robustness and §9 Siamese** are P2/P3 and are cut if the P0
  measurement consumes the budget. Their status is reported, not faked.
