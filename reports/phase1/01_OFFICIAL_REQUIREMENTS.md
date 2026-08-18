# 01 — Official Track 2 Requirements (verified against source)

**Source:** <https://i4c.in/hackathon-2026/>
**Accessed:** 2026-08-18
**Track:** Track 2 — Applied Materials — *Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools*

Evidence tags: `VERIFIED FACT` = quoted or directly paraphrased from the official page.
`HYPOTHESIS` / `RECOMMENDATION` = this audit's interpretation.

---

## 1. Task definition

| Item | Official wording | Tag |
|---|---|---|
| Core task | "Find the location inside the Search Image where the Reference Image pattern appears (shrunk 10x)." | `VERIFIED FACT` |
| Output | "Return the center coordinates (x, y) in pixels of the matching region within the Search Image." | `VERIFIED FACT` |
| Multi-match rule | "If more than one matching region is found, return the one closest to the center of the Search Image." | `VERIFIED FACT` |

**Implication.** The output contract is a single `(x, y)` pair in Search-image pixel coordinates.
The tie rule is a *disambiguation* rule that only activates once a set of matching regions
has been established. It is **not** a prior that pulls every prediction toward the image
centre. Any implementation that biases scoring by distance-to-centre before establishing
match equivalence is misreading this clause — see `07_RISK_REGISTER.md` R5.

## 2. Structures

| Item | Official wording | Tag |
|---|---|---|
| DRAM-style | "Periodic horizontal word-lines and vertical bit-lines crossing at right angles, with a small contact/via dot at every intersection." | `VERIFIED FACT` |
| FinFET-style | "Dense parallel vertical fin lines, crossed by one or two horizontal gate bars at the intersection region." | `VERIFIED FACT` |

### 2.1 The DRAM/FinFET ambiguity — exact wording and resolution

The page frames **dataset generation** as a participant choice of architecture style,
while the **test-data description** states Applied Materials' hidden cases span both
DRAM and FinFET.

- `VERIFIED FACT` — the structural definitions above are given for *both* families, and the
  hidden test description references both.
- `VERIFIED FACT` — the sponsor's own starter generator ships **six DRAM presets and six
  FinFET presets** (`src/presets.py`) and its evaluation harness cycles
  `["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]`. The organiser therefore
  treats both families as in-scope by default.
- `RECOMMENDATION` — **Safest interpretation: support both, unconditionally, with no
  architecture flag required at inference.** Treat "choose a style" as permission to
  scope *your generator*, never as permission to scope *your solver*. The inference CLI
  takes two images and no architecture hint, so a DRAM-only or FinFET-only solver is a
  silent 50% failure risk on hidden data.
- `RECOMMENDATION` — architecture may be *inferred* internally (as a gating feature) but
  must never be *required* as an input.

**Note on the contact rule.** The official DRAM wording says a contact "at every
intersection". DriftForge renders contacts on a **checkerboard** (`parity = (ix+iy) even`,
`scene.py:142`), as does the sponsor generator ("one contact per 2 cells, matching real
folded-bitline layouts rather than a naive full grid"). The sponsor deviates from the
literal official wording in the same direction DriftForge does, and for a stated physical
reason (6F² folded-bitline). This deviation is **material to alias structure**: with a
checkerboard, a lattice translation `(m·pitch_x, n·pitch_y)` only preserves the contact
sub-lattice when `m+n` is even. This is measured and confirmed in `05_FAILURE_AUTOPSY.md`.

## 3. Dataset generation requirements (participant-side)

| Requirement | Official wording | Tag |
|---|---|---|
| Independent noise | "Generate independent sensor noise for each image — do NOT reuse the same noise on both images." | `VERIFIED FACT` |
| Edge brightening | "Apply edge-brightening to mimic real SEM (Scanning Electron Microscope) image behaviour." | `VERIFIED FACT` |
| Degradation | "Include realistic degradation: blur, rotation, and scaling variations." | `VERIFIED FACT` |
| Self-eval size | "Generate a minimum of 30 randomized image pairs for self-evaluation." | `VERIFIED FACT` |

## 4. Hidden test data (sponsor-side)

| Requirement | Official wording | Tag |
|---|---|---|
| Relative noise | "The search image will have more noise than the reference image on actual test data." | `VERIFIED FACT` |
| Test noise level | test set will "be MORE noisy than the training examples" | `VERIFIED FACT` |
| Periodic stress case | test set will "include at least one highly periodic array region where correct localization is genuinely difficult" | `VERIFIED FACT` |

**Implication.** The organiser has pre-announced that the decisive test case is a
*periodic-ambiguity* case. Combined with the sponsor's own slide deck admitting its ZNCC
baseline "can lock onto the wrong periodic repeat", the single highest-value engineering
target for this competition is **catastrophic periodic aliasing**, not sub-pixel
precision. This audit's measurements (`05_FAILURE_AUTOPSY.md`) show DriftForge v1 fails
in exactly this pre-announced way.

## 5. Submission requirements

| Requirement | Official wording | Tag |
|---|---|---|
| Inference script | "The localization inference script is the most important file in your repository...must run without manual edits, must accept a reference image path and search image path as inputs." | `VERIFIED FACT` |
| Citations | "Every augmentation choice, noise model, and structural parameter...must be justified with at least 2-3 credible public references." | `VERIFIED FACT` |
| Reproducibility | public GitHub repository | `VERIFIED FACT` |

**Implication for DriftForge.** `scripts/inference.py` already takes two positional image
paths and prints `(x, y)`, satisfying the interface. Two gaps:
1. It prints `(x, y)` with parentheses; the sponsor baseline prints bare `x,y`. The
   official text does not mandate a format, but `RECOMMENDATION`: emit the bare
   `x,y` form as the default stdout line to match the sponsor's demonstrated convention,
   keeping the rich form behind `--json`.
2. `--strict` exits code 2 on abstention. `RECOMMENDATION`: never make abstention the
   default behaviour of the submitted entry point — the contest requires a coordinate for
   every pair.

## 6. Timeline

`VERIFIED FACT` — the roadmap spans **24 July 2026 – 18 September 2026**. The page as
fetched does not label discrete "Phase 1"/"Phase 2" deliverable dates within that window.

`RECOMMENDATION` — do not plan against inferred internal milestones. Re-check the page
for a dated Phase-2 deliverable before freezing the submission schedule. This is an
open item in `10_ARCHITECTURE_FREEZE.md`.

## 7. Judging context

`VERIFIED FACT` — the page identifies **Aayush Raina**, "Senior Algorithm Manager —
Applied Materials", with "Over a decade of experience in semiconductor manufacturing,
specializing in application development and machine learning."

`RECOMMENDATION` — use this only as a coarse signal that industrial robustness,
deployability and honest evaluation are valued. Do **not** build features targeting an
individual. See `02_SPONSOR_BASELINE_AUDIT.md` §7 for the defensible convergence argument.

---

## 8. Requirement → DriftForge v1 compliance matrix

| # | Requirement | DriftForge v1 status | Evidence |
|---|---|---|---|
| 1 | Reference/Search 10× relationship | **Met** — 1 nm/px vs 10 nm/px, both 1000² | `config.py:6-8` |
| 2 | Output `(x, y)` centre | **Met** | `scripts/inference.py:31` |
| 3 | Nearest-to-centre tie rule | **Partially met** — applied only inside `tie_delta=0.015` of the top score, on an NMS-decimated candidate set | `baseline.py:165-166`; see R5, H1 |
| 4 | DRAM structures | **Met** | `scene.py:146-149` |
| 5 | FinFET structures | **Met** | `scene.py:150-153` |
| 6 | Independent sensor noise | **Met** — separate RNG streams | `generator.py:76-77` |
| 7 | Edge brightening | **Met** | `sem.py:50-57` |
| 8 | Blur | **Met** — anisotropic Gaussian PSF | `sem.py:60-63` |
| 9 | Rotation | **Met** | `sem.py:15,22` |
| 10 | Scaling | **Met** | `sem.py:16,23` |
| 11 | Search noisier than Reference | **Met** — search dose 70–700 vs ref 900–4200 | `sem.py:13,20` |
| 12 | ≥30 self-eval pairs | **Met by manifest** (5,100 defined) but the *evidenced* gate is only 20 pairs | `RESULTS.md`; R8 |
| 13 | Standalone 2-image CLI | **Met** | `scripts/inference.py` |
| 14 | Citations 2–3 per choice | **Partially met** — `SOURCES.md` cites families, not per-parameter | see `06_GENERATOR_PHYSICS_AUDIT.md` |
| 15 | Handles highly periodic case | **FAILS** — 4/20 catastrophic aliases | `05_FAILURE_AUTOPSY.md` |

**The single non-compliance that matters is #15**, which is precisely the case the
organiser pre-announced as decisive.
