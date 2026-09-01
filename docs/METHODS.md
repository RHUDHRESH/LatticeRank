# LatticeRank Phase 2: Methods and Engineering Reference

This chapter describes the system that is actually executed by
`python register.py --input pairs.csv --output predictions.csv`. It separates
the scored runtime path from the Phase 1 and experimental modules that remain
in the repository, states every coordinate and pose convention, explains how
the synthetic data is constructed, and connects measured design choices to the
code that implements them.

The short version is:

```text
pairs.csv
   |
   +-- decode Reference and Search images
   |
   +-- 45-pose full-resolution DoG + ZNCC sweep
   |      scales: 8.0 ... 12.0, step 0.5
   |      rotations: -6, -3, 0, 3, 6 degrees
   |
   +-- local translation / scale / rotation refinement
   |
   +-- presence model --------------------------> found
   |
   +-- correctness-ranking model --------------> score
   |
   +-- contract enforcement and atomic CSV write
```

The active Phase 2 entry point is deliberately simpler than the complete
research tree. The Phase 1 periodic-residual pipeline and many Phase 2
experiment modules are retained for evidence and reproducibility, but
`register.py` does not call them. This distinction matters when reproducing a
score, profiling runtime, or deciding where a change must be made.

## 1. Task definition

Each input pair contains:

- a 1,000 x 1,000 high-magnification **Reference** image;
- a 1,000 x 1,000 wide-field **Search** image;
- a pair identifier and paths to both images.

For a present pair, the solver estimates the Reference centre in Search-image
pixels and its pose. For an absent pair, it rejects the match. The output schema
is fixed:

```text
pair_id,x,y,theta,scale,found,score
```

The columns have independent meanings:

| Column | Meaning |
|---|---|
| `x` | horizontal coordinate of the match centre; columns increase rightward |
| `y` | vertical coordinate of the match centre; rows increase downward |
| `theta` | template rotation in degrees; counter-clockwise is positive |
| `scale` | down-scaling factor `s`, not `1/s`; the template side is about `1000/s` pixels |
| `found` | binary present/absent decision |
| `score` | fitted probability used to rank whether the reported answer is correct |

When `found == 0`, `x`, `y`, `theta`, and `scale` are all zero. `score` is not
forced to zero because presence and localization correctness are different
questions. A reference may be present while the selected lattice site is
wrong.

## 2. Coordinate and pose conventions

### 2.1 Position

Correlation surfaces index the top-left placement of a template. The reported
position is its centre:

```text
x = correlation_column + (template_width  - 1) / 2
y = correlation_row    + (template_height - 1) / 2
```

This convention is implemented in `driftforge/dense.py::_correlate`. Fractional
centres are valid and are preserved through the CSV writer.

### 2.2 Scale

`scale` is the physical down-scaling factor between the two images. A value of
`s = 10` means the 1,000-pixel Reference is converted to a roughly 100-pixel
template before it is searched in the wide field:

```text
template_side ~= reference_side / s
```

`driftforge.pose.build_template` first applies Gaussian anti-aliasing with
`sigma = clip(0.35 * s, 0.8, 5.0)`, then samples at `1/s`. A caller may pin the
output shape to prevent integer template-size changes from creating a
saw-tooth response as scale varies.

This convention differs from older Phase 1 helpers that multiply a fixed 0.1
zoom. New code should use `build_template` and should never pass `1/s` as the
reported scale.

### 2.3 Rotation

The template builder applies `scipy.ndimage.rotate` with counter-clockwise
angles in the array plane and `reshape=False`. The generator's label is not
trusted from an analytic sign formula alone. It is measured by a local ZNCC
oracle at the known site, and `scripts/verify_conventions.py` checks that the
generator and the inference template builder recover the same sign.

The stage range is nominally +/-5 degrees. The production sweep reaches +/-6
degrees and refinement permits +/-8.5 degrees because the reportable relative
angle also contains independent reference/search acquisition jitter.

## 3. Synthetic world and acquisition model

The Phase 2 generator is not a crop-and-resize generator. It constructs a
latent physical scene, then simulates two independent acquisitions.

### 3.1 Latent scene

`driftforge.scene` renders DRAM- and FinFET-like periodic structures in physical
coordinates. The scene includes pitch, line/contact geometry, sparse defects,
roughness, structural variation, and optional near-duplicate sites. The Search
image covers the full world. The Reference covers a smaller physical field
centred on the sampled target.

The generator uses separate random streams for:

- the master scene and geometry;
- the present/absent label (`seed ^ 0xC1A55`);
- severity (`seed ^ 0x5E7`);
- Reference acquisition (`seed * 7,919 + 101`);
- Search acquisition (`seed * 1,000,003 + 313`);
- the foreign Reference scene on absent pairs (`seed ^ 0x5EED`).

The dedicated label and severity streams exist because a shared draw sequence
created weak, learnable associations between labels and image histograms even
when every individual seed appeared independent.

### 3.2 Present and absent pairs

The default absent fraction is 20%. A present Reference is rendered from the
same latent world as the Search image. An absent Reference is rendered from a
different structural realization while keeping architecture, profile, and
preset family matched. This prevents rejection from collapsing into a trivial
architecture classifier.

Decoys are sampled before the present/absent branch and occur on 40% of pairs by
default. On absent pairs, decoys follow the Search scene's motif, not the
foreign Reference. Therefore decoy presence cannot be used as a proxy for the
label.

### 3.3 Zoom is physical, not post-render resizing

The sampled down-scaling factor is `s in [8, 12]`. The Reference field of view
is `WORLD_FOV_NM / s`, while the Search field of view remains
`WORLD_FOV_NM`. Both outputs are rendered to 1,000 x 1,000 pixels. No completed
Reference image is resized to manufacture the zoom label.

### 3.4 Independent degradations

Reference and Search acquisitions receive independent point-spread functions,
dose/noise draws, rotations, scan distortion, charging, vignetting, gamma,
gain, offsets, hot pixels, and acquisition jitter. Four severity levels widen
these distributions. The difficult end includes stronger line-edge roughness,
CD bias, contact dropout, local damage, occlusion, anisotropic blur, charging,
streaks, row jitter, scan shear, and radial distortion.

RGB mode applies independent channel reflectance, gain, colour cast, and
chromatic shifts. Approximately 18% of RGB pairs are deliberately close to
grayscale.

### 3.5 Ground truth is measured

The target mask is warped through the same Search transformation as the image.
Its centroid gives `(gt_x, gt_y)`. At that site, alternating rotation and scale
oracles read the ZNCC response until the pose stabilizes. A final scale pass
removes the one-step lag. This makes the label a measurement of the implemented
image formation and template conventions rather than a fragile algebraic
approximation.

## 4. Runtime entry point and failure contract

`register.py` is both the command-line interface and the contract boundary.

### 4.1 Input handling

The parser accepts common header variants for pair IDs and image paths. If no
exact image column name matches, a guarded hint pass considers names such as
`template`, `query`, `search`, or `scene`, but accepts them only when the value
looks like an image path. This avoids mistakes such as interpreting
`preferred_scale` as a Reference path because it contains `ref`.

Relative paths are resolved in this order:

1. as written, relative to the process working directory;
2. relative to the directory containing `pairs.csv`;
3. relative to the submission root.

The second and third rules allow the judge to execute from a foreign working
directory without turning every pair into a file-not-found rejection.

Images are decoded with Pillow. Common colour modes are converted to grayscale.
Numeric single-plane modes are linearly normalized to `uint8`. Multi-frame or
unsupported images become explicit per-pair failures.

### 4.2 One input row, one output row

Every pair produces a row even when decoding, inference, or model loading fails.
Unexpected exceptions are printed to stderr and receive a finite on-scale floor
score of `1e-6`. Duplicate `pair_id` values keep the first result and emit a
warning. Output is first written to `predictions.csv.tmp`, then atomically
replaced, preventing a killed run from leaving a truncated destination file.

### 4.3 Thread and time controls

BLAS/OpenMP thread counts are pinned to four before NumPy or SciPy is imported.
The scored reference machine has four cores; uncontrolled oversubscription was
slower.

Each pair receives a 12-second `Deadline`, leaving headroom below the 20-second
hard limit for image decoding, CSV work, a slower judge machine, and a stage
that finishes after its last boundary check. The watchdog does not interrupt
NumPy/SciPy calls. Instead it checks stage boundaries and records explicit
degradations:

- stop adding dense pose hypotheses when the remaining budget is insufficient;
- skip local refinement if approximately 0.5 seconds cannot be afforded;
- skip the learned presence features if approximately 1 second cannot be
  afforded and use the raw-peak fallback.

The priority is to return a weaker valid answer rather than time out and score
zero.

## 5. Active localization method

### 5.1 Difference-of-Gaussians signal

Both Search and template are transformed as:

```text
DoG(I) = Gaussian(I, sigma=2) - Gaussian(I, sigma=8)
```

The operation removes slowly varying charging and illumination while retaining
the mid-scale structural response. This was the strongest confirmed change in
the Phase 2 campaign: on 280 present synthetic pairs, <=1-pixel localization
rose from 18.6% to 29.6%, with a paired McNemar `p = 0.0002`. The largest gain
occurred at severity 3.

### 5.2 Pose sweep

`driftforge.dense.dense_pose_search` evaluates 45 hypotheses:

```text
scales    = 8.0, 8.5, ..., 12.0       (9 values)
rotations = -6, -3, 0, 3, 6 degrees   (5 values)
```

For each hypothesis:

1. anti-alias and down-scale the Reference;
2. rotate the template;
3. apply the DoG transform;
4. compute a valid zero-mean normalized cross-correlation surface;
5. keep the global peak `(x, y, score)`;
6. append the per-pose record for the learned heads.

ZNCC for a template `T` and Search window `W` is:

```text
rho(W,T) = sum((W-mean(W)) (T-mean(T)))
           --------------------------------
           sqrt(sum((W-mean(W))^2) sum((T-mean(T))^2))
```

The best score across all 45 surfaces is the coarse pose. The full sweep is
used because measured candidate funnels and hard caps repeatedly discarded the
true site; the true peak can rank in the hundreds under cheap proxy scores.

### 5.3 What is not active

`driftforge.pipeline`, `driftforge.residual`,
`driftforge.structural_descriptor`, `driftforge.pose_search`, and the
`driftforge.phys` experiment family remain useful for Phase 1 reproduction and
research evidence. They are not called by the scored `register.py` path.

In particular, the current Phase 2 prediction is the global best DoG+ZNCC pose,
not a re-ranked periodic-residual candidate pool. The experiment record shows
why: multiple candidate-pooling, lattice-relative, unit-cell, edge-profile,
photometric, and sibling-relative alternatives either lost the true site,
failed to improve full-pool top-1 selection, or added unacceptable runtime.

## 6. Local pose refinement

`driftforge.refine.refine_candidate` alternates translation, scale, and
rotation refinement for up to two passes.

### 6.1 Translation

A local Search window holds the current template plus 32 pixels of slack on
every side. A full-resolution DoG+ZNCC surface is evaluated only inside this
window. The peak is refined independently along rows and columns by fitting a
three-point parabola.

Periodic images may contain several nearly equal peaks. Among peaks within
0.02 correlation of the maximum, refinement chooses the one nearest the
incoming location. This prevents an already-correct coarse estimate from
jumping a full lattice period during local optimization.

### 6.2 Scale and rotation

After translation, scale is searched within +/-1.0 of the current estimate and
clamped to `[7.5, 12.5]`. Rotation is searched within +/-3 degrees and clamped
to +/-8.5 degrees. The loop stops early after the second pass if the position
shift is at most 0.05 pixels.

Pose accuracy is conditional on site selection. Measurements show that when
the correct site is chosen, scale and rotation are usually already inside the
scoring tolerances. The dominant failure is choosing the wrong periodic site,
not subpixel fitting.

## 7. Presence decision (`found`)

Presence and localization correctness use separate models.

The presence model is a 17-feature `HistGradientBoostingClassifier`. Its
features summarize:

- peak scores from raw, mid-band, and directionality channels;
- best-versus-runner-up margins and peak-to-sidelobe ratio;
- candidate count, score entropy, and equivalent-mode fraction;
- agreement among channels, scales, and rotations;
- refinement gain, displacement, and final score;
- local peak curvature.

The operating threshold is 0.48. It was selected for stability across severity
strata rather than for the maximum calibration-split F1. The packaged metadata
records a 51-scene lockbox result of F1 0.9024, with a 90% bootstrap interval of
0.8378 to 0.9535. The interval is important: the point estimate is useful, but
it does not certify a guaranteed >=0.90 blind-set result.

If the model cannot be loaded or the remaining deadline is too short, the
fallback declares presence when the best raw sweep score is at least 0.45.

## 8. Correctness ranking (`score`)

The score model is logistic regression over 20 features with a
`StandardScaler`. It estimates whether the reported answer is correct, not
whether the Reference merely exists.

Its inputs include:

- top sweep score, top-two margin, top-median margin, and score entropy;
- refined score, gain, displacement, scale change, rotation change, and
  convergence flag;
- local Hessian curvature and condition number at the final peak;
- Search-image blur, low-frequency energy, and noise estimates;
- four coarse-pool features imputed at their training means.

The four imputed values would require a second coarse search costing about 1.3
seconds per pair. Their measured removal cost 0.0098 AUC, so the runtime trade
was rejected.

On its 74-pair lockbox, the fitted model reached AUC 0.892 versus 0.760 for the
raw ZNCC peak on the same pairs. It is best treated as a ranking score expressed
on a probability scale. The project evaluates its discrimination by AUC; it
does not claim perfect probability calibration.

## 9. Data splits and leakage control

Phase 2 splits use disjoint seed bases separated by hundreds of thousands or
millions. `driftforge.splits.assert_disjoint` checks not only raw seeds but all
derived scene and acquisition seeds. This catches the subtle case where one
split's raw seed equals another split's derived realization.

The packaged presence and correctness models were trained only on generated
`p2_val` and `p2_stress` data. Their metadata states that organizer data and
`p2_test` were not used. Features are computed at inference-estimated pose,
never at ground-truth pose.

## 10. Validation gates G1-G12

`scripts/validate_phase2.py` turns generator claims into executable gates:

| Gate | Check | Pass condition |
|---|---|---|
| G1 | scale-label recovery by an independent oracle | <=0.5% error on at least 99% |
| G2 | rotation-label recovery by an independent oracle | <=0.15 degrees on at least 99% |
| G3 | sponsor-style ZNCC difficulty | 5-pixel accuracy between 0.40 and 0.85 |
| G4 | present/absent leakage from pixel histograms | cross-validated AUC <=0.55 |
| G5 | severity leakage from pixel histograms | macro one-vs-rest AUC <=0.70 |
| G6 | crop-and-paste leakage | zero bit-identical 32 x 32 patches |
| G7 | pose marginals | KS `p > 0.05` for scale and rotation |
| G8 | index correlation | `abs(r) < 0.1`, or permutation `p > 0.01` |
| G9 | split leakage | no raw or derived seed collision |
| G10 | absent/present parity | KS `p > 0.05` for scale, severity, family, architecture |
| G11 | image contract | every output is 1,000 x 1,000 with expected mode |
| G12 | determinism | sampled images and labels regenerate byte-identically |

The offline gate installs an audit hook that rejects network access, copies the
entry point to a temporary foreign working directory, and confirms that the
documented command completes there. `scripts/build_submission.py` packages from
an explicit allow-list, checks required files and citations, constrains the
failure-analysis PDF, extracts the archive, and smoke-tests the extracted copy.

## 11. Reproduction map

| Goal | Command or file |
|---|---|
| Run unit/integration tests | `python -m pytest -q` |
| Verify evidence files | `python scripts/verify_evidence.py` |
| Verify pose conventions | `python scripts/verify_conventions.py` |
| Run Phase 2 gates | `python scripts/validate_phase2.py --data-root data/phase2 --splits p2_val --quick` |
| Verify offline/foreign-cwd behavior | `python scripts/verify_offline.py --entry register.py --args "--input pairs.csv --output out.csv"` |
| Build and smoke-test the submission | `python scripts/build_submission.py` |
| Review measured findings | `docs/PHASE2_FINDINGS.md` |
| Review failure analysis | `docs/failure_analysis.md` |

## 12. Module map

| Module | Responsibility | Active in `register.py`? |
|---|---|---|
| `register.py` | I/O, failure contract, orchestration, atomic output | yes |
| `driftforge/dense.py` | 45-hypothesis full-resolution DoG+ZNCC sweep | yes |
| `driftforge/pose.py` | template construction, DoG, pose oracles | yes |
| `driftforge/refine.py` | local subpixel translation and pose refinement | yes |
| `driftforge/presence_model.py` | present/absent features and classifier | yes |
| `driftforge/correctness_model.py` | correctness-ranking features and model | yes |
| `driftforge/budget.py` | per-pair deadline and graceful degradation | yes |
| `driftforge/generator.py` | Phase 1/2 sample generation | training/validation |
| `driftforge/phase2.py` | Phase 2 distributions and acquisition functions | training/validation |
| `driftforge/scene.py` | latent DRAM/FinFET scene rendering | training/validation |
| `driftforge/splits.py` | deterministic split construction and leakage checks | training/validation |
| `driftforge/pipeline.py` | Phase 1 periodic-aware candidate ranking | no |
| `driftforge/pose_search.py` | coarse decimated candidate-pool experiment | no |
| `driftforge/phys/*` | measured experimental feature families | no |

## 13. Known limits and honest interpretation

1. **Site selection remains the bottleneck.** Errors are almost binary: a
   correct site is subpixel, while a wrong site is often hundreds of pixels
   away at a periodic alias.
2. **The full sweep is expensive.** Deadline degradation protects the hard
   timeout, but a truncated sweep can miss late hypotheses.
3. **Lockboxes are small.** Presence and correctness point estimates are useful
   but their uncertainty must accompany them.
4. **Evidence is synthetic.** No sponsor blind test images were used for model
   training or reported internal experiments.
5. **The generator is a model of the domain, not the domain itself.** DRAM and
   FinFET share reusable rendering primitives; generalization to unseen process
   families remains an external-validity risk.
6. **Experimental modules are not production stages.** Their presence in the
   repository does not imply that the entry point executes them.

The engineering objective is therefore not to present the method as solved. It
is to make the exact scored behavior reproducible, keep rejection and confidence
semantically correct, prevent leakage and timeout failures, and preserve the
evidence needed to improve the remaining top-1 site-selection problem.
