# 05 — Catastrophic failure classification

**Gate under test.** The brief sets a hard rule:

> `if periodic_alias_fraction < 0.80: STOP — architecture freeze must be reopened`

**Result: the gate FAILS under every measurement method available.**
Best-supported estimate **70.7%** (95% CI 65.0% – 75.7%). The CI upper bound is
below 0.80, so this is not a borderline pass.

Data: [`catastrophic_failures.csv`](catastrophic_failures.csv) (276 rows),
[`alias_null_control.json`](alias_null_control.json).

---

## 1. First, the instrument had to be fixed

Phase 1 classified aliases by lattice arithmetic: solve
`(dx, dy) = m·a + n·b` for the Search-image lattice basis and check whether
`(m, n)` is near-integer. Applied at scale, that instrument turned out to be
close to worthless here, and it is worth being explicit about why.

`diagnostics/alias_null_control.py` compares observed residuals against two
nulls — displacements drawn uniformly, and real displacements permuted across
rows (preserving the magnitude distribution, destroying only the lattice
pairing):

| Distribution | median | p95 | max | <1 px | <3 px |
|---|---:|---:|---:|---:|---:|
| **Observed** | 1.99 px | 4.79 | 7.97 | 22.3% | 75.5% |
| Null: uniform | 2.77 px | 5.84 | 9.59 | 7.5% | 55.8% |
| Null: shuffled | 2.60 px | 5.74 | 9.34 | 8.9% | 60.1% |

Effect size **0.39–0.50 σ**. Statistically detectable (permutation p = 0.005 at
n=220) but far too weak to classify individual failures.

The reason is geometric, and it is decisive:

- **half a lattice cell diagonal: 5.59 px (median)** — the largest a residual
  can ever be;
- **non-affine warp budget: 4.96 px median, 8.97 px p95** — raster shear, row
  jitter and radial distortion, none of which are in the affine lattice basis.

The unmodelled warps are as large as the cell itself. A true alias therefore
*cannot* have a small residual, and a random landing often does. **Any
threshold on lattice residual is measuring noise.** The Phase-1 conclusion
("4/4 catastrophic failures are aliases, residuals 0.50–2.12 px") rested on
this instrument at n=4 and was over-confident.

## 2. The replacement instrument: direct visual equivalence

Instead of asking *"is the offset an integer number of periods?"*, ask the
question that actually determines the architecture:

> **Does the predicted location look like the true location?**

For every catastrophic failure we extract the 100×100 Search crop at the true
location and at the predicted location, compute their ZNCC, and calibrate it
against a **per-image null** (60 random in-bounds locations in the same
texture). "Equivalent" means `ZNCC ≥ 0.60` **and** above that image's own null
p95.

This is assumption-free: no lattice model, no knowledge of pitch, no affine
basis. It also maps directly onto what can be fixed:

- **equivalent** → a genuine repeat. No local matcher can separate the two;
  this needs lattice-aware candidates, global context, and the official centre
  rule.
- **not equivalent** → the locator picked something that does not look like the
  target. Better evidence *can* fix this.

## 3. Result

276 catastrophic failures out of 400 pairs (69.0%).

| Measurement | Alias fraction | Verdict vs 0.80 gate |
|---|---:|---|
| **Visual equivalence (primary)** | **195/276 = 70.7%** (CI 65.0–75.7) | **FAIL** |
| Above per-image null p95 (looser) | 231/276 = 83.7% | passes, but "above null" is a weak bar |
| Legacy lattice arithmetic | 204/276 = 73.9% | FAIL — and the instrument is unreliable (§1) |
| Mechanism-first bucketing (as coded) | 144/276 = 52.2% | FAIL |

Reported honestly: **three of four measurements fail the gate**, and the one
that passes uses the weakest possible criterion ("scored above the 95th
percentile of random locations"), which admits partial-context matches that are
plainly not equivalent on inspection.

### Class breakdown (mechanism-first bucketing, as implemented)

| Class | n | % | median patch ZNCC | its null p95 | also visually equivalent |
|---|---:|---:|---:|---:|---:|
| `periodic_alias` | 144 | 52.2% | 0.783 | 0.350 | 100% |
| `wrong_transform` | 47 | 17.0% | 0.751 | 0.360 | **77%** |
| `matcher_error` | 35 | 12.7% | 0.365 | 0.548 | 0% |
| `boundary_context` | 29 | 10.5% | 0.487 | 0.303 | 0% |
| `low_snr` | 19 | 6.9% | 0.763 | 0.565 | **79%** |
| `other` | 2 | 0.7% | — | — | 0% |

**`wrong_transform` and `low_snr` are not competing explanations.** Their patch
similarity (0.751, 0.763) is statistically indistinguishable from the alias
class (0.783) and far above their own nulls, and 77–79% of them meet the
equivalence criterion outright. A slightly-wrong transform variant winning the
global arg-max is *how* the locator lands on the wrong repeat — it is a
co-occurring condition, not an alternative mechanism. Bucketing them separately
splits one phenomenon across three rows and is why the mechanism-first ordering
reports only 52.2%.

This is also why the primary estimate is 70.7% rather than 52.2%: the
equivalence test is applied to all 276 rows regardless of co-occurring
mechanical conditions.

## 4. Independent corroboration: the parity test

Contacts sit on a checkerboard, so only translates with `m + n` **even**
preserve them. Parity is *discrete* — chance landings give 50/50 — so it is
immune to the residual-threshold problem entirely.

| Population | parity-even | z vs 50% |
|---|---:|---:|
| All 276 catastrophic | **71.4%** | **+7.10** |
| Visually equivalent (195) | 71.3% | |
| **Not** visually equivalent (81) | **71.8%** | |

`z = 7.10` is overwhelming (p ≪ 10⁻¹⁰). **Lattice structure is real and pervades
every catastrophic failure**, including the ones that are *not* visually
equivalent.

That last row is the most informative number in this report. Predictions land
on lattice-consistent sites even when the surroundings differ — the fine
lattice matches, the larger-scale context does not. That is precisely the
`boundary_context` / `matcher_error` population: **fixable by better evidence**,
because something visibly distinguishes the right cell from the wrong one.

## 5. What this means for the architecture

The v2 pipeline is **not falsified**, but the freeze rested on a premise —
"catastrophic failure is ~entirely periodic aliasing" — that the data does not
support at the stated threshold. Splitting the 276:

| Population | n | % of catastrophic | What fixes it |
|---|---:|---:|---|
| Visually equivalent repeats | 195 | 70.7% | lattice candidates + centre rule over a calibrated equivalence set. **Local matching cannot help.** |
| Not equivalent (context mismatch) | 81 | 29.3% | better evidence: mid-band context, directionality, ensemble ranking. **Local matching can help.** |

Every component in the frozen architecture attacks one of these two, so nothing
needs discarding. What changes is the **weighting and the acceptance criteria**:
a second, explicit workstream for the 29.3% is required, and the Phase-2B
success criterion cannot be candidate recall alone.

Also note the 40 `ambiguous` pairs (100% catastrophic, 0% accurate) are 14.5%
of all catastrophic failures and are a **tie-rule defect**, not a matcher
defect — the cheapest identified win in the whole phase.

## 6. Verdict

**STOP. The architecture freeze must be reopened** — per the brief's own rule,
and I am not going to argue around a threshold that was set in advance.

The reopening is an **amendment, not a rewrite**: the alias workstream stands,
and a non-alias workstream must be added alongside it with its own measurable
target. Recommended amendments are in `07_PHASE2A_VERDICT.md`.
