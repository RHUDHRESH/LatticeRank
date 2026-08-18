# 01 — Generator sanity gate

**DRAM CLEAN: PASS · FinFET CLEAN: PASS.** Locator work proceeds.

Figures: `generator_gallery/{dram,finfet}_{clean,hard}.png`
Metadata: `generator_gallery/metadata.json`
Script: `reports/war45/gen_sanity.py` (reproducible from seed alone)

Each panel shows, left to right: the **latent master** (analytic render, *no
acquisition whatsoever*), the Reference (1 nm/px), the Search (10 nm/px) with
the ground-truth box, and the true 100×100 target crop. Showing the latent
master matters — the hard-case images are heavily degraded, and degraded output
alone cannot demonstrate that the underlying world model is credible.

| Case | seed | preset | pitch (nm) | linewidth (nm) | contact (nm) | rel. rot | rel. scale | search dose |
|---|---:|---|---|---|---:|---:|---:|---:|
| DRAM CLEAN | 950001 | `dram_dense` | 49.0 × 78.0 | 22.4 / 29.8 | 23.9 | −1.46° | 1.0092 | 155 |
| DRAM HARD | 950003 | `dram_dense` | 61.8 × 102.8 | 18.6 / 25.0 | 24.1 | −1.22° | 1.0224 | 531 |
| FinFET CLEAN | 950002 | `finfet_dense` | 28.5 × 61.4 | 7.8 / 28.0 | 16.0 | −2.06° | 1.0010 | 248 |
| FinFET HARD | 950004 | `finfet_dense` | 40.0 × 67.3 | 15.0 / 30.8 | 25.9 | −1.24° | 0.9849 | 160 |

## DRAM CLEAN — PASS

Visible in the latent master:

- **repeated orthogonal line structure** — horizontal word-lines crossing
  vertical bit-lines at right angles;
- **contact/via-like repeating elements** — bright dots at intersections, on a
  **checkerboard** (one per two cells), not a naive full grid;
- **consistent pitch** — 49.0 nm × 78.0 nm, close to a 2F/3F ratio at F ≈ 25 nm,
  which is the 6F² folded-bitline convention the sponsor's own presets use
  (`dram_dense` there is 48/72 nm);
- **Manhattan geometry** — all features axis-aligned, no curves.
- A **mat/periphery boundary** is visible in the upper part of the crop: the
  array stops and a flatter routing region begins. Real dies are not one
  infinite wallpaper, and this is the context the locator relies on.

Verdict: a defensible 6F² DRAM abstraction, not arbitrary texture.

## FinFET CLEAN — PASS

Visible in the latent master:

- **dense parallel fin-like structures** at 28.5 nm pitch (2.85 px at Search
  scale — genuinely at the resolution limit, which is realistic and hard);
- **gate-crossing structure** — wider perpendicular bars at 61.4 nm pitch,
  i.e. contacted-poly-pitch style, crossing the fins;
- **contacts on a checkerboard** between gates;
- **consistent pitch and Manhattan geometry**;
- a **periphery strip** on the left edge with distinct routing material.

Fin width 7.8 nm vs gate width 28.0 nm is the correct asymmetry (fins are the
finest feature). Verdict: PASS.

## Hard cases — degradations of the same masters, not different worlds

Both hard panels remain traceable to a valid clean master. What changes is
acquisition: lower dose and heavier noise, stronger PSF, relative rotation of
~1.2°, relative scale error up to 2.2%, plus charging and raster effects. In
the Search image the finest structures partially dissolve — expected and
correct, since a 28–62 nm pitch is only 2.8–6.2 px at 10 nm/px and the beam has
band-limited the signal before sampling.

**This is the honest difficulty of the problem, not a generator defect.** The
sponsor's own deck makes the same point: dense pitch "aliases into a
near-random-looking texture" at Search resolution, and that is a free
consequence of the 10× relationship rather than an added fudge factor.

## Caveat carried forward

The clean masters are credible *as 2D layout abstractions*. They are not
Monte-Carlo electron-transport simulations, and `SOURCES.md` has always said
so. One known rendering limitation stands (Phase-1 physics audit §3.1): scene
geometry is hard-thresholded with no geometric anti-aliasing, so at
`supersample=2` a 28 nm fin pitch is sampled at 5.6 points per period. The
supersample 2-vs-4 study found no material accuracy difference, but it could
not settle this because `ss=4` does not change the underlying binary-threshold
rendering. Fixing it means anti-aliasing in `render_scene`, which is out of
scope here.
