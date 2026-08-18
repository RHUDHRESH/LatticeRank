"""Traceable visual evidence.

Every figure filename encodes outcome, architecture and sample_id, and every
figure carries a caption panel naming the sample, seed, manifest and system.
An image that cannot be traced back to a seed cannot go in the deck.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import REFERENCE_FOOTPRINT_PX, render  # noqa: E402

DPI = 190
TOL_PX = 5.0
CATASTROPHIC_PX = 25.0


def outcome_tag(error_px: float, failure_class: str | None = None) -> str:
    if error_px <= 1.0:
        return "success_easy"
    if error_px <= TOL_PX:
        return "success_hard"
    if failure_class == "periodic_alias":
        return "failure_alias"
    # Name every class explicitly: a figure called "failure_other" cannot be
    # traced back to what actually went wrong, which defeats the point.
    if failure_class in ("low_snr", "wrong_transform", "matcher_error",
                         "boundary_context"):
        return f"failure_{failure_class}"
    return "failure_other"


def _caption(ax, row: dict, manifest: str, system: str) -> None:
    ax.axis("off")
    txt = (f"sample_id : {row['sample_id']}\n"
           f"seed      : {row.get('seed', '?')}\n"
           f"manifest  : {manifest}\n"
           f"system    : {system}\n"
           f"arch      : {row['architecture']}\n"
           f"profile   : {row['profile']}\n"
           f"error     : {row['error_px']:.2f} px\n"
           f"class     : {row.get('failure_class', 'ok')}\n"
           f"runtime   : {row.get('runtime_ms', float('nan')):.0f} ms")
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=9)


def localization_overlay(sample, row: dict, outdir: Path, manifest: str,
                         system: str) -> Path:
    """Search image with truth, prediction and the error vector."""
    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(sample.search, cmap="gray", interpolation="nearest")
    tx, ty = row["true_x"], row["true_y"]
    px, py = row["pred_x"], row["pred_y"]
    h = REFERENCE_FOOTPRINT_PX / 2.0
    ax.add_patch(plt.Rectangle((tx - h, ty - h), 2 * h, 2 * h, fill=False,
                               ec="lime", lw=2.0))
    ax.plot(tx, ty, "o", mfc="none", mec="lime", ms=14, mew=2.4, label="ground truth")
    ax.plot(px, py, "x", color="red", ms=14, mew=2.4, label="prediction")
    if row["error_px"] > 1.0:
        ax.annotate("", xy=(px, py), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8))
    ax.set_title(f"Search 10 nm/px — error {row['error_px']:.2f} px", fontsize=11)
    ax.set_xlabel("x (Search px)"); ax.set_ylabel("y (Search px)")
    ax.legend(loc="upper right", fontsize=9)
    _caption(fig.add_subplot(gs[0, 1]), row, manifest, system)
    fig.tight_layout()

    tag = outcome_tag(row["error_px"], row.get("failure_class"))
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{tag}_{row['architecture']}_{row['sample_id']}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def pair_panel(sample, row: dict, outdir: Path, manifest: str, system: str) -> Path:
    """Reference, Search, and the true target crop side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.85]})
    axes[0].imshow(sample.reference, cmap="gray")
    axes[0].set_title("Reference — 1000×1000 @ 1 nm/px\n(1 µm FOV)", fontsize=10)

    axes[1].imshow(sample.search, cmap="gray")
    h = REFERENCE_FOOTPRINT_PX / 2.0
    axes[1].add_patch(plt.Rectangle((row["true_x"] - h, row["true_y"] - h), 2 * h, 2 * h,
                                    fill=False, ec="lime", lw=1.8))
    axes[1].plot(row["pred_x"], row["pred_y"], "x", color="red", ms=12, mew=2.2)
    axes[1].set_title("Search — 1000×1000 @ 10 nm/px\n(10 µm FOV)", fontsize=10)

    x0 = int(round(row["true_x"] - h)); y0 = int(round(row["true_y"] - h))
    x0 = max(0, min(x0, sample.search.shape[1] - REFERENCE_FOOTPRINT_PX))
    y0 = max(0, min(y0, sample.search.shape[0] - REFERENCE_FOOTPRINT_PX))
    crop = sample.search[y0:y0 + REFERENCE_FOOTPRINT_PX, x0:x0 + REFERENCE_FOOTPRINT_PX]
    axes[2].imshow(crop, cmap="gray", interpolation="nearest")
    axes[2].set_title("True target region\n100×100 px at Search scale", fontsize=10)

    for a in axes[:3]:
        a.set_xticks([]); a.set_yticks([])
    _caption(axes[3], row, manifest, system)
    fig.tight_layout()

    tag = outcome_tag(row["error_px"], row.get("failure_class"))
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"pair_{tag}_{row['architecture']}_{row['sample_id']}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def alias_comparison(sample, row: dict, outdir: Path) -> Path:
    """The decisive picture: what the locator picked vs what it should have.

    If these two crops look the same, the failure is a genuine repeat and no
    local matcher can separate them - that is the argument for lattice-aware
    candidates plus the official centre rule, made visually.
    """
    h = REFERENCE_FOOTPRINT_PX // 2

    def crop(cx, cy):
        x0 = int(np.clip(round(cx) - h, 0, sample.search.shape[1] - 2 * h))
        y0 = int(np.clip(round(cy) - h, 0, sample.search.shape[0] - 2 * h))
        return sample.search[y0:y0 + 2 * h, x0:x0 + 2 * h]

    a, b = crop(row["true_x"], row["true_y"]), crop(row["pred_x"], row["pred_y"])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    axes[0].imshow(a, cmap="gray", interpolation="nearest")
    axes[0].set_title("TRUE location", fontsize=11, color="green")
    axes[1].imshow(b, cmap="gray", interpolation="nearest")
    axes[1].set_title("PREDICTED location", fontsize=11, color="red")
    d = a.astype(float) - b.astype(float)
    im = axes[2].imshow(d, cmap="coolwarm", vmin=-80, vmax=80)
    z = row.get("patch_zncc", float("nan"))
    axes[2].set_title(f"difference — patch ZNCC {z:.3f}", fontsize=11)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for a_ in axes:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(
        f"{row['sample_id']} ({row['architecture']}/{row['profile']}) — "
        f"error {row['error_px']:.0f} px, class {row.get('failure_class', '?')}",
        fontsize=12)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"aliascmp_{row['architecture']}_{row['sample_id']}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def _pick(rows: list[dict]) -> dict:
    """One representative sample per interesting outcome, both architectures."""
    chosen: dict[str, dict] = {}

    def take(key, cands, sort=None):
        if key in chosen or not cands:
            return
        chosen[key] = min(cands, key=sort) if sort else cands[0]

    for arch in ("dram", "finfet"):
        sub = [r for r in rows if r["architecture"] == arch]
        take(f"success_easy_{arch}", [r for r in sub if r["error_px"] <= 1.0],
             sort=lambda r: r["error_px"])
        take(f"success_hard_{arch}",
             [r for r in sub if 1.0 < r["error_px"] <= TOL_PX],
             sort=lambda r: -r["error_px"])
        take(f"failure_alias_{arch}",
             [r for r in sub if r.get("failure_class") == "periodic_alias"],
             sort=lambda r: -r["error_px"])
    take("failure_low_snr",
         [r for r in rows if r.get("failure_class") == "low_snr"],
         sort=lambda r: -r["error_px"])
    take("failure_wrong_transform",
         [r for r in rows if r.get("failure_class") == "wrong_transform"],
         sort=lambda r: -r["error_px"])
    take("failure_matcher_error",
         [r for r in rows if r.get("failure_class") == "matcher_error"],
         sort=lambda r: -r["error_px"])
    return chosen


def build_gallery(rows: list[dict], outdir: Path, manifest: str, system: str,
                  supersample: int = 2, verbose: bool = True) -> dict:
    """Render the standard evidence set. Returns {slot: [paths]}."""
    picks = _pick(rows)
    made: dict[str, list[str]] = {}
    for slot, row in picks.items():
        try:
            sample = render({"seed": row["seed"], "architecture": row["architecture"],
                             "profile": row["profile"]}, supersample)
            paths = [localization_overlay(sample, row, outdir, manifest, system),
                     pair_panel(sample, row, outdir, manifest, system)]
            if row["error_px"] > CATASTROPHIC_PX:
                paths.append(alias_comparison(sample, row, outdir))
            made[slot] = [str(p) for p in paths]
            if verbose:
                print(f"  gallery {slot:28s} {row['sample_id']} "
                      f"err={row['error_px']:.2f}px")
        except Exception as exc:  # figures are best-effort, never fatal
            if verbose:
                print(f"  gallery {slot}: FAILED ({exc})")
    return made
