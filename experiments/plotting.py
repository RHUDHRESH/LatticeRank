"""Presentation-grade figures. matplotlib only, no seaborn.

Rules followed here: every axis is labelled with units, nothing is styled in a
way that hides data, and no plot smooths or clips outliers - the catastrophic
tail is the subject of this project, so it is never cosmetically removed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 200
CATASTROPHIC_PX = 25.0
TOL_PX = 5.0


def _save(fig, outdir: Path, name: str, also_vector: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    if also_vector:
        fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _ref_lines(ax):
    ax.axvline(TOL_PX, color="tab:green", ls="--", lw=1.2, label=f"{TOL_PX:g} px tolerance")
    ax.axvline(CATASTROPHIC_PX, color="tab:red", ls="--", lw=1.2,
               label=f"{CATASTROPHIC_PX:g} px catastrophic")


def error_cdf(errors, outdir: Path, name: str = "error_cdf", label: str | None = None) -> Path:
    e = np.sort(np.asarray(errors, dtype=float))
    y = np.arange(1, e.size + 1) / e.size
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.step(np.maximum(e, 1e-3), y, where="post", lw=2.0, label=label or "system")
    ax.set_xscale("log")
    ax.set_xlabel("localization error (Search pixels, log scale)")
    ax.set_ylabel("fraction of pairs with error ≤ x")
    ax.set_title("Localization error CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    _ref_lines(ax)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, outdir, name)


def error_cdf_multi(series: dict, outdir: Path, name: str = "error_cdf_all_models") -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for lbl, errors in series.items():
        e = np.sort(np.asarray(errors, dtype=float))
        y = np.arange(1, e.size + 1) / e.size
        ax.step(np.maximum(e, 1e-3), y, where="post", lw=1.9, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("localization error (Search pixels, log scale)")
    ax.set_ylabel("fraction of pairs with error ≤ x")
    ax.set_title("Localization error CDF by system")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    _ref_lines(ax)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, outdir, name)


def error_histogram(errors, outdir: Path, name: str = "error_histogram") -> Path:
    e = np.asarray(errors, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    pos = np.maximum(e, 1e-2)
    bins = np.logspace(np.log10(pos.min()), np.log10(max(pos.max(), 1.0)), 45)
    ax.hist(pos, bins=bins, color="tab:blue", edgecolor="black", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("localization error (Search pixels, log scale)")
    ax.set_ylabel("number of pairs")
    ax.set_title(f"Error distribution (n={e.size}) — note the bimodality")
    ax.grid(alpha=0.3, axis="y")
    _ref_lines(ax)
    ax.legend(loc="upper center", fontsize=9)
    return _save(fig, outdir, name)


def _grouped_accuracy(ax, groups: dict, tol: float):
    names = list(groups)
    acc = [groups[g]["acc"] for g in names]
    lo = [groups[g]["acc"] - groups[g]["ci95_lo"] for g in names]
    hi = [groups[g]["ci95_hi"] - groups[g]["acc"] for g in names]
    n = [groups[g]["n"] for g in names]
    xs = np.arange(len(names))
    ax.bar(xs, acc, yerr=[lo, hi], capsize=4, color="tab:blue",
           edgecolor="black", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(names, n)], fontsize=9)
    ax.set_ylabel(f"accuracy ≤ {tol:g} px")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    for x, a in zip(xs, acc):
        ax.text(x, a + 0.02, f"{a:.1%}", ha="center", fontsize=9)


def accuracy_by_profile(summary: dict, outdir: Path, tol: float = TOL_PX,
                        name: str = "accuracy_by_profile") -> Path:
    groups = {k: v["accuracy"][str(int(tol))]
              for k, v in summary["by_profile"].items() if v.get("pairs")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _grouped_accuracy(ax, groups, tol)
    ax.set_title("Accuracy by scene profile (95% Wilson intervals)")
    return _save(fig, outdir, name)


def accuracy_dram_vs_finfet(summary: dict, outdir: Path, tol: float = TOL_PX,
                            name: str = "accuracy_dram_vs_finfet") -> Path:
    groups = {k: v["accuracy"][str(int(tol))]
              for k, v in summary["by_architecture"].items() if v.get("pairs")}
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _grouped_accuracy(ax, groups, tol)
    ax.set_title("DRAM vs FinFET (95% Wilson intervals)")
    return _save(fig, outdir, name)


def runtime_distribution(runtime_ms, outdir: Path, name: str = "runtime_distribution") -> Path:
    r = np.asarray(runtime_ms, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(r, bins=40, color="tab:orange", edgecolor="black", lw=0.4)
    ax.axvline(float(np.median(r)), color="black", ls="--", lw=1.4,
               label=f"median {np.median(r):.0f} ms")
    ax.axvline(1000.0, color="tab:green", ls=":", lw=1.6, label="1 s/pair target")
    ax.set_xlabel("locator runtime per pair (ms)")
    ax.set_ylabel("number of pairs")
    ax.set_title("Runtime distribution")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, outdir, name)


def runtime_vs_error(runtime_ms, errors, outdir: Path, name: str = "runtime_vs_error") -> Path:
    r = np.asarray(runtime_ms, dtype=float)
    e = np.maximum(np.asarray(errors, dtype=float), 1e-2)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ok = e <= TOL_PX
    ax.scatter(r[ok], e[ok], s=14, alpha=0.6, color="tab:green", label=f"≤ {TOL_PX:g} px")
    ax.scatter(r[~ok], e[~ok], s=14, alpha=0.6, color="tab:red", label=f"> {TOL_PX:g} px")
    ax.set_yscale("log")
    ax.set_xlabel("runtime per pair (ms)")
    ax.set_ylabel("localization error (px, log scale)")
    ax.set_title("Runtime vs error — accuracy here is not bought with time")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    return _save(fig, outdir, name)


def failure_class_bar(counts: dict, outdir: Path, name: str = "failure_classes") -> Path:
    names = list(counts)
    vals = [counts[k] for k in names]
    total = max(sum(vals), 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(np.arange(len(names)), vals, color="tab:purple", edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("number of catastrophic failures (>25 px)")
    ax.set_title(f"Catastrophic failure classes (n={total})")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v}\n{v/total:.0%}", ha="center", fontsize=9)
    return _save(fig, outdir, name)


def model_comparison_bar(values: dict, ylabel: str, title: str, outdir: Path,
                         name: str, errbars: dict | None = None,
                         percent: bool = False) -> Path:
    names = list(values)
    vals = [values[k] for k in names]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(names) + 2), 4.4))
    yerr = None
    if errbars:
        yerr = [[values[k] - errbars[k][0] for k in names],
                [errbars[k][1] - values[k] for k in names]]
    ax.bar(np.arange(len(names)), vals, yerr=yerr, capsize=4,
           color="tab:blue", edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1%}" if percent else f"{v:.3g}",
                ha="center", va="bottom", fontsize=9)
    return _save(fig, outdir, name)


def runtime_accuracy_frontier(points: dict, outdir: Path,
                              name: str = "runtime_accuracy_frontier") -> Path:
    """points: {label: (runtime_ms, accuracy_at_5px)}"""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for lbl, (r, a) in points.items():
        ax.scatter(r, a, s=70, edgecolor="black", zorder=3)
        ax.annotate(lbl, (r, a), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.axvline(1000.0, color="tab:green", ls=":", lw=1.6, label="1 s/pair target")
    ax.set_xscale("log")
    ax.set_xlabel("runtime per pair (ms, log scale)")
    ax.set_ylabel("accuracy ≤ 5 px")
    ax.set_ylim(0, 1.0)
    ax.set_title("Runtime / accuracy frontier")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    return _save(fig, outdir, name)


# ------------------------------------------------------- training curves (2C)


def training_curves(log_rows: list[dict], outdir: Path) -> list[Path]:
    """Plots for the Phase-2C training log contract (see experiments/README)."""
    if not log_rows:
        return []
    ep = [r["epoch"] for r in log_rows]
    out = []

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ep, [r["train_loss"] for r in log_rows], lw=1.8, label="train")
    ax.plot(ep, [r["val_loss"] for r in log_rows], lw=1.8, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training / validation loss"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    out.append(_save(fig, outdir, "training_loss"))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, lbl, c in (("positive_similarity_mean", "positive", "tab:green"),
                        ("hard_negative_similarity_mean", "hard negative", "tab:red"),
                        ("random_negative_similarity_mean", "random negative", "tab:gray")):
        if key in log_rows[0]:
            m = np.array([r[key] for r in log_rows], dtype=float)
            ax.plot(ep, m, lw=1.8, label=lbl, color=c)
            sk = key.replace("_mean", "_std")
            if sk in log_rows[0]:
                s = np.array([r[sk] for r in log_rows], dtype=float)
                ax.fill_between(ep, m - s, m + s, alpha=0.15, color=c)
    ax.set_xlabel("epoch"); ax.set_ylabel("embedding similarity")
    ax.set_title("Embedding separation — the hard-negative gap is the point")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    out.append(_save(fig, outdir, "embedding_separation"))

    if "candidate_rank_accuracy" in log_rows[0]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(ep, [r["candidate_rank_accuracy"] for r in log_rows], lw=1.8, color="tab:purple")
        ax.set_xlabel("epoch"); ax.set_ylabel("top-1 candidate rank accuracy")
        ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
        ax.set_title("Candidate ranking accuracy")
        out.append(_save(fig, outdir, "candidate_rank_accuracy"))
    return out


# ------------------------------------------------------------- Taguchi (2E)


def taguchi_main_effects(effects: dict, ylabel: str, outdir: Path,
                         name: str = "taguchi_main_effects") -> Path:
    """effects: {factor: {level: value}} -> one panel per factor."""
    facs = list(effects)
    ncol = min(4, max(1, len(facs)))
    nrow = int(np.ceil(len(facs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), squeeze=False)
    allv = [v for f in facs for v in effects[f].values()]
    lo, hi = min(allv), max(allv)
    pad = 0.08 * (hi - lo + 1e-9)
    for i, f in enumerate(facs):
        ax = axes[i // ncol][i % ncol]
        lv = sorted(effects[f])
        ax.plot(range(len(lv)), [effects[f][k] for k in lv], "o-", lw=1.8, color="tab:blue")
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([str(k) for k in lv], fontsize=8)
        ax.set_title(f, fontsize=10)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
        if i % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=9)
    for j in range(len(facs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Taguchi main effects — {ylabel}", fontsize=12)
    fig.tight_layout()
    return _save(fig, outdir, name)


def anova_contribution(rows, outdir: Path, name: str = "anova_contribution") -> Path:
    rows = [r for r in rows if r.source != "residual"]
    names = [r.source for r in rows]
    pct = [r.pct_contribution for r in rows]
    order = np.argsort(pct)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(np.arange(len(names)), [pct[i] for i in order],
           color="tab:teal", edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("percent contribution to total sum of squares")
    ax.set_title("ANOVA factor contribution")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, outdir, name)
