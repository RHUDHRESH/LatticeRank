"""Localization metrics with honest uncertainty.

Every proportion carries a Wilson interval and every median a bootstrap
interval, because Phase-1 conclusions were drawn from 20 pairs and the whole
point of Phase 2A is to stop doing that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CATASTROPHIC_PX = 25.0
DEFAULT_TOLERANCES = (1.0, 2.0, 5.0, 10.0)


# ------------------------------------------------------------------ intervals


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because accuracies here land near
    0 and near 1, where the normal interval leaves the [0, 1] range.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values, stat=np.median, n_boot: int = 4000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = stat(values[idx], axis=1)
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


# -------------------------------------------------------------------- metrics


def euclidean_error(pred_x, pred_y, true_x, true_y) -> np.ndarray:
    return np.hypot(np.asarray(pred_x, dtype=float) - np.asarray(true_x, dtype=float),
                    np.asarray(pred_y, dtype=float) - np.asarray(true_y, dtype=float))


def accuracy_at_tolerance(errors, tol: float) -> dict:
    errors = np.asarray(errors, dtype=float)
    k = int(np.sum(errors <= tol))
    p, lo, hi = wilson(k, errors.size)
    return {"tolerance_px": tol, "k": k, "n": int(errors.size),
            "acc": p, "ci95_lo": lo, "ci95_hi": hi}


def error_percentiles(errors, qs=(50, 90, 95, 99)) -> dict:
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {f"p{q}_error_px": float("nan") for q in qs}
    out = {f"p{q}_error_px": float(np.percentile(errors, q)) for q in qs}
    out["mean_error_px"] = float(errors.mean())
    out["max_error_px"] = float(errors.max())
    lo, hi = bootstrap_ci(errors)
    out["median_error_px_ci95"] = [lo, hi]
    return out


def catastrophic_rate(errors, threshold: float = CATASTROPHIC_PX) -> dict:
    errors = np.asarray(errors, dtype=float)
    k = int(np.sum(errors > threshold))
    p, lo, hi = wilson(k, errors.size)
    return {"threshold_px": threshold, "count": k, "n": int(errors.size),
            "rate": p, "ci95_lo": lo, "ci95_hi": hi}


def catastrophic_alias_rate(errors, failure_classes, threshold: float = CATASTROPHIC_PX) -> dict:
    """Fraction of catastrophic failures that are periodic aliases.

    Reported with an interval: the Phase-2A gate is a threshold on this number,
    and a point estimate near the threshold with a wide interval is not a pass.
    """
    errors = np.asarray(errors, dtype=float)
    cls = np.asarray(list(failure_classes), dtype=object)
    cat = errors > threshold
    n = int(cat.sum())
    if n == 0:
        return {"catastrophic_total": 0, "alias_count": 0, "alias_fraction": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    k = int(np.sum(cls[cat] == "periodic_alias"))
    p, lo, hi = wilson(k, n)
    return {"catastrophic_total": n, "alias_count": k, "alias_fraction": p,
            "ci95_lo": lo, "ci95_hi": hi}


def runtime_summary(runtime_ms) -> dict:
    r = np.asarray(runtime_ms, dtype=float)
    if r.size == 0:
        return {}
    return {"mean_runtime_ms": float(r.mean()), "median_runtime_ms": float(np.median(r)),
            "p95_runtime_ms": float(np.percentile(r, 95)),
            "max_runtime_ms": float(r.max())}


def _slice(errors, tolerances) -> dict:
    errors = np.asarray(errors, dtype=float)
    out = {"pairs": int(errors.size)}
    if errors.size == 0:
        return out
    out["accuracy"] = {str(int(t)): accuracy_at_tolerance(errors, t) for t in tolerances}
    out.update(error_percentiles(errors))
    out["catastrophic"] = catastrophic_rate(errors)
    return out


def _breakdown(rows, key, tolerances) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r["error_px"])
    return {k: _slice(v, tolerances) for k, v in sorted(groups.items())}


def architecture_breakdown(rows, tolerances=DEFAULT_TOLERANCES) -> dict:
    return _breakdown(rows, "architecture", tolerances)


def profile_breakdown(rows, tolerances=DEFAULT_TOLERANCES) -> dict:
    return _breakdown(rows, "profile", tolerances)


def summarize(rows: list[dict], tolerances=DEFAULT_TOLERANCES,
              tolerance_px: float = 5.0) -> dict:
    """Full standardized summary for one system on one manifest."""
    errors = np.array([r["error_px"] for r in rows], dtype=float)
    out = {"pairs": len(rows)}
    out.update(_slice(errors, tolerances))
    out.update(runtime_summary([r.get("runtime_ms", float("nan")) for r in rows]))

    if any("accepted" in r for r in rows):
        acc = np.array([bool(r.get("accepted", True)) for r in rows])
        p, lo, hi = wilson(int(acc.sum()), acc.size)
        out["accepted_coverage"] = p
        out["accepted_coverage_ci95"] = [lo, hi]
        out["accepted_incorrect_at_tolerance"] = int(np.sum(acc & (errors > tolerance_px)))
        out["accepted_incorrect_gt25px"] = int(np.sum(acc & (errors > CATASTROPHIC_PX)))
    if any("failure_class" in r for r in rows):
        out["catastrophic_alias"] = catastrophic_alias_rate(
            errors, [r.get("failure_class", "") for r in rows])

    out["by_architecture"] = architecture_breakdown(rows, tolerances)
    out["by_profile"] = profile_breakdown(rows, tolerances)
    archs = sorted({r["architecture"] for r in rows})
    profs = sorted({r["profile"] for r in rows})
    out["by_architecture_profile"] = {
        f"{a}|{p}": _slice([r["error_px"] for r in rows
                            if r["architecture"] == a and r["profile"] == p], tolerances)
        for a in archs for p in profs
    }
    return out


# ------------------------------------------------------------ Taguchi helpers


def sn_smaller_is_better(errors) -> float:
    """Taguchi S/N for a smaller-is-better response.

    SNR = -10 log10(mean(y^2)). Higher is better. Applied to localization
    error in pixels.
    """
    y = np.asarray(errors, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(-10.0 * np.log10(np.mean(y * y)))


def sn_larger_is_better(values) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return float("nan")
    return float(-10.0 * np.log10(np.mean(1.0 / (y * y))))


@dataclass
class AnovaRow:
    source: str
    ss: float
    dof: int
    ms: float
    f: float
    p: float
    pct_contribution: float


def anova_oneway_factors(design: list[dict], responses: list[float],
                         factors: list[str]) -> list[AnovaRow]:
    """Main-effects ANOVA for an orthogonal design.

    Assumptions, stated because they matter for interpretation:
      * main effects only - two-factor interactions in an L27 with 8 assigned
        factors are confounded with unassigned columns and land in residual;
      * balanced/orthogonal design, so factor sums of squares are independent
        and no adjustment for order is needed;
      * residual is used as the error term. With 8 factors this leaves 26 - 16
        = 10 df, which is thin - report F with that caveat rather than
        treating p-values as precise.
    """
    y = np.asarray(responses, dtype=float)
    n = y.size
    grand = float(y.mean())
    ss_total = float(np.sum((y - grand) ** 2))

    rows, ss_factors, dof_factors = [], 0.0, 0
    per_factor = []
    for f in factors:
        levels = sorted({d[f] for d in design})
        ss = 0.0
        for lv in levels:
            idx = [i for i, d in enumerate(design) if d[f] == lv]
            if not idx:
                continue
            ss += len(idx) * (float(y[idx].mean()) - grand) ** 2
        dof = len(levels) - 1
        per_factor.append((f, ss, dof))
        ss_factors += ss
        dof_factors += dof

    ss_err = max(ss_total - ss_factors, 0.0)
    dof_err = max(n - 1 - dof_factors, 0)
    ms_err = ss_err / dof_err if dof_err > 0 else float("nan")

    try:
        from scipy import stats as _st
    except ImportError:
        _st = None

    for f, ss, dof in per_factor:
        ms = ss / dof if dof else float("nan")
        fstat = ms / ms_err if (ms_err and math.isfinite(ms_err) and ms_err > 0) else float("nan")
        if _st is not None and math.isfinite(fstat) and dof_err > 0:
            p = float(_st.f.sf(fstat, dof, dof_err))
        else:
            p = float("nan")
        rows.append(AnovaRow(f, ss, dof, ms, fstat, p,
                             100.0 * ss / ss_total if ss_total > 0 else float("nan")))
    rows.append(AnovaRow("residual", ss_err, dof_err, ms_err, float("nan"), float("nan"),
                         100.0 * ss_err / ss_total if ss_total > 0 else float("nan")))
    return rows
