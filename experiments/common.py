"""Shared contracts for the LatticeRank experiment suite.

Two things live here and nothing else: the data contracts every system and
every future model must speak, and the registry that lets systems be compared
without rewriting the benchmark runner.

Phase 2A implements exactly one system (`driftforge_v1_leakfree`). The rest are
declared with `implemented=False` so the registry is the single honest record
of what exists. `benchmark.py` refuses to run an unimplemented system rather
than emitting a placeholder row that could later be mistaken for a result.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"

CATASTROPHIC_PX = 25.0
SEARCH_NM_PER_PX = 10.0
REFERENCE_FOOTPRINT_PX = 100  # Reference FOV expressed at Search scale


# ------------------------------------------------------------------ contracts


@dataclass
class Prediction:
    """What every system returns for one (reference, search) pair."""

    x: float
    y: float
    runtime_ms: float = float("nan")
    confidence: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {"pred_x": self.x, "pred_y": self.y,
                "runtime_ms": self.runtime_ms, "confidence": self.confidence}


@dataclass
class CandidateRecord:
    """One candidate location, the unit the meta-ranker ensemble learns on.

    This is the ML research data contract. Fields that no current expert
    produces are left as NaN on purpose: a placeholder number here would be
    indistinguishable from a measurement later, and the Phase-1 audit exists
    because exactly that kind of contamination went unnoticed.

    `is_positive` is derived from `error_px` against a tolerance at build
    time, never hand-set.
    """

    scene_id: str
    sample_id: str
    architecture: str
    profile: str

    candidate_x: float
    candidate_y: float
    true_x: float
    true_y: float
    error_px: float
    is_positive: bool

    # --- lattice geometry -------------------------------------------------
    lattice_m: float = float("nan")
    lattice_n: float = float("nan")
    lattice_residual: float = float("nan")

    # --- classical evidence (available now) -------------------------------
    zncc_score: float = float("nan")
    midband_score: float = float("nan")
    peak_gap: float = float("nan")
    psr: float = float("nan")
    scale: float = float("nan")
    rotation: float = float("nan")

    # --- acquisition / context estimates ----------------------------------
    noise_estimate: float = float("nan")
    blur_estimate: float = float("nan")
    local_entropy: float = float("nan")
    boundary_context_score: float = float("nan")

    # --- experts that do not exist yet (Phase 2B/2C) ----------------------
    directionality_score: float = float("nan")
    phase_score: float = float("nan")
    siamese_similarity: float = float("nan")
    expert_vote_count: float = float("nan")
    lattice_confidence: float = float("nan")

    #: Never a feature. Distance-to-centre is the official *tie-break*, applied
    #: after an equivalence set exists. Letting a ranker see it turns "which
    #: candidate looks right" into "which candidate is central" (Phase-1 R5).
    FORBIDDEN_FEATURES = ("distance_to_centre", "distance_to_center", "centre_dist")

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def feature_names(cls) -> list[str]:
        """Columns a model may train on."""
        skip = {"scene_id", "sample_id", "architecture", "profile",
                "candidate_x", "candidate_y", "true_x", "true_y",
                "error_px", "is_positive"}
        return [f.name for f in fields(cls) if f.name not in skip]


def build_candidate(sample_id: str, scene_id: str, architecture: str, profile: str,
                    cx: float, cy: float, true_x: float, true_y: float,
                    tolerance_px: float = 5.0, **evidence) -> CandidateRecord:
    err = math.hypot(cx - true_x, cy - true_y)
    known = {f.name for f in fields(CandidateRecord)}
    unknown = set(evidence) - known
    if unknown:
        raise ValueError(f"unknown candidate evidence fields: {sorted(unknown)}")
    for bad in CandidateRecord.FORBIDDEN_FEATURES:
        if bad in evidence:
            raise ValueError(f"{bad} may not be a candidate feature (Phase-1 R5)")
    return CandidateRecord(
        scene_id=scene_id, sample_id=sample_id, architecture=architecture,
        profile=profile, candidate_x=cx, candidate_y=cy, true_x=true_x,
        true_y=true_y, error_px=err, is_positive=bool(err <= tolerance_px),
        **evidence)


# ------------------------------------------------------------------- registry


@dataclass
class SystemSpec:
    key: str
    label: str
    description: str
    implemented: bool
    factory: Callable[[], Callable] | None = None
    phase: str = ""


SYSTEMS: dict[str, SystemSpec] = {}


def register(spec: SystemSpec) -> SystemSpec:
    SYSTEMS[spec.key] = spec
    return spec


def _classical_v1():
    """The shipped v1 locator, unchanged. Phase 2A measures this."""
    from driftforge.baseline import locate

    def predict(reference: np.ndarray, search: np.ndarray) -> Prediction:
        import time
        t0 = time.perf_counter()
        r = locate(reference, search)
        dt = (time.perf_counter() - t0) * 1000.0
        return Prediction(
            x=r.x, y=r.y, runtime_ms=dt, confidence=float(r.score),
            diagnostics={"peak_gap": r.peak_gap, "psr": r.psr, "scale": r.scale,
                         "rotation_deg": r.rotation_deg, "accepted": bool(r.accepted),
                         "ambiguity_detected": bool(r.ambiguity_detected)})
    return predict


register(SystemSpec("sponsor_zncc", "B0 sponsor ZNCC",
                    "Organiser baseline: multi-scale cv2 ZNCC, global arg-max, no tie rule. "
                    "Needs OpenCV, which DriftForge deliberately does not depend on.",
                    implemented=False, phase="2B"))
register(SystemSpec("driftforge_v1_leakfree", "B1 DriftForge v1 (leak-free)",
                    "Shipped v1 locator, unmodified, on the de-correlated generator.",
                    implemented=True, factory=_classical_v1, phase="2A"))
register(SystemSpec("lattice_candidates", "B2 lattice candidate system",
                    "Lattice estimator + centred-rectangular candidate expansion.",
                    implemented=False, phase="2B"))
register(SystemSpec("lattice_directionality", "B3 lattice + directionality",
                    "B2 plus the polarity-insensitive edge-orientation channel.",
                    implemented=False, phase="2B"))
register(SystemSpec("siamese_random_neg", "B4 Siamese (random negatives)",
                    "Metric-learning control arm: negatives sampled at random.",
                    implemented=False, phase="2C"))
register(SystemSpec("siamese_hard_neg", "B5 Siamese (periodic hard negatives)",
                    "Treatment arm: negatives are integer lattice translates.",
                    implemented=False, phase="2C"))
register(SystemSpec("classical_ensemble", "B6 classical ensemble ranker",
                    "Multi-model tabular ensemble (HGB + RF + ExtraTrees + calibrated "
                    "linear) stacked over classical candidate evidence.",
                    implemented=False, phase="2D"))
register(SystemSpec("siamese_plus_ensemble", "B7 Siamese + ensemble ranker",
                    "B6 with learned patch similarity added as a feature.",
                    implemented=False, phase="2D"))
register(SystemSpec("calibrated_ensemble", "B8 calibrated ensemble",
                    "B7 plus probability calibration and the evidential equivalence "
                    "set feeding the official nearest-centre tie rule.",
                    implemented=False, phase="2D"))
register(SystemSpec("ensemble_plus_subpixel", "B9 ensemble + subpixel",
                    "B8 plus upsampled-DFT refinement of the selected cell.",
                    implemented=False, phase="2D"))


def get_system(key: str) -> SystemSpec:
    if key not in SYSTEMS:
        raise SystemExit(f"unknown system {key!r}. known: {sorted(SYSTEMS)}")
    spec = SYSTEMS[key]
    if not spec.implemented:
        raise SystemExit(
            f"system {key!r} ({spec.label}) is declared but not implemented "
            f"(planned for Phase {spec.phase}). Refusing to emit placeholder results.")
    return spec


# ---------------------------------------------------------------- data access


def load_manifest(path: str | Path) -> list[dict]:
    from driftforge.splits import read_manifest
    return read_manifest(path)


def render(record: dict, supersample: int = 2):
    from driftforge.generator import generate_sample
    return generate_sample(int(record["seed"]), record["architecture"],
                           record["profile"], supersample)


def local_entropy(patch: np.ndarray, bins: int = 32) -> float:
    h, _ = np.histogram(np.asarray(patch, dtype=float).ravel(), bins=bins, range=(0, 255))
    p = h.astype(float) / max(h.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def estimate_noise(img: np.ndarray) -> float:
    """Robust high-frequency noise estimate (MAD of a Laplacian residual)."""
    a = np.asarray(img, dtype=float)
    lap = (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(np.median(np.abs(lap - np.median(lap))) * 1.4826 / math.sqrt(20.0))


def estimate_blur(img: np.ndarray) -> float:
    """Larger value = sharper image (variance of the Laplacian)."""
    a = np.asarray(img, dtype=float)
    lap = (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(lap.var())
