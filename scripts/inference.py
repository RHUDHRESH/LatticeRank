#!/usr/bin/env python3
"""DriftForge v2 inference CLI.

    python scripts/inference.py reference.png search.png
    -> 537.42,481.18

Pipeline (every stage measured on the 80-pair disjoint validation split):
3-channel adaptive candidate harvesting -> scene-relative + structural
identity features -> HGB ranker -> periodic-cancellation residual evidence
-> probability-equivalence set -> official nearest-Search-centre tie rule.
Weights load automatically from driftforge/models/. Always prints a
coordinate; never needs the network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.pipeline import locate_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftForge v2 navigation-error recovery.")
    parser.add_argument("reference", type=Path, help="1000x1000 high-mag Reference PNG")
    parser.add_argument("search", type=Path, help="1000x1000 low-mag Search PNG")
    parser.add_argument("--json", action="store_true", help="print full diagnostics as JSON")
    parser.add_argument("--no-residual", action="store_true",
                        help="disable the residual evidence stage (debug only)")
    args = parser.parse_args()
    reference = np.asarray(Image.open(args.reference).convert("L"))
    search = np.asarray(Image.open(args.search).convert("L"))
    r = locate_v2(reference, search, use_residual=not args.no_residual)
    if args.json:
        print(json.dumps({"x": r.x, "y": r.y, "probability": r.probability,
                          "eq_set_size": r.eq_set_size, "n_candidates": r.n_candidates,
                          "used_residual": r.used_residual, **r.diagnostics}, indent=2))
    else:
        print(f"{r.x:.2f},{r.y:.2f}")


if __name__ == "__main__":
    main()
