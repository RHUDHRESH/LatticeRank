#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone DriftForge classical baseline inference.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("search", type=Path)
    parser.add_argument("--json", action="store_true", help="Print diagnostics as JSON")
    parser.add_argument("--strict", action="store_true", help="Exit 2 when confidence gate abstains")
    args = parser.parse_args()
    reference = np.asarray(Image.open(args.reference).convert("L"))
    search = np.asarray(Image.open(args.search).convert("L"))
    result = locate(reference, search)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"({result.x:.6f}, {result.y:.6f})")
    if args.strict and not result.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

