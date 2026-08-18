#!/usr/bin/env python3
"""Emit the Phase-2A deliverable filenames required by the brief.

`run_leakfree_baseline.py` writes tag-prefixed files so several runs can
coexist without overwriting each other. The Phase-2A brief names specific
artifacts, so this copies the authoritative run to those names rather than
renaming the generator's output and losing that property.

Copies, not moves: the tagged originals stay as the provenance record of which
run produced which file.

Usage:
    python reports/phase2a/diagnostics/emit_required_artifacts.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]

#: source stem -> required deliverable name
MAPPING = {
    "leakfree_baseline_predictions.csv": "leakfree_predictions.csv",
    "leakfree_baseline_catastrophic.csv": "catastrophic_failures.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=PROJECT / "reports/phase2a")
    args = ap.parse_args()

    missing = []
    for src_name, dst_name in MAPPING.items():
        src = args.dir / src_name
        if not src.exists():
            missing.append(src_name)
            continue
        shutil.copyfile(src, args.dir / dst_name)
        print(f"{src_name}  ->  {dst_name}")

    if missing:
        print(f"\nmissing sources: {missing}", file=sys.stderr)
        raise SystemExit(1)
    print("\nrequired artifacts emitted")


if __name__ == "__main__":
    main()
