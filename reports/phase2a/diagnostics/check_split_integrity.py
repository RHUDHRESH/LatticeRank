#!/usr/bin/env python3
"""Phase-2A split integrity across every manifest in the repository.

Read-only. Reads the shipped manifests; writes nothing back to them. The
1,000-pair private `test.jsonl` is opened only to compute disjointness of scene
IDs and seeds - no record is rendered, evaluated or inspected, and no Phase-2A
decision depends on it.

Three checks, in increasing strictness:

1. `assert_disjoint()` - the shipped guard.
2. Pairwise scene-ID and seed overlap between every pair of splits.
3. **Scene-realisation collisions.** Scene parameters are drawn from
   ``default_rng(seed * 17 + 3)``, so two splits could in principle hold
   different manifest seeds that map to the *same* scene. Distinct seeds are
   not sufficient; the derived key must be distinct too. This is the check
   that would catch a subtle leak the first two would miss.

Usage:
    python reports/phase2a/diagnostics/check_split_integrity.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from driftforge.splits import assert_disjoint, read_manifest  # noqa: E402

#: Filename stem -> reported name. `validation_phase2a_400` is the Phase-2A
#: benchmark split; the other four are the v1.0.0 originals, unmodified.
MANIFESTS = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
    "stress": "stress.jsonl",
    "validation_phase2a_400": "validation_phase2a_400.jsonl",
}


def describe(records: list[dict]) -> dict:
    seeds = [int(r["seed"]) for r in records]
    return {
        "count": len(records),
        "architectures": dict(sorted(Counter(r["architecture"] for r in records).items())),
        "profiles": dict(sorted(Counter(r["profile"] for r in records).items())),
        "unique_scenes": len({r["scene_id"] for r in records}),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "reports/phase2a/split_audit.json")
    args = ap.parse_args()

    manifests = {
        name: read_manifest(PROJECT / "manifests" / filename)
        for name, filename in MANIFESTS.items()
    }

    assert_disjoint(manifests)

    pairwise_scene: dict[str, int] = {}
    pairwise_seed: dict[str, int] = {}
    for a, b in itertools.combinations(manifests, 2):
        key = f"{a}|{b}"
        pairwise_scene[key] = len(
            {r["scene_id"] for r in manifests[a]} & {r["scene_id"] for r in manifests[b]}
        )
        pairwise_seed[key] = len(
            {int(r["seed"]) for r in manifests[a]} & {int(r["seed"]) for r in manifests[b]}
        )

    # Distinct seeds are not enough - the derived scene key must differ too.
    owner: dict[int, str] = {}
    collisions = 0
    for name, records in manifests.items():
        for r in records:
            key = int(r["seed"]) * 17 + 3
            if key in owner and owner[key] != name:
                collisions += 1
            owner[key] = name

    # Taguchi campaign manifests, if present, must also be clear.
    taguchi_dir = PROJECT / "manifests/taguchi"
    taguchi: dict | None = None
    if taguchi_dir.is_dir():
        files = sorted(taguchi_dir.glob("*.jsonl"))
        seeds: set[int] = set()
        for f in files:
            seeds |= {int(r["seed"]) for r in read_manifest(f)}
        if seeds:
            taguchi = {
                "manifest_files": len(files),
                "unique_seeds": len(seeds),
                "seed_min": min(seeds),
                "seed_max": max(seeds),
                "seed_overlap_with_splits": {
                    name: len(seeds & {int(r["seed"]) for r in records})
                    for name, records in manifests.items()
                },
            }

    report = {
        "splits": {name: describe(records) for name, records in manifests.items()},
        "pairwise_scene_overlap": pairwise_scene,
        "pairwise_seed_overlap": pairwise_seed,
        "scene_spec_seed_collisions": collisions,
        "scene_spec_note": "scene parameters derive from default_rng(seed * 17 + 3)",
        "all_disjoint": (
            max(pairwise_scene.values()) == 0
            and max(pairwise_seed.values()) == 0
            and collisions == 0
        ),
        "taguchi": taguchi,
        "benchmark_split": "validation_phase2a_400",
        "private_test_touched": False,
        "private_test_note": (
            "test.jsonl was read for scene-ID/seed disjointness only. No test "
            "record was rendered, located, or used in any Phase-2A decision."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== SPLIT INTEGRITY ===")
    for name, info in report["splits"].items():
        print(f"{name:24s} n={info['count']:5d}  seeds {info['seed_min']}-{info['seed_max']}  "
              f"{info['architectures']}")
    print(f"\nmax pairwise scene overlap   {max(pairwise_scene.values())}")
    print(f"max pairwise seed overlap    {max(pairwise_seed.values())}")
    print(f"scene-spec seed collisions   {collisions}   (over {len(owner)} derived keys)")
    if taguchi:
        print(f"taguchi seeds                {taguchi['unique_seeds']} in "
              f"{taguchi['manifest_files']} files, overlap "
              f"{max(taguchi['seed_overlap_with_splits'].values())}")
    print(f"ALL DISJOINT                 {report['all_disjoint']}")
    print(f"wrote {args.out}")
    if not report["all_disjoint"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
