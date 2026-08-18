#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample
from driftforge.splits import read_manifest


def _save_png_atomic(array, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array).save(temporary, format="PNG", optimize=True)
    # Decode/CRC verification happens before the atomic rename. A partial file
    # can therefore never masquerade as a completed dataset item.
    with Image.open(temporary) as image:
        image.verify()
    temporary.replace(path)


def _records_from_args(args: argparse.Namespace) -> list[dict]:
    if args.manifest:
        return read_manifest(args.manifest)
    architectures = [args.architecture] if args.architecture != "both" else ["dram", "finfet"]
    return [
        {
            "id": f"custom-{i:06d}",
            "split": "custom",
            "seed": args.seed_start + i,
            "scene_id": f"scene-{args.seed_start + i}",
            "architecture": architectures[i % len(architectures)],
            "profile": args.profile,
        }
        for i in range(args.count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize paired synthetic SEM images and coordinate labels.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="JSONL manifest made by make_manifests.py")
    source.add_argument("--count", type=int, default=30, help="Number of custom pairs when no manifest is supplied")
    parser.add_argument("--architecture", choices=("dram", "finfet", "both"), default="both")
    parser.add_argument("--profile", choices=("standard", "hard", "boundary", "ambiguous", "ood"), default="standard")
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hide-labels", action="store_true", help="Create a public test folder without ground-truth files")
    args = parser.parse_args()

    records = _records_from_args(args)[args.start:]
    if args.limit is not None:
        records = records[:args.limit]
    refs = args.output_dir / "reference"
    searches = args.output_dir / "search"
    metadata_dir = args.output_dir / "metadata"
    refs.mkdir(parents=True, exist_ok=True)
    searches.mkdir(parents=True, exist_ok=True)
    if not args.hide_labels:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for index, record in enumerate(records, start=1):
        sample = generate_sample(
            seed=int(record["seed"]),
            architecture=record["architecture"],
            profile=record["profile"],
            search_supersample=args.search_supersample,
        )
        sample_id = record["id"]
        _save_png_atomic(sample.reference, refs / f"{sample_id}.png")
        _save_png_atomic(sample.search, searches / f"{sample_id}.png")
        if not args.hide_labels:
            meta = sample.metadata()
            meta["id"] = sample_id
            meta["split"] = record["split"]
            meta["scene_id"] = record["scene_id"]
            (metadata_dir / f"{sample_id}.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            labels.append({
                "id": sample_id,
                "reference": f"reference/{sample_id}.png",
                "search": f"search/{sample_id}.png",
                "x": f"{sample.gt_x:.6f}",
                "y": f"{sample.gt_y:.6f}",
                "architecture": sample.architecture,
                "profile": sample.profile,
                "scene_id": record["scene_id"],
            })
        print(f"[{index}/{len(records)}] {sample_id}", flush=True)

    if not args.hide_labels:
        with (args.output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["id", "reference", "search", "x", "y", "architecture", "profile", "scene_id"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(labels)
    (args.output_dir / "DATASET_INFO.json").write_text(
        json.dumps({
            "count": len(records),
            "labels_included": not args.hide_labels,
            "search_supersample": args.search_supersample,
            "coordinate_convention": "x=column, y=row, origin at top-left, units=search pixels",
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
