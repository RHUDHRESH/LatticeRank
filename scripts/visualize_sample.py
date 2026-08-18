#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a labelled generator QA panel.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=("dram", "finfet"), default="dram")
    parser.add_argument("--profile", choices=("standard", "hard", "boundary", "ambiguous", "ood"), default="standard")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = generate_sample(args.seed, args.architecture, args.profile, search_supersample=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(sample.reference, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Reference: 1000 nm FOV")
    axes[1].imshow(sample.search, cmap="gray", vmin=0, vmax=255)
    axes[1].add_patch(Rectangle((sample.gt_x - 50, sample.gt_y - 50), 100, 100, fill=False, color="#ff3b30", lw=1.5))
    axes[1].plot(sample.gt_x, sample.gt_y, "+", color="#ff3b30", ms=12, mew=2)
    axes[1].set_title(f"Search: 10,000 nm FOV — GT ({sample.gt_x:.1f}, {sample.gt_y:.1f})")
    crop = sample.search[
        max(0, int(sample.gt_y) - 80):min(1000, int(sample.gt_y) + 80),
        max(0, int(sample.gt_x) - 80):min(1000, int(sample.gt_x) + 80),
    ]
    axes[2].imshow(crop, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Search neighbourhood")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"DriftForge | {args.architecture.upper()} | {args.profile} | seed {args.seed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()

