"""Command-line interface for the public TE subfamily clustering package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import FixedBinaryConfig
from .recursive import RecursiveConfig
from .workflows import (
    run_fixed_binary_baseline,
    run_full_recursive,
    run_quick_start,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quick = subparsers.add_parser("quick-start", help="Run one-shot spectral clustering")
    quick.add_argument("--mldist", required=True, type=Path)
    quick.add_argument("--labels", required=True, type=Path)
    quick.add_argument("--output-dir", required=True, type=Path)

    recursive = subparsers.add_parser(
        "recursive", help="Run silhouette-selected recursive clustering"
    )
    recursive.add_argument("--family", default="family")
    recursive.add_argument("--mldist", required=True, type=Path)
    recursive.add_argument("--alignment", required=True, type=Path)
    recursive.add_argument("--labels", required=True, type=Path)
    recursive.add_argument("--output-dir", required=True, type=Path)
    recursive.add_argument("--swipe-threads", type=int, default=2)
    recursive.add_argument("--temporary-parent", type=Path, default=None)

    fixed = subparsers.add_parser(
        "fixed-binary", help="Run the repeated fixed-k=2 recursive baseline"
    )
    fixed.add_argument("--family", default="family")
    fixed.add_argument("--mldist", required=True, type=Path)
    fixed.add_argument("--alignment", required=True, type=Path)
    fixed.add_argument("--labels", required=True, type=Path)
    fixed.add_argument("--output-dir", required=True, type=Path)
    fixed.add_argument("--swipe-threads", type=int, default=2)
    fixed.add_argument("--temporary-parent", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quick-start":
        metrics = run_quick_start(
            mldist_path=args.mldist,
            labels_path=args.labels,
            output_directory=args.output_dir,
        )
    elif args.command == "recursive":
        metrics = run_full_recursive(
            family_name=args.family,
            mldist_path=args.mldist,
            alignment_path=args.alignment,
            labels_path=args.labels,
            output_directory=args.output_dir,
            config=RecursiveConfig(
                swipe_threads=args.swipe_threads,
                temporary_parent=args.temporary_parent,
            ),
        )
    else:
        metrics = run_fixed_binary_baseline(
            family_name=args.family,
            mldist_path=args.mldist,
            alignment_path=args.alignment,
            labels_path=args.labels,
            output_directory=args.output_dir,
            config=FixedBinaryConfig(
                swipe_threads=args.swipe_threads,
                temporary_parent=args.temporary_parent,
            ),
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
