#!/usr/bin/env python3
"""Run sequence-validated recursive clustering, using family 308 by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from te_subfamily_clustering.recursive import RecursiveConfig
from te_subfamily_clustering.sequence_validation import require_external_tools
from te_subfamily_clustering.workflows import run_full_recursive


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["372", "308"], default="308")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--swipe-threads", type=int, default=2)
    args = parser.parse_args()

    tool_paths = require_external_tools()
    print("External tools:")
    for name, path in tool_paths.items():
        print(f"  {name}: {path}")

    data_directory = REPOSITORY_ROOT / "examples" / "data" / f"family_{args.family}"
    output_directory = args.output_dir or (
        REPOSITORY_ROOT / "outputs" / f"full_recursive_family_{args.family}"
    )
    metrics = run_full_recursive(
        family_name=f"family_{args.family}",
        mldist_path=data_directory / "pairwise_ml_distance.mldist",
        alignment_path=data_directory / "alignment.fasta",
        labels_path=data_directory / "labels.csv",
        output_directory=output_directory,
        config=RecursiveConfig(swipe_threads=args.swipe_threads),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
