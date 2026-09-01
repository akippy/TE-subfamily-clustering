#!/usr/bin/env python3
"""Run the Python-only example, using family 372 by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from te_subfamily_clustering.workflows import run_quick_start


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["372", "308"], default="372")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data_directory = REPOSITORY_ROOT / "examples" / "data" / f"family_{args.family}"
    output_directory = args.output_dir or (
        REPOSITORY_ROOT / "outputs" / f"quick_start_family_{args.family}"
    )
    metrics = run_quick_start(
        mldist_path=data_directory / "pairwise_ml_distance.mldist",
        labels_path=data_directory / "labels.csv",
        output_directory=output_directory,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
