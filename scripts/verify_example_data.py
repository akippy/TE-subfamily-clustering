#!/usr/bin/env python3
"""Verify that public ID replacement changed no distances, sequences, or labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def _rows(path: Path) -> tuple[int, list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return int(lines[0].strip()), [line.split() for line in lines[1:] if line.split()]


def _used_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def verify(
    family: str,
    original_mldist: Path,
    original_alignment: Path,
    original_annotation: Path,
    public_directory: Path,
) -> None:
    original_size, original_rows = _rows(original_mldist)
    public_size, public_rows = _rows(public_directory / "pairwise_ml_distance.mldist")
    expected_ids = [
        f"family{family}_seq{index:03d}" for index in range(1, original_size + 1)
    ]
    assert original_size == public_size == len(original_rows) == len(public_rows)
    assert [row[0] for row in public_rows] == expected_ids
    assert [row[1:] for row in original_rows] == [row[1:] for row in public_rows]

    original_records = list(SeqIO.parse(original_alignment, "fasta"))
    public_records = list(SeqIO.parse(public_directory / "alignment.fasta", "fasta"))
    assert [record.id for record in public_records] == expected_ids
    assert [str(record.seq) for record in original_records] == [
        str(record.seq) for record in public_records
    ]

    annotation = pd.read_csv(original_annotation)
    annotation = annotation.loc[_used_mask(annotation["used"])].copy()
    annotation["canonical_id"] = annotation["name"].astype(str).str.replace(
        ":", "_", regex=False
    )
    annotation = annotation.set_index("canonical_id").loc[
        [row[0] for row in original_rows]
    ]
    public_labels = pd.read_csv(public_directory / "labels.csv", dtype=str)
    assert public_labels["sequence_id"].tolist() == expected_ids
    assert public_labels["known_subfamily"].tolist() == annotation["subfamily"].astype(str).tolist()

    forbidden_markers = ("DF", "CM", ":")
    assert not any(
        marker in sequence_id
        for sequence_id in expected_ids
        for marker in forbidden_markers
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--original-mldist", required=True, type=Path)
    parser.add_argument("--original-alignment", required=True, type=Path)
    parser.add_argument("--original-annotation", required=True, type=Path)
    parser.add_argument("--public-dir", required=True, type=Path)
    args = parser.parse_args()
    verify(
        args.family,
        args.original_mldist,
        args.original_alignment,
        args.original_annotation,
        args.public_dir,
    )
    print(f"family {args.family}: ID replacement integrity verified")


if __name__ == "__main__":
    main()
