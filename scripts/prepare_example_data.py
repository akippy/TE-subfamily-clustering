#!/usr/bin/env python3
"""Create a coordinate-free public example while preserving data values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def used_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def prepare_example(
    family: str,
    mldist_path: Path,
    alignment_path: Path,
    annotation_path: Path,
    output_directory: Path,
) -> None:
    records = list(SeqIO.parse(alignment_path, "fasta"))
    original_alignment_ids = [record.id for record in records]
    canonical_alignment_ids = [sequence_id.replace(":", "_") for sequence_id in original_alignment_ids]

    lines = mldist_path.read_text(encoding="utf-8").splitlines()
    expected_size = int(lines[0].strip())
    parsed_rows = [line.split() for line in lines[1:] if line.split()]
    mldist_ids = [row[0] for row in parsed_rows]
    if expected_size != len(records) or expected_size != len(parsed_rows):
        raise ValueError("FASTA and .mldist sequence counts differ")
    if canonical_alignment_ids != mldist_ids:
        raise ValueError("FASTA and .mldist sequence IDs or order differ")

    annotation = pd.read_csv(annotation_path)
    required = {"name", "subfamily", "subfamily_ID", "used"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"Annotation is missing columns: {sorted(missing)}")
    annotation = annotation.loc[used_mask(annotation["used"])].copy()
    annotation["canonical_id"] = annotation["name"].astype(str).str.replace(
        ":", "_", regex=False
    )
    if annotation["canonical_id"].duplicated().any():
        raise ValueError("Annotation contains duplicate used sequence IDs")
    annotation = annotation.set_index("canonical_id").loc[mldist_ids]

    new_ids = [f"family{family}_seq{index:03d}" for index in range(1, expected_size + 1)]
    mapping = dict(zip(mldist_ids, new_ids))
    output_directory.mkdir(parents=True, exist_ok=True)

    public_mldist_lines = [str(expected_size)]
    for row in parsed_rows:
        public_mldist_lines.append(" ".join([mapping[row[0]], *row[1:]]))
    (output_directory / "pairwise_ml_distance.mldist").write_text(
        "\n".join(public_mldist_lines) + "\n",
        encoding="utf-8",
    )

    for record, new_id in zip(records, new_ids):
        record.id = new_id
        record.name = new_id
        record.description = ""
    SeqIO.write(records, output_directory / "alignment.fasta", "fasta")

    labels = pd.DataFrame(
        {
            "sequence_id": new_ids,
            "known_subfamily": annotation["subfamily"].astype(str).to_numpy(),
        }
    )
    labels.to_csv(output_directory / "labels.csv", index=False)

    label_counts = labels["known_subfamily"].value_counts().sort_index().to_dict()
    metadata = {
        "family_id": int(family),
        "organism": "Homo sapiens",
        "taxonomy_id": 9606,
        "reference_assembly": "GRCh38.p14",
        "reference_assembly_accession": "GCA_000001405.29",
        "Dfam_release": "3.9",
        "sequence_count": expected_size,
        "known_subfamily_counts": label_counts,
        "distance_source": "IQ-TREE 2 pairwise maximum-likelihood distances from the filtered MSA",
        "alignment_source": "MAFFT alignment of TE copies used for recursive consensus validation",
        "id_policy": (
            "Original Dfam accessions and genomic coordinates were replaced consistently "
            "with family-specific sequential IDs. No ID mapping is distributed."
        ),
        "integrity": (
            "Only sequence identifiers were replaced; distance strings, aligned sequences, "
            "record order, and known-subfamily labels were preserved."
        ),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--mldist", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare_example(
        args.family,
        args.mldist,
        args.alignment,
        args.annotation,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
