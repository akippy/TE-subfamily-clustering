"""Sequence-based validation used as recursive-clustering stopping criteria."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def require_external_tools() -> dict[str, str]:
    """Return external-tool paths or raise with installation guidance."""

    paths = {name: shutil.which(name) for name in ("makeblastdb", "swipe")}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise RuntimeError(
            "Full recursive clustering requires external commands: "
            + ", ".join(missing)
        )
    return {name: str(path) for name, path in paths.items()}


def consensus_and_cleaned_records(
    sequence_ids: Sequence[str],
    aligned_sequences: Mapping[str, str],
) -> tuple[str, list[SeqRecord]]:
    """Build a majority-rule consensus and remove its gap columns from copies."""

    sequences = [aligned_sequences[sequence_id] for sequence_id in sequence_ids]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"MSA sequences must have equal lengths: {sorted(lengths)[:5]}")

    sequence_array = np.array([list(sequence) for sequence in sequences])
    consensus_bases = [
        Counter(sequence_array[:, column]).most_common(1)[0][0]
        for column in range(sequence_array.shape[1])
    ]
    gap_positions = {
        position for position, base in enumerate(consensus_bases) if base == "-"
    }
    consensus = "".join(base for base in consensus_bases if base != "-")
    cleaned_records = [
        SeqRecord(
            Seq(
                "".join(
                    base
                    for position, base in enumerate(aligned_sequences[sequence_id])
                    if position not in gap_positions
                )
            ),
            id=sequence_id,
            description="",
        )
        for sequence_id in sequence_ids
    ]
    return consensus, cleaned_records


def build_cluster_records(
    labels: pd.Series,
    aligned_sequences: Mapping[str, str],
) -> tuple[list[SeqRecord], list[SeqRecord]]:
    """Create cluster-consensus records and cleaned copy records."""

    consensus_records: list[SeqRecord] = []
    copy_records: list[SeqRecord] = []
    for cluster_label in sorted(labels.unique().tolist()):
        sequence_ids = labels[labels == cluster_label].index.tolist()
        consensus, cleaned = consensus_and_cleaned_records(sequence_ids, aligned_sequences)
        consensus_records.append(
            SeqRecord(Seq(consensus), id=str(int(cluster_label)), description="")
        )
        copy_records.extend(cleaned)
    return consensus_records, copy_records


def consensus_similarity(first: str, second: str) -> float:
    """Calculate identity after global alignment, ignoring all gap columns."""

    if not first or not second:
        return 0.0
    alignments = pairwise2.align.globalms(
        first,
        second,
        5,
        -4,
        -10,
        -0.5,
        one_alignment_only=True,
    )
    if not alignments:
        return 0.0
    aligned_first, aligned_second, _score, _start, _end = alignments[0]
    matches = 0
    valid = 0
    for first_base, second_base in zip(aligned_first, aligned_second):
        if first_base == "-" or second_base == "-":
            continue
        valid += 1
        matches += first_base == second_base
    return matches / valid if valid else 0.0


def pairwise_consensus_similarities(
    consensus_records: Sequence[SeqRecord],
) -> dict[tuple[int, int], float]:
    """Calculate similarities for every pair of cluster consensuses."""

    similarities: dict[tuple[int, int], float] = {}
    for first_index in range(len(consensus_records)):
        for second_index in range(first_index + 1, len(consensus_records)):
            first = consensus_records[first_index]
            second = consensus_records[second_index]
            similarities[(int(first.id), int(second.id))] = consensus_similarity(
                str(first.seq), str(second.seq)
            )
    return similarities


def _subject_cluster(subject: object) -> Optional[int]:
    try:
        return int(str(subject).split("|")[-1])
    except ValueError:
        return None


def reassignment_accuracy(
    family_name: str,
    labels: pd.Series,
    consensus_records: Sequence[SeqRecord],
    copy_records: Sequence[SeqRecord],
    *,
    threads: int = 2,
    temporary_parent: str | Path | None = None,
) -> float:
    """Reassign copies to consensuses with SWIPE and return the match rate."""

    tools = require_external_tools()
    safe_family = re.sub(r"[^A-Za-z0-9_.-]", "_", family_name)
    columns = [
        "Query",
        "Subject",
        "Identity%",
        "Length",
        "Mismatches",
        "Gap openings",
        "Q Start",
        "Q End",
        "S Start",
        "S End",
        "Score",
    ]
    with tempfile.TemporaryDirectory(
        prefix=f"{safe_family}_{os.getpid()}_",
        dir=str(temporary_parent) if temporary_parent is not None else None,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        consensus_path = temporary / "consensus.fasta"
        copies_path = temporary / "copies.fasta"
        database_prefix = temporary / "consensus_db"
        swipe_output = temporary / "swipe.tsv"
        SeqIO.write(consensus_records, consensus_path, "fasta")
        SeqIO.write(copy_records, copies_path, "fasta")

        subprocess.run(
            [
                tools["makeblastdb"],
                "-in",
                str(consensus_path),
                "-dbtype",
                "nucl",
                "-out",
                str(database_prefix),
                "-blastdb_version",
                "4",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                tools["swipe"],
                "-p",
                "0",
                "-i",
                str(copies_path),
                "-d",
                str(database_prefix),
                "-S",
                "1",
                "-r",
                "5",
                "-q",
                "-4",
                "-G",
                "10",
                "-E",
                "0.5",
                "-m",
                "9",
                f"--num_threads={threads}",
                "--num_descriptions",
                "10000",
                "-o",
                str(swipe_output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if not swipe_output.exists() or swipe_output.stat().st_size == 0:
            return 0.0
        alignments = pd.read_csv(
            swipe_output,
            sep="\t",
            comment="#",
            names=columns,
        )

    if alignments.empty:
        return 0.0
    alignments = alignments.sort_values("Score", ascending=False)
    alignments["reassigned_cluster"] = alignments["Subject"].apply(_subject_cluster)
    alignments = alignments.dropna(subset=["reassigned_cluster"])
    if alignments.empty:
        return 0.0
    best_hits = alignments.drop_duplicates(subset=["Query"], keep="first")
    correct = 0
    for row in best_hits.itertuples(index=False):
        if row.Query in labels.index:
            correct += int(int(row.reassigned_cluster) == int(labels.loc[row.Query]))
    return correct / len(labels)
