"""Small end-to-end workflows used by the command line and examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .affinity import distance_to_affinity
from .baselines import FixedBinaryConfig, run_fixed_binary_recursive
from .evaluation import evaluate_clustering
from .io import (
    read_alignment,
    read_labels,
    read_mldist,
    validate_sequence_ids,
    write_dataframe,
)
from .recursive import RecursiveConfig, run_recursive_clustering
from .spectral import run_spectral_clustering


def load_inputs(
    mldist_path: str | Path,
    labels_path: str | Path,
    alignment_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, str] | None]:
    """Load inputs and enforce one sequence-ID order across every data type."""

    distance = read_mldist(mldist_path)
    labels = read_labels(labels_path)
    collections: list[tuple[str, list[str]]] = [
        ("labels.csv", labels.index.tolist())
    ]
    alignment = None
    if alignment_path is not None:
        alignment = read_alignment(alignment_path)
        collections.append(("alignment FASTA", list(alignment.keys())))
    validate_sequence_ids(distance.index, *collections, require_same_order=True)
    return distance, labels, alignment


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_quick_start(
    *,
    mldist_path: str | Path,
    labels_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run distance parsing, affinity conversion, one-shot SC, and ARI."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    distance, labels, _alignment = load_inputs(mldist_path, labels_path)
    affinity = distance_to_affinity(distance)
    result = run_spectral_clustering(affinity, labels)
    metrics = evaluate_clustering(result.membership["predicted_cluster"], labels)
    metrics.update(
        {
            "selected_k": result.selected_k,
            "selected_mean_silhouette": float(
                result.candidate_scores.loc[result.selected_k, "mean_silhouette"]
            ),
            "maximum_distance": float(distance.to_numpy().max()),
        }
    )
    write_dataframe(distance, output / "distance_matrix.csv")
    write_dataframe(affinity, output / "affinity_matrix.csv")
    write_dataframe(result.membership, output / "cluster_membership.csv")
    write_dataframe(result.candidate_scores, output / "candidate_scores.csv")
    write_json(metrics, output / "metrics.json")
    return metrics


def run_full_recursive(
    *,
    family_name: str,
    mldist_path: str | Path,
    alignment_path: str | Path,
    labels_path: str | Path,
    output_directory: str | Path,
    config: RecursiveConfig | None = None,
) -> dict[str, Any]:
    """Run the complete sequence-validated recursive method and ARI."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    distance, labels, alignment = load_inputs(
        mldist_path,
        labels_path,
        alignment_path,
    )
    if alignment is None:  # Narrow the type for static checkers.
        raise RuntimeError("Alignment loading unexpectedly returned no sequences")
    affinity = distance_to_affinity(distance)
    result = run_recursive_clustering(
        affinity,
        alignment,
        labels,
        family_name=family_name,
        config=config,
    )
    metrics = evaluate_clustering(result.membership["predicted_cluster"], labels)
    metrics["accepted_splits"] = int(result.diagnostics["accepted"].fillna(False).sum())
    write_dataframe(affinity, output / "affinity_matrix.csv")
    write_dataframe(result.membership, output / "recursive_membership.csv")
    write_dataframe(result.diagnostics, output / "recursive_diagnostics.csv", index=False)
    write_json(metrics, output / "metrics.json")
    return metrics


def run_fixed_binary_baseline(
    *,
    family_name: str,
    mldist_path: str | Path,
    alignment_path: str | Path,
    labels_path: str | Path,
    output_directory: str | Path,
    config: FixedBinaryConfig | None = None,
) -> dict[str, Any]:
    """Run the repeated fixed-k=2 recursive comparison baseline."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    distance, labels, alignment = load_inputs(
        mldist_path,
        labels_path,
        alignment_path,
    )
    if alignment is None:
        raise RuntimeError("Alignment loading unexpectedly returned no sequences")
    affinity = distance_to_affinity(distance)
    result = run_fixed_binary_recursive(
        affinity,
        alignment,
        labels,
        family_name=family_name,
        config=config,
    )
    metrics = evaluate_clustering(result.membership["predicted_cluster"], labels)
    metrics["accepted_splits"] = int(result.diagnostics["accepted"].fillna(False).sum())
    write_dataframe(result.membership, output / "fixed_binary_membership.csv")
    write_dataframe(
        result.diagnostics,
        output / "fixed_binary_diagnostics.csv",
        index=False,
    )
    write_json(metrics, output / "metrics.json")
    return metrics
