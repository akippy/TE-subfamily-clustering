"""Repeated fixed-k=2 recursive spectral clustering baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import spectral_embedding

from ..evaluation import align_known_labels
from ..io import validate_sequence_ids, validate_square_matrix
from ..recursive import RecursiveResult
from ..sequence_validation import (
    build_cluster_records,
    pairwise_consensus_similarities,
    reassignment_accuracy,
)


@dataclass(frozen=True)
class FixedBinaryConfig:
    """Parameters retained from the repeated binary-split baseline."""

    minimum_cluster_size: int = 10
    consensus_similarity_threshold: float = 0.95
    reassignment_accuracy_threshold: float = 0.95
    max_recursion_counter: int = 10
    spectral_random_state: int = 1
    kmeans_random_state: int = 0
    swipe_threads: int = 2
    temporary_parent: Optional[Path] = None


def _binary_membership(affinity: pd.DataFrame, config: FixedBinaryConfig) -> pd.Series:
    n_sequences = len(affinity)
    component_count = n_sequences - 1 if n_sequences < 21 else 20
    spectral_model = SpectralClustering(
        n_clusters=component_count,
        affinity="precomputed",
        random_state=config.spectral_random_state,
    )
    spectral_model.fit(affinity)
    embedding = spectral_embedding(
        spectral_model.affinity_matrix_,
        n_components=component_count,
        random_state=config.spectral_random_state,
    )
    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=config.kmeans_random_state,
    )
    labels = kmeans.fit_predict(embedding[:, :2])
    return pd.Series(labels, index=affinity.index, name="predicted_cluster")


def _recurse_binary(
    *,
    family: str,
    affinity: pd.DataFrame,
    recursion_counter: int,
    path: tuple[int, ...],
    upper_similarity: float,
    cluster_paths: dict[str, tuple[int, ...]],
    aligned_sequences: Mapping[str, str],
    config: FixedBinaryConfig,
    diagnostics: list[dict[str, object]],
) -> None:
    n_sequences = len(affinity)
    if recursion_counter + 1 > config.max_recursion_counter:
        diagnostics.append(
            {
                "family": family,
                "recursion_counter": recursion_counter,
                "path": "-".join(map(str, path)) or "root",
                "n_sequences": n_sequences,
                "accepted": False,
                "rejection_reason": "maximum_recursion_counter_reached",
                "consensus_similarity": None,
                "reassignment_accuracy": None,
                "child_sizes": None,
            }
        )
        return
    if n_sequences < config.minimum_cluster_size * 2:
        diagnostics.append(
            {
                "family": family,
                "recursion_counter": recursion_counter,
                "path": "-".join(map(str, path)) or "root",
                "n_sequences": n_sequences,
                "accepted": False,
                "rejection_reason": "too_few_sequences_to_split",
                "consensus_similarity": None,
                "reassignment_accuracy": None,
                "child_sizes": None,
            }
        )
        return

    labels = _binary_membership(affinity, config)
    child_sizes = {
        int(label): int(size)
        for label, size in labels.value_counts().sort_index().items()
    }
    if not all(size >= config.minimum_cluster_size for size in child_sizes.values()):
        diagnostics.append(
            {
                "family": family,
                "recursion_counter": recursion_counter,
                "path": "-".join(map(str, path)) or "root",
                "n_sequences": n_sequences,
                "accepted": False,
                "rejection_reason": "cluster_below_minimum_size",
                "consensus_similarity": None,
                "reassignment_accuracy": None,
                "child_sizes": ";".join(
                    f"{label}:{size}" for label, size in sorted(child_sizes.items())
                ),
            }
        )
        return

    consensus_records, copy_records = build_cluster_records(labels, aligned_sequences)
    similarities = pairwise_consensus_similarities(consensus_records)
    similarity = next(iter(similarities.values()))
    if similarity >= config.consensus_similarity_threshold:
        reason = "consensus_similarity_threshold_failed"
    elif similarity < upper_similarity:
        reason = "layer_similarity_failed"
    else:
        reason = ""
    if reason:
        diagnostics.append(
            {
                "family": family,
                "recursion_counter": recursion_counter,
                "path": "-".join(map(str, path)) or "root",
                "n_sequences": n_sequences,
                "accepted": False,
                "rejection_reason": reason,
                "consensus_similarity": similarity,
                "reassignment_accuracy": None,
                "child_sizes": ";".join(
                    f"{label}:{size}" for label, size in sorted(child_sizes.items())
                ),
            }
        )
        return

    accuracy = reassignment_accuracy(
        family,
        labels,
        consensus_records,
        copy_records,
        threads=config.swipe_threads,
        temporary_parent=config.temporary_parent,
    )
    accepted = accuracy >= config.reassignment_accuracy_threshold
    diagnostics.append(
        {
            "family": family,
            "recursion_counter": recursion_counter,
            "path": "-".join(map(str, path)) or "root",
            "n_sequences": n_sequences,
            "accepted": accepted,
            "rejection_reason": "" if accepted else "reassignment_accuracy_failed",
            "consensus_similarity": similarity,
            "reassignment_accuracy": accuracy,
            "child_sizes": ";".join(
                f"{label}:{size}" for label, size in sorted(child_sizes.items())
            ),
        }
    )
    if not accepted:
        return

    for cluster_label in sorted(labels.unique().tolist()):
        child_label = int(cluster_label)
        sequence_ids = labels[labels == child_label].index.tolist()
        child_path = path + (child_label,)
        for sequence_id in sequence_ids:
            cluster_paths[sequence_id] = child_path
        _recurse_binary(
            family=family,
            affinity=affinity.loc[sequence_ids, sequence_ids],
            recursion_counter=recursion_counter + 1,
            path=child_path,
            upper_similarity=similarity,
            cluster_paths=cluster_paths,
            aligned_sequences=aligned_sequences,
            config=config,
            diagnostics=diagnostics,
        )


def run_fixed_binary_recursive(
    affinity_matrix: pd.DataFrame,
    aligned_sequences: Mapping[str, str],
    known_labels: Optional[pd.Series] = None,
    *,
    family_name: str = "family",
    config: Optional[FixedBinaryConfig] = None,
) -> RecursiveResult:
    """Run the fixed-k=2 recursive comparison baseline."""

    active_config = config or FixedBinaryConfig()
    affinity = validate_square_matrix(
        affinity_matrix,
        source="affinity matrix",
        require_symmetric=True,
        require_nonnegative=True,
    )
    collections = [("alignment FASTA", list(aligned_sequences.keys()))]
    truth = None
    if known_labels is not None:
        truth = align_known_labels(affinity.index, known_labels)
        collections.append(("known labels", truth.index.tolist()))
    validate_sequence_ids(affinity.index, *collections, require_same_order=True)

    paths = {sequence_id: tuple() for sequence_id in affinity.index}
    diagnostics: list[dict[str, object]] = []
    _recurse_binary(
        family=family_name,
        affinity=affinity,
        recursion_counter=1,
        path=tuple(),
        upper_similarity=0.0,
        cluster_paths=paths,
        aligned_sequences=aligned_sequences,
        config=active_config,
        diagnostics=diagnostics,
    )
    membership = pd.DataFrame(
        {
            "predicted_cluster": [
                "-".join(map(str, paths[sequence_id])) if paths[sequence_id] else "0"
                for sequence_id in affinity.index
            ]
        },
        index=affinity.index,
    )
    membership.index.name = "sequence_id"
    if truth is not None:
        membership["known_subfamily"] = truth
    return RecursiveResult(membership, pd.DataFrame(diagnostics))
