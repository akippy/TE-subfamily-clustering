"""Silhouette-selected recursive spectral clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics import silhouette_samples

from .evaluation import align_known_labels
from .io import validate_sequence_ids, validate_square_matrix
from .sequence_validation import (
    build_cluster_records,
    pairwise_consensus_similarities,
    reassignment_accuracy,
)


@dataclass(frozen=True)
class RecursiveConfig:
    """Parameters of the primary recursive method used in the comparison."""

    minimum_cluster_size: int = 10
    consensus_similarity_threshold: float = 0.95
    reassignment_accuracy_threshold: float = 0.95
    max_depth: int = 5
    top_k_candidates: int = 3
    global_k_max: int = 20
    random_state: int = 1
    swipe_threads: int = 2
    temporary_parent: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.minimum_cluster_size < 1:
            raise ValueError("minimum_cluster_size must be at least 1")
        if not 0 <= self.consensus_similarity_threshold <= 1:
            raise ValueError("consensus_similarity_threshold must be between 0 and 1")
        if not 0 <= self.reassignment_accuracy_threshold <= 1:
            raise ValueError("reassignment_accuracy_threshold must be between 0 and 1")
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        if self.top_k_candidates < 1:
            raise ValueError("top_k_candidates must be at least 1")
        if self.global_k_max < 2:
            raise ValueError("global_k_max must be at least 2")
        if self.swipe_threads < 1:
            raise ValueError("swipe_threads must be at least 1")


@dataclass
class Candidate:
    k: int
    labels: pd.Series
    silhouette_score: float
    rank: Optional[int] = None
    error: Optional[str] = None


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    pass_minimum_size: Optional[bool]
    pass_similarity_threshold: Optional[bool]
    pass_layer_similarity: Optional[bool]
    pass_reassignment: Optional[bool]
    accepted: bool
    rejection_reason: str
    reassignment_accuracy: Optional[float]
    minimum_child_similarity: Optional[float]
    maximum_child_similarity: Optional[float]
    child_sizes: dict[int, int]
    child_upper_similarity: dict[int, float]


@dataclass(frozen=True)
class RecursiveResult:
    membership: pd.DataFrame
    diagnostics: pd.DataFrame


DIAGNOSTIC_COLUMNS = [
    "family",
    "depth",
    "path",
    "n_sequences",
    "tested_k",
    "silhouette_score",
    "rank",
    "pass_minimum_size",
    "pass_similarity_threshold",
    "pass_layer_similarity",
    "pass_reassignment",
    "accepted",
    "rejection_reason",
    "upper_similarity",
    "reassignment_accuracy",
    "minimum_child_similarity",
    "maximum_child_similarity",
    "child_sizes",
]


def _path_string(path: Sequence[int], *, root: str = "root") -> str:
    return "-".join(map(str, path)) if path else root


def _log_row(
    rows: list[dict[str, object]],
    *,
    family: str,
    depth: int,
    path: Sequence[int],
    n_sequences: int,
    tested_k: Optional[int] = None,
    silhouette_score: Optional[float] = None,
    rank: Optional[int] = None,
    pass_minimum_size: Optional[bool] = None,
    pass_similarity_threshold: Optional[bool] = None,
    pass_layer_similarity: Optional[bool] = None,
    pass_reassignment: Optional[bool] = None,
    accepted: bool = False,
    rejection_reason: str,
    upper_similarity: float,
    reassignment: Optional[float] = None,
    minimum_child_similarity: Optional[float] = None,
    maximum_child_similarity: Optional[float] = None,
    child_sizes: Optional[dict[int, int]] = None,
) -> None:
    rows.append(
        {
            "family": family,
            "depth": depth,
            "path": _path_string(path),
            "n_sequences": n_sequences,
            "tested_k": tested_k,
            "silhouette_score": silhouette_score,
            "rank": rank,
            "pass_minimum_size": pass_minimum_size,
            "pass_similarity_threshold": pass_similarity_threshold,
            "pass_layer_similarity": pass_layer_similarity,
            "pass_reassignment": pass_reassignment,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "upper_similarity": upper_similarity,
            "reassignment_accuracy": reassignment,
            "minimum_child_similarity": minimum_child_similarity,
            "maximum_child_similarity": maximum_child_similarity,
            "child_sizes": (
                ";".join(f"{label}:{size}" for label, size in sorted(child_sizes.items()))
                if child_sizes
                else None
            ),
        }
    )


def _spectral_candidates(
    affinity: pd.DataFrame,
    config: RecursiveConfig,
) -> tuple[list[Candidate], int]:
    n_sequences = len(affinity)
    k_max = min(
        config.global_k_max,
        math.floor(n_sequences / config.minimum_cluster_size),
    )
    if k_max < 2:
        return [], k_max

    component_count = min(config.global_k_max, n_sequences - 1)
    embedding = spectral_embedding(
        affinity.to_numpy(dtype=float),
        n_components=component_count,
        random_state=config.random_state,
    )
    candidates: list[Candidate] = []
    for k in range(2, k_max + 1):
        selected_embedding = embedding[:, :k]
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=config.random_state,
        )
        try:
            predicted = kmeans.fit_predict(selected_embedding)
            unique = np.unique(predicted)
            labels = pd.Series(predicted, index=affinity.index, name="predicted_cluster")
            if len(unique) < 2 or len(unique) >= n_sequences:
                candidates.append(
                    Candidate(k, labels, float("nan"), error="invalid_number_of_clusters")
                )
                continue
            silhouettes = silhouette_samples(
                selected_embedding,
                predicted,
                metric="euclidean",
            )
            candidates.append(Candidate(k, labels, float(np.mean(silhouettes))))
        except Exception as exc:  # Preserve per-candidate failure handling.
            candidates.append(
                Candidate(
                    k,
                    pd.Series(dtype=int),
                    float("nan"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    ranked = sorted(
        [candidate for candidate in candidates if np.isfinite(candidate.silhouette_score)],
        key=lambda candidate: candidate.silhouette_score,
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank
    return candidates, k_max


def _evaluate_candidate(
    *,
    family: str,
    candidate: Candidate,
    aligned_sequences: Mapping[str, str],
    upper_similarity: float,
    config: RecursiveConfig,
) -> CandidateEvaluation:
    labels = candidate.labels.astype(int)
    child_sizes = {
        int(label): int(size)
        for label, size in labels.value_counts().sort_index().items()
    }
    minimum_size_ok = all(
        size >= config.minimum_cluster_size for size in child_sizes.values()
    )
    if not minimum_size_ok:
        return CandidateEvaluation(
            candidate,
            False,
            None,
            None,
            None,
            False,
            "cluster_below_minimum_size",
            None,
            None,
            None,
            child_sizes,
            {},
        )

    consensus_records, copy_records = build_cluster_records(labels, aligned_sequences)
    empty_labels = [int(record.id) for record in consensus_records if len(record.seq) == 0]
    if empty_labels:
        return CandidateEvaluation(
            candidate,
            True,
            False,
            None,
            None,
            False,
            f"empty_consensus:{empty_labels}",
            None,
            None,
            None,
            child_sizes,
            {},
        )

    similarities = pairwise_consensus_similarities(consensus_records)
    values = list(similarities.values())
    minimum_similarity = min(values) if values else None
    maximum_similarity = max(values) if values else None
    threshold_ok = all(
        similarity < config.consensus_similarity_threshold for similarity in values
    )
    layer_ok = all(similarity >= upper_similarity for similarity in values)
    if not threshold_ok or not layer_ok:
        reasons = []
        if not threshold_ok:
            reasons.append("consensus_similarity_threshold_failed")
        if not layer_ok:
            reasons.append("layer_similarity_failed")
        return CandidateEvaluation(
            candidate,
            True,
            threshold_ok,
            layer_ok,
            None,
            False,
            ";".join(reasons),
            None,
            minimum_similarity,
            maximum_similarity,
            child_sizes,
            {},
        )

    accuracy = reassignment_accuracy(
        family,
        labels,
        consensus_records,
        copy_records,
        threads=config.swipe_threads,
        temporary_parent=config.temporary_parent,
    )
    reassignment_ok = accuracy >= config.reassignment_accuracy_threshold
    child_upper_similarity: dict[int, float] = {}
    for cluster_label in child_sizes:
        sibling_similarities = [
            similarity
            for pair, similarity in similarities.items()
            if cluster_label in pair
        ]
        child_upper_similarity[cluster_label] = (
            max(sibling_similarities) if sibling_similarities else upper_similarity
        )
    return CandidateEvaluation(
        candidate,
        True,
        True,
        True,
        reassignment_ok,
        reassignment_ok,
        "" if reassignment_ok else "reassignment_accuracy_failed",
        accuracy,
        minimum_similarity,
        maximum_similarity,
        child_sizes,
        child_upper_similarity,
    )


def _log_candidates(
    rows: list[dict[str, object]],
    *,
    family: str,
    depth: int,
    path: Sequence[int],
    n_sequences: int,
    candidates: Sequence[Candidate],
    evaluations: Mapping[int, CandidateEvaluation],
    accepted_k: Optional[int],
    upper_similarity: float,
    config: RecursiveConfig,
) -> None:
    for candidate in sorted(candidates, key=lambda value: value.k):
        evaluation = evaluations.get(candidate.k)
        if evaluation is not None:
            _log_row(
                rows,
                family=family,
                depth=depth,
                path=path,
                n_sequences=n_sequences,
                tested_k=candidate.k,
                silhouette_score=candidate.silhouette_score,
                rank=candidate.rank,
                pass_minimum_size=evaluation.pass_minimum_size,
                pass_similarity_threshold=evaluation.pass_similarity_threshold,
                pass_layer_similarity=evaluation.pass_layer_similarity,
                pass_reassignment=evaluation.pass_reassignment,
                accepted=evaluation.accepted,
                rejection_reason=evaluation.rejection_reason,
                upper_similarity=upper_similarity,
                reassignment=evaluation.reassignment_accuracy,
                minimum_child_similarity=evaluation.minimum_child_similarity,
                maximum_child_similarity=evaluation.maximum_child_similarity,
                child_sizes=evaluation.child_sizes,
            )
            continue
        if candidate.error:
            reason = candidate.error
        elif candidate.rank is None:
            reason = "invalid_silhouette"
        elif candidate.rank > config.top_k_candidates:
            reason = "not_in_top_k_candidates"
        elif accepted_k is not None:
            reason = "not_evaluated_after_acceptance"
        else:
            reason = "not_evaluated"
        _log_row(
            rows,
            family=family,
            depth=depth,
            path=path,
            n_sequences=n_sequences,
            tested_k=candidate.k,
            silhouette_score=(
                candidate.silhouette_score
                if np.isfinite(candidate.silhouette_score)
                else None
            ),
            rank=candidate.rank,
            rejection_reason=reason,
            upper_similarity=upper_similarity,
        )


def _recurse(
    *,
    family: str,
    affinity: pd.DataFrame,
    depth: int,
    path: tuple[int, ...],
    upper_similarity: float,
    cluster_paths: dict[str, tuple[int, ...]],
    aligned_sequences: Mapping[str, str],
    config: RecursiveConfig,
    diagnostics: list[dict[str, object]],
) -> None:
    n_sequences = len(affinity)
    if depth >= config.max_depth:
        _log_row(
            diagnostics,
            family=family,
            depth=depth,
            path=path,
            n_sequences=n_sequences,
            rejection_reason="maximum_depth_reached",
            upper_similarity=upper_similarity,
        )
        return
    if n_sequences < config.minimum_cluster_size * 2:
        _log_row(
            diagnostics,
            family=family,
            depth=depth,
            path=path,
            n_sequences=n_sequences,
            rejection_reason="too_few_sequences_to_split",
            upper_similarity=upper_similarity,
        )
        return

    candidates, k_max = _spectral_candidates(affinity, config)
    if k_max < 2:
        _log_row(
            diagnostics,
            family=family,
            depth=depth,
            path=path,
            n_sequences=n_sequences,
            rejection_reason="maximum_candidate_k_below_two",
            upper_similarity=upper_similarity,
        )
        return
    ranked = sorted(
        [candidate for candidate in candidates if candidate.rank is not None],
        key=lambda candidate: int(candidate.rank),
    )
    if not ranked:
        _log_candidates(
            diagnostics,
            family=family,
            depth=depth,
            path=path,
            n_sequences=n_sequences,
            candidates=candidates,
            evaluations={},
            accepted_k=None,
            upper_similarity=upper_similarity,
            config=config,
        )
        return

    evaluations: dict[int, CandidateEvaluation] = {}
    accepted: Optional[CandidateEvaluation] = None
    for candidate in ranked[: config.top_k_candidates]:
        evaluation = _evaluate_candidate(
            family=family,
            candidate=candidate,
            aligned_sequences=aligned_sequences,
            upper_similarity=upper_similarity,
            config=config,
        )
        evaluations[candidate.k] = evaluation
        if evaluation.accepted:
            accepted = evaluation
            break
    _log_candidates(
        diagnostics,
        family=family,
        depth=depth,
        path=path,
        n_sequences=n_sequences,
        candidates=candidates,
        evaluations=evaluations,
        accepted_k=accepted.candidate.k if accepted else None,
        upper_similarity=upper_similarity,
        config=config,
    )
    if accepted is None:
        return

    labels = accepted.candidate.labels.astype(int)
    for cluster_label in sorted(labels.unique().tolist()):
        child_label = int(cluster_label)
        sequence_ids = labels[labels == child_label].index.tolist()
        child_path = path + (child_label,)
        for sequence_id in sequence_ids:
            cluster_paths[sequence_id] = child_path
        _recurse(
            family=family,
            affinity=affinity.loc[sequence_ids, sequence_ids],
            depth=depth + 1,
            path=child_path,
            upper_similarity=accepted.child_upper_similarity[child_label],
            cluster_paths=cluster_paths,
            aligned_sequences=aligned_sequences,
            config=config,
            diagnostics=diagnostics,
        )


def run_recursive_clustering(
    affinity_matrix: pd.DataFrame,
    aligned_sequences: Mapping[str, str],
    known_labels: Optional[pd.Series] = None,
    *,
    family_name: str = "family",
    config: Optional[RecursiveConfig] = None,
) -> RecursiveResult:
    """Run the primary silhouette-based recursive spectral method."""

    active_config = config or RecursiveConfig()
    affinity = validate_square_matrix(
        affinity_matrix,
        source="affinity matrix",
        require_symmetric=True,
        require_nonnegative=True,
    )
    collections: list[tuple[str, Sequence[str]]] = [
        ("alignment FASTA", list(aligned_sequences.keys()))
    ]
    truth = None
    if known_labels is not None:
        truth = align_known_labels(affinity.index, known_labels)
        collections.append(("known labels", truth.index.tolist()))
    validate_sequence_ids(affinity.index, *collections, require_same_order=True)

    cluster_paths = {sequence_id: tuple() for sequence_id in affinity.index}
    diagnostic_rows: list[dict[str, object]] = []
    _recurse(
        family=family_name,
        affinity=affinity,
        depth=0,
        path=tuple(),
        upper_similarity=0.0,
        cluster_paths=cluster_paths,
        aligned_sequences=aligned_sequences,
        config=active_config,
        diagnostics=diagnostic_rows,
    )
    membership = pd.DataFrame(
        {
            "predicted_cluster": [
                _path_string(cluster_paths[sequence_id], root="0")
                for sequence_id in affinity.index
            ]
        },
        index=affinity.index,
    )
    membership.index.name = "sequence_id"
    if truth is not None:
        membership["known_subfamily"] = truth
    diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
    return RecursiveResult(membership=membership, diagnostics=diagnostics)
