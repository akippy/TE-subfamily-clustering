"""One-shot spectral clustering with silhouette-based cluster selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from .evaluation import align_known_labels
from .io import validate_square_matrix


@dataclass(frozen=True)
class SpectralResult:
    """Selected membership and the complete candidate-k score table."""

    membership: pd.DataFrame
    candidate_scores: pd.DataFrame
    selected_k: int


def embedding_component_count(n_sequences: int) -> int:
    """Return the component count used by the original one-shot implementation."""

    component_count = n_sequences - 1
    if component_count < 21:
        component_count -= 1
    else:
        component_count = 20
    if component_count < 2:
        raise ValueError("At least four sequences are required for the original k sweep")
    return component_count


def run_spectral_clustering(
    affinity_matrix: pd.DataFrame,
    known_labels: Optional[pd.Series] = None,
    *,
    spectral_random_state: int = 1,
    kmeans_random_state: int = 0,
) -> SpectralResult:
    """Select cluster count by mean silhouette and return the memberships.

    The two random states intentionally differ because this mirrors the
    research implementation used for the one-shot spectral comparison.
    """

    affinity = validate_square_matrix(
        affinity_matrix,
        source="affinity matrix",
        require_symmetric=True,
        require_nonnegative=True,
    )
    component_count = embedding_component_count(len(affinity))

    # Keep the original model call before extracting the spectral embedding.
    spectral_model = SpectralClustering(
        n_clusters=component_count,
        affinity="precomputed",
        random_state=spectral_random_state,
    )
    spectral_model.fit(affinity)
    embedding = spectral_embedding(
        spectral_model.affinity_matrix_,
        n_components=component_count,
        random_state=spectral_random_state,
    )

    truth = None
    if known_labels is not None:
        truth = align_known_labels(affinity.index, known_labels)

    score_rows: list[dict[str, float | int]] = []
    memberships: dict[int, pd.DataFrame] = {}
    for k in range(2, component_count + 1):
        selected_embedding = embedding[:, :k]
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=kmeans_random_state,
        )
        predictions = kmeans.fit_predict(selected_embedding)
        silhouettes = silhouette_samples(
            selected_embedding,
            predictions,
            metric="euclidean",
        )
        membership = pd.DataFrame(
            {
                "predicted_cluster": predictions.astype(int),
                "silhouette": silhouettes,
            },
            index=affinity.index,
        )
        membership.index.name = "sequence_id"
        memberships[k] = membership

        row: dict[str, float | int] = {
            "k": k,
            "mean_silhouette": float(np.mean(silhouettes)),
        }
        if truth is not None:
            row["adjusted_rand_index"] = float(
                adjusted_rand_score(truth, membership["predicted_cluster"])
            )
        score_rows.append(row)

    scores = pd.DataFrame(score_rows).set_index("k")
    selected_k = int(scores["mean_silhouette"].idxmax())
    selected = memberships[selected_k].copy()
    selected["selected_k"] = selected_k
    if truth is not None:
        selected["known_subfamily"] = truth
    return SpectralResult(
        membership=selected,
        candidate_scores=scores,
        selected_k=selected_k,
    )
