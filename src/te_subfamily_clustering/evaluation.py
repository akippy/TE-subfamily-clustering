"""Evaluation helpers for predicted TE subfamily clusters."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def align_known_labels(sequence_ids: pd.Index, known_labels: pd.Series) -> pd.Series:
    """Align known labels to an expected sequence-ID order after strict checks."""

    expected = pd.Index(sequence_ids.astype(str))
    labels = known_labels.copy()
    labels.index = labels.index.astype(str)
    if not labels.index.is_unique:
        raise ValueError("Known-label sequence IDs are not unique")
    missing = expected.difference(labels.index)
    extra = labels.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"Known labels do not match clustering IDs: missing={missing[:5].tolist()}, "
            f"extra={extra[:5].tolist()}"
        )
    return labels.loc[expected].astype(str)


def evaluate_clustering(
    predicted_labels: pd.Series,
    known_labels: pd.Series,
) -> dict[str, float | int]:
    """Calculate ARI and predicted/known cluster counts."""

    predictions = predicted_labels.copy()
    predictions.index = predictions.index.astype(str)
    if not predictions.index.is_unique or predictions.isna().any():
        raise ValueError("Predicted labels must be complete and uniquely indexed")
    truth = align_known_labels(predictions.index, known_labels)
    return {
        "n_sequences": int(len(predictions)),
        "predicted_clusters": int(predictions.nunique()),
        "known_subfamilies": int(truth.nunique()),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predictions)),
    }


def evaluate_dbscan_with_unique_noise(
    predicted_labels: pd.Series,
    known_labels: pd.Series,
    *,
    noise_label: int = -1,
) -> dict[str, float | int]:
    """Evaluate DBSCAN while treating each noise point as a singleton cluster."""

    labels = pd.to_numeric(predicted_labels, errors="raise").astype(int)
    truth = align_known_labels(labels.index, known_labels)
    noise_mask = labels.eq(noise_label)
    unique_noise = labels.copy()
    non_noise = labels.loc[~noise_mask]
    next_label = int(non_noise.max()) + 1 if len(non_noise) else 0
    unique_noise.loc[noise_mask] = np.arange(next_label, next_label + int(noise_mask.sum()))
    return {
        "n_sequences": int(len(labels)),
        "predicted_clusters_excluding_noise": int(len(set(labels) - {noise_label})),
        "noise_count": int(noise_mask.sum()),
        "noise_rate": float(noise_mask.mean()),
        "adjusted_rand_index": float(adjusted_rand_score(truth, unique_noise)),
    }


def summarize_methods(
    family_results: pd.DataFrame,
    method_columns: Mapping[str, str],
    *,
    subset_name: str,
) -> pd.DataFrame:
    """Summarize family-level ARI values using the research comparison rules."""

    if "n_sequences" not in family_results:
        raise ValueError("family_results must contain n_sequences")
    weights = family_results["n_sequences"].to_numpy(dtype=float)
    rows = []
    for column, label in method_columns.items():
        if column not in family_results:
            raise ValueError(f"family_results is missing {column}")
        values = pd.to_numeric(family_results[column], errors="raise")
        rows.append(
            {
                "subset": subset_name,
                "method": label,
                "n_families": int(values.notna().sum()),
                "mean_ARI": float(values.mean()),
                "sequence_weighted_mean_ARI": float(np.average(values, weights=weights)),
                "median_ARI": float(values.median()),
                "Q1": float(values.quantile(0.25)),
                "Q3": float(values.quantile(0.75)),
                "ARI_ge_0.9": int(values.ge(0.9).sum()),
                "ARI_lt_0.5": int(values.lt(0.5).sum()),
            }
        )
    return pd.DataFrame(rows)
