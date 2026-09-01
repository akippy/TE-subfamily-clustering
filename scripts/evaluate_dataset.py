#!/usr/bin/env python3
"""Rebuild the seven-method family-level and aggregate ARI summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from te_subfamily_clustering.evaluation import summarize_methods


SEQUENCE_RULES = (80, 90, 95)
TREE_CUTOFFS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
PRIMARY_METHODS = {
    "sequence_80_ARI": "Sequence similarity (rule 80)",
    "tree_0.02_ARI": "Phylogenetic tree (cutoff 0.02)",
    "dbscan_ARI": "DBSCAN (eps 0.1)",
    "umap_silhouette_ARI": "UMAP + K-means (silhouette k)",
    "spectral_silhouette_ARI": "Spectral (silhouette k)",
    "recursive_k2_ARI": "Recursive Spectral (repeated k=2)",
    "recursive_silhouette_ARI": "Recursive Spectral (silhouette)",
}


def numeric_csv_families(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.csv") if path.stem.isdigit()}


def used_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def read_matrix_ids(matrix_directory: Path, family: str) -> list[str]:
    path = matrix_directory / f"{family}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
    sequence_ids = [str(value) for value in header[1:]]
    if not sequence_ids or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError(f"Invalid or duplicate matrix IDs: family={family}")
    return sequence_ids


def load_truth(matrix_directory: Path, annotation_directory: Path, family: str) -> pd.Series:
    matrix_ids = read_matrix_ids(matrix_directory, family)
    path = annotation_directory / f"{family}.csv"
    annotation = pd.read_csv(path)
    required = {"name", "subfamily", "used"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    annotation = annotation.loc[used_mask(annotation["used"])].copy()
    annotation["sequence_id"] = annotation["name"].astype(str).str.replace(
        ":", "_", regex=False
    )
    if annotation["sequence_id"].duplicated().any():
        raise ValueError(f"Duplicate used annotation IDs: family={family}")
    truth = annotation.set_index("sequence_id")["subfamily"]
    truth.index = truth.index.astype(str)
    if set(matrix_ids) != set(truth.index):
        raise ValueError(f"Matrix/annotation ID mismatch: family={family}")
    return truth.loc[matrix_ids].astype(str)


def align_output(
    path: Path,
    truth: pd.Series,
    label_column: str,
    known_column: str | None = None,
) -> pd.DataFrame:
    result = pd.read_csv(path, index_col=0)
    result.index = result.index.astype(str)
    if label_column not in result or not result.index.is_unique:
        raise ValueError(f"Invalid result table: {path}")
    if set(result.index) != set(truth.index):
        raise ValueError(f"Output/truth ID mismatch: {path}")
    result = result.loc[truth.index]
    if known_column is not None:
        if known_column not in result:
            raise ValueError(f"{path} is missing {known_column}")
        if result[known_column].astype(str).tolist() != truth.tolist():
            raise ValueError(f"Saved known labels differ: {path}")
    if result[label_column].isna().any():
        raise ValueError(f"Missing predicted label: {path}")
    return result


def candidate_choices(path: Path) -> dict[str, float | int]:
    candidates = pd.read_csv(path, index_col=0)
    required = {"ARI score", "silhouettes"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(candidates.columns))}")
    candidates.index = pd.to_numeric(candidates.index, errors="raise").astype(int)
    silhouette_k = int(candidates["silhouettes"].idxmax())
    upper_k = int(candidates["ARI score"].idxmax())
    return {
        "silhouette_ari": float(candidates.loc[silhouette_k, "ARI score"]),
        "silhouette_k": silhouette_k,
        "silhouette_value": float(candidates.loc[silhouette_k, "silhouettes"]),
        "upper_ari": float(candidates.loc[upper_k, "ARI score"]),
        "upper_k": upper_k,
    }


def validate_family_sets(paths: dict[str, Path], expected_count: int) -> list[str]:
    family_sets = {
        name: numeric_csv_families(directory)
        for name, directory in paths.items()
        if name not in {"annotation", "family_metadata"}
    }
    reference_name = "matrix"
    reference = family_sets[reference_name]
    if len(reference) != expected_count:
        raise ValueError(
            f"Expected {expected_count} matrix families, observed {len(reference)}"
        )
    for name, observed in family_sets.items():
        if observed != reference:
            missing = sorted(reference - observed, key=int)
            extra = sorted(observed - reference, key=int)
            raise ValueError(
                f"Family set differs for {name}: missing={missing[:10]}, extra={extra[:10]}"
            )
    annotation_families = numeric_csv_families(paths["annotation"])
    if not reference.issubset(annotation_families):
        missing = sorted(reference - annotation_families, key=int)
        raise ValueError(f"Annotation families are missing: {missing[:10]}")
    return sorted(reference, key=int)


def evaluate_all(paths: dict[str, Path], families: list[str]) -> pd.DataFrame:
    metadata = pd.read_csv(paths["family_metadata"])
    metadata["family_ID"] = metadata["family_ID"].astype(int).astype(str)
    metadata = metadata.set_index("family_ID")
    rows: list[dict[str, object]] = []
    for position, family in enumerate(families, start=1):
        truth = load_truth(paths["matrix"], paths["annotation"], family)
        family_metadata = metadata.loc[family]
        row: dict[str, object] = {
            "family": family,
            "order": family_metadata["order"],
            "TE": family_metadata["TE"],
            "n_sequences": len(truth),
            "known_cluster_number": int(truth.nunique()),
        }

        sequence = pd.read_csv(paths["sequence"] / f"{family}.csv", index_col=0)
        sequence.index = sequence.index.astype(str)
        if set(sequence.index) != set(truth.index):
            raise ValueError(f"Sequence-similarity ID mismatch: family={family}")
        sequence = sequence.loc[truth.index]
        if sequence["subfamily"].astype(str).tolist() != truth.tolist():
            raise ValueError(f"Sequence-similarity known labels differ: family={family}")
        for rule in SEQUENCE_RULES:
            labels = sequence[str(rule)]
            row[f"sequence_{rule}_ARI"] = adjusted_rand_score(truth, labels)
            row[f"sequence_{rule}_clusters"] = int(labels.nunique())

        tree = pd.read_csv(paths["tree"] / f"{family}.csv", index_col=0)
        tree.index = tree.index.astype(str)
        if set(tree.index) != set(truth.index):
            raise ValueError(f"Phylogenetic-tree ID mismatch: family={family}")
        tree = tree.loc[truth.index]
        for cutoff in TREE_CUTOFFS:
            label = str(cutoff)
            row[f"tree_{label}_ARI"] = adjusted_rand_score(truth, tree[label])
            row[f"tree_{label}_clusters"] = int(tree[label].nunique())

        dbscan = align_output(
            paths["dbscan"] / f"{family}.csv",
            truth,
            "cluster_number",
            "known_cluster",
        )
        db_labels = pd.to_numeric(dbscan["cluster_number"], errors="raise").astype(int)
        noise = db_labels.eq(-1)
        row["dbscan_noise_count"] = int(noise.sum())
        row["dbscan_noise_rate"] = float(noise.mean())
        row["dbscan_clusters"] = len(set(db_labels) - {-1})
        row["dbscan_noise_single_label_ARI"] = adjusted_rand_score(truth, db_labels)
        unique_noise = db_labels.copy()
        non_noise = db_labels.loc[~noise]
        next_label = int(non_noise.max()) + 1 if len(non_noise) else 0
        unique_noise.loc[noise] = np.arange(next_label, next_label + int(noise.sum()))
        row["dbscan_ARI"] = adjusted_rand_score(truth, unique_noise)
        row["dbscan_noise_unique_ARI"] = row["dbscan_ARI"]
        keep = ~noise
        row["dbscan_noise_removed_ARI"] = (
            adjusted_rand_score(truth.loc[keep], db_labels.loc[keep])
            if int(keep.sum()) >= 2
            else np.nan
        )

        umap = candidate_choices(paths["umap"] / f"{family}.csv")
        row["umap_silhouette_ARI"] = umap["silhouette_ari"]
        row["umap_silhouette_k"] = umap["silhouette_k"]
        row["umap_silhouette_value"] = umap["silhouette_value"]

        spectral = candidate_choices(paths["spectral"] / f"{family}.csv")
        row["spectral_silhouette_ARI"] = spectral["silhouette_ari"]
        row["spectral_silhouette_k"] = spectral["silhouette_k"]
        row["spectral_silhouette_value"] = spectral["silhouette_value"]
        row["spectral_upper_ARI"] = spectral["upper_ari"]
        row["spectral_upper_k"] = spectral["upper_k"]

        recursive_k2 = align_output(
            paths["recursive_k2"] / f"{family}.csv",
            truth,
            "cluster_number",
            "known_cluster",
        )
        row["recursive_k2_ARI"] = adjusted_rand_score(
            truth, recursive_k2["cluster_number"]
        )
        row["recursive_k2_clusters"] = int(recursive_k2["cluster_number"].nunique())

        recursive_silhouette = align_output(
            paths["recursive_silhouette"] / f"{family}.csv",
            truth,
            "cluster_number",
            "known_cluster",
        )
        row["recursive_silhouette_ARI"] = adjusted_rand_score(
            truth, recursive_silhouette["cluster_number"]
        )
        row["recursive_silhouette_clusters"] = int(
            recursive_silhouette["cluster_number"].nunique()
        )
        rows.append(row)
        if position % 100 == 0 or position == len(families):
            print(f"evaluated {position}/{len(families)}", flush=True)
    return pd.DataFrame(rows).set_index("family")


def build_method_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    informative = comparison.loc[comparison["known_cluster_number"] > 1]
    informative_te = informative.loc[informative["TE"].eq("TE")]
    summaries = [
        summarize_methods(comparison, PRIMARY_METHODS, subset_name="all families"),
        summarize_methods(informative, PRIMARY_METHODS, subset_name="known k > 1"),
        summarize_methods(
            informative_te,
            PRIMARY_METHODS,
            subset_name="known k > 1 and TE",
        ),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    numeric = ["mean_ARI", "sequence_weighted_mean_ARI", "median_ARI", "Q1", "Q3"]
    summary[numeric] = summary[numeric].round(4)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "matrix",
        "annotation",
        "sequence",
        "tree",
        "dbscan",
        "umap",
        "spectral",
        "recursive-k2",
        "recursive-silhouette",
    ):
        parser.add_argument(f"--{name}-dir", required=True, type=Path)
    parser.add_argument("--family-metadata", required=True, type=Path)
    parser.add_argument("--expected-family-count", required=True, type=int)
    parser.add_argument("--family-output", required=True, type=Path)
    parser.add_argument("--method-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "matrix": args.matrix_dir,
        "annotation": args.annotation_dir,
        "family_metadata": args.family_metadata,
        "sequence": args.sequence_dir,
        "tree": args.tree_dir,
        "dbscan": args.dbscan_dir,
        "umap": args.umap_dir,
        "spectral": args.spectral_dir,
        "recursive_k2": args.recursive_k2_dir,
        "recursive_silhouette": args.recursive_silhouette_dir,
    }
    families = validate_family_sets(paths, args.expected_family_count)
    comparison = evaluate_all(paths, families)
    if len(comparison) != args.expected_family_count:
        raise ValueError("Final comparison count changed during evaluation")
    method_summary = build_method_summary(comparison)
    args.family_output.parent.mkdir(parents=True, exist_ok=True)
    args.method_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.family_output)
    method_summary.to_csv(args.method_output, index=False)
    print(f"family summary: {args.family_output}")
    print(f"method summary: {args.method_output}")


if __name__ == "__main__":
    main()
