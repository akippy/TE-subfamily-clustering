"""Input/output helpers with strict sequence-ID validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from Bio import SeqIO


def _string_index(values: Iterable[object], *, source: str) -> pd.Index:
    index = pd.Index([str(value) for value in values], dtype="object")
    if index.empty:
        raise ValueError(f"{source} contains no sequence IDs")
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique().tolist()
        raise ValueError(f"{source} contains duplicate sequence IDs: {duplicates[:5]}")
    if any(not value.strip() for value in index):
        raise ValueError(f"{source} contains an empty sequence ID")
    return index


def validate_square_matrix(
    matrix: pd.DataFrame,
    *,
    source: str = "matrix",
    require_symmetric: bool = True,
    require_nonnegative: bool = False,
) -> pd.DataFrame:
    """Return a numeric square matrix after validating labels and values."""

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{source} is not square: shape={matrix.shape}")
    matrix = matrix.copy()
    matrix.index = _string_index(matrix.index, source=f"{source} row labels")
    matrix.columns = _string_index(matrix.columns, source=f"{source} column labels")
    if not matrix.index.equals(matrix.columns):
        missing = matrix.index.difference(matrix.columns).tolist()
        extra = matrix.columns.difference(matrix.index).tolist()
        raise ValueError(
            f"{source} row/column IDs or order differ: "
            f"missing_columns={missing[:5]}, extra_columns={extra[:5]}"
        )

    numeric = matrix.apply(pd.to_numeric, errors="raise").astype(float)
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains NaN or infinite values")
    if require_nonnegative and (values < 0).any():
        raise ValueError(f"{source} contains negative values")
    if require_symmetric and not np.allclose(values, values.T, rtol=1e-7, atol=1e-10):
        difference = float(np.max(np.abs(values - values.T)))
        raise ValueError(f"{source} is not symmetric: max_abs_difference={difference}")
    return numeric


def read_mldist(path: str | Path) -> pd.DataFrame:
    """Read an IQ-TREE ``.mldist`` file as a labeled distance matrix."""

    input_path = Path(path)
    with input_path.open(encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        try:
            expected_size = int(first_line)
        except ValueError as exc:
            raise ValueError(
                f"{input_path} has an invalid first-line matrix size: {first_line!r}"
            ) from exc

        sequence_ids: list[str] = []
        rows: list[list[float]] = []
        for line_number, line in enumerate(handle, start=2):
            fields = line.split()
            if not fields:
                continue
            sequence_ids.append(fields[0])
            try:
                rows.append([float(value) for value in fields[1:]])
            except ValueError as exc:
                raise ValueError(
                    f"{input_path}:{line_number} contains a non-numeric distance"
                ) from exc

    if len(sequence_ids) != expected_size:
        raise ValueError(
            f"{input_path} declares {expected_size} rows but contains {len(sequence_ids)}"
        )
    invalid_widths = [index + 1 for index, row in enumerate(rows) if len(row) != expected_size]
    if invalid_widths:
        raise ValueError(
            f"{input_path} rows have the wrong number of distances: {invalid_widths[:5]}"
        )

    matrix = pd.DataFrame(rows, index=sequence_ids, columns=sequence_ids, dtype=float)
    return validate_square_matrix(
        matrix,
        source=str(input_path),
        require_symmetric=True,
        require_nonnegative=True,
    )


def read_matrix_csv(path: str | Path) -> pd.DataFrame:
    """Read a labeled square CSV matrix."""

    input_path = Path(path)
    matrix = pd.read_csv(input_path, index_col=0)
    return validate_square_matrix(matrix, source=str(input_path))


def read_labels(path: str | Path) -> pd.Series:
    """Read ``sequence_id,known_subfamily`` labels in their saved order."""

    input_path = Path(path)
    frame = pd.read_csv(input_path, dtype=str)
    required = {"sequence_id", "known_subfamily"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{input_path} contains missing sequence IDs or labels")
    index = _string_index(frame["sequence_id"], source=str(input_path))
    return pd.Series(
        frame["known_subfamily"].astype(str).to_numpy(),
        index=index,
        name="known_subfamily",
    )


def read_alignment(path: str | Path) -> dict[str, str]:
    """Read an aligned FASTA without assuming any sequence-ID format."""

    input_path = Path(path)
    records = list(SeqIO.parse(input_path, "fasta"))
    if not records:
        raise ValueError(f"{input_path} contains no FASTA records")
    sequence_ids = _string_index((record.id for record in records), source=str(input_path))
    sequences = [str(record.seq) for record in records]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"{input_path} is not an alignment: lengths={sorted(lengths)[:5]}")
    return dict(zip(sequence_ids, sequences))


def validate_sequence_ids(
    reference_ids: Iterable[object],
    *collections: tuple[str, Iterable[object]],
    require_same_order: bool = True,
) -> None:
    """Validate sequence-ID equality across matrices, FASTA files, and labels."""

    reference = _string_index(reference_ids, source="reference IDs")
    for source, values in collections:
        observed = _string_index(values, source=source)
        if require_same_order:
            matches = reference.equals(observed)
        else:
            matches = set(reference) == set(observed)
        if not matches:
            missing = reference.difference(observed).tolist()
            extra = observed.difference(reference).tolist()
            order_only = not missing and not extra
            raise ValueError(
                f"Sequence IDs differ for {source}: missing={missing[:5]}, "
                f"extra={extra[:5]}, order_only_difference={order_only}"
            )


def write_dataframe(frame: pd.DataFrame, path: str | Path, *, index: bool = True) -> None:
    """Write a dataframe after creating its parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=index)
