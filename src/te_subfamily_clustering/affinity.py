"""Convert within-family ML distances into a spectral affinity matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import validate_square_matrix


def distance_to_affinity(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply ``W = 1 - D / D_max`` and set the diagonal to zero.

    ``D_max`` is calculated independently for each TE family. The zero
    diagonal intentionally preserves the behavior of the research code.
    """

    distance = validate_square_matrix(
        distance_matrix,
        source="distance matrix",
        require_symmetric=True,
        require_nonnegative=True,
    )
    maximum_distance = float(distance.to_numpy().max())
    if maximum_distance <= 0:
        raise ValueError("The maximum within-family distance must be greater than zero")

    affinity = 1.0 - (distance / maximum_distance)
    np.fill_diagonal(affinity.values, 0.0)
    return validate_square_matrix(
        affinity,
        source="affinity matrix",
        require_symmetric=True,
        require_nonnegative=True,
    )
