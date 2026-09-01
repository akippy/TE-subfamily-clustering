from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from te_subfamily_clustering.affinity import distance_to_affinity
from te_subfamily_clustering.io import read_labels, read_mldist
from te_subfamily_clustering.spectral import run_spectral_clustering


ROOT = Path(__file__).resolve().parents[1]


class SpectralRegressionTests(unittest.TestCase):
    def test_family_372_matches_verified_expected_output(self) -> None:
        data = ROOT / "examples" / "data" / "family_372"
        expected_directory = ROOT / "examples" / "expected" / "family_372"
        affinity = distance_to_affinity(read_mldist(data / "pairwise_ml_distance.mldist"))
        labels = read_labels(data / "labels.csv")
        result = run_spectral_clustering(affinity, labels)
        expected_membership = pd.read_csv(
            expected_directory / "quick_start_membership.csv", index_col=0
        )
        expected_scores = pd.read_csv(
            expected_directory / "quick_start_candidate_scores.csv", index_col=0
        )
        self.assertEqual(result.selected_k, 2)
        np.testing.assert_array_equal(
            result.membership["predicted_cluster"].to_numpy(),
            expected_membership["predicted_cluster"].to_numpy(),
        )
        np.testing.assert_allclose(
            result.membership["silhouette"].to_numpy(),
            expected_membership["silhouette"].to_numpy(),
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.candidate_scores.to_numpy(),
            expected_scores.to_numpy(),
            rtol=1e-9,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
