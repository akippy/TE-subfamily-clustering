from __future__ import annotations

import unittest

import pandas as pd

from te_subfamily_clustering.evaluation import (
    evaluate_clustering,
    evaluate_dbscan_with_unique_noise,
)


class EvaluationTests(unittest.TestCase):
    def test_adjusted_rand_index(self) -> None:
        truth = pd.Series(["A", "A", "B", "B"], index=list("abcd"))
        predictions = pd.Series([1, 1, 0, 0], index=list("abcd"))
        result = evaluate_clustering(predictions, truth)
        self.assertEqual(result["adjusted_rand_index"], 1.0)
        self.assertEqual(result["predicted_clusters"], 2)

    def test_dbscan_noise_points_become_singletons(self) -> None:
        truth = pd.Series(["A", "A", "B", "B"], index=list("abcd"))
        predictions = pd.Series([0, -1, 1, -1], index=list("abcd"))
        result = evaluate_dbscan_with_unique_noise(predictions, truth)
        self.assertEqual(result["noise_count"], 2)
        self.assertEqual(result["predicted_clusters_excluding_noise"], 2)


if __name__ == "__main__":
    unittest.main()
