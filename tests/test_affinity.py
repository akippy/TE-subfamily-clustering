from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from te_subfamily_clustering.affinity import distance_to_affinity


class AffinityTests(unittest.TestCase):
    def test_family_maximum_transform_and_zero_diagonal(self) -> None:
        distance = pd.DataFrame(
            [[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        observed = distance_to_affinity(distance)
        expected = np.array(
            [[0.0, 0.75, 0.0], [0.75, 0.0, 0.5], [0.0, 0.5, 0.0]]
        )
        np.testing.assert_allclose(observed.to_numpy(), expected)
        self.assertEqual(observed.index.tolist(), ["a", "b", "c"])
        self.assertEqual(observed.columns.tolist(), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
