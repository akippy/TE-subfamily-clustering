from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from te_subfamily_clustering.affinity import distance_to_affinity
from te_subfamily_clustering.baselines import run_fixed_binary_recursive
from te_subfamily_clustering.io import read_alignment, read_labels, read_mldist
from te_subfamily_clustering.recursive import run_recursive_clustering


ROOT = Path(__file__).resolve().parents[1]
RUN_EXTERNAL = os.environ.get("RUN_FULL_RECURSIVE_TESTS") == "1"
TOOLS_AVAILABLE = shutil.which("swipe") is not None and shutil.which("makeblastdb") is not None


@unittest.skipUnless(
    RUN_EXTERNAL and TOOLS_AVAILABLE,
    "set RUN_FULL_RECURSIVE_TESTS=1 with SWIPE and makeblastdb available",
)
class RecursiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = ROOT / "examples" / "data" / "family_308"
        cls.affinity = distance_to_affinity(
            read_mldist(data / "pairwise_ml_distance.mldist")
        )
        cls.labels = read_labels(data / "labels.csv")
        cls.alignment = read_alignment(data / "alignment.fasta")

    def test_primary_recursive_membership(self) -> None:
        result = run_recursive_clustering(
            self.affinity,
            self.alignment,
            self.labels,
            family_name="family_308",
        )
        expected = pd.read_csv(
            ROOT / "examples" / "expected" / "family_308" / "recursive_membership.csv",
            index_col=0,
            dtype={"predicted_cluster": str},
        )
        np.testing.assert_array_equal(
            result.membership["predicted_cluster"].to_numpy(),
            expected["predicted_cluster"].to_numpy(),
        )
        self.assertEqual(int(result.diagnostics["accepted"].sum()), 2)

    def test_fixed_binary_membership(self) -> None:
        result = run_fixed_binary_recursive(
            self.affinity,
            self.alignment,
            self.labels,
            family_name="family_308",
        )
        expected = pd.read_csv(
            ROOT
            / "examples"
            / "expected"
            / "family_308"
            / "fixed_binary_membership.csv",
            index_col=0,
            dtype={"predicted_cluster": str},
        )
        np.testing.assert_array_equal(
            result.membership["predicted_cluster"].to_numpy(),
            expected["predicted_cluster"].to_numpy(),
        )


if __name__ == "__main__":
    unittest.main()
