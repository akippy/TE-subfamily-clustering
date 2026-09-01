from __future__ import annotations

import re
import unittest
from pathlib import Path

from te_subfamily_clustering.io import (
    read_alignment,
    read_labels,
    read_mldist,
    validate_sequence_ids,
)


ROOT = Path(__file__).resolve().parents[1]


class ExampleInputTests(unittest.TestCase):
    def test_all_public_ids_match_in_order(self) -> None:
        for family, expected_size in (("372", 122), ("308", 233)):
            directory = ROOT / "examples" / "data" / f"family_{family}"
            distance = read_mldist(directory / "pairwise_ml_distance.mldist")
            labels = read_labels(directory / "labels.csv")
            alignment = read_alignment(directory / "alignment.fasta")
            self.assertEqual(len(distance), expected_size)
            validate_sequence_ids(
                distance.index,
                ("labels", labels.index),
                ("alignment", alignment.keys()),
                require_same_order=True,
            )

    def test_public_ids_are_family_specific_sequential_strings(self) -> None:
        for family in ("372", "308"):
            directory = ROOT / "examples" / "data" / f"family_{family}"
            sequence_ids = read_mldist(directory / "pairwise_ml_distance.mldist").index
            pattern = re.compile(rf"^family{family}_seq\d{{3}}$")
            self.assertTrue(all(pattern.fullmatch(value) for value in sequence_ids))
            self.assertNotIn(":", "".join(sequence_ids))
            self.assertNotIn("DF", "".join(sequence_ids))


if __name__ == "__main__":
    unittest.main()
