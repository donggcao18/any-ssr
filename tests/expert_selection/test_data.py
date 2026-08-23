import unittest

from expert_selection.config import ExperimentConfig
from expert_selection.data import build_calibration_pool, resolved_sample_indices


class FakeDataset:
    column_names = ["prompt", "answer"]
    _fingerprint = "fake-fingerprint"

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


class DataTests(unittest.TestCase):
    def test_count_is_capped_before_indices(self):
        indices = resolved_sample_indices(272, 500, 1234)
        self.assertEqual(len(indices), 272)
        self.assertEqual(max(indices), 271)

    def test_sampling_is_deterministic(self):
        self.assertEqual(resolved_sample_indices(1000, 256, 7), resolved_sample_indices(1000, 256, 7))
        self.assertNotEqual(resolved_sample_indices(1000, 256, 7), resolved_sample_indices(1000, 256, 8))

    def test_calibration_views_are_disjoint_and_gca_is_support_subset(self):
        dataset = FakeDataset([{"prompt": str(index), "answer": "x"} for index in range(256)])
        config = ExperimentConfig(targets=("BFP",), methods=("gmm", "gca", "oia", "slu"))
        pool = build_calibration_pool(dataset, config, "BFP")
        self.assertEqual(len(pool.support_positions), 96)
        self.assertEqual(len(pool.verification_positions), 96)
        self.assertTrue(set(pool.support_positions).isdisjoint(pool.verification_positions))
        self.assertTrue(set(pool.gca_positions).issubset(pool.support_positions))


if __name__ == "__main__":
    unittest.main()

