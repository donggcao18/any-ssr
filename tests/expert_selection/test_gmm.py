import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_selection.config import ExperimentConfig
from expert_selection.methods.gmm import DiagonalGMM, GMMArtifact, load_artifact, monte_carlo_jsd, write_artifact


class GMMTests(unittest.TestCase):
    def distribution(self, shift=0.0):
        return DiagonalGMM(
            weights=np.array([1.0]),
            means=np.array([[shift, shift]], dtype=np.float64),
            covariances=np.array([[1.0, 1.0]], dtype=np.float64),
            diagnostics={"converged": True},
        )

    def test_self_is_more_similar_than_shifted(self):
        target = self.distribution()
        target_samples = target.sample(20000, 11)
        same = monte_carlo_jsd(target, target, n_mc=20000, chunk_size=256, source_seed=12, target_samples=target_samples)
        shifted = monte_carlo_jsd(self.distribution(5.0), target, n_mc=20000, chunk_size=256, source_seed=12, target_samples=target_samples)
        self.assertGreater(same["similarity"], shifted["similarity"])
        self.assertGreaterEqual(same["similarity"], 0.0)
        self.assertLessEqual(same["similarity"], 1.0)

    def test_estimator_is_approximately_symmetric(self):
        left, right = self.distribution(), self.distribution(1.0)
        lr = monte_carlo_jsd(left, right, n_mc=30000, chunk_size=512, source_seed=4, target_seed=5)
        rl = monte_carlo_jsd(right, left, n_mc=30000, chunk_size=512, source_seed=5, target_seed=4)
        self.assertLess(abs(lr["jsd"] - rl["jsd"]), 0.025)
        self.assertTrue(math.isfinite(lr["standard_error"]))

    def test_invalid_variance_is_rejected(self):
        invalid = self.distribution()
        invalid.covariances[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "variances"):
            invalid.validate()

    def test_durable_artifact_contains_only_compact_arrays_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                targets=("BFP",), methods=("gmm",), gmm_artifact_root=Path(directory)
            )
            metadata = {
                "format_version": 1,
                "task": "CONCODE",
                "task_order": list(config.task_order),
                "task_order_id": config.task_order_id,
                "representation_role": "future_source",
                "source_cap": 10000,
                "resolved_count": 100,
                "dataset_fingerprint": "aggregate",
                "sample_checksum": "non-reconstructive",
                "seed": 1,
                "sampling_seed": 2,
                "provenance": "online_current_task",
                "pipeline": {"dimension": 2},
                "dtype": "float32",
                "fit_diagnostics": {"converged": True},
            }
            path = write_artifact(config, "CONCODE", GMMArtifact(self.distribution(), metadata))
            loaded = load_artifact(path)
            self.assertEqual(loaded.metadata["representation_role"], "future_source")
            self.assertEqual(set(item.name for item in path.iterdir()), {"gmm.npz", "metadata.json"})
            durable_text = (path / "metadata.json").read_text()
            for forbidden in ("prompt", "answer", "input_ids", "row_ids", "representations"):
                self.assertNotIn(forbidden, durable_text)


if __name__ == "__main__":
    unittest.main()
