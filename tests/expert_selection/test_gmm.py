import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_selection.config import ExperimentConfig
from expert_selection.methods.gmm import (
    DiagonalGMM,
    GMMArtifact,
    calibration_log_likelihood,
    load_artifact,
    write_artifact,
)


class GMMTests(unittest.TestCase):
    def distribution(self, shift=0.0):
        return DiagonalGMM(
            weights=np.array([1.0]),
            means=np.array([[shift, shift]], dtype=np.float64),
            covariances=np.array([[1.0, 1.0]], dtype=np.float64),
            diagnostics={"converged": True},
        )

    def test_matching_source_has_higher_calibration_likelihood_than_shifted_source(self):
        target_vectors = np.array([[0.0, 0.0], [0.2, -0.1], [-0.2, 0.1]], dtype=np.float64)
        matching = calibration_log_likelihood(self.distribution(), target_vectors, chunk_size=2)
        shifted = calibration_log_likelihood(self.distribution(5.0), target_vectors, chunk_size=2)
        self.assertGreater(
            matching["log_likelihood_per_dimension"],
            shifted["log_likelihood_per_dimension"],
        )
        self.assertAlmostEqual(matching["mean_nll"], -matching["mean_log_likelihood"])

    def test_calibration_likelihood_is_chunk_size_invariant(self):
        vectors = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]], dtype=np.float64)
        one = calibration_log_likelihood(self.distribution(), vectors, chunk_size=1)
        all_at_once = calibration_log_likelihood(self.distribution(), vectors, chunk_size=32)
        self.assertAlmostEqual(one["mean_log_likelihood"], all_at_once["mean_log_likelihood"])

    def test_invalid_variance_is_rejected(self):
        invalid = self.distribution()
        invalid.covariances[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "variances"):
            invalid.validate()

    def test_validation_normalizes_harmless_weight_sum_drift(self):
        distribution = DiagonalGMM(
            weights=np.array([0.2, 0.3, 0.50000002], dtype=np.float64),
            means=np.zeros((3, 2), dtype=np.float64),
            covariances=np.ones((3, 2), dtype=np.float64),
            diagnostics={"converged": True},
        )
        distribution.validate()
        self.assertEqual(distribution.weights.sum(dtype=np.float64), 1.0)

    def test_material_weight_corruption_is_rejected(self):
        distribution = DiagonalGMM(
            weights=np.array([0.2, 0.3, 0.6], dtype=np.float64),
            means=np.zeros((3, 2), dtype=np.float64),
            covariances=np.ones((3, 2), dtype=np.float64),
            diagnostics={"converged": True},
        )
        with self.assertRaisesRegex(ValueError, "materially differ"):
            distribution.validate()

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
