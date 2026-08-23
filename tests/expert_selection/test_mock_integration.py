import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from expert_selection.adapters import AdapterSchema, AdapterValidation
from expert_selection.config import ExperimentConfig
from expert_selection.records import MethodScore
from expert_selection.run import execute_target_with_resources, make_run_state


class FakeDataset:
    def __len__(self):
        return 8


class FakeCalibration:
    fingerprint = "fake"
    dataset = tuple(range(8))

    def aggregate_summary(self):
        return {"pool_count": 8, "support_count": 3, "verification_count": 3, "gca_count": 2}


class FakeMethod:
    def __init__(self, name):
        self.name = name

    def prepare_target(self, context):
        return context.shared.get("lookahead", object())

    def score_candidate(self, context, candidate, artifacts):
        return MethodScore(1.0 / (candidate.index + 1), "ok", {"mock": True})


class MockIntegrationTests(unittest.TestCase):
    def test_any_method_subset_uses_same_candidate_rows(self):
        schema = AdapterSchema({}, {"r": 8}, "schema")

        def fake_validate(candidates, _model, _revision=None):
            return [
                AdapterValidation(item.task, item.adapter_path, "ok", None, schema, "hash", item.adapter_path / "adapter_model.bin", ())
                for item in candidates
            ], schema

        fake_lookahead = SimpleNamespace(
            fresh_lora_seed=42,
            fresh_adapter_hash="fresh-hash",
            fresh=SimpleNamespace(nll={0: 2.0, 1: 1.0}, supervised_tokens={0: 10, 1: 10}),
            sources={},
        )
        for methods in (("gmm",), ("gca",), ("oia", "slu"), ("gmm", "gca", "oia", "slu")):
            with self.subTest(methods=methods), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(
                    targets=("BFP",), methods=methods, adapter_root=Path(directory), continue_on_error=False
                )
                state = make_run_state(config)
                bundle = SimpleNamespace(model=object(), tokenizer=object(), device=SimpleNamespace(type="cpu"), dtype_name="float32")
                with (
                    patch("expert_selection.run.validate_adapter_set", side_effect=fake_validate),
                    patch("expert_selection.run.build_calibration_pool", return_value=FakeCalibration()),
                    patch("expert_selection.run.build_methods", side_effect=lambda names: [FakeMethod(name) for name in names]),
                    patch("expert_selection.run.run_lookahead", return_value=fake_lookahead),
                ):
                    execute_target_with_resources(state, "BFP", FakeDataset(), bundle)
                self.assertEqual(len(state.candidate_rows), 3)
                self.assertEqual({row["source_task"] for row in state.candidate_rows}, {"CONCODE", "CodeTrans", "CodeSearchNet"})
                self.assertEqual(len(state.records), 3 * len(methods))


if __name__ == "__main__":
    unittest.main()
