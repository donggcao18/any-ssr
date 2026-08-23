import unittest
from pathlib import Path

from expert_selection.config import CANONICAL_TASK_ORDER, ExperimentConfig
from expert_selection.tasks import eligible_sources, method_applicable, sequence_tasks


class TaskConfigTests(unittest.TestCase):
    def config(self, **overrides):
        values = {"targets": ("BFP",), "methods": ("gmm", "gca", "oia", "slu")}
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_canonical_eligibility(self):
        config = self.config()
        self.assertEqual([item.task for item in eligible_sources(config, "BFP")], list(CANONICAL_TASK_ORDER[:3]))

    def test_method_start_points(self):
        config = self.config(targets=("CONCODE", "CodeTrans", "CodeSearchNet"))
        self.assertFalse(method_applicable(config, "CONCODE", "gca"))
        self.assertTrue(method_applicable(config, "CodeTrans", "gca"))
        self.assertFalse(method_applicable(config, "CodeTrans", "gmm"))
        self.assertTrue(method_applicable(config, "CodeSearchNet", "gmm"))

    def test_reordered_sequence_is_configuration_only(self):
        order = ("CodeTrans", "CONCODE", "BFP", "CodeSearchNet", "KodCode", "RunBugRun", "TheVault_Csharp", "CoST")
        config = self.config(task_order=order)
        self.assertEqual([item.task for item in eligible_sources(config, "BFP")], ["CodeTrans", "CONCODE"])
        self.assertTrue(method_applicable(config, "BFP", "gmm"))
        self.assertNotEqual(config.task_order_id, self.config().task_order_id)

    def test_duplicate_unknown_and_missing_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.config(task_order=("CONCODE", "CONCODE", "BFP"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.config(task_order=("CONCODE", "Unknown", "BFP"))
        with self.assertRaisesRegex(ValueError, "absent"):
            self.config(task_order=("CONCODE", "CodeTrans"))

    def test_sequence_traverses_through_last_target_for_gmm(self):
        config = self.config(prepare_artifacts="gmm")
        self.assertEqual(sequence_tasks(config), CANONICAL_TASK_ORDER[:4])
        non_gmm = self.config(methods=("gca",), prepare_artifacts="none")
        self.assertEqual(sequence_tasks(non_gmm), ("BFP",))

    def test_fresh_seed_does_not_depend_on_task_position_or_methods(self):
        schema = {"r": 8, "target_modules": ["q_proj"]}
        first = self.config(methods=("oia",))
        order = ("CodeTrans", "CONCODE", "BFP", "CodeSearchNet", "KodCode", "RunBugRun", "TheVault_Csharp", "CoST")
        second = self.config(task_order=order, methods=("gmm", "gca", "oia", "slu"))
        self.assertEqual(first.fresh_lora_seed("BFP", schema), second.fresh_lora_seed("BFP", schema))


if __name__ == "__main__":
    unittest.main()

