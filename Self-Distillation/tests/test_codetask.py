"""Offline contract tests; no Hub access, model download, or GPU required."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codetask_data as data
import run_codetask_sequential as sequence
import main as entrypoint


class MemoryDataset:
    """Small datasets API stand-in to check loader calls without HF dependencies."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else ["input", "output"]
        self._fingerprint = "offline-fixture"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        if isinstance(key, slice):
            return {name: [row[name] for row in self.rows[key]] for name in self.column_names}
        return self.rows[key]

    def select_columns(self, columns):
        return MemoryDataset([{name: row[name] for name in columns} for row in self.rows])

    def add_column(self, name, values):
        return MemoryDataset([dict(row, **{name: value}) for row, value in zip(self.rows, values)])

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return MemoryDataset(rows)

    def select(self, indices):
        return MemoryDataset([self.rows[index] for index in indices])

    def map(self, function, remove_columns, **kwargs):
        assert set(remove_columns) == set(self.column_names)
        return MemoryDataset([function(row) for row in self.rows])


def fixture_rows():
    return [{"input": f"Translate function {i}\n  keep indentation",
             "output": f"REFERENCE_ONLY_{i}\n  return {i};", "unused": i} for i in range(20)]


class LoaderTests(unittest.TestCase):
    def load(self, rows=None, **kwargs):
        loader = Mock(return_value=MemoryDataset(fixture_rows() if rows is None else rows))
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=loader)}):
            result = data.load_codetask_dataset("CodeTrans", **kwargs)
        return result, loader

    def test_subset_is_reproducible_and_reference_is_teacher_only(self):
        (dataset, manifest), loader = self.load(num_train=5, seed=7, revision="fixed-commit")
        (again, same), _ = self.load(num_train=5, seed=7, revision="fixed-commit")
        (_, different), _ = self.load(num_train=5, seed=8)
        self.assertEqual(manifest["source_indices"], same["source_indices"])
        self.assertNotEqual(manifest["source_indices"], different["source_indices"])
        self.assertEqual(len(dataset), 5)
        self.assertEqual(set(dataset.column_names), {"prompt", "teacher_prompt"})
        loader.assert_called_once_with(data.CODETASK_REPO,
            data_files={"train": "CodeTrans/train-*.parquet"}, split="train", revision="fixed-commit")
        for index, row in zip(manifest["source_indices"], dataset.rows):
            self.assertEqual(row["prompt"][0]["content"], fixture_rows()[index]["input"])
            self.assertNotIn("REFERENCE_ONLY", row["prompt"][0]["content"])
            self.assertIn(fixture_rows()[index]["output"], row["teacher_prompt"][0]["content"])

    def test_all_rows_and_invalid_counts(self):
        (dataset, manifest), _ = self.load(num_train=-1)
        self.assertEqual(manifest["source_indices"], list(range(20)))
        self.assertEqual(len(dataset), 20)
        for count in (0, -2, 21):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.load(num_train=count)

    def test_bad_schema_and_empty_or_invalid_rows(self):
        for rows in ([], [{"input": "x"}], [{"input": None, "output": "x"}],
                     [{"input": "x", "output": "   "}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(rows=rows, num_train=-1)

    def test_prompt_length_counts_cover_teacher(self):
        (dataset, _), _ = self.load(num_train=2)
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"]
        tokenizer.side_effect = lambda texts, **kwargs: {"input_ids": [list(text) for text in texts]}
        stats = data.prompt_length_stats(dataset, tokenizer, 100)
        self.assertEqual(stats["prompt"]["rows_over_limit"], 0)
        self.assertEqual(stats["teacher_prompt"]["rows_over_limit"], 2)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "Hugging Face datasets is not installed")
    def test_real_hf_sampling_matches_original_loader(self):
        from datasets import Dataset
        source = Dataset.from_list(fixture_rows())
        with patch("datasets.load_dataset", return_value=source):
            dataset, manifest = data.load_codetask_dataset("BFP", num_train=7, seed=1234)
        expected = source.shuffle(seed=1234).select(range(7))
        self.assertEqual([row[0]["content"] for row in dataset["prompt"]], list(expected["input"]))
        self.assertEqual(manifest["source_indices"], list(expected["unused"]))


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    def args(self, *extra):
        return sequence.parse_args(["--output_dir", str(self.root), *extra])

    def test_eight_tasks_counts_and_checkpoint_chain(self):
        args = self.args("--num_train", "1,2,3,4,5,6,7,8")
        stages = sequence.build_stages(args)
        self.assertEqual([stage["task"] for stage in stages], list(data.CODETASK_TASKS))
        self.assertEqual([stage["num_train"] for stage in stages], list(range(1, 9)))
        self.assertEqual(stages[0]["source_model"], "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        for previous, current in zip(stages, stages[1:]):
            self.assertEqual(current["source_model"], previous["final_dir"])
            self.assertIn(previous["final_dir"], current["command"])

    def test_no_full_dataset_counts(self):
        self.assertEqual(sequence.positive_counts("100", 8), [100] * 8)
        for value in ("-1", "0", "1,2", "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sequence.positive_counts(value, 8)

    def test_dry_run_has_no_side_effects(self):
        with patch.object(sequence.subprocess, "run") as launch, contextlib.redirect_stdout(io.StringIO()):
            sequence.run(self.args("--dry_run"))
        launch.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_stages_run_in_order_and_validate_saves(self):
        observed = []

        def simulate_training(command, check):
            source = command[command.index("--model_name") + 1]
            if observed:
                sequence.validate_checkpoint(source)
            observed.append(source)
            final = Path(command[command.index("--output_dir") + 1]) / "final"
            final.mkdir(parents=True)
            for name in ("adapter_config.json", "tokenizer_config.json", "training_complete.json", "adapter_model.safetensors"):
                (final / name).write_text("{}")

        with patch.object(sequence.subprocess, "run", side_effect=simulate_training), contextlib.redirect_stdout(io.StringIO()):
            sequence.run(self.args())
        self.assertEqual(len(observed), 8)
        manifest = json.loads((self.root / "sequence_manifest.json").read_text())
        self.assertEqual(manifest["teacher_at_task_boundary"], "reset_from_previous_final_student")

    def test_failure_and_missing_checkpoint_stop_sequence(self):
        for error in (subprocess.CalledProcessError(1, "training"), None):
            args = self.args("--output_dir", str(self.root / str(error is None)))
            with patch.object(sequence.subprocess, "run", side_effect=error) as launch, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
                    sequence.run(args)
            self.assertEqual(launch.call_count, 1)

    def test_existing_outputs_are_preserved(self):
        self.root.mkdir()
        marker = self.root / "existing.txt"
        marker.write_text("keep")
        with patch.object(sequence.subprocess, "run") as launch, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                sequence.run(self.args())
        launch.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")


class EntryPointTests(unittest.TestCase):
    def test_lora_training_saves_adapter_and_disables_vllm(self):
        tokenizer, model, trainer = Mock(), Mock(), Mock()
        trainer.state.global_step = 4
        trainer.model = model
        trainer.is_world_process_zero.return_value = True
        trainer.save_model.side_effect = lambda path: Path(path).mkdir(parents=True)
        config_type = Mock(side_effect=lambda **kw: types.SimpleNamespace(
            generation_batch_size=kw["steps_per_generation"], **kw))
        modules = {
            "transformers": types.SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer)),
                                                  AutoModelForCausalLM=Mock()),
            "torch": types.SimpleNamespace(float16="fp16"),
            "distil_config": types.SimpleNamespace(DistilConfig=config_type),
            "distil_trainer": types.SimpleNamespace(DistilTrainer=Mock(return_value=trainer)),
        }
        with tempfile.TemporaryDirectory() as temp:
            argv = ["main.py", "--dataset_name", "codetask", "--codetask_task", "BFP",
                    "--model_name", "previous/final", "--output_dir", temp, "--num_train", "100"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, modules), \
                 patch.dict(entrypoint.os.environ, {"SDFT_PRECISION": "float16"}), \
                 patch.object(entrypoint, "load_lora_model", return_value=model) as load, \
                 patch.object(entrypoint, "freeze_except_lora", return_value=["lora_A"]) as freeze, \
                 patch.object(entrypoint, "load_codetask_dataset", return_value=([0]*100, {"source_rows": 200})), \
                 patch.object(entrypoint, "prompt_length_stats", return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                entrypoint.main()
            load.assert_called_once_with("previous/final", tokenizer, "fp16", trainable=True)
            freeze.assert_called_once_with(model)
            self.assertFalse(config_type.call_args.kwargs["use_vllm"])
            self.assertTrue(config_type.call_args.kwargs["save_only_model"])
            trainer.train.assert_called_once()
            marker = json.loads((Path(temp) / "final" / "training_complete.json").read_text())
            self.assertEqual(marker["training_mode"], "lora_only")

    def test_preparation_exits_before_loading_training_dependencies(self):
        tokenizer = Mock()
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        stats = {"prompt": {"rows_over_limit": 0}, "teacher_prompt": {"rows_over_limit": 0}}
        with tempfile.TemporaryDirectory() as temp:
            args = ["main.py", "--dataset_name", "codetask", "--codetask_task", "BFP",
                    "--output_dir", temp, "--prepare_only", "--num_train", "3"]
            with patch.object(sys, "argv", args), \
                 patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto_tokenizer)}), \
                 patch.object(entrypoint, "load_codetask_dataset", return_value=([1, 2, 3], {"source_rows": 20})), \
                 patch.object(entrypoint, "prompt_length_stats", return_value=stats), \
                 contextlib.redirect_stdout(io.StringIO()):
                entrypoint.main()
            config = json.loads((Path(temp) / "run_config.json").read_text())
            self.assertEqual(config["num_loss_tokens_to_skip"], 0)
            self.assertTrue((Path(temp) / "data_manifest.json").is_file())
            self.assertFalse((Path(temp) / "final").exists())


if __name__ == "__main__":
    unittest.main()
