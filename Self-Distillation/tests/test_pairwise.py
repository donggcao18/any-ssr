"""Pairwise orchestration and split isolation tests, without server/GPU access."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codetask_data as data
import eval_codetask as evaluation
import prepare_pairwise as preparation
import run_codetask_pairwise as pairwise
import main as training
from local_model import resolve_local_model
import local_model
from test_codetask import MemoryDataset, fixture_rows


class PairwiseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, *extra):
        return pairwise.parse_args(["--output_dir", str(self.root / "run"), *extra])

    def test_six_pairs_start_from_same_source_and_evaluate_both(self):
        args = self.args()
        plan = pairwise.build_plan(args)
        self.assertEqual(args.tasks, ["CodeSearchNet", "BFP", "KodCode", "RunBugRun", "TheVault_Csharp", "CoST"])
        training = [job for job in plan if job["name"].startswith("train_")]
        self.assertEqual(len(training), 6)
        initializers = [job["command"][job["command"].index("--model_name") + 1] for job in training]
        self.assertEqual(len(set(initializers)), 1)
        self.assertTrue(initializers[0].endswith("source_model"))
        self.assertIn(pairwise.DEFAULT_SOURCE, plan[0]["command"])
        for task in args.tasks:
            cmd = next(job["command"] for job in plan if job["name"] == f"evaluate_{task}")
            self.assertEqual(cmd[cmd.index("--tasks") + 1], f"CodeTrans,{task}")
        prepare = plan[1]["command"]
        for name, value in (("--num_train", "20000"), ("--num_validation", "1000"), ("--num_test", "2000")):
            self.assertEqual(prepare[prepare.index(name) + 1], value)

    def test_dry_run_never_inspects_server_checkpoint(self):
        with patch.object(preparation, "checkpoint_kind") as inspect, \
             patch.object(pairwise.subprocess, "run") as launch, contextlib.redirect_stdout(io.StringIO()):
            pairwise.run(self.args("--dry_run"))
        inspect.assert_not_called()
        launch.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_distributed_launch_only_wraps_training_and_forwards_batch(self):
        args = self.args("--num_gpus", "2", "--per_device_train_batch_size", "3",
                         "--gradient_accumulation_steps", "4")
        for job in pairwise.build_plan(args):
            cmd = job["command"]
            if job["name"].startswith("train_"):
                self.assertIn("torch.distributed.run", cmd)
                self.assertIn("--nproc_per_node=2", cmd)
                self.assertEqual(cmd[cmd.index("--per_device_train_batch_size") + 1], "3")
                self.assertEqual(cmd[cmd.index("--num_prompts_per_batch") + 1], "4")
            else:
                self.assertNotIn("torch.distributed.run", cmd)

    def test_distributed_batch_plan_retains_full_microbatches(self):
        for rows, world, batch, accum in ((20000, 2, 1, 16), (19, 2, 3, 4), (7, 1, 1, 32)):
            usable, steps = training.training_batch_plan(rows, world, batch, accum)
            self.assertLessEqual(usable, rows)
            self.assertLess(rows - usable, world * batch)
            self.assertEqual(usable % (world * batch * steps), 0)
            self.assertEqual(accum % steps, 0)
        with self.assertRaises(ValueError):
            training.training_batch_plan(3, 2, 2, 16)

    def test_caps_and_source_task_are_enforced(self):
        for extra in (("--num_train", "20001"), ("--num_validation", "1001"), ("--num_test", "2001"),
                      ("--num_train", "-1"), ("--tasks", "CodeTrans"), ("--tasks", "BFP,BFP")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_split_caps_use_all_if_smaller_but_never_exceed_cap(self):
        loader = Mock(return_value=MemoryDataset(fixture_rows()))
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=loader)}):
            selected, manifest = data.load_codetask_split("BFP", "validation", 1000, seed=1234)
            small, small_manifest = data.load_codetask_split("BFP", "test", 5, seed=1234)
        self.assertEqual(len(selected), 20)
        self.assertEqual(manifest["requested_rows"], 1000)
        self.assertEqual(manifest["split"], "validation")
        self.assertEqual(len(small), 5)
        self.assertEqual(small_manifest["split"], "test")
        self.assertEqual(loader.call_args.kwargs["data_files"], {"test": "BFP/test-*.parquet"})

    def test_legacy_prompt_and_evaluation_never_receive_reference(self):
        row = {"input": "Translate the function", "output": "SECRET_REFERENCE"}
        formatted = data.format_codetask_example(row, "legacy")
        self.assertEqual(formatted["prompt"], "input: Translate the function\noutput: ")
        self.assertNotIn(row["output"], formatted["prompt"])
        self.assertIn(row["output"], formatted["teacher_prompt"])
        tokenizer = Mock(side_effect=lambda text, **kwargs: {"input_ids": list(range(len(text)))})
        ids, truncated = evaluation.encode_prompt(tokenizer, row["input"], "legacy", 10)
        self.assertTrue(truncated)
        self.assertEqual(len(ids), 10)
        self.assertEqual(tokenizer.call_args.args[0], formatted["prompt"])
        tokenizer.apply_chat_template.assert_not_called()

    def test_checkpoint_detection_and_adapter_merge_use_recorded_base(self):
        source = self.root / "adapter"
        source.mkdir()
        (source / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "recorded/base"}))
        with self.assertRaises(ValueError):
            preparation.checkpoint_kind(source)
        (source / "adapter_model.safetensors").write_text("fixture")
        self.assertEqual(preparation.checkpoint_kind(source), "adapter")
        (source / "tokenizer_config.json").write_text("{}")
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 17
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        auto_model, peft_model, merged = Mock(), Mock(), Mock()
        peft_model.from_pretrained.return_value.merge_and_unload.return_value = merged
        merged.save_pretrained.side_effect = lambda path, **kwargs: Path(path).mkdir()
        modules = {"torch": types.SimpleNamespace(bfloat16="bf16"),
                   "transformers": types.SimpleNamespace(AutoTokenizer=auto_tokenizer, AutoModelForCausalLM=auto_model),
                   "peft": types.SimpleNamespace(PeftModel=peft_model)}
        with patch.dict(sys.modules, modules), \
             patch.object(preparation, "resolve_local_model", return_value="/cached/base") as resolve:
            preparation.export_checkpoint(str(source), self.root / "export")
        resolve.assert_called_once_with("recorded/base")
        self.assertEqual(auto_model.from_pretrained.call_args.args[0], "/cached/base")
        for loader in (auto_model, auto_tokenizer, peft_model):
            self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])
        auto_model.from_pretrained.return_value.resize_token_embeddings.assert_called_once_with(24)
        peft_model.from_pretrained.return_value.merge_and_unload.assert_called_once_with(safe_merge=True)
        merged.save_pretrained.assert_called_once()

    def test_local_model_directory_needs_no_hub_call(self):
        model = self.root / "local_model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        download = Mock(side_effect=AssertionError("Hub must not be called"))
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(resolve_local_model(str(model)), str(model.resolve()))
        download.assert_not_called()

    def test_model_cache_lookup_is_local_only_and_missing_cache_is_actionable(self):
        model = self.root / "snapshot"
        model.mkdir()
        (model / "config.json").write_text("{}")
        download = Mock(return_value=str(model))
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(resolve_local_model("Qwen/Qwen2.5-Coder-1.5B"), str(model))
            download.assert_called_once_with(repo_id="Qwen/Qwen2.5-Coder-1.5B", local_files_only=True)
            download.side_effect = OSError("cache miss")
            with self.assertRaisesRegex(FileNotFoundError, "--base_model"):
                resolve_local_model("Qwen/Qwen2.5-Coder-1.5B")

    def test_online_switch_allows_snapshot_download(self):
        model = self.root / "downloaded"
        model.mkdir()
        (model / "config.json").write_text("{}")
        download = Mock(return_value=str(model))
        with patch.dict(local_model.os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), \
             patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(resolve_local_model("Qwen/Qwen2.5-Coder-1.5B"), str(model))
        download.assert_called_once_with(repo_id="Qwen/Qwen2.5-Coder-1.5B", local_files_only=False)

    def test_frozen_preparation_separates_training_and_eval_seeds(self):
        calls = []

        def load(task, split, cap, seed, repo, revision):
            calls.append((task, split, cap, seed, revision))
            ds = MagicMock()
            ds.__len__.return_value = min(cap, 10)
            ds.save_to_disk.side_effect = lambda path: Path(path).mkdir(parents=True)
            return ds, {"source_rows": 10, "selected_rows": min(cap, 10)}

        args = self.args("--tasks", "BFP")
        args.source_task = "CodeTrans"
        args.tasks = "BFP"
        api = Mock()
        api.return_value.dataset_info.return_value.sha = "resolved-commit"
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=api)}), \
             patch.object(preparation, "load_codetask_split", side_effect=load), contextlib.redirect_stdout(io.StringIO()):
            preparation.prepare_data(args)
        self.assertEqual(len(calls), 5)
        self.assertNotIn(("CodeTrans", "train"), [(task, split) for task, split, *_ in calls])
        for task, split, cap, seed, revision in calls:
            self.assertEqual(seed, args.seed if split == "train" else args.eval_seed)
            self.assertEqual(revision, "resolved-commit")

    def test_offline_data_preparation_never_queries_hub(self):
        args = self.args("--tasks", "BFP")
        args.source_task = "CodeTrans"
        args.tasks = "BFP"
        api = Mock(side_effect=AssertionError("Offline preparation must not query Hub"))
        ds = MagicMock()
        ds.__len__.return_value = 3
        ds.save_to_disk.side_effect = lambda path: Path(path).mkdir(parents=True)
        with patch.dict(preparation.os.environ, {"HF_HUB_OFFLINE": "1"}), \
             patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=api)}), \
             patch.object(preparation, "load_codetask_split", return_value=(ds, {"source_rows": 3})) as load, \
             contextlib.redirect_stdout(io.StringIO()):
            preparation.prepare_data(args)
        api.assert_not_called()
        self.assertEqual(load.call_count, 5)
        self.assertTrue(all(call.args[-1] is None for call in load.call_args_list))
        manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
        self.assertFalse(manifest["revision_resolved_online"])

    def test_failed_training_stops_before_pair_evaluation(self):
        args = self.args("--tasks", "BFP")
        failure = subprocess.CalledProcessError(1, "train")
        with patch.object(preparation, "checkpoint_kind", return_value="full"), \
             patch.object(pairwise.subprocess, "run", side_effect=[None, None, None, failure]) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                pairwise.run(args)
        self.assertEqual(launch.call_count, 4)

    def test_summary_compares_same_subsets_and_records_deltas(self):
        def result(score):
            return {"metrics": {"bleu": score}, "num_samples": 2, "sampling": {"source_indices": [1, 3]}}
        baseline = {task: {split: result(50) for split in ("validation", "test")} for task in ("CodeTrans", "BFP")}
        after = {task: {split: result(40 if task == "CodeTrans" else 70) for split in ("validation", "test")}
                 for task in ("CodeTrans", "BFP")}
        for relative, results in (("baseline", baseline), ("CodeTrans_to_BFP/eval", after)):
            directory = self.root / relative
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps({"results": results}))
        pairwise.summarize(self.root, ["BFP"])
        summary = json.loads((self.root / "pairwise_results.json").read_text())["CodeTrans_to_BFP"]
        self.assertEqual(summary["CodeTrans"]["test"]["delta_after_minus_before"]["bleu"], -10)
        self.assertEqual(summary["BFP"]["test"]["delta_after_minus_before"]["bleu"], 20)

    def test_prepared_training_uses_only_train_and_legacy_format(self):
        raw = MemoryDataset(fixture_rows()[:2])
        prepared = self.root / "data" / "BFP" / "train"
        prepared.mkdir(parents=True)
        (prepared / "sampling_manifest.json").write_text(json.dumps({
            "task": "BFP", "split": "train", "selected_rows": 2, "source_rows": 20}))
        loader = Mock(return_value=raw)
        stats = {"prompt": {"rows_over_limit": 0}, "teacher_prompt": {"rows_over_limit": 0}}
        length_check = Mock(return_value=stats)
        modules = {"datasets": types.SimpleNamespace(load_from_disk=loader),
                   "transformers": types.SimpleNamespace(AutoTokenizer=Mock())}
        argv = ["main.py", "--dataset_name", "codetask", "--codetask_task", "BFP",
                "--prepared_train", str(prepared), "--prompt_format", "legacy", "--prepare_only",
                "--num_train", "20000", "--output_dir", str(self.root / "training")]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, modules), \
             patch.object(training, "prompt_length_stats", length_check), \
             patch.object(training, "load_codetask_dataset") as hub_load, contextlib.redirect_stdout(io.StringIO()):
            training.main()
        loader.assert_called_once_with(str(prepared))
        hub_load.assert_not_called()
        dataset = length_check.call_args.args[0]
        self.assertEqual(dataset[0]["prompt"], data.student_prompt(fixture_rows()[0]["input"], "legacy"))

    def test_eval_generates_both_splits_and_saves_predictions(self):
        rows = [{"input": "QUESTION", "output": "SECRET_REFERENCE", "_source_index": 7}]
        raw = MemoryDataset(rows)
        data_dir = self.root / "data"
        for split in ("validation", "test"):
            path = data_dir / "CodeTrans" / split
            path.mkdir(parents=True)
            (path / "sampling_manifest.json").write_text(json.dumps({
                "task": "CodeTrans", "split": split, "selected_rows": 1}))
        tokenizer = Mock(side_effect=lambda text, **kwargs: {"input_ids": [ord(c) for c in text]})
        tokenizer.eos_token_id = 0
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        llm_type = Mock()
        llm_type.return_value.generate.return_value = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="GENERATED")])]
        metrics = Mock(return_value={"bleu": 25})
        modules = {"datasets": types.SimpleNamespace(load_from_disk=Mock(return_value=raw)),
                   "transformers": types.SimpleNamespace(AutoTokenizer=auto_tokenizer),
                   "vllm": types.SimpleNamespace(LLM=llm_type, SamplingParams=Mock()),
                   "evaluator.compute_metrics": types.SimpleNamespace(compute_metrics=metrics, DATASET_TO_OUTPUT_LANG={"CodeTrans": "c_sharp"})}
        args = types.SimpleNamespace(model_path="pair/final", data_dir=str(data_dir), tasks="CodeTrans",
                                    splits="validation,test", output_dir=str(self.root / "eval"),
                                    seed=1234, max_prompt_length=100, max_completion_length=10,
                                    batch_size=2, gpu_memory_utilization=0.8, prompt_format="legacy")
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            evaluation.evaluate(args)
        self.assertEqual(llm_type.return_value.generate.call_count, 2)
        for call in llm_type.return_value.generate.call_args_list:
            text = "".join(chr(i) for i in call.args[0][0]["prompt_token_ids"])
            self.assertNotIn("SECRET_REFERENCE", text)
        prediction = json.loads((self.root / "eval" / "CodeTrans" / "test" / "predictions.json").read_text())[0]
        self.assertEqual(prediction["source_index"], 7)
        self.assertEqual(prediction["prediction"], "GENERATED")
        self.assertEqual(prediction["ground-truth"], "SECRET_REFERENCE")


if __name__ == "__main__":
    unittest.main()
