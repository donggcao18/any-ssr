"""Independent CodeTrans SFT -> target SDFT experiments, with before/after evaluation."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from codetask_data import CODETASK_REPO, CODETASK_TASKS
from run_codetask_sequential import validate_checkpoint

DEFAULT_SOURCE = "/research/cbim/vast/qt60/any-ssr/output/CodeTrans/0"
DEFAULT_TARGETS = CODETASK_TASKS[CODETASK_TASKS.index("CodeTrans") + 1:]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--base_model", help="Override adapter base only if its recorded path is unavailable on the server")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--output_dir", default="outputs/sdft_pairwise_codetrans")
    parser.add_argument("--num_train", type=int, default=20000)
    parser.add_argument("--num_validation", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--dataset_repo", default=CODETASK_REPO)
    parser.add_argument("--dataset_revision")
    parser.add_argument("--prompt_format", choices=["legacy", "chat"], default="legacy")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_prompts_per_batch", "--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", help="Existing run directory with completed baseline; reuse its saved configuration")
    parser.add_argument("--resume_runtime_settings", action="store_true",
                        help="Use current microbatch/accumulation/vLLM memory settings when resuming")
    parser.add_argument("--restart_incomplete", action="store_true",
                        help="Archive unfinished training and restart that task from the source model")
    args = parser.parse_args(argv)
    args.tasks = [task.strip() for task in args.tasks.split(",")]
    if len(set(args.tasks)) != len(args.tasks) or any(task not in CODETASK_TASKS or task == "CodeTrans" for task in args.tasks):
        parser.error("--tasks must be unique CodeTask names other than CodeTrans")
    for name, limit in (("num_train", 20000), ("num_validation", 1000), ("num_test", 2000)):
        if not 0 < getattr(args, name) <= limit:
            parser.error(f"--{name} must be between 1 and {limit}")
    if min(args.num_train_epochs, args.num_prompts_per_batch, args.max_prompt_length,
           args.max_completion_length, args.eval_batch_size, args.num_gpus,
           args.per_device_train_batch_size, args.save_steps) <= 0:
        parser.error("Epochs, batch sizes, and token limits must be positive")
    if args.learning_rate <= 0 or not 0 <= args.ref_model_mixup_alpha <= 1:
        parser.error("Learning rate must be positive and EMA alpha must be in [0, 1]")
    if not 0 < args.vllm_gpu_memory_utilization < 1 or not 0 <= args.warmup_ratio <= 1:
        parser.error("vLLM memory fraction must be in (0, 1); warmup ratio must be in [0, 1]")
    return args


def build_plan(args):
    scripts = Path(__file__).resolve().parent
    root = Path(args.output_dir).resolve()
    source = str(root / "source_model")
    data = str(root / "data")

    def command(script, *options):
        return [sys.executable, str(scripts / script), *map(str, options)]

    def eval_command(model, tasks, output):
        return command("eval_codetask.py", "--model_path", model, "--data_dir", data,
                       "--tasks", ",".join(tasks), "--output_dir", output,
                       "--prompt_format", args.prompt_format, "--seed", args.eval_seed,
                       "--max_prompt_length", args.max_prompt_length,
                       "--max_completion_length", args.max_completion_length,
                       "--batch_size", args.eval_batch_size)

    export = command("prepare_pairwise.py", "checkpoint", "--source_checkpoint", args.source_checkpoint,
                     "--output_dir", source)
    if args.base_model:
        export += ["--base_model", args.base_model]
    prepare = command("prepare_pairwise.py", "data", "--tasks", ",".join(args.tasks),
                      "--output_dir", data, "--num_train", args.num_train,
                      "--num_validation", args.num_validation, "--num_test", args.num_test,
                      "--seed", args.seed, "--eval_seed", args.eval_seed,
                      "--dataset_repo", args.dataset_repo)
    if args.dataset_revision:
        prepare += ["--dataset_revision", args.dataset_revision]
    jobs = [{"name": "export_source", "command": export}, {"name": "freeze_subsets", "command": prepare},
            {"name": "baseline", "command": eval_command(source, ["CodeTrans"] + args.tasks, root / "baseline")}]
    for task in args.tasks:
        pair = root / f"CodeTrans_to_{task}"
        train = command("main.py", "--dataset_name", "codetask", "--codetask_task", task,
                        "--model_name", source, "--prepared_train", Path(data) / task / "train",
                        "--output_dir", pair / "train", "--num_train", args.num_train,
                        "--seed", args.seed, "--prompt_format", args.prompt_format,
                        "--learning_rate", args.learning_rate, "--num_train_epochs", args.num_train_epochs,
                        "--num_prompts_per_batch", args.num_prompts_per_batch,
                        "--ref_model_mixup_alpha", args.ref_model_mixup_alpha,
                        "--max_prompt_length", args.max_prompt_length,
                        "--max_completion_length", args.max_completion_length,
                        "--num_loss_tokens_to_skip", 0, "--report_to", args.report_to)
        train += ["--per_device_train_batch_size", str(args.per_device_train_batch_size),
                  "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
                  "--warmup_ratio", str(args.warmup_ratio), "--save_steps", str(args.save_steps)]
        if args.num_gpus > 1:
            train = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                     "--nnodes=1", f"--nproc_per_node={args.num_gpus}", *train[1:]]
        final = str(pair / "train" / "final")
        jobs.append({"name": f"train_{task}", "command": train, "checkpoint": final})
        jobs.append({"name": f"evaluate_{task}", "command": eval_command(final, ["CodeTrans", task], pair / "eval")})
    return jobs


def summarize(root, tasks):
    root = Path(root)
    baseline = json.loads((root / "baseline" / "summary.json").read_text(encoding="utf-8"))["results"]
    result = {}
    for task in tasks:
        after = json.loads((root / f"CodeTrans_to_{task}" / "eval" / "summary.json").read_text(encoding="utf-8"))["results"]
        pair = {}
        for evaluated in ("CodeTrans", task):
            pair[evaluated] = {}
            for split in ("validation", "test"):
                before_result, after_result = baseline[evaluated][split], after[evaluated][split]
                before, current = before_result["metrics"], after_result["metrics"]
                if before_result["sampling"] != after_result["sampling"]:
                    raise ValueError(f"Baseline and pair evaluated different subsets: {task}/{evaluated}/{split}")
                pair[evaluated][split] = {
                    "num_samples": after_result["num_samples"], "before": before, "after": current,
                    "delta_after_minus_before": {key: round(current[key] - before[key], 4) for key in current},
                }
        result[f"CodeTrans_to_{task}"] = pair
    (root / "pairwise_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def resume_jobs(args):
    """Resume at stage boundaries; never overwrite unfinished training."""
    root = Path(args.resume).resolve()
    saved = json.loads((root / "pairwise_manifest.json").read_text(encoding="utf-8"))["config"]
    runtime = {key: getattr(args, key) for key in (
        "per_device_train_batch_size", "num_prompts_per_batch", "vllm_gpu_memory_utilization")}
    for key, value in saved.items():
        if key not in ("resume", "dry_run", "output_dir", "resume_runtime_settings", "restart_incomplete") and hasattr(args, key):
            setattr(args, key, value)
    if args.resume_runtime_settings:
        for key, value in runtime.items():
            setattr(args, key, value)
    args.output_dir = str(root)
    from prepare_pairwise import checkpoint_kind
    if checkpoint_kind(root / "source_model") != "full":
        raise ValueError("Resume requires the exported full source model")
    if not (root / "source_model" / "tokenizer_config.json").is_file():
        raise ValueError("Exported tokenizer is missing")
    baseline = json.loads((root / "baseline" / "summary.json").read_text(encoding="utf-8"))
    for task in ["CodeTrans", *args.tasks]:
        splits = ("validation", "test") if task == "CodeTrans" else ("train", "validation", "test")
        for split in splits:
            path = root / "data" / task / split
            sampling = json.loads((path / "sampling_manifest.json").read_text(encoding="utf-8"))
            if not (path / "state.json").is_file():
                raise ValueError(f"Saved dataset is missing: {path}")
            if split != "train":
                result = baseline["results"][task][split]
                if result["sampling"] != sampling or result["num_samples"] != sampling["selected_rows"]:
                    raise ValueError(f"Baseline/subset mismatch: {task}/{split}")
    jobs = []
    for job in build_plan(args)[3:]:
        if "checkpoint" in job:
            final = Path(job["checkpoint"])
            if (final / "training_complete.json").is_file():
                validate_checkpoint(str(final))
                print(f"Skipping completed {job['name']}", flush=True)
                continue
            if final.parent.exists() and any(final.parent.iterdir()):
                if args.restart_incomplete:
                    job["archive_train_dir"] = str(final.parent)
                else:
                    raise ValueError(f"Unfinished training exists in {final.parent}. "
                                 "This resume mode resumes completed stages, not optimizer steps; "
                                 "move that unfinished train directory aside before restarting this task.")
        if job["name"].startswith("evaluate_"):
            task = job["name"][len("evaluate_"):]
            summary_path = root / f"CodeTrans_to_{task}" / "eval" / "summary.json"
            if summary_path.is_file():
                results = json.loads(summary_path.read_text(encoding="utf-8"))["results"]
                for evaluated in ("CodeTrans", task):
                    for split in ("validation", "test"):
                        if results[evaluated][split]["sampling"] != baseline["results"][evaluated][split]["sampling"]:
                            raise ValueError(f"Completed evaluation subset mismatch: {task}/{evaluated}/{split}")
                validate_checkpoint(str(root / f"CodeTrans_to_{task}" / "train" / "final"))
                print(f"Skipping completed {job['name']}", flush=True)
                continue
        jobs.append(job)
    print(f"Reusing source model, frozen data, and completed baseline from {root}", flush=True)
    return jobs


def run(args):
    jobs = resume_jobs(args) if args.resume else build_plan(args)
    for job in jobs:
        print(f"\n[{job['name']}]\n{shlex.join(job['command'])}", flush=True)
    if args.dry_run:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run the pairwise script with plain bash/python; it launches distributed training itself")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(args.num_gpus)]
    if len(devices) < args.num_gpus:
        raise ValueError("CUDA_VISIBLE_DEVICES contains fewer GPUs than --num_gpus")
    # This check runs on the server. A local dry run never accesses the source.
    from prepare_pairwise import checkpoint_kind
    if not args.resume:
        checkpoint_kind(args.source_checkpoint)
    root = Path(args.output_dir).resolve()
    if not args.resume and root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"config": vars(args), "objective": "SDFT on B initialized independently from SFT(A)",
                "validation_policy": "post-training generation; final checkpoint, no model selection",
                "jobs": jobs}
    if not args.resume:
        (root / "pairwise_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for job in jobs:
        print(f"\nRunning {job['name']}", flush=True)
        if "archive_train_dir" in job:
            old = Path(job["archive_train_dir"]).resolve()
            if not old.is_relative_to(root) or old.name != "train":
                raise ValueError(f"Unexpected archive path: {old}")
            archived = old.with_name(f"train_interrupted_{time.time_ns()}")
            old.rename(archived)
            print(f"Preserved interrupted training in {archived}", flush=True)
        if args.resume:
            (root / "resume_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        env = os.environ.copy()
        count = args.num_gpus if job["name"].startswith("train_") else 1
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:count])
        subprocess.run(job["command"], check=True, env=env)
        if "checkpoint" in job:
            validate_checkpoint(job["checkpoint"])
    summarize(root, args.tasks)
    print(f"\nPairwise results: {root / 'pairwise_results.json'}", flush=True)


if __name__ == "__main__":
    run(parse_args())
