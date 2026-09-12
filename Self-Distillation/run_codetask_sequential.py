"""Run one fresh process per CodeTask, carrying forward the final student weights."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from codetask_data import CODETASK_REPO, CODETASK_TASKS


def positive_counts(value, task_count):
    try:
        counts = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--num_train must be a positive integer or comma-separated integers") from exc
    if len(counts) == 1:
        counts *= task_count
    if len(counts) != task_count or any(count <= 0 for count in counts):
        raise ValueError("--num_train needs one positive count or one per task; full-dataset (-1) runs are disabled")
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--tasks", default=",".join(CODETASK_TASKS), help="Ordered comma-separated task names")
    parser.add_argument("--num_train", default="1000", help="Positive count shared by all tasks or comma-separated counts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_repo", default=CODETASK_REPO)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--output_dir", default="outputs/sdft_codetask")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_prompts_per_batch", type=int, default=32)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--dry_run", action="store_true", help="Print checkpoint chain without downloads, directories, or training")
    args = parser.parse_args(argv)
    args.tasks = [task.strip() for task in args.tasks.split(",")]
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or any(task not in CODETASK_TASKS for task in args.tasks):
        parser.error(f"--tasks must contain unique task names from {CODETASK_TASKS}")
    try:
        args.num_train = positive_counts(args.num_train, len(args.tasks))
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.max_prompt_length, args.max_completion_length, args.num_train_epochs, args.num_prompts_per_batch) <= 0:
        parser.error("Token limits, epochs, and batch size must be positive")
    if args.learning_rate <= 0 or not 0 <= args.ref_model_mixup_alpha <= 1:
        parser.error("Learning rate must be positive and teacher mixup alpha must be in [0, 1]")
    return args


def build_stages(args):
    root = Path(args.output_dir).resolve()
    source_model = args.model_name
    stages = []
    for index, (task, count) in enumerate(zip(args.tasks, args.num_train), start=1):
        stage_dir = root / f"{index:02d}_{task}"
        command = [
            sys.executable, str(Path(__file__).resolve().with_name("main.py")),
            "--dataset_name", "codetask", "--codetask_task", task,
            "--model_name", source_model, "--output_dir", str(stage_dir),
            "--num_train", str(count), "--seed", str(args.seed),
            "--dataset_repo", args.dataset_repo,
            "--learning_rate", str(args.learning_rate),
            "--num_train_epochs", str(args.num_train_epochs),
            "--num_prompts_per_batch", str(args.num_prompts_per_batch),
            "--ref_model_mixup_alpha", str(args.ref_model_mixup_alpha),
            "--max_prompt_length", str(args.max_prompt_length),
            "--max_completion_length", str(args.max_completion_length),
            "--num_loss_tokens_to_skip", "0", "--report_to", args.report_to,
        ]
        if args.dataset_revision is not None:
            command.extend(["--dataset_revision", args.dataset_revision])
        final_dir = stage_dir / "final"
        stages.append({"task": task, "num_train": count, "source_model": source_model,
                       "final_dir": str(final_dir), "command": command})
        source_model = str(final_dir)
    return stages


def validate_checkpoint(final_dir):
    final_dir = Path(final_dir)
    required = ("config.json", "tokenizer_config.json", "training_complete.json")
    if any(not (final_dir / name).is_file() for name in required):
        raise RuntimeError(f"Incomplete final checkpoint: {final_dir}")
    if not any((final_dir / name).is_file() for name in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )):
        raise RuntimeError(f"No full model weights saved in {final_dir}")


def run(args):
    stages = build_stages(args)
    for stage in stages:
        print(f"\n{stage['task']}: {stage['num_train']} training examples", flush=True)
        print(shlex.join(stage["command"]), flush=True)
    if args.dry_run:
        return

    root = Path(args.output_dir).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {root}. Choose another --output_dir.")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": vars(args),
        "teacher_at_task_boundary": "reset_from_previous_final_student",
        "optimizer_at_task_boundary": "fresh",
        "stages": stages,
    }
    (root / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for stage in stages:
        print(f"\nTraining {stage['task']} from {stage['source_model']}", flush=True)
        # A separate process releases the previous trainer/vLLM GPU allocations.
        subprocess.run(stage["command"], check=True)
        validate_checkpoint(stage["final_dir"])
    print(f"\nAll tasks completed. Final student: {stages[-1]['final_dir']}", flush=True)


if __name__ == "__main__":
    run(parse_args())
