from string import Template
import argparse
import json
from math import gcd
from pathlib import Path

from codetask_data import CODETASK_REPO, CODETASK_TASKS, load_codetask_dataset, prompt_length_stats

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "codetask"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--codetask_task", choices=CODETASK_TASKS)
    parser.add_argument("--dataset_repo", default=CODETASK_REPO)
    parser.add_argument("--dataset_revision", default=None, help="HF dataset revision; pin a commit for reproducibility")
    parser.add_argument("--prepared_train", help="Saved raw CodeTask train subset produced by the pairwise preparation script")
    parser.add_argument("--prompt_format", choices=["chat", "legacy"], default="chat")
    parser.add_argument("--num_train", type=int, default=1000, help="CodeTask training subset size; -1 explicitly uses all rows")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=None,
                        help="Defaults to 0 for CodeTask, 3 for the original tasks")
    parser.add_argument("--report_to", default="wandb", help="Use none to disable external logging")
    parser.add_argument("--prepare_only", action="store_true",
                        help="CodeTask: load subset and tokenizer, save manifest and lengths, then exit before loading models")
    args = parser.parse_args()
    if args.dataset_name == "codetask" and not args.codetask_task:
        parser.error("--codetask_task is required for --dataset_name codetask")
    if args.prepare_only and args.dataset_name != "codetask":
        parser.error("--prepare_only requires --dataset_name codetask")
    if args.num_train != -1 and args.num_train <= 0:
        parser.error("--num_train must be positive or -1")
    if min(args.max_prompt_length, args.max_completion_length, args.num_prompts_per_batch, args.num_train_epochs) <= 0:
        parser.error("Token limits, batch size, and epochs must be positive")
    if args.num_loss_tokens_to_skip is None:
        args.num_loss_tokens_to_skip = 0 if args.dataset_name == "codetask" else 3
    if not 0 <= args.num_loss_tokens_to_skip < args.max_completion_length:
        parser.error("--num_loss_tokens_to_skip must be nonnegative and less than --max_completion_length")
    return args

def load_tooluse_dataset(seed=42):
    """Load and prepare tooluse dataset with formatted prompts."""
    from datasets import load_from_disk
    train_dir = Path(__file__).resolve().parent / 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(str(train_dir))

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42):
    """Load and prepare science dataset with formatted prompts."""
    from datasets import load_from_disk
    path = Path(__file__).resolve().parent / 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(str(path))

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    manifest = None
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "codetask":
        if args.prepared_train:
            from datasets import load_from_disk
            from functools import partial
            from codetask_data import format_codetask_example
            prepared = Path(args.prepared_train)
            manifest = json.loads((prepared / "sampling_manifest.json").read_text(encoding="utf-8"))
            if manifest["task"] != args.codetask_task or manifest["split"] != "train":
                raise ValueError("Prepared subset task/split does not match the requested training task")
            raw = load_from_disk(str(prepared))
            if len(raw) != manifest["selected_rows"] or (args.num_train != -1 and len(raw) > args.num_train):
                raise ValueError("Prepared subset count does not match its manifest or exceeds --num_train")
            dataset = raw.map(partial(format_codetask_example, prompt_format=args.prompt_format),
                              remove_columns=raw.column_names)
            manifest["prompt_format"] = args.prompt_format
            manifest["teacher_template"] = "codetask_output_only_v1"
        else:
            kwargs = {"prompt_format": args.prompt_format} if args.prompt_format != "chat" else {}
            dataset, manifest = load_codetask_dataset(
                args.codetask_task, args.num_train, args.seed,
                args.dataset_repo, args.dataset_revision, **kwargs,
            )
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        manifest["prompt_lengths"] = prompt_length_stats(dataset, tokenizer, args.max_prompt_length)
        manifest["max_prompt_length"] = args.max_prompt_length
        (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"CodeTask {args.codetask_task}: selected {len(dataset)} / {manifest['source_rows']} train rows")
        print(f"Prompt lengths: {json.dumps(manifest['prompt_lengths'])}")
        if any(stats["rows_over_limit"] for stats in manifest["prompt_lengths"].values()):
            print("WARNING: prompts exceed max_prompt_length and will be left-truncated. "
                  "Increase --max_prompt_length to preserve the task input and teacher reference.", flush=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.prepare_only:
        return

    from distil_trainer import DistilTrainer
    from distil_config import DistilConfig
    from transformers import AutoModelForCausalLM
    import torch

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        # RepeatSampler groups complete generation batches. Choose a divisor
        # of the sampled size so small/non-multiple subsets retain every row.
        steps_per_generation = (gcd(len(dataset), args.num_prompts_per_batch)
                                if args.dataset_name == "codetask" else None),
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        num_iterations = 1,
        num_generations = 1,
        save_steps = 100,
        max_grad_norm = 1,
        report_to = args.report_to,
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = args.num_loss_tokens_to_skip,
    )
    if args.dataset_name == "codetask" and len(dataset) % config.generation_batch_size:
        raise ValueError("CodeTask subset size must be divisible by the global generation batch. "
                         "Use the single-GPU sequential script or adjust the subset/batch size.")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, local_files_only=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, local_files_only=True)
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        # Written only after the final student and tokenizer have been saved.
        (final_dir / "training_complete.json").write_text(json.dumps({
            "source_model": args.model_name,
            "dataset": args.dataset_name,
            "task": args.codetask_task,
            "global_step": trainer.state.global_step,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
