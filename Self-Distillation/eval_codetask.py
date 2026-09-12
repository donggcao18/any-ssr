"""Generate and score frozen CodeTask validation/test subsets without demonstrations."""

import argparse
import json
from pathlib import Path
import sys

from codetask_data import CODETASK_TASKS, student_prompt


def encode_prompt(tokenizer, instruction, prompt_format, max_prompt_length):
    prompt = student_prompt(instruction, prompt_format)
    text = prompt if isinstance(prompt, str) else tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return ids[-max_prompt_length:], len(ids) > max_prompt_length


def evaluate(args):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Import before generation so missing metric dependencies fail immediately.
    from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    llm = LLM(model=args.model_path, dtype="bfloat16", seed=args.seed,
              max_model_len=args.max_prompt_length + args.max_completion_length,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_completion_length,
                              stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None)
    summary = {"config": vars(args), "results": {}}
    for task in args.tasks.split(","):
        if task not in CODETASK_TASKS:
            raise ValueError(f"Unknown task: {task}")
        summary["results"][task] = {}
        for split in args.splits.split(","):
            if split not in ("validation", "test"):
                raise ValueError("Evaluation only accepts validation/test splits")
            path = Path(args.data_dir) / task / split
            manifest = json.loads((path / "sampling_manifest.json").read_text(encoding="utf-8"))
            dataset = load_from_disk(str(path))
            if manifest["task"] != task or manifest["split"] != split or len(dataset) != manifest["selected_rows"]:
                raise ValueError(f"Subset does not match its sampling manifest: {path}")
            rows, truncated = [], 0
            for offset in range(0, len(dataset), args.batch_size):
                batch = [dataset[index] for index in range(offset, min(offset + args.batch_size, len(dataset)))]
                encoded = [encode_prompt(tokenizer, row["input"], args.prompt_format, args.max_prompt_length)
                           for row in batch]
                truncated += sum(was_truncated for _, was_truncated in encoded)
                outputs = llm.generate([{"prompt_token_ids": ids} for ids, _ in encoded], sampling)
                if len(outputs) != len(batch):
                    raise RuntimeError("Generation count does not match evaluation batch size")
                rows.extend({"source_index": row["_source_index"], "source": row["input"],
                             "ground-truth": row["output"], "prediction": output.outputs[0].text}
                            for row, output in zip(batch, outputs))
            destination = output_dir / task / split
            destination.mkdir(parents=True, exist_ok=True)
            # Save raw generations even if metric computation subsequently fails.
            (destination / "predictions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            metrics = compute_metrics([row["prediction"] for row in rows], [row["ground-truth"] for row in rows],
                                      calc_codebleu=task not in ("CodeSearchNet", "TheVault_Csharp"),
                                      language=DATASET_TO_OUTPUT_LANG[task])
            result = {"metrics": metrics, "num_samples": len(rows), "truncated_prompts": truncated,
                      "sampling": manifest, "model_path": args.model_path,
                      "prompt_format": args.prompt_format, "temperature": 0.0}
            (destination / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary["results"][task][split] = result
            print(f"{task}/{split}: {metrics} ({len(rows)} samples)", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_format", choices=["chat", "legacy"], default="legacy")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if min(args.max_prompt_length, args.max_completion_length, args.batch_size) <= 0:
        parser.error("Token limits and batch size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
