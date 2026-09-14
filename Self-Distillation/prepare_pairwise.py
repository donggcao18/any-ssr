"""Server-side checkpoint export and frozen CodeTask subsets for pairwise runs."""

import argparse
import json
import os
from pathlib import Path

from codetask_data import CODETASK_REPO, CODETASK_TASKS, load_codetask_split
from local_model import resolve_local_model


def checkpoint_kind(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Source checkpoint is not available on this machine: {path}")
    if (path / "adapter_config.json").is_file():
        if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"Adapter weights are missing from {path}")
        return "adapter"
    if (path / "config.json").is_file() and any((path / name).is_file() for name in (
        "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"
    )):
        return "full"
    raise ValueError(f"Expected a PEFT adapter or Hugging Face full checkpoint: {path}")


def export_checkpoint(source, output, base_model=None):
    kind = checkpoint_kind(source)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Checkpoint output must be empty: {output}")
    if kind != "adapter":
        raise ValueError("LoRA-only export requires the original CodeTrans adapter checkpoint, not merged weights")
    import torch
    from transformers import AutoTokenizer
    from lora_runtime import load_lora_model, save_lora
    adapter_config = json.loads((Path(source) / "adapter_config.json").read_text())
    base = resolve_local_model(base_model or adapter_config["base_model_name_or_path"])
    tokenizer_source = source if (Path(source) / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    model = load_lora_model(source, tokenizer, torch.float32, trainable=False, base_override=base)
    save_lora(model, output)
    tokenizer.save_pretrained(str(output))
    (output / "source_manifest.json").write_text(json.dumps({
        "source_checkpoint": source, "kind": "adapter", "base_model": base,
        "tokenizer_source": tokenizer_source, "adapter_merged": False,
        "training_mode": "lora_only",
    }, indent=2), encoding="utf-8")


def prepare_data(args):
    offline = any(os.environ.get(name, "").upper() in ("1", "TRUE", "YES", "ON")
                  for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))
    if offline:
        # Preserve the original request so datasets can reuse its Arrow cache.
        revision = args.dataset_revision
        print("Offline mode: loading CodeTask from cache; skipping Hub revision lookup.", flush=True)
    else:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(args.dataset_repo, revision=args.dataset_revision).sha
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tasks = [args.source_task] + args.tasks.split(",")
    manifests = {}
    for task in dict.fromkeys(tasks):
        splits = ("validation", "test") if task == args.source_task else ("train", "validation", "test")
        for split in splits:
            cap = {"train": args.num_train, "validation": args.num_validation, "test": args.num_test}[split]
            seed = args.seed if split == "train" else args.eval_seed
            selected, manifest = load_codetask_split(task, split, cap, seed, args.dataset_repo, revision)
            path = root / task / split
            if path.exists():
                raise ValueError(f"Prepared subset already exists: {path}")
            selected.save_to_disk(str(path))
            (path / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests[f"{task}/{split}"] = manifest
            print(f"{task}/{split}: {len(selected)} / {manifest['source_rows']} rows", flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "revision": revision, "revision_resolved_online": not offline, "subsets": manifests,
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--source_checkpoint", required=True)
    checkpoint.add_argument("--output_dir", required=True)
    checkpoint.add_argument("--base_model")
    data = sub.add_parser("data")
    data.add_argument("--source_task", choices=CODETASK_TASKS, default="CodeTrans")
    data.add_argument("--tasks", required=True)
    data.add_argument("--output_dir", required=True)
    data.add_argument("--dataset_repo", default=CODETASK_REPO)
    data.add_argument("--dataset_revision")
    data.add_argument("--num_train", type=int, default=20000)
    data.add_argument("--num_validation", type=int, default=1000)
    data.add_argument("--num_test", type=int, default=2000)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--eval_seed", type=int, default=1234)
    args = parser.parse_args()
    if args.action == "checkpoint":
        export_checkpoint(args.source_checkpoint, args.output_dir, args.base_model)
    else:
        if min(args.num_train, args.num_validation, args.num_test) <= 0:
            parser.error("All sample caps must be positive")
        prepare_data(args)


if __name__ == "__main__":
    main()
