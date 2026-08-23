from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .config import ExperimentConfig, stable_seed


PROMPT_FORMAT_VERSION = "codetask-input-output-v1"


def resolved_sample_indices(length: int, requested: int, seed: int) -> list[int]:
    if length < 0 or requested < 0:
        raise ValueError("Dataset length and requested count must be nonnegative")
    count = min(requested, length)
    if count == length:
        return list(range(length))
    generator = np.random.default_rng(seed)
    return generator.permutation(length)[:count].tolist()


def load_train_dataset(config: ExperimentConfig, task: str) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_repo,
        data_files={"train": f"{task}/train-*.parquet"},
        split="train",
        cache_dir=str(config.cache_root),
    )
    columns = set(dataset.column_names)
    if {"input", "output"}.issubset(columns):
        keep = {"input", "output"}
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep])
        dataset = dataset.rename_columns({"input": "prompt", "output": "answer"})
    elif not {"prompt", "answer"}.issubset(columns):
        raise ValueError(f"Unexpected CodeTask columns for {task}: {sorted(columns)}")
    return dataset


def dataset_fingerprint(dataset: Any) -> str:
    value = getattr(dataset, "_fingerprint", None)
    if value:
        return str(value)
    return hashlib.sha256(f"rows={len(dataset)};columns={getattr(dataset, 'column_names', [])}".encode()).hexdigest()


def select_dataset(dataset: Any, requested: int, seed: int) -> tuple[Any, list[int]]:
    indices = resolved_sample_indices(len(dataset), requested, seed)
    return dataset.select(indices), indices


@dataclass(slots=True)
class CalibrationPool:
    dataset: Any
    source_indices: list[int]
    support_positions: tuple[int, ...]
    verification_positions: tuple[int, ...]
    gca_positions: tuple[int, ...]
    remainder_positions: tuple[int, ...]
    fingerprint: str

    def rows(self, positions: Sequence[int]) -> list[dict[str, str]]:
        return [self.dataset[int(position)] for position in positions]

    @property
    def all_positions(self) -> tuple[int, ...]:
        return tuple(range(len(self.dataset)))

    def aggregate_summary(self) -> dict[str, Any]:
        return {
            "pool_count": len(self.dataset),
            "support_count": len(self.support_positions),
            "verification_count": len(self.verification_positions),
            "gca_count": len(self.gca_positions),
            "gmm_only_count": len(self.remainder_positions),
            "dataset_fingerprint": self.fingerprint,
        }


def _allocate_views(total: int, support: int, verification: int, remainder: int) -> tuple[int, int, int]:
    requested_total = support + verification + remainder
    if total >= requested_total:
        return support, verification, remainder
    if total == 0:
        return 0, 0, 0
    weights = np.asarray([support, verification, remainder], dtype=np.float64)
    if remainder == 0 and total < support + verification:
        weights = np.asarray([3.0, 3.0, 2.0])
    if weights.sum() <= 0:
        weights = np.asarray([3.0, 3.0, 2.0])
    quotas = total * weights / weights.sum()
    counts = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def build_calibration_pool(dataset: Any, config: ExperimentConfig, task: str) -> CalibrationPool:
    seed = stable_seed(config.data_seed, task.casefold(), "calibration")
    subset, source_indices = select_dataset(dataset, config.calibration_size, seed)
    expected_remainder = max(0, config.calibration_size - config.support_size - config.verification_size)
    support_count, verification_count, remainder_count = _allocate_views(
        len(subset), config.support_size, config.verification_size, expected_remainder
    )
    support_positions = tuple(range(support_count))
    verification_start = support_count
    verification_positions = tuple(range(verification_start, verification_start + verification_count))
    remainder_start = verification_start + verification_count
    remainder_positions = tuple(range(remainder_start, remainder_start + remainder_count))
    gca_positions = support_positions[: min(config.gca_size, len(support_positions))]
    return CalibrationPool(
        subset,
        source_indices,
        support_positions,
        verification_positions,
        gca_positions,
        remainder_positions,
        dataset_fingerprint(dataset),
    )


def format_prompt(instruction: str) -> str:
    return f"input: {instruction}\noutput: "


def _tokenize_no_special(tokenizer: Any, text: str, max_length: int) -> dict[str, list[int]]:
    result = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        padding=False,
        return_tensors=None,
    )
    return {"input_ids": list(result["input_ids"]), "attention_mask": list(result["attention_mask"])}


def collate_prompts(rows: Sequence[dict[str, str]], tokenizer: Any, max_prompt_len: int) -> dict[str, Any]:
    import torch

    encoded = [_tokenize_no_special(tokenizer, format_prompt(row["prompt"]), max_prompt_len) for row in rows]
    if not encoded:
        raise ValueError("Cannot collate an empty prompt batch")
    maximum = max(len(item["input_ids"]) for item in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = [], []
    for item in encoded:
        padding = maximum - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def collate_supervised(
    rows: Sequence[dict[str, str]],
    tokenizer: Any,
    max_prompt_len: int,
    max_ans_len: int,
) -> dict[str, Any]:
    import torch

    encoded: list[dict[str, list[int]]] = []
    for row in rows:
        prompt = _tokenize_no_special(tokenizer, format_prompt(row["prompt"]), max_prompt_len)
        answer = _tokenize_no_special(tokenizer, row["answer"], max_ans_len)
        if len(answer["input_ids"]) < max_ans_len and tokenizer.eos_token_id is not None:
            answer["input_ids"].append(tokenizer.eos_token_id)
            answer["attention_mask"].append(1)
        prompt_len = len(prompt["input_ids"])
        encoded.append({
            "input_ids": prompt["input_ids"] + answer["input_ids"],
            "attention_mask": prompt["attention_mask"] + answer["attention_mask"],
            "labels": [-100] * prompt_len + list(answer["input_ids"]),
        })
    if not encoded:
        raise ValueError("Cannot collate an empty supervised batch")
    maximum = max(len(item["input_ids"]) for item in encoded)
    pad_id = tokenizer.pad_token_id
    for item in encoded:
        padding = maximum - len(item["input_ids"])
        item["input_ids"].extend([pad_id] * padding)
        item["attention_mask"].extend([0] * padding)
        item["labels"].extend([-100] * padding)
    batch = {key: torch.tensor([item[key] for item in encoded], dtype=torch.long) for key in ("input_ids", "attention_mask", "labels")}
    if supervised_token_count(batch["labels"]) <= 0:
        raise ValueError("Supervised batch has no shifted answer tokens")
    return batch


def supervised_token_count(labels: Any) -> int:
    return int((labels[..., 1:] != -100).sum().item())


def causal_loss_sum(logits: Any, labels: Any) -> tuple[Any, int]:
    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    count = int((shift_labels != -100).sum().item())
    if count <= 0:
        raise ValueError("No supervised shifted answer tokens")
    loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count


def iter_chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
