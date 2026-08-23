from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig


@dataclass(slots=True)
class BackboneBundle:
    model: Any
    tokenizer: Any
    device: Any
    dtype_name: str
    resolved_model_revision: str | None


def _resolve_device(config: ExperimentConfig) -> Any:
    import torch

    if config.device != "auto":
        return torch.device(config.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: ExperimentConfig, device: Any) -> Any:
    import torch

    if config.dtype == "float32":
        return torch.float32
    if config.dtype == "float16":
        return torch.float16
    if config.dtype == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_backbone(config: ExperimentConfig) -> BackboneBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(config)
    dtype = _resolve_dtype(config, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.tokenizer_revision or config.model_revision,
        cache_dir=str(config.cache_root),
        local_files_only=config.local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Existing CodeTask training tokenizes with the repository tokenizer's
    # left-truncation policy, while its collator pads the resulting sequences
    # on the right. Keep those two choices distinct.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        cache_dir=str(config.cache_root),
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    return BackboneBundle(model, tokenizer, device, str(dtype).replace("torch.", ""), resolved_revision)


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
