from __future__ import annotations

from typing import Any, Sequence

from .backbone import move_batch
from .data import causal_loss_sum, collate_supervised, iter_chunks


def find_base_weight_parameters(model: Any, modules: Sequence[str]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    resolved: dict[str, Any] = {}
    for module in modules:
        candidates = (f"{module}.weight", f"model.{module}.weight")
        matches = [name for name in candidates if name in parameters]
        if len(matches) != 1:
            suffix = f"{module}.weight"
            suffix_matches = [name for name in parameters if name.endswith(suffix)]
            if len(suffix_matches) == 1:
                matches = suffix_matches
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map LoRA module {module!r} to a base weight")
        resolved[module] = parameters[matches[0]]
    return resolved


def collect_target_gradients(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    modules: Sequence[str],
    *,
    batch_size: int,
    max_prompt_len: int,
    max_ans_len: int,
    device: Any,
) -> tuple[dict[str, Any], int]:
    weights = find_base_weight_parameters(model, modules)
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.eval()
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in weights.values():
        parameter.requires_grad_(True)
    total_tokens = 0
    gradient_numerators = {
        module: None for module in weights
    }
    try:
        for chunk in iter_chunks(rows, batch_size):
            model.zero_grad(set_to_none=True)
            batch = move_batch(collate_supervised(chunk, tokenizer, max_prompt_len, max_ans_len), device)
            output = model(**batch, use_cache=False, return_dict=True)
            loss_sum, tokens = causal_loss_sum(output.logits, batch["labels"])
            loss_sum.backward()
            total_tokens += tokens
            for module, parameter in weights.items():
                if parameter.grad is None:
                    raise RuntimeError(f"No target gradient was produced for {module}")
                contribution = parameter.grad.detach().float().cpu()
                if gradient_numerators[module] is None:
                    gradient_numerators[module] = contribution
                else:
                    gradient_numerators[module].add_(contribution)
        if total_tokens <= 0:
            raise ValueError("GCA view contains no supervised answer tokens")
        gradients: dict[str, Any] = {}
        for module, numerator in gradient_numerators.items():
            if numerator is None:
                raise RuntimeError(f"No accumulated target gradient was produced for {module}")
            gradients[module] = numerator / float(total_tokens)
        return gradients, total_tokens
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
