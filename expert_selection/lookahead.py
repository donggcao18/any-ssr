from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapters import adapter_tensor_hash
from .backbone import move_batch
from .data import causal_loss_sum, collate_supervised, iter_chunks
from .records import TargetContext


@dataclass(slots=True)
class Trajectory:
    nll: dict[int, float]
    supervised_tokens: dict[int, int]
    elapsed_seconds: float


@dataclass(slots=True)
class LookaheadArtifacts:
    fresh: Trajectory
    sources: dict[str, Trajectory]
    fresh_lora_seed: int
    fresh_adapter_hash: str
    lookahead_seed: int
    checkpoints: tuple[int, ...]
    schedule_microbatches: int
    step0_equivalence_error: float | None
    settings: dict[str, Any]


def resolved_checkpoints(methods: tuple[str, ...], slu_checkpoints: tuple[int, ...]) -> tuple[int, ...]:
    checkpoints = {0}
    if "oia" in methods:
        checkpoints.add(1)
    if "slu" in methods:
        checkpoints.update(slu_checkpoints)
    return tuple(sorted(checkpoints))


def build_support_schedule(
    support_count: int,
    *,
    batch_size: int,
    optimizer_steps: int,
    gradient_accumulation: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if support_count <= 0:
        raise ValueError("Lookahead support set is empty")
    required = optimizer_steps * gradient_accumulation
    generator = np.random.default_rng(seed)
    schedule: list[tuple[int, ...]] = []
    epoch_batches: list[tuple[int, ...]] = []
    while len(schedule) < required:
        if not epoch_batches:
            order = generator.permutation(support_count).tolist()
            epoch_batches = [tuple(order[start : start + batch_size]) for start in range(0, support_count, batch_size)]
        schedule.append(epoch_batches.pop(0))
    return schedule


def _active_parameters(model: Any, adapter_name: str) -> list[Any]:
    active: list[Any] = []
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        enabled = "lora_" in name and marker in name
        parameter.requires_grad_(enabled)
        if enabled:
            active.append(parameter)
    if not active:
        raise RuntimeError(f"No trainable LoRA tensors found for adapter {adapter_name!r}")
    return active


def _adapter_state(model: Any, adapter_name: str) -> dict[str, Any]:
    marker = f".{adapter_name}."
    state: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "lora_" in name and marker in name:
            state[name.replace(marker, ".__adapter__.")] = parameter.detach().cpu().clone()
    return state


def _set_rng(seed: int, device: Any) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def evaluate_verification_nll(
    model: Any,
    rows: list[dict[str, str]],
    context: TargetContext,
) -> tuple[float, int]:
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for chunk in iter_chunks(rows, context.config.verification_batch_size):
            batch = move_batch(
                collate_supervised(chunk, context.tokenizer, context.config.max_prompt_len, context.config.max_ans_len),
                context.shared["bundle"].device,
            )
            output = model(**batch, use_cache=False, return_dict=True)
            loss_sum, tokens = causal_loss_sum(output.logits, batch["labels"])
            total_loss += float(loss_sum.detach().double().item())
            total_tokens += tokens
    if total_tokens <= 0:
        raise ValueError("Verification view has no supervised answer tokens")
    return total_loss / total_tokens, total_tokens


def _run_trajectory(
    model: Any,
    adapter_name: str,
    context: TargetContext,
    support_rows: list[dict[str, str]],
    verification_rows: list[dict[str, str]],
    schedule: list[tuple[int, ...]],
    checkpoints: tuple[int, ...],
    seed: int,
) -> Trajectory:
    import torch

    started = time.perf_counter()
    config = context.config
    model.set_adapter(adapter_name)
    trainable = _active_parameters(model, adapter_name)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lookahead_learning_rate,
        betas=config.lookahead_betas,
        eps=config.lookahead_epsilon,
        weight_decay=config.lookahead_weight_decay,
        amsgrad=False,
    )
    use_scaler = context.shared["bundle"].dtype_name == "float16" and context.shared["bundle"].device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    _set_rng(seed, context.shared["bundle"].device)
    nll: dict[int, float] = {}
    token_counts: dict[int, int] = {}
    nll[0], token_counts[0] = evaluate_verification_nll(model, verification_rows, context)
    model.train()
    accumulation = config.lookahead_gradient_accumulation
    microbatch_cursor = 0
    maximum = max(checkpoints)
    for step in range(1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_tokens = 0
        for _ in range(accumulation):
            positions = schedule[microbatch_cursor]
            microbatch_cursor += 1
            rows = [support_rows[position] for position in positions]
            batch = move_batch(
                collate_supervised(rows, context.tokenizer, config.max_prompt_len, config.max_ans_len),
                context.shared["bundle"].device,
            )
            output = model(**batch, use_cache=False, return_dict=True)
            loss_sum, tokens = causal_loss_sum(output.logits, batch["labels"])
            accumulated_tokens += tokens
            scaler.scale(loss_sum).backward() if use_scaler else loss_sum.backward()
        if accumulated_tokens <= 0:
            raise RuntimeError("A scheduled optimizer update has zero supervised tokens")
        if use_scaler:
            scaler.unscale_(optimizer)
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(float(accumulated_tokens))
        torch.nn.utils.clip_grad_norm_(trainable, config.lookahead_max_grad_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if step in checkpoints:
            nll[step], token_counts[step] = evaluate_verification_nll(model, verification_rows, context)
            model.train()
    if set(nll) != set(checkpoints):
        raise RuntimeError(f"Lookahead missed checkpoints: expected={checkpoints}, got={sorted(nll)}")
    if not all(math.isfinite(value) for value in nll.values()):
        raise FloatingPointError("Nonfinite verification NLL in lookahead trajectory")
    return Trajectory(nll, token_counts, time.perf_counter() - started)


def run_lookahead(context: TargetContext) -> LookaheadArtifacts:
    import torch
    from peft import PeftConfig, get_peft_model

    if not context.candidates:
        raise ValueError("Lookahead requires at least one eligible source expert")
    config = context.config
    peft_config = PeftConfig.from_pretrained(str(context.candidates[0].adapter_path))
    peft_config = copy.deepcopy(peft_config)
    peft_config.inference_mode = False
    canonical_config = context.adapter_schema.canonical
    fresh_seed = config.fresh_lora_seed(context.target_task, canonical_config)
    _set_rng(fresh_seed, context.shared["bundle"].device)
    model = get_peft_model(context.model, peft_config, adapter_name="fresh")
    fresh_state = _adapter_state(model, "fresh")
    fresh_hash = adapter_tensor_hash(fresh_state, context.adapter_schema.config_hash)
    checkpoints = resolved_checkpoints(config.methods, config.slu_checkpoints)
    lookahead_seed = config.lookahead_seed(context.target_task)
    support_rows = context.calibration.rows(context.calibration.support_positions)
    verification_rows = context.calibration.rows(context.calibration.verification_positions)
    schedule = build_support_schedule(
        len(support_rows),
        batch_size=config.lookahead_batch_size,
        optimizer_steps=max(checkpoints),
        gradient_accumulation=config.lookahead_gradient_accumulation,
        seed=lookahead_seed,
    )

    step0_error: float | None = None
    if hasattr(model, "disable_adapter"):
        model.set_adapter("fresh")
        fresh_nll, _ = evaluate_verification_nll(model, verification_rows[:1], context)
        with model.disable_adapter():
            base_nll, _ = evaluate_verification_nll(model, verification_rows[:1], context)
        step0_error = abs(fresh_nll - base_nll)
        tolerance = 1e-4 if context.shared["bundle"].dtype_name in {"float16", "bfloat16"} else 1e-6
        if step0_error > tolerance:
            raise RuntimeError(
                f"Fresh LoRA is not a null update at step 0: NLL difference={step0_error}"
            )

    fresh = _run_trajectory(
        model,
        "fresh",
        context,
        support_rows,
        verification_rows,
        schedule,
        checkpoints,
        lookahead_seed,
    )
    sources: dict[str, Trajectory] = {}
    for index, candidate in enumerate(context.candidates):
        adapter_name = f"source_{index}"
        model.load_adapter(str(candidate.adapter_path), adapter_name=adapter_name, is_trainable=True)
        sources[candidate.task] = _run_trajectory(
            model,
            adapter_name,
            context,
            support_rows,
            verification_rows,
            schedule,
            checkpoints,
            lookahead_seed,
        )
        if not hasattr(model, "delete_adapter"):
            raise RuntimeError("Installed PEFT cannot delete disposable source adapters safely")
        model.delete_adapter(adapter_name)
        if context.shared["bundle"].device.type == "cuda":
            torch.cuda.empty_cache()
    return LookaheadArtifacts(
        fresh,
        sources,
        fresh_seed,
        fresh_hash,
        lookahead_seed,
        checkpoints,
        len(schedule),
        step0_error,
        {
            "optimizer": "torch.optim.AdamW",
            "learning_rate": config.lookahead_learning_rate,
            "betas": list(config.lookahead_betas),
            "adam_epsilon": config.lookahead_epsilon,
            "weight_decay": config.lookahead_weight_decay,
            "amsgrad": False,
            "scheduler": "constant",
            "warmup_steps": 0,
            "gradient_accumulation": config.lookahead_gradient_accumulation,
            "nominal_effective_batch_size": config.lookahead_batch_size * config.lookahead_gradient_accumulation,
            "max_gradient_norm": config.lookahead_max_grad_norm,
            "dtype": context.shared["bundle"].dtype_name,
            "fp16_grad_scaler": context.shared["bundle"].dtype_name == "float16" and context.shared["bundle"].device.type == "cuda",
            "update_reduction": "global_supervised_token_sum_per_optimizer_step",
            "trainable_tensor_count": len(fresh_state),
            "trainable_parameter_count": sum(int(tensor.numel()) for tensor in fresh_state.values()),
        },
    )
