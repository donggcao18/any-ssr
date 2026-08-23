from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_FACTOR_PATTERN = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$")


@dataclass(frozen=True, slots=True)
class AdapterSchema:
    config: dict[str, Any]
    canonical: dict[str, Any]
    config_hash: str


@dataclass(frozen=True, slots=True)
class AdapterValidation:
    task: str
    path: Path
    status: str
    reason: str | None
    schema: AdapterSchema | None
    adapter_hash: str | None
    weight_path: Path | None
    factor_signature: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] | None


def _canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    target_modules = config.get("target_modules") or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    layers = config.get("layers_to_transform")
    if isinstance(layers, int):
        layers = [layers]
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", 0.0))
    use_rslora = bool(config.get("use_rslora", False))
    scale = alpha / (math.sqrt(rank) if use_rslora and rank > 0 else rank) if rank > 0 else float("nan")
    return {
        "peft_type": str(config.get("peft_type", "")).upper(),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "revision": config.get("revision"),
        "r": rank,
        "lora_alpha": alpha,
        "scale": scale,
        "lora_dropout": float(config.get("lora_dropout", 0.0)),
        "target_modules": sorted(str(item) for item in target_modules),
        "layers_to_transform": sorted(int(item) for item in layers) if layers else None,
        "layers_pattern": config.get("layers_pattern"),
        "bias": config.get("bias", "none"),
        "fan_in_fan_out": bool(config.get("fan_in_fan_out", False)),
        "use_dora": bool(config.get("use_dora", False)),
        "use_rslora": use_rslora,
        "init_lora_weights": config.get("init_lora_weights", True),
        "modules_to_save": config.get("modules_to_save"),
        "task_type": config.get("task_type"),
    }


def read_adapter_schema(path: Path) -> AdapterSchema:
    config_path = path / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    canonical = _canonical_config(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return AdapterSchema(config, canonical, hashlib.sha256(payload.encode()).hexdigest())


def resolve_weight_path(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(
    task: str,
    path: Path,
    model_name_or_path: str,
    model_revision: str | None = None,
) -> AdapterValidation:
    if not path.is_dir():
        return AdapterValidation(task, path, "missing", "adapter directory does not exist", None, None, None, None)
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return AdapterValidation(task, path, "missing", "adapter_config.json is absent", None, None, None, None)
    weight_path = resolve_weight_path(path)
    if weight_path is None:
        return AdapterValidation(task, path, "missing", "adapter_model.bin/safetensors is absent", None, None, None, None)
    try:
        schema = read_adapter_schema(path)
        canonical = schema.canonical
        if canonical["peft_type"] != "LORA":
            raise ValueError(f"unsupported peft_type={canonical['peft_type']!r}")
        if canonical["r"] <= 0 or not math.isfinite(canonical["lora_alpha"]) or canonical["lora_alpha"] <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not math.isfinite(canonical["scale"]) or canonical["scale"] <= 0:
            raise ValueError("LoRA applied-weight scale must be positive and finite")
        if canonical["bias"] != "none":
            raise ValueError("only bias='none' adapters are supported")
        if canonical["modules_to_save"]:
            raise ValueError("modules_to_save adapters are not supported")
        if canonical["fan_in_fan_out"]:
            raise ValueError("fan_in_fan_out=True is unsupported")
        if canonical["use_dora"]:
            raise ValueError("DoRA adapters are unsupported")
        recorded_model = canonical["base_model_name_or_path"]
        if recorded_model and str(recorded_model).rstrip("/") != str(model_name_or_path).rstrip("/"):
            raise ValueError(f"base model mismatch: adapter={recorded_model}, run={model_name_or_path}")
        recorded_revision = canonical["revision"]
        if recorded_revision and model_revision and str(recorded_revision) != str(model_revision):
            raise ValueError(f"base model revision mismatch: adapter={recorded_revision}, run={model_revision}")
        factors = extract_lora_factors(load_adapter_state(path))
        factor_signature = tuple(
            (module, tuple(int(value) for value in factor_a.shape), tuple(int(value) for value in factor_b.shape))
            for module, (factor_a, factor_b) in sorted(factors.items())
        )
        if any(a_shape[0] != canonical["r"] or b_shape[1] != canonical["r"] for _, a_shape, b_shape in factor_signature):
            raise ValueError("Adapter factor shapes do not match configured LoRA rank")
        adapter_hash = sha256_file(weight_path)
        return AdapterValidation(task, path, "ok", None, schema, adapter_hash, weight_path, factor_signature)
    except Exception as exc:
        return AdapterValidation(task, path, "incompatible", str(exc), None, None, weight_path, None)


def validate_adapter_set(
    candidates: Iterable[Any],
    model_name_or_path: str,
    model_revision: str | None = None,
) -> tuple[list[AdapterValidation], AdapterSchema | None]:
    results = [
        validate_adapter(candidate.task, candidate.adapter_path, model_name_or_path, model_revision)
        for candidate in candidates
    ]
    successful = [result for result in results if result.status == "ok" and result.schema is not None]
    reference = successful[0].schema if successful else None
    reference_signature = successful[0].factor_signature if successful else None
    if reference is not None:
        for index, result in enumerate(results):
            if (
                result.status == "ok"
                and result.schema is not None
                and (result.schema.canonical != reference.canonical or result.factor_signature != reference_signature)
            ):
                results[index] = AdapterValidation(
                    result.task,
                    result.path,
                    "incompatible",
                    "LoRA schema differs from the first eligible expert",
                    result.schema,
                    result.adapter_hash,
                    result.weight_path,
                    result.factor_signature,
                )
    return results, reference


def load_adapter_state(path: Path) -> dict[str, Any]:
    weight_path = resolve_weight_path(path)
    if weight_path is None:
        raise FileNotFoundError(f"No adapter weights found in {path}")
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to read this adapter") from exc
        return load_file(str(weight_path), device="cpu")
    import torch

    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Adapter file does not contain a state dictionary: {weight_path}")
    return state


def extract_lora_factors(state: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    for key, tensor in state.items():
        match = _FACTOR_PATTERN.match(key)
        if not match:
            continue
        module = match.group("module")
        for prefix in ("base_model.model.", "base_model."):
            if module.startswith(prefix):
                module = module[len(prefix):]
                break
        collected.setdefault(module, {})[match.group("factor")] = tensor
    incomplete = [module for module, factors in collected.items() if set(factors) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"Incomplete LoRA A/B pairs for modules: {incomplete[:5]}")
    if not collected:
        raise ValueError("No standard LoRA A/B tensors were found")
    result = {module: (factors["A"], factors["B"]) for module, factors in collected.items()}
    for module, (factor_a, factor_b) in result.items():
        if factor_a.ndim != 2 or factor_b.ndim != 2:
            raise ValueError(f"Non-matrix LoRA factors for {module}")
        if factor_a.shape[0] != factor_b.shape[1]:
            raise ValueError(f"LoRA rank mismatch for {module}: A={tuple(factor_a.shape)}, B={tuple(factor_b.shape)}")
    return result


def adapter_tensor_hash(state: dict[str, Any], config_hash: str) -> str:
    import torch

    digest = hashlib.sha256(config_hash.encode("ascii"))
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
