from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..adapters import extract_lora_factors, load_adapter_state
from ..gradients import collect_target_gradients
from ..records import MethodScore, TargetContext
from ..tasks import SourceExpert
from .registry import register


def numerical_basis(matrix: Any, tolerance_multiplier: float = 1.0) -> tuple[Any, dict[str, Any]]:
    import torch

    matrix = matrix.detach().double().cpu()
    if matrix.ndim != 2:
        raise ValueError("Numerical basis input must be a matrix")
    if matrix.numel() == 0:
        return torch.empty((matrix.shape[0], 0), dtype=torch.float64), {"rank": 0, "tolerance": 0.0, "singular_values": []}
    left, singular_values, _right = torch.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = tolerance_multiplier * max(matrix.shape) * torch.finfo(matrix.dtype).eps * largest
    rank = int((singular_values > tolerance).sum().item())
    basis = left[:, :rank].contiguous() if rank else torch.empty((matrix.shape[0], 0), dtype=torch.float64)
    return basis, {
        "rank": rank,
        "tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular_values.tolist()],
    }


def captured_energy(gradient: Any, factor_a: Any, factor_b: Any, tolerance_multiplier: float = 1.0) -> tuple[float, float, dict[str, Any]]:
    gradient = gradient.detach().double().cpu()
    factor_a = factor_a.detach().double().cpu()
    factor_b = factor_b.detach().double().cpu()
    if factor_a.ndim != 2 or factor_b.ndim != 2 or gradient.ndim != 2:
        raise ValueError("GCA expects matrix gradient and LoRA factors")
    if factor_a.shape[0] != factor_b.shape[1]:
        raise ValueError("LoRA A/B rank mismatch")
    if tuple(gradient.shape) != (factor_b.shape[0], factor_a.shape[1]):
        raise ValueError(
            f"Applied-weight geometry mismatch: G={tuple(gradient.shape)}, "
            f"B={tuple(factor_b.shape)}, A={tuple(factor_a.shape)}"
        )
    basis_u, info_u = numerical_basis(factor_b, tolerance_multiplier)
    basis_v, info_v = numerical_basis(factor_a.T, tolerance_multiplier)
    total = float((gradient * gradient).sum().item())
    first = basis_u.T @ gradient if basis_u.shape[1] else gradient.new_zeros((0, gradient.shape[1]))
    gv = gradient @ basis_v if basis_v.shape[1] else gradient.new_zeros((gradient.shape[0], 0))
    correction = basis_u @ (basis_u.T @ gv) if basis_u.shape[1] and basis_v.shape[1] else gradient.new_zeros(gv.shape)
    captured = float((first * first).sum().item() + ((gv - correction) ** 2).sum().item())
    tolerance = 1e-9 * max(1.0, total)
    if captured < -tolerance or captured > total + tolerance:
        raise RuntimeError(f"Captured energy {captured} lies outside [0, {total}]")
    captured = min(max(captured, 0.0), total)
    return captured, total, {"left_basis": info_u, "right_basis": info_v}


@dataclass(slots=True)
class GCATargetArtifacts:
    gradients: dict[str, Any]
    token_count: int


class GCASelectionMethod:
    name = "gca"
    minimum_history_tasks = 1

    def prepare_target(self, context: TargetContext) -> GCATargetArtifacts:
        if not context.candidates:
            return GCATargetArtifacts({}, 0)
        reference_state = load_adapter_state(context.candidates[0].adapter_path)
        reference_factors = extract_lora_factors(reference_state)
        rows = context.calibration.rows(context.calibration.gca_positions)
        gradients, token_count = collect_target_gradients(
            context.model,
            context.tokenizer,
            rows,
            sorted(reference_factors),
            batch_size=context.config.gca_batch_size,
            max_prompt_len=context.config.max_prompt_len,
            max_ans_len=context.config.max_ans_len,
            device=context.shared["bundle"].device,
        )
        return GCATargetArtifacts(gradients, token_count)

    def score_candidate(
        self,
        context: TargetContext,
        candidate: SourceExpert,
        artifacts: GCATargetArtifacts,
    ) -> MethodScore:
        factors = extract_lora_factors(load_adapter_state(candidate.adapter_path))
        if set(factors) != set(artifacts.gradients):
            return MethodScore(None, "incompatible_modules", {"reason": "candidate module universe differs"})
        captured_total = 0.0
        gradient_total = 0.0
        per_module: dict[str, Any] = {}
        for module in sorted(artifacts.gradients):
            factor_a, factor_b = factors[module]
            captured, total, info = captured_energy(
                artifacts.gradients[module],
                factor_a,
                factor_b,
                context.config.gca_rank_tolerance_multiplier,
            )
            captured_total += captured
            gradient_total += total
            per_module[module] = {"captured_energy": captured, "total_energy": total, **info}
        if gradient_total <= context.config.score_epsilon:
            return MethodScore(None, "no_target_gradient", {"target_token_count": artifacts.token_count})
        score = captured_total / (gradient_total + context.config.score_epsilon)
        if not math.isfinite(score):
            return MethodScore(None, "nonfinite_score", {})
        return MethodScore(float(min(max(score, 0.0), 1.0)), "ok", {
            "captured_energy": captured_total,
            "total_gradient_energy": gradient_total,
            "target_token_count": artifacts.token_count,
            "modules": per_module,
        })


@register("gca")
def _factory() -> GCASelectionMethod:
    return GCASelectionMethod()

