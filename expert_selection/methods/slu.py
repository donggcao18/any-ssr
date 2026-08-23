from __future__ import annotations

import math
from typing import Any

from ..records import MethodScore, TargetContext
from ..tasks import SourceExpert
from .registry import register


def slu_score(
    source: dict[int, float],
    fresh: dict[int, float],
    checkpoints: tuple[int, ...],
    weights: tuple[float, ...],
    epsilon: float,
) -> tuple[float, dict[str, Any]]:
    if len(checkpoints) != len(weights):
        raise ValueError("SLU checkpoints and weights differ in length")
    utilities: dict[int, float] = {}
    score = 0.0
    for checkpoint, weight in zip(checkpoints, weights):
        if checkpoint not in source or checkpoint not in fresh:
            raise ValueError(f"Missing SLU checkpoint {checkpoint}")
        utility = 1.0 - source[checkpoint] / (fresh[checkpoint] + epsilon)
        if not math.isfinite(utility):
            raise FloatingPointError(f"Nonfinite SLU utility at step {checkpoint}")
        utilities[checkpoint] = utility
        score += weight * utility
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite SLU score")
    return score, {
        "source_nll": {str(key): value for key, value in source.items()},
        "fresh_nll": {str(key): value for key, value in fresh.items()},
        "utilities": {str(key): value for key, value in utilities.items()},
        "resolved_weights": list(weights),
    }


class SLUSelectionMethod:
    name = "slu"
    minimum_history_tasks = 1

    def prepare_target(self, context: TargetContext) -> Any:
        return context.shared["lookahead"]

    def score_candidate(self, context: TargetContext, candidate: SourceExpert, artifacts: Any) -> MethodScore:
        try:
            score, diagnostics = slu_score(
                artifacts.sources[candidate.task].nll,
                artifacts.fresh.nll,
                context.config.slu_checkpoints,
                context.config.normalized_slu_weights,
                context.config.score_epsilon,
            )
            diagnostics["verification_supervised_tokens"] = artifacts.fresh.supervised_tokens
            diagnostics["requested_weights"] = list(
                context.config.slu_weights
                or tuple(1.0 for _ in context.config.slu_checkpoints)
            )
            diagnostics.update({
                "lookahead_seed": artifacts.lookahead_seed,
                "resolved_checkpoints": list(artifacts.checkpoints),
                "schedule_microbatches": artifacts.schedule_microbatches,
                "fresh_step0_equivalence_error": artifacts.step0_equivalence_error,
                "source_trajectory_seconds": artifacts.sources[candidate.task].elapsed_seconds,
                "fresh_trajectory_seconds": artifacts.fresh.elapsed_seconds,
                "lookahead_settings": artifacts.settings,
            })
            return MethodScore(score, "ok", diagnostics)
        except (ValueError, FloatingPointError) as exc:
            return MethodScore(None, "invalid_trajectory", {"reason": str(exc)})


@register("slu")
def _factory() -> SLUSelectionMethod:
    return SLUSelectionMethod()
