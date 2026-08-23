from __future__ import annotations

import math
from typing import Any

from ..records import MethodScore, TargetContext
from ..tasks import SourceExpert
from .registry import register


def oia_score(source: dict[int, float], fresh: dict[int, float], epsilon: float) -> tuple[float, dict[str, float]]:
    required = {0, 1}
    if not required.issubset(source) or not required.issubset(fresh):
        raise ValueError("OIA requires NLL checkpoints 0 and 1")
    source_contraction = math.log((source[0] + epsilon) / (source[1] + epsilon))
    fresh_contraction = math.log((fresh[0] + epsilon) / (fresh[1] + epsilon))
    score = source_contraction - fresh_contraction
    values = {
        "source_nll_step0": source[0],
        "source_nll_step1": source[1],
        "fresh_nll_step0": fresh[0],
        "fresh_nll_step1": fresh[1],
        "source_contraction": source_contraction,
        "fresh_contraction": fresh_contraction,
    }
    if not all(math.isfinite(value) for value in (score, *values.values())):
        raise FloatingPointError("Nonfinite OIA value")
    return score, values


class OIASelectionMethod:
    name = "oia"
    minimum_history_tasks = 1

    def prepare_target(self, context: TargetContext) -> Any:
        return context.shared["lookahead"]

    def score_candidate(self, context: TargetContext, candidate: SourceExpert, artifacts: Any) -> MethodScore:
        try:
            score, diagnostics = oia_score(
                artifacts.sources[candidate.task].nll,
                artifacts.fresh.nll,
                context.config.score_epsilon,
            )
            diagnostics["verification_supervised_tokens"] = artifacts.fresh.supervised_tokens
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


@register("oia")
def _factory() -> OIASelectionMethod:
    return OIASelectionMethod()
