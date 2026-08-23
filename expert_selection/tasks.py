from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ExperimentConfig


METHOD_MINIMUM_HISTORY = {"gmm": 2, "gca": 1, "oia": 1, "slu": 1}


@dataclass(frozen=True, slots=True)
class SourceExpert:
    task: str
    index: int
    adapter_path: Path


def target_index(config: ExperimentConfig, target: str) -> int:
    return config.task_order.index(target)


def eligible_sources(config: ExperimentConfig, target: str) -> list[SourceExpert]:
    index = target_index(config, target)
    allowed = set(config.candidates) if config.candidates else None
    return [
        SourceExpert(task, source_index, config.adapter_root / task / "0")
        for source_index, task in enumerate(config.task_order[:index])
        if allowed is None or task in allowed
    ]


def method_applicable(config: ExperimentConfig, target: str, method: str) -> bool:
    return target_index(config, target) >= METHOD_MINIMUM_HISTORY[method]


def sequence_tasks(config: ExperimentConfig) -> tuple[str, ...]:
    if config.prepare_artifacts == "gmm" and "gmm" in config.methods:
        last = max(target_index(config, target) for target in config.targets)
        return config.task_order[: last + 1]
    return tuple(task for task in config.task_order if task in set(config.targets))


def validate_requested_candidates(config: ExperimentConfig) -> None:
    if not config.candidates:
        return
    for target in config.targets:
        eligible = {source.task for source in eligible_sources(config, target)}
        invalid = set(config.candidates) - eligible
        if invalid:
            raise ValueError(f"Candidates not older than target {target}: {sorted(invalid)}")


def chronological_rows(config: ExperimentConfig, targets: Iterable[str] | None = None) -> dict[str, list[str]]:
    chosen = config.targets if targets is None else tuple(targets)
    return {target: [source.task for source in eligible_sources(config, target)] for target in chosen}

