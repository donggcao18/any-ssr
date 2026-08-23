from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MethodScore:
    score: float | None
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreRecord:
    run_id: str
    task_order: list[str]
    task_order_id: str
    target_task: str
    target_index: int
    source_task: str
    source_index: int
    adapter_path: str
    method: str
    score: float | None
    rank: int | None
    status: str
    seed: int
    fresh_lora_seed: int | None = None
    fresh_adapter_hash: str | None = None
    elapsed_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TargetContext:
    config: Any
    target_task: str
    target_index: int
    candidates: list[Any]
    dataset: Any
    calibration: Any
    tokenizer: Any
    model: Any
    adapter_schema: Any
    shared: dict[str, Any] = field(default_factory=dict)

