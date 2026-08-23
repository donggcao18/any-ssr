from __future__ import annotations

from typing import Any, Protocol

from ..records import MethodScore, TargetContext
from ..tasks import SourceExpert


class SelectionMethod(Protocol):
    name: str
    minimum_history_tasks: int

    def prepare_target(self, context: TargetContext) -> Any:
        ...

    def score_candidate(self, context: TargetContext, candidate: SourceExpert, artifacts: Any) -> MethodScore:
        ...


class HistoricalArtifactBuilder(Protocol):
    name: str

    def prepare_current_task(self, context: Any) -> Any:
        ...

