"""Task registry."""

from __future__ import annotations

from .base import Case, DimensionScores, Evidence, Task
from .conflict_resolution import ConflictResolutionTask
from .constraint_following import ConstraintFollowingTask
from .domain_probe import (
    FinanceQATask,
    LegalQATask,
    MedicalQATask,
    TechnicalQATask,
)
from .dual_battery import (
    DualConflictTask,
    DualConstraintTask,
    DualFactualTask,
    DualMultihopTask,
    DualStateTask,
)
from .dual_instrumented import DualInstrumentedTask
from .factual_qa import FactualQATask
from .state_consistency import StateConsistencyTask

_REGISTRY: dict[str, type[Task]] = {
    FactualQATask.name: FactualQATask,
    StateConsistencyTask.name: StateConsistencyTask,
    ConflictResolutionTask.name: ConflictResolutionTask,
    ConstraintFollowingTask.name: ConstraintFollowingTask,
    DualInstrumentedTask.name: DualInstrumentedTask,
    TechnicalQATask.name: TechnicalQATask,
    MedicalQATask.name: MedicalQATask,
    LegalQATask.name: LegalQATask,
    FinanceQATask.name: FinanceQATask,
    DualFactualTask.name: DualFactualTask,
    DualMultihopTask.name: DualMultihopTask,
    DualStateTask.name: DualStateTask,
    DualConflictTask.name: DualConflictTask,
    DualConstraintTask.name: DualConstraintTask,
}

__all__ = [
    "Case",
    "Evidence",
    "DimensionScores",
    "Task",
    "get_task",
    "available_tasks",
]


def get_task(name: str) -> Task:
    """Instantiate the task registered under ``name``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return cls()


def available_tasks() -> list[str]:
    return sorted(_REGISTRY)
