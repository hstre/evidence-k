"""Task registry."""

from __future__ import annotations

from .base import Case, DimensionScores, Evidence, Task
from .conflict_resolution import ConflictResolutionTask
from .constraint_following import ConstraintFollowingTask
from .factual_qa import FactualQATask
from .state_consistency import StateConsistencyTask

_REGISTRY: dict[str, type[Task]] = {
    FactualQATask.name: FactualQATask,
    StateConsistencyTask.name: StateConsistencyTask,
    ConflictResolutionTask.name: ConflictResolutionTask,
    ConstraintFollowingTask.name: ConstraintFollowingTask,
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
