"""Turn a single model answer into a scored record (without the cost term).

The cost penalty depends on a run-wide token reference, so reliability is finalised later
in the sweep (see :func:`evidence_k.runners.scorer.compute_reliability`). The evaluator is
the thin glue between a :class:`~evidence_k.tasks.base.Task` and the scoring primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tasks.base import Case, Evidence, Task


@dataclass(frozen=True)
class CaseEvaluation:
    dimensions: dict[str, float]
    constraint_detail: dict[str, bool]


def evaluate_answer(
    task: Task, case: Case, answer: str, selected: tuple[Evidence, ...]
) -> CaseEvaluation:
    scores = task.evaluate(case, answer, selected)
    return CaseEvaluation(
        dimensions=scores.as_dict(),
        constraint_detail=scores.constraint_detail,
    )
