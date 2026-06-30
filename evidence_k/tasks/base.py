"""Task abstraction: dataset shape, prompt construction and per-case evaluation.

A *task* owns three things:

1. how to parse a dataset row into a :class:`Case`,
2. how to select the first ``k`` evidence fragments and build a :class:`Prompt`,
3. how to turn a model answer into per-dimension scores (:class:`DimensionScores`).

Generic scoring lives in :mod:`evidence_k.runners.scorer`; subclasses override only the
dimensions that need task-specific logic (e.g. state consistency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import FULL, KValue
from ..models.base import Prompt
from ..prompts.default_prompt import build_system, build_user
from ..runners import scorer


@dataclass(frozen=True)
class Evidence:
    rank: int
    text: str
    label: str = "supporting"


@dataclass(frozen=True)
class Case:
    id: str
    task: str
    question: str
    evidence: tuple[Evidence, ...]
    expected_answer: str | None = None
    constraints: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensionScores:
    correctness: float
    grounding: float
    constraint_adherence: float
    state_consistency: float
    hallucination_rate: float
    constraint_detail: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "grounding": self.grounding,
            "constraint_adherence": self.constraint_adherence,
            "state_consistency": self.state_consistency,
            "hallucination_rate": self.hallucination_rate,
        }


def parse_case(row: dict[str, Any], default_task: str) -> Case:
    """Validate a raw dataset row into a :class:`Case`. Raises on missing fields."""
    if "id" not in row:
        raise ValueError(f"dataset row missing 'id': {row!r}")
    if "question" not in row:
        raise ValueError(f"case {row['id']!r} missing 'question'")

    raw_evidence = row.get("evidence", []) or []
    evidence: list[Evidence] = []
    for e in raw_evidence:
        if "text" not in e:
            raise ValueError(f"case {row['id']!r} has an evidence item without 'text'")
        evidence.append(
            Evidence(
                rank=int(e.get("rank", len(evidence) + 1)),
                text=str(e["text"]),
                label=str(e.get("label", "supporting")),
            )
        )
    evidence.sort(key=lambda e: e.rank)

    constraints = tuple(str(c) for c in (row.get("constraints", []) or []))
    known = {"id", "task", "question", "evidence", "expected_answer", "constraints"}
    extra = {k: v for k, v in row.items() if k not in known}

    return Case(
        id=str(row["id"]),
        task=str(row.get("task", default_task)),
        question=str(row["question"]),
        evidence=tuple(evidence),
        expected_answer=(
            None if row.get("expected_answer") is None else str(row["expected_answer"])
        ),
        constraints=constraints,
        extra=extra,
    )


class Task:
    """Base task. Subclasses set ``name`` and may override scoring hooks.

    Not instantiated directly — :func:`evidence_k.tasks.get_task` returns a concrete
    subclass — but it carries the full default behaviour so subclasses stay tiny.
    """

    name: str = "base"

    def parse(self, row: dict[str, Any]) -> Case:
        return parse_case(row, default_task=self.name)

    def select_evidence(self, case: Case, k: KValue) -> tuple[Evidence, ...]:
        """Return the first ``k`` evidence fragments by rank (all for ``"full"``)."""
        if k == FULL:
            return case.evidence
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"invalid k value: {k!r}")
        return case.evidence[:k]

    def build_prompt(
        self, case: Case, k: KValue, selected: tuple[Evidence, ...], seed: int = 0
    ) -> Prompt:
        system = build_system(case.constraints)
        user = build_user(case.question, [e.text for e in selected])
        meta = {
            "task": self.name,
            "case_id": case.id,
            "k": k,
            "seed": seed,
            "expected_answer": case.expected_answer,
            "selected_evidence": [
                {"rank": e.rank, "text": e.text, "label": e.label} for e in selected
            ],
            "extra": case.extra,
        }
        return Prompt(system=system, user=user, meta=meta)

    # --- scoring hooks -----------------------------------------------------------------

    def score_correctness(self, case: Case, answer: str) -> float:
        return scorer.score_correctness(answer, case.expected_answer)

    def score_state_consistency(
        self, case: Case, answer: str, selected: tuple[Evidence, ...]
    ) -> float:
        """Neutral by default; tasks that track state override this."""
        return 1.0

    def evaluate(
        self, case: Case, answer: str, selected: tuple[Evidence, ...]
    ) -> DimensionScores:
        evidence_texts = [e.text for e in selected]
        adherence, detail = scorer.score_constraints(answer, case.constraints)
        return DimensionScores(
            correctness=self.score_correctness(case, answer),
            grounding=scorer.score_grounding(answer, evidence_texts),
            constraint_adherence=adherence,
            state_consistency=self.score_state_consistency(case, answer, selected),
            hallucination_rate=scorer.hallucination_rate(
                answer, evidence_texts, case.expected_answer
            ),
            constraint_detail=detail,
        )
