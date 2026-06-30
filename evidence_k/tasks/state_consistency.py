"""State consistency: track the *current* value of a state across updates.

The expected answer is the current, consistent value. Distractor evidence often carries a
stale/superseded value; the optional ``stale_values`` field lists values the answer must
NOT regress to. State consistency is violated if the answer is wrong or echoes a stale
value.
"""

from __future__ import annotations

from ..runners import scorer
from .base import Case, Evidence, Task


class StateConsistencyTask(Task):
    name = "state_consistency"

    def score_state_consistency(
        self, case: Case, answer: str, selected: tuple[Evidence, ...]
    ) -> float:
        if self.score_correctness(case, answer) < 1.0:
            return 0.0
        stale = case.extra.get("stale_values", []) or []
        na = scorer.normalize_text(answer)
        for value in stale:
            nv = scorer.normalize_text(str(value))
            if nv and nv in na:
                return 0.0
        return 1.0
