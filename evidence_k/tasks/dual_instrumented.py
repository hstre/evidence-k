"""Dual-instrumented task: one input, one response, scored on BOTH axes.

This is the core of the dual-axis benchmark. Each case carries a factual question with a
correct answer (correctness axis) *and* evidence fragments written in a manipulative
framework register (contamination axis). As k grows, more register-laden fragments enter
the window: the correctness axis can stay flat (the model still answers correctly) while
the contamination axis bends (the model starts echoing the source's vocabulary/register).

Running both axes on the *same* input and the *same* k removes the task confound that a
naive "QA sweep vs. contamination sweep" comparison would suffer from.
"""

from __future__ import annotations

from typing import Any

from ..contamination.score import score_contamination
from .base import Case, Evidence, Task


class DualInstrumentedTask(Task):
    name = "dual_instrumented"

    def contamination(
        self, case: Case, answer: str, selected: tuple[Evidence, ...]
    ) -> dict[str, Any] | None:
        # Contamination is a property of the model's own generated text (does it adopt the
        # source register?), so it is scored on the answer alone — the same vendored DESi
        # heuristics, normalised to a [0,1] severity.
        return score_contamination(answer)
