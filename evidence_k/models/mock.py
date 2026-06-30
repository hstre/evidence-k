"""Deterministic offline model simulator.

``MockModel`` does not call any network. It reads the evaluation side-channel in
``prompt.meta`` to produce a *realistic* evidence-saturation curve so the whole pipeline
(sweep → score → profile) can be exercised and tested offline and reproducibly.

The simulated behaviour encodes the core hypothesis Evidence-k measures:

* with **no supporting evidence** in the window the model mostly guesses (low accuracy);
* with the supporting fragment present and few distractors it is reliable;
* as **distractors accumulate** the model is increasingly pulled toward a grounded but
  *wrong* answer — accuracy degrades even though the answer stays "in context".

It is a simulator, not a claim about any specific real model. Real adapters ignore
``meta`` entirely.
"""

from __future__ import annotations

import re
from typing import Any

from ..utils.hashing import seeded_random
from ..utils.tokens import count_tokens
from .base import Model, ModelResponse, Prompt

_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
_HALLUCINATION_POOL = ["Atlantis", "Xanadu", "Eldorado", "Zubron", "Qarth", "Lemuria"]


def _salient_token(text: str, avoid: str) -> str | None:
    """Pick a representative wrong token from a distractor's text."""
    avoid_l = avoid.lower()
    words = _WORD.findall(text)
    # Prefer a capitalised / longer token that is not part of the expected answer.
    candidates = sorted(
        (w for w in words if len(w) > 3 and w.lower() not in avoid_l),
        key=lambda w: (w[0].isupper(), len(w)),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return words[0] if words else None


class MockModel(Model):
    provider = "mock"

    def __init__(
        self,
        name: str = "mock-model",
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> None:
        super().__init__(name=name, temperature=temperature, max_tokens=max_tokens)
        self.seed = seed

    def generate(self, prompt: Prompt, *, repetition: int = 0) -> ModelResponse:
        meta: dict[str, Any] = prompt.meta or {}
        expected = str(meta.get("expected_answer", "") or "")
        evidence = list(meta.get("selected_evidence", []) or [])
        case_id = meta.get("case_id", "?")
        k_label = meta.get("k", "?")

        supporting = [e for e in evidence if str(e.get("label", "")).lower() == "supporting"]
        distractors = [e for e in evidence if str(e.get("label", "")).lower() != "supporting"]

        rng = seeded_random(self.seed, meta.get("seed", 0), case_id, k_label, repetition)

        if supporting:
            # Corroborating support raises reliability; distractors erode it. This encodes
            # the saturation hypothesis: a little evidence helps, more support helps more,
            # but accumulating distractors eventually drowns the signal.
            p_correct = 0.50 + 0.22 * len(supporting) - 0.08 * len(distractors)
            p_correct = max(0.20, min(0.97, p_correct))
        else:
            # No supporting fragment in the window: the model falls back on prior knowledge.
            p_correct = 0.30

        draw = rng.random()
        if draw < p_correct and expected:
            answer = expected
            mode = "correct"
        elif distractors:
            chosen = rng.choice(distractors)
            token = _salient_token(str(chosen.get("text", "")), expected)
            answer = token if token else rng.choice(_HALLUCINATION_POOL)
            mode = "distracted"
        else:
            answer = rng.choice(_HALLUCINATION_POOL)
            mode = "hallucinated"

        prompt_tokens = self._estimate_prompt_tokens(prompt)
        completion_tokens = max(1, count_tokens(answer))
        # Illustrative simulated latency that grows with prompt size (not wall-clock).
        latency_s = round(0.01 + prompt_tokens * 0.0004, 6)

        return ModelResponse(
            text=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            raw={"simulated": True, "mode": mode, "p_correct": round(p_correct, 4)},
        )
