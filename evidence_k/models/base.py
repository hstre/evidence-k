"""Model adapter interface.

A model takes a :class:`Prompt` and returns a :class:`ModelResponse`. The prompt carries
``system`` and ``user`` text (what a real model sees) plus an opaque ``meta`` mapping.

``meta`` is a side-channel used only by the deterministic :class:`~evidence_k.models.mock.MockModel`
simulator so it can produce a realistic evidence-saturation curve offline. Real adapters
(e.g. OpenAI-compatible) MUST ignore ``meta`` and send only ``system`` + ``user``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..utils.tokens import count_tokens


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Model(abc.ABC):
    """Abstract base for all model adapters."""

    def __init__(self, name: str, temperature: float = 0.0, max_tokens: int = 512) -> None:
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def generate(self, prompt: Prompt, *, repetition: int = 0) -> ModelResponse:
        """Produce a single completion for ``prompt``.

        ``repetition`` is the 0-based index within a repeated sample of the same prompt;
        adapters may use it (e.g. the mock) to vary stochastic behaviour deterministically.
        """

    @staticmethod
    def _estimate_prompt_tokens(prompt: Prompt) -> int:
        return count_tokens(prompt.system) + count_tokens(prompt.user)
