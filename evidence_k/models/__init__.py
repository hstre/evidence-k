"""Model adapter registry."""

from __future__ import annotations

from ..config import ModelConfig
from .base import Model, ModelResponse, Prompt
from .mock import MockModel
from .openai_compatible import ModelAdapterError, OpenAICompatibleModel

__all__ = [
    "Model",
    "ModelResponse",
    "Prompt",
    "MockModel",
    "OpenAICompatibleModel",
    "ModelAdapterError",
    "build_model",
]


def build_model(cfg: ModelConfig, *, seed: int = 0) -> Model:
    """Instantiate the model adapter named by ``cfg.provider``."""
    if cfg.provider == "mock":
        return MockModel(
            name=cfg.name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=seed,
        )
    if cfg.provider in {"openai_compatible", "openai"}:
        return OpenAICompatibleModel(
            name=cfg.name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            extra=cfg.extra,
        )
    raise ValueError(f"unknown model provider: {cfg.provider!r}")
