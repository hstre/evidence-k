"""Adapter for any OpenAI-compatible Chat Completions endpoint.

Works with the OpenAI API and compatible providers (DeepSeek, OpenRouter, Together,
local servers like vLLM / Ollama's OpenAI shim, …). It uses only the Python standard
library (``urllib``) so Evidence-k keeps a tiny dependency footprint.

API keys are **never** hardcoded and never read from the config file: they come from an
environment variable. If the key is missing the adapter raises a clear error instead of
sending an unauthenticated request.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import Model, ModelResponse, Prompt

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"


class ModelAdapterError(RuntimeError):
    """Raised for configuration or transport errors talking to the provider."""


class OpenAICompatibleModel(Model):
    provider = "openai_compatible"

    def __init__(
        self,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, temperature=temperature, max_tokens=max_tokens)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key_env = api_key_env or DEFAULT_KEY_ENV
        self.timeout = timeout
        self.extra = extra or {}
        self._api_key: str | None = None

    def _resolve_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ModelAdapterError(
                f"No API key found. Set the '{self.api_key_env}' environment variable "
                f"(or change model.api_key_env in your config). Keys are never read from "
                f"the config file or stored in code."
            )
        self._api_key = key
        return key

    def generate(self, prompt: Prompt, *, repetition: int = 0) -> ModelResponse:
        key = self._resolve_key()
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ModelAdapterError(
                f"Provider returned HTTP {exc.code} for model {self.name!r}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ModelAdapterError(
                f"Could not reach provider at {self.base_url!r}: {exc.reason}"
            ) from exc
        latency_s = time.perf_counter() - start

        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAdapterError(f"Unexpected response shape from provider: {raw!r}") from exc

        usage = raw.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", self._estimate_prompt_tokens(prompt)))
        completion_tokens = int(usage.get("completion_tokens", 0))

        return ModelResponse(
            text=text or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            raw={"provider": self.provider, "model": self.name},
        )
