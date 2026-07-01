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

    # HTTP statuses worth retrying (transient gateway/provider errors). 4xx client
    # errors (400/401/403/404) are NOT retried — they will not fix themselves.
    _RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        extra: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> None:
        super().__init__(name=name, temperature=temperature, max_tokens=max_tokens)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key_env = api_key_env or DEFAULT_KEY_ENV
        self.timeout = timeout
        self.retries = max(1, retries)
        # extra may carry a ``provider_pin`` helper key (OpenRouter provider routing);
        # it is translated into the payload's ``provider`` field, never sent verbatim.
        self.extra = dict(extra or {})
        self._provider_pin = self.extra.pop("provider_pin", None)
        self._api_key: str | None = None
        # served-backend provenance, filled as calls return (so a pin can be verified)
        self.providers_seen: set[str] = set()
        self.served_models_seen: set[str] = set()
        self._logged_backend = False

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

    def _build_payload(self, prompt: Prompt) -> dict[str, Any]:
        """Assemble the Chat Completions request body (no network; unit-testable).

        A ``provider_pin`` (e.g. ``{"order": ["DeepInfra"], "allow_fallbacks": false}``)
        is translated into OpenRouter's ``provider`` routing field so the served
        backend is controlled; ``allow_fallbacks`` defaults to ``false`` under a pin.
        """
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
        if self._provider_pin:
            pin = self._provider_pin
            payload["provider"] = {
                "order": list(pin["order"]),
                "allow_fallbacks": bool(pin.get("allow_fallbacks", False)),
            }
        return payload

    def generate(self, prompt: Prompt, *, repetition: int = 0) -> ModelResponse:
        key = self._resolve_key()
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(self._build_payload(prompt)).encode("utf-8")
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
        raw = self._request_with_retry(req)
        latency_s = time.perf_counter() - start

        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAdapterError(f"Unexpected response shape from provider: {raw!r}") from exc

        # served-backend provenance (OpenRouter reports the routed provider + served
        # model id, often a provider-specific quantization). Recorded so a pin is
        # verifiable after the fact; printed once so it lands in run logs.
        served_provider = raw.get("provider")
        served_model = raw.get("model")
        if served_provider:
            self.providers_seen.add(str(served_provider))
        if served_model:
            self.served_models_seen.add(str(served_model))
        if not self._logged_backend and (served_provider or served_model):
            self._logged_backend = True
            pinned = self._provider_pin["order"] if self._provider_pin else None
            print(
                f"[served backend] provider={served_provider} model={served_model}"
                + (f" (pinned to {pinned})" if pinned else " (unpinned)")
            )

        usage = raw.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", self._estimate_prompt_tokens(prompt)))
        completion_tokens = int(usage.get("completion_tokens", 0))

        return ModelResponse(
            text=text or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            raw={
                "provider": self.provider,
                "model": self.name,
                "served_provider": served_provider,
                "served_model": served_model,
            },
        )

    def _request_with_retry(self, req: urllib.request.Request) -> dict[str, Any]:
        """POST with exponential backoff on transient (5xx/429/network) failures.

        Non-retryable client errors (400/401/403/404) raise immediately. A pinned
        provider with fallbacks disabled makes transient upstream errors more likely
        to surface, so retrying here is what keeps a pinned sweep from dying on one
        blip — without ever silently re-routing to a different backend.
        """
        last = ""
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                last = f"HTTP {exc.code} for model {self.name!r}: {detail[:300]}"
                if exc.code not in self._RETRYABLE:
                    raise ModelAdapterError(f"Provider returned {last}") from exc
            except urllib.error.URLError as exc:
                last = f"network error reaching {self.base_url!r}: {exc.reason}"
            if attempt + 1 < self.retries:
                time.sleep(2**attempt)
        raise ModelAdapterError(f"Provider request failed after {self.retries} attempts: {last}")
