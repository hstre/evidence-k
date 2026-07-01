"""Offline unit tests for the OpenAI-compatible adapter's payload construction."""

from __future__ import annotations

from evidence_k.models.base import Prompt
from evidence_k.models.openai_compatible import OpenAICompatibleModel

PROMPT = Prompt(system="sys", user="hello")


def test_payload_has_no_provider_field_without_pin():
    m = OpenAICompatibleModel(name="qwen/qwen-2.5-7b-instruct", base_url="https://x/v1")
    payload = m._build_payload(PROMPT)
    assert "provider" not in payload
    assert payload["model"] == "qwen/qwen-2.5-7b-instruct"
    assert payload["messages"][0]["content"] == "sys"


def test_provider_pin_is_translated_into_routing_field():
    m = OpenAICompatibleModel(
        name="qwen/qwen-2.5-7b-instruct",
        base_url="https://openrouter.ai/api/v1",
        extra={"provider_pin": {"order": ["DeepInfra"], "allow_fallbacks": False}},
    )
    payload = m._build_payload(PROMPT)
    assert payload["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False}
    # the helper key is consumed, never sent verbatim
    assert "provider_pin" not in payload


def test_provider_pin_defaults_fallbacks_off():
    m = OpenAICompatibleModel(
        name="m", extra={"provider_pin": {"order": ["Together"]}}
    )
    payload = m._build_payload(PROMPT)
    assert payload["provider"] == {"order": ["Together"], "allow_fallbacks": False}
