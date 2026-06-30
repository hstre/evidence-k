"""Lightweight token estimation.

We deliberately avoid a hard dependency on a tokenizer library. ``count_tokens`` is a
provider-agnostic approximation (~4 characters per token) that is good enough for the
relative cost comparisons Evidence-k makes between k-values. A real token count from the
model API, when available, always takes precedence (see the model adapters).
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Approximate the number of tokens in ``text`` (~4 chars/token, min 0)."""
    if not text:
        return 0
    # Blend a char-based and word-based estimate for a slightly more stable number.
    char_estimate = len(text) / 4.0
    word_estimate = len(text.split()) / 0.75
    return max(1, round((char_estimate + word_estimate) / 2))
