"""Normalise the vendored contamination metrics into one severity score in [0, 1].

The vendored `metrics.py` returns raw, auditable counts (adopted framing terms, drift in
[0,1], attribution-failure count, role-adoption count). For the dual-instrumented benchmark
we need a single comparable *contamination severity* per response, on the same [0,1] scale
as the correctness-axis reliability, so the two axes can be plotted and contrasted.

This wrapper is Evidence-k's own (NOT vendored): the saturation caps and weights are a
calibration choice and live here, separate from the vendored heuristics.
"""

from __future__ import annotations

from typing import Any

from .metrics import score_response

# Saturation caps: at/above these raw counts the sub-signal is treated as fully contaminated.
_FRAMING_SATURATION = 8
_ATTRIB_SATURATION = 3
_ROLE_SATURATION = 3

# Weights over the four single-turn sub-signals (framing leakage is the dominant, most
# reliable channel — it is what DESi's published density curves are built on). Loop
# detection is multi-turn only and is therefore not part of the single-response severity.
_WEIGHTS = {
    "framing": 0.45,
    "drift": 0.25,
    "attribution": 0.20,
    "role": 0.10,
}


def contamination_metrics(text: str) -> dict[str, Any]:
    """Slim, auditable per-response metrics (adopted framing, drift, attribution, role)."""
    s = score_response(text)
    return {
        "framing_adopted": s["framing_leakage"]["adopted"],
        "framing_quoted": s["framing_leakage"]["quoted"],
        "register_drift": s["register_drift"]["score"],
        "attribution_failures": s["attribution"]["failures"],
        "role_adoption": s["role_adoption"]["count"],
        "framing_terms": s["framing_leakage"]["adopted_terms"],
    }


def contamination_severity(text: str) -> float:
    """Single contamination severity in [0, 1] (higher = more contaminated)."""
    m = contamination_metrics(text)
    framing = min(1.0, m["framing_adopted"] / _FRAMING_SATURATION)
    drift = m["register_drift"]  # already [0, 1]
    attribution = min(1.0, m["attribution_failures"] / _ATTRIB_SATURATION)
    role = min(1.0, m["role_adoption"] / _ROLE_SATURATION)
    severity = (
        _WEIGHTS["framing"] * framing
        + _WEIGHTS["drift"] * drift
        + _WEIGHTS["attribution"] * attribution
        + _WEIGHTS["role"] * role
    )
    return round(max(0.0, min(1.0, severity)), 6)


def score_contamination(text: str) -> dict[str, Any]:
    """Severity + the underlying auditable metrics for one response."""
    return {"severity": contamination_severity(text), "metrics": contamination_metrics(text)}
