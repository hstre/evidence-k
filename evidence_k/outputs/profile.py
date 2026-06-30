"""Build the portable ``k_profile.json``.

The profile is the deliverable other systems (DESi routers, RAG layers, governance) load.
It is intentionally small and self-describing and does NOT depend on any Evidence-k
internals to be consumed.
"""

from __future__ import annotations

from typing import Any

from .. import PROFILE_VERSION


def build_profile(summary: dict[str, Any]) -> dict[str, Any]:
    """Construct the k-profile dict from a run summary (see :mod:`outputs.reports`)."""
    tasks_out: dict[str, Any] = {}
    for task_name, tsum in summary.get("tasks", {}).items():
        per_k = tsum.get("per_k", {})
        score_curve = {
            kl: per_k[kl]["reliability"]
            for kl in _ordered_k_labels(summary.get("tested_k_values", []), per_k)
        }
        tasks_out[task_name] = {
            "recommended_k": tsum.get("recommended_k"),
            "best_score": tsum.get("best_score"),
            "score_curve": score_curve,
        }

    return {
        "profile_version": PROFILE_VERSION,
        "model": summary.get("model"),
        "provider": summary.get("provider"),
        "created_at": summary.get("created_at"),
        "tested_k_values": list(summary.get("tested_k_values", [])),
        "tasks": tasks_out,
        "global_recommendation": summary.get(
            "global_recommendation", {"default_k": None, "notes": ""}
        ),
    }


def _ordered_k_labels(tested_k_values: list[Any], per_k: dict[str, Any]) -> list[str]:
    """Yield k-labels in the configured order, restricted to those that have data."""
    labels = []
    for k in tested_k_values:
        kl = "full" if k == "full" else str(k)
        if kl in per_k:
            labels.append(kl)
    # Include any stray keys not in tested order (defensive), preserving dict order.
    for kl in per_k:
        if kl not in labels:
            labels.append(kl)
    return labels
