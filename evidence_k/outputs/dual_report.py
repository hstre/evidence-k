"""Dual-axis report: contrast the correctness axis against the contamination axis.

The headline metric is **blind-spot severity**: the contamination incurred at the
reliability-optimal k (k\\*, = argmax reliability, the operating point a router would deploy).
At that k\\* the correctness axis reads a flat 1.000, so a correctness-only profile signs off on
it while this contamination goes unflagged — the harm a correctness-only calibration walks a
router into, which is the whole point of the dual-instrumented benchmark. (Correctness alone
saturates early and cannot distinguish most of the k-range, so it is not itself an informative
"optimal k"; k\\* is the reliability argmax reused from the per-task recommendation.)
"""

from __future__ import annotations

from typing import Any

from ..config import FULL
from ..runners.sweep import SweepResult, k_label, k_sort_key


def _unlabel(kl: str) -> int | str:
    return FULL if kl == FULL else int(kl)


def _is_dual(per_k: dict[str, dict[str, Any]]) -> bool:
    return any(m.get("contamination") is not None for m in per_k.values())


def build_dual_report(result: SweepResult) -> dict[str, Any]:
    """Per dual-instrumented task: both curves, both optima, and blind-spot severity."""
    out: dict[str, Any] = {}
    for task_name, per_k in result.aggregates.items():
        if not _is_dual(per_k):
            continue
        klabels = [
            k_label(k) for k in result.config.benchmark.k_values if k_label(k) in per_k
        ]
        curve = {
            kl: {
                "reliability": per_k[kl]["reliability"],
                "correctness": per_k[kl]["correctness"],
                "contamination": per_k[kl].get("contamination"),
            }
            for kl in klabels
        }

        # Reliability-optimal k (k*, = argmax reliability): reuse the per-task recommendation.
        # This is the k a router deploys and the k the blind-spot severity is measured at; it is
        # NOT a correctness argmax (correctness saturates early and cannot rank most of the range).
        rel_k = result.task_summaries.get(task_name, {}).get("recommended_k")
        rel_kl = k_label(rel_k) if rel_k is not None else None

        # Contamination-optimal k: argmin contamination, tie-break toward smaller k.
        cont_items = [
            (kl, per_k[kl]["contamination"])
            for kl in klabels
            if per_k[kl].get("contamination") is not None
        ]
        cont_kl = (
            min(cont_items, key=lambda x: (x[1], k_sort_key(_unlabel(x[0]))))[0]
            if cont_items
            else None
        )

        blind = per_k[rel_kl].get("contamination") if rel_kl in per_k else None
        out[task_name] = {
            "reliability_optimal_k": rel_k,
            "contamination_optimal_k": _unlabel(cont_kl) if cont_kl else None,
            "blind_spot_severity": blind,
            "max_contamination": max((v for _, v in cont_items), default=None),
            "contamination_at_full": per_k.get(FULL, {}).get("contamination"),
            "curve": curve,
        }
    return out


def format_dual_report(dual: dict[str, Any]) -> str:
    """Human-readable dual-axis table(s)."""
    if not dual:
        return ""
    lines = ["", "Dual-axis (correctness vs. contamination):"]
    for task_name, d in dual.items():
        lines.append(
            f"\n  {task_name}: reliability-optimal k* = {d['reliability_optimal_k']} "
            f"→ contamination there = {_fmt(d['blind_spot_severity'])} "
            f"(blind-spot severity); contamination-optimal k = {d['contamination_optimal_k']}"
        )
        lines.append(f"    {'k':>5} | {'reliab':>7} | {'correct':>7} | {'CONTAM':>7}")
        for kl, m in d["curve"].items():
            lines.append(
                f"    {kl:>5} | {m['reliability']:>7.3f} | {m['correctness']:>7.3f} | "
                f"{_fmt(m['contamination']):>7}"
            )
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"
