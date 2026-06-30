"""The k-sweep: run every (task, k, case, repetition) and aggregate the results.

This is the deterministic core. Given the same config and seed it produces the same raw
outputs and the same aggregates (the mock model is fully seeded; for real models only the
provider introduces nondeterminism).
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import FULL, Config, KValue
from ..models import build_model
from ..models.base import Model
from ..tasks import get_task
from ..tasks.base import Case, Task
from ..utils.hashing import short_hash
from ..utils.jsonl import read_jsonl
from . import scorer
from .evaluator import evaluate_answer

_DIMENSIONS = (
    "correctness",
    "grounding",
    "constraint_adherence",
    "state_consistency",
    "hallucination_rate",
)


def k_label(k: KValue) -> str:
    """JSON/-CSV friendly string label for a k value."""
    return FULL if k == FULL else str(k)


def k_sort_key(k: KValue) -> tuple[int, float]:
    """Order k values numerically with ``"full"`` last; used for tie-breaking."""
    if k == FULL:
        return (1, float("inf"))
    return (0, float(k))  # type: ignore[arg-type]


@dataclass
class SweepResult:
    config: Config
    run_id: str
    created_at: str
    raw_records: list[dict[str, Any]] = field(default_factory=list)
    score_records: list[dict[str, Any]] = field(default_factory=list)
    # aggregates[task][k_label] -> metrics dict
    aggregates: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    task_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_recommendation: dict[str, Any] = field(default_factory=dict)


def _load_cases(task: Task, dataset_path: str, base_dir: Path) -> list[Case]:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = base_dir / path
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"dataset {path} is empty")
    return [task.parse(row) for row in rows]


def run_sweep(
    config: Config, *, base_dir: str | Path = ".", model: Model | None = None
) -> SweepResult:
    """Execute the full sweep and return aggregated results (nothing written to disk)."""
    base = Path(base_dir)
    seed = config.benchmark.random_seed
    mdl = model if model is not None else build_model(config.model, seed=seed)

    created = datetime.now(UTC).replace(microsecond=0).isoformat()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + short_hash(
        config.model.name, config.model.provider, seed, [t.name for t in config.tasks]
    )

    result = SweepResult(config=config, run_id=run_id, created_at=created)

    for task_cfg in config.tasks:
        task = get_task(task_cfg.name)
        cases = _load_cases(task, task_cfg.dataset, base)

        for k in config.benchmark.k_values:
            for case in cases:
                selected = task.select_evidence(case, k)
                prompt = task.build_prompt(case, k, selected, seed=seed)
                for rep in range(config.benchmark.repetitions):
                    resp = mdl.generate(prompt, repetition=rep)
                    ev = evaluate_answer(task, case, resp.text, selected)
                    # Second axis (None for correctness-only tasks).
                    cont = task.contamination(case, resp.text, selected)
                    cont_sev = cont["severity"] if cont else None

                    result.raw_records.append(
                        {
                            "task": task.name,
                            "k": k,
                            "case_id": case.id,
                            "repetition": rep,
                            "answer": resp.text,
                            "prompt_tokens": resp.prompt_tokens,
                            "completion_tokens": resp.completion_tokens,
                            "total_tokens": resp.total_tokens,
                            "latency_s": resp.latency_s,
                            "n_evidence": len(selected),
                            "mode": resp.raw.get("mode"),
                            "contamination": cont_sev,
                        }
                    )
                    result.score_records.append(
                        {
                            "task": task.name,
                            "k": k,
                            "case_id": case.id,
                            "repetition": rep,
                            "total_tokens": resp.total_tokens,
                            "latency_s": resp.latency_s,
                            "contamination": cont_sev,
                            **{d: ev.dimensions[d] for d in _DIMENSIONS},
                        }
                    )

    _finalise(result)
    return result


def _finalise(result: SweepResult) -> None:
    """Compute reliability, aggregates, per-task recommendations and global default."""
    weights = result.config.scoring.weights
    reference_tokens = max((r["total_tokens"] for r in result.score_records), default=1) or 1

    # Attach reliability (now that the run-wide token reference is known).
    for rec in result.score_records:
        dims = {d: rec[d] for d in _DIMENSIONS}
        normalized_cost = rec["total_tokens"] / reference_tokens
        rec["normalized_cost"] = round(normalized_cost, 6)
        rec["reliability"] = round(
            scorer.compute_reliability(dims, weights, normalized_cost), 6
        )

    # Group by (task, k_label).
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in result.score_records:
        grouped[(rec["task"], k_label(rec["k"]))].append(rec)

    k_values = result.config.benchmark.k_values
    tasks = [t.name for t in result.config.tasks]

    for task_name in tasks:
        result.aggregates[task_name] = {}
        for k in k_values:
            kl = k_label(k)
            recs = grouped.get((task_name, kl), [])
            if not recs:
                continue
            metrics: dict[str, Any] = {
                "reliability": round(statistics.fmean(r["reliability"] for r in recs), 6),
                "answer_variance": _answer_variance(recs),
                "mean_total_tokens": round(
                    statistics.fmean(r["total_tokens"] for r in recs), 2
                ),
                "mean_latency_s": round(statistics.fmean(r["latency_s"] for r in recs), 6),
                "n_samples": len(recs),
            }
            for d in _DIMENSIONS:
                metrics[d] = round(statistics.fmean(r[d] for r in recs), 6)
            # Second axis: mean contamination severity, if this task scores it.
            cont_vals = [r["contamination"] for r in recs if r.get("contamination") is not None]
            if cont_vals:
                metrics["contamination"] = round(statistics.fmean(cont_vals), 6)
            result.aggregates[task_name][kl] = metrics

    # Per-task recommendation: argmax reliability, tie-break toward the smaller k.
    for task_name in tasks:
        per_k = result.aggregates[task_name]
        if not per_k:
            continue
        ordered = sorted(per_k.items(), key=lambda kv: k_sort_key(_unlabel(kv[0])))
        best_kl, best_metrics = max(
            ordered,
            key=lambda kv: (kv[1]["reliability"], -k_sort_key(_unlabel(kv[0]))[1]),
        )
        result.task_summaries[task_name] = {
            "recommended_k": _unlabel(best_kl),
            "best_score": best_metrics["reliability"],
            "per_k": per_k,
        }

    result.global_recommendation = _global_recommendation(result)


def _unlabel(kl: str) -> KValue:
    return FULL if kl == FULL else int(kl)


def _answer_variance(recs: list[dict[str, Any]]) -> float:
    """Mean disagreement across repetitions, per case (0 = perfectly stable)."""
    by_case: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_case[r["case_id"]].append(r["correctness"])
    disagreements = []
    for values in by_case.values():
        if len(values) <= 1:
            disagreements.append(0.0)
            continue
        counts = Counter(values)
        agreement = max(counts.values()) / len(values)
        disagreements.append(1.0 - agreement)
    return round(statistics.fmean(disagreements), 6) if disagreements else 0.0


def _global_recommendation(result: SweepResult) -> dict[str, Any]:
    """Pick the k that maximises mean reliability averaged across tasks."""
    per_k_scores: dict[str, list[float]] = defaultdict(list)
    for task_name in result.aggregates:
        for kl, metrics in result.aggregates[task_name].items():
            per_k_scores[kl].append(metrics["reliability"])
    if not per_k_scores:
        return {"default_k": None, "notes": "No results produced."}
    means = {kl: statistics.fmean(v) for kl, v in per_k_scores.items()}
    best_kl = max(
        means.items(),
        key=lambda kv: (kv[1], -k_sort_key(_unlabel(kv[0]))[1]),
    )[0]
    return {
        "default_k": _unlabel(best_kl),
        "notes": "Use task-specific k where available; k* is model- and task-specific.",
    }
