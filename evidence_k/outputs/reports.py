"""Assemble and persist all per-run artifacts under ``runs/<run_id>/``."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .. import PROFILE_VERSION
from ..config import config_to_dict, dump_resolved_yaml
from ..runners.sweep import SweepResult
from ..utils.jsonl import write_jsonl
from .dual_report import build_dual_report
from .profile import build_profile

_CSV_COLUMNS = [
    "task",
    "k",
    "reliability",
    "correctness",
    "grounding",
    "constraint_adherence",
    "state_consistency",
    "hallucination_rate",
    "answer_variance",
    "mean_total_tokens",
    "mean_latency_s",
    "n_samples",
]


def build_summary(result: SweepResult) -> dict[str, Any]:
    """The machine-readable run summary (``summary.json``)."""
    return {
        "profile_version": PROFILE_VERSION,
        "run_id": result.run_id,
        "created_at": result.created_at,
        "model": result.config.model.name,
        "provider": result.config.model.provider,
        "config_path": result.config.source_path,
        "repetitions": result.config.benchmark.repetitions,
        "random_seed": result.config.benchmark.random_seed,
        "weights": dict(result.config.scoring.weights),
        "tested_k_values": list(result.config.benchmark.k_values),
        "tasks": result.task_summaries,
        "global_recommendation": result.global_recommendation,
        "dual": build_dual_report(result),
    }


def summary_to_csv(summary: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    tested = summary.get("tested_k_values", [])
    for task_name, tsum in summary.get("tasks", {}).items():
        per_k = tsum.get("per_k", {})
        for k in tested:
            kl = "full" if k == "full" else str(k)
            if kl not in per_k:
                continue
            m = per_k[kl]
            writer.writerow(
                {
                    "task": task_name,
                    "k": kl,
                    "reliability": m["reliability"],
                    "correctness": m["correctness"],
                    "grounding": m["grounding"],
                    "constraint_adherence": m["constraint_adherence"],
                    "state_consistency": m["state_consistency"],
                    "hallucination_rate": m["hallucination_rate"],
                    "answer_variance": m["answer_variance"],
                    "mean_total_tokens": m["mean_total_tokens"],
                    "mean_latency_s": m["mean_latency_s"],
                    "n_samples": m["n_samples"],
                }
            )
    return buf.getvalue()


def build_run_readme(summary: dict[str, Any]) -> str:
    lines = [
        f"# Evidence-k run `{summary['run_id']}`",
        "",
        f"- **Model:** `{summary['model']}` (provider: `{summary['provider']}`)",
        f"- **Created:** {summary['created_at']}",
        f"- **Repetitions:** {summary['repetitions']} · **Seed:** {summary['random_seed']}",
        f"- **k values tested:** {summary['tested_k_values']}",
        "",
        "## Recommended k per task",
        "",
        "| Task | Recommended k | Best reliability |",
        "| --- | --- | --- |",
    ]
    for task_name, tsum in summary.get("tasks", {}).items():
        lines.append(
            f"| {task_name} | {tsum['recommended_k']} | {tsum['best_score']:.3f} |"
        )
    gr = summary.get("global_recommendation", {})
    lines += [
        "",
        f"**Global default_k:** {gr.get('default_k')} — {gr.get('notes', '')}",
        "",
        "## Reliability curve (per task)",
        "",
    ]
    tested = summary.get("tested_k_values", [])
    for task_name, tsum in summary.get("tasks", {}).items():
        per_k = tsum.get("per_k", {})
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append("| k | reliability | correctness | grounding | halluc. | variance |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for k in tested:
            kl = "full" if k == "full" else str(k)
            if kl not in per_k:
                continue
            m = per_k[kl]
            lines.append(
                f"| {kl} | {m['reliability']:.3f} | {m['correctness']:.3f} | "
                f"{m['grounding']:.3f} | {m['hallucination_rate']:.3f} | "
                f"{m['answer_variance']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Files",
        "",
        "- `config.resolved.yaml` — the exact, validated config used for this run",
        "- `raw_outputs.jsonl` — one row per (task, k, case, repetition) model call",
        "- `scores.jsonl` — per-sample dimension scores + reliability",
        "- `summary.json` / `summary.csv` — aggregated per-(task, k) metrics",
        "- `k_profile.json` — portable profile for routers / RAG / governance layers",
        "",
        "> Evidence-k is a reasoning aid for context sizing. `k*` is model- and "
        "task-specific, not a universal constant.",
        "",
    ]
    return "\n".join(lines)


def write_run(result: SweepResult, runs_root: str | Path = "runs") -> Path:
    """Write every artifact for ``result`` and return the run directory."""
    run_dir = Path(runs_root) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(result)
    profile = build_profile(summary)

    (run_dir / "config.resolved.yaml").write_text(
        dump_resolved_yaml(result.config), encoding="utf-8"
    )
    write_jsonl(run_dir / "raw_outputs.jsonl", result.raw_records)
    write_jsonl(run_dir / "scores.jsonl", result.score_records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.csv").write_text(summary_to_csv(summary), encoding="utf-8")
    (run_dir / "k_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if summary.get("dual"):
        (run_dir / "dual_report.json").write_text(
            json.dumps(summary["dual"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (run_dir / "README_run.md").write_text(build_run_readme(summary), encoding="utf-8")

    # Keep a copy of the input config dict for provenance (handy when source moves).
    (run_dir / "config.input.json").write_text(
        json.dumps(config_to_dict(result.config), indent=2), encoding="utf-8"
    )
    return run_dir
