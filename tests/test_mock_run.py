"""End-to-end k-sweep with the deterministic MockModel."""

from __future__ import annotations

from pathlib import Path

from evidence_k.config import parse_config
from evidence_k.runners.sweep import run_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "model": {"provider": "mock", "name": "mock-model", "temperature": 0, "max_tokens": 256},
    "benchmark": {"k_values": [0, 1, 3, "full"], "repetitions": 3, "random_seed": 42},
    "tasks": [
        {"name": "factual_qa", "dataset": "data/examples/factual_qa.jsonl"},
        {"name": "constraint_following", "dataset": "data/examples/constraint_following.jsonl"},
    ],
    "scoring": {"weights": {}},
}


def _run():
    cfg = parse_config(CONFIG)
    return run_sweep(cfg, base_dir=REPO_ROOT)


def test_sweep_produces_all_combinations():
    result = _run()
    # 2 tasks; factual_qa has 6 cases, constraint_following 5 -> 11 cases
    # 4 k-values * 3 reps
    n_cases = 6 + 5
    assert len(result.raw_records) == n_cases * 4 * 3
    assert len(result.score_records) == len(result.raw_records)


def test_every_score_has_reliability_in_range():
    result = _run()
    for rec in result.score_records:
        assert 0.0 <= rec["reliability"] <= 1.0
        assert 0.0 <= rec["correctness"] <= 1.0


def test_k0_worse_than_peak():
    result = _run()
    agg = result.aggregates["factual_qa"]
    peak = max(m["reliability"] for m in agg.values())
    assert agg["0"]["reliability"] < peak
    # more context is not free: "full" should not beat the peak
    assert agg["full"]["reliability"] <= peak


def test_determinism_same_seed_same_results():
    a = _run()
    b = _run()
    assert a.task_summaries == b.task_summaries
    assert [r["answer"] for r in a.raw_records] == [r["answer"] for r in b.raw_records]


def test_recommended_k_is_tested_value():
    result = _run()
    for task, tsum in result.task_summaries.items():
        assert tsum["recommended_k"] in [0, 1, 3, "full"], task
