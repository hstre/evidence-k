"""Dual-instrumented (correctness + contamination) axis."""

from __future__ import annotations

from pathlib import Path

from evidence_k.config import parse_config
from evidence_k.contamination.score import contamination_severity, score_contamination
from evidence_k.outputs.dual_report import build_dual_report
from evidence_k.runners.sweep import run_sweep
from evidence_k.tasks import get_task

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "model": {"provider": "mock", "name": "mock-model"},
    "benchmark": {"k_values": [0, 1, 3, "full"], "repetitions": 3, "random_seed": 42},
    "tasks": [{"name": "dual_instrumented", "dataset": "data/dual/dual_instrumented.jsonl"}],
    "scoring": {"weights": {}},
}


def test_contamination_scorer_detects_adopted_framework_terms():
    assert contamination_severity("The answer is ten minutes.") == 0.0
    sev = contamination_severity("ten. We move into field-awareness, resonance, harvest together.")
    assert sev > 0.0
    detail = score_contamination("we co-create resonance and attunement")
    assert detail["metrics"]["framing_adopted"] >= 2


def test_quoted_terms_are_not_counted_as_adopted():
    # Attributed/quoted use is expected in analysis and must not score as adoption.
    quoted = contamination_severity('The source uses the term "resonance" and "attunement".')
    assert quoted == 0.0


def test_dual_task_exposes_contamination_hook():
    task = get_task("dual_instrumented")
    out = task.contamination(None, "we move into harvest and resonance together", ())
    assert out is not None and out["severity"] > 0.0
    # correctness-only task returns None
    assert get_task("factual_qa").contamination(None, "anything", ()) is None


def test_dual_sweep_records_contamination_and_blind_spot():
    result = run_sweep(parse_config(CONFIG), base_dir=REPO_ROOT)
    agg = result.aggregates["dual_instrumented"]
    # contamination present in aggregates, correctness still scored
    assert all("contamination" in m for m in agg.values())
    # contamination is zero before register fragments enter (k=0) and positive once they do
    assert agg["0"]["contamination"] == 0.0
    assert agg["full"]["contamination"] > 0.0

    dual = build_dual_report(result)["dual_instrumented"]
    assert dual["reliability_optimal_k"] in [0, 1, 3, "full"]
    assert dual["blind_spot_severity"] is not None
    assert 0.0 <= dual["max_contamination"] <= 1.0
