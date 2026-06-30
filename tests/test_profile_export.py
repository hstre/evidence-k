"""Profile export and on-disk run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from evidence_k.config import parse_config
from evidence_k.outputs.profile import build_profile
from evidence_k.outputs.reports import build_summary, write_run
from evidence_k.runners.sweep import run_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "model": {"provider": "mock", "name": "mock-model"},
    "benchmark": {"k_values": [0, 1, 3, "full"], "repetitions": 2, "random_seed": 1},
    "tasks": [{"name": "factual_qa", "dataset": "data/examples/factual_qa.jsonl"}],
    "scoring": {"weights": {}},
}


def _result():
    return run_sweep(parse_config(CONFIG), base_dir=REPO_ROOT)


def test_profile_shape():
    summary = build_summary(_result())
    profile = build_profile(summary)
    assert profile["profile_version"] == "0.1"
    assert profile["model"] == "mock-model"
    assert profile["tested_k_values"] == [0, 1, 3, "full"]
    fq = profile["tasks"]["factual_qa"]
    assert fq["recommended_k"] in [0, 1, 3, "full"]
    assert set(fq["score_curve"]) == {"0", "1", "3", "full"}
    assert 0.0 <= fq["best_score"] <= 1.0
    assert profile["global_recommendation"]["default_k"] in [0, 1, 3, "full"]


def test_write_run_creates_all_files(tmp_path: Path):
    result = _result()
    run_dir = write_run(result, runs_root=tmp_path)
    expected = [
        "config.resolved.yaml",
        "raw_outputs.jsonl",
        "scores.jsonl",
        "summary.json",
        "summary.csv",
        "k_profile.json",
        "README_run.md",
    ]
    for name in expected:
        assert (run_dir / name).exists(), name

    profile = json.loads((run_dir / "k_profile.json").read_text())
    assert profile["tasks"]["factual_qa"]["score_curve"]["0"] >= 0.0
