"""Config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evidence_k.config import ConfigError, dump_resolved_yaml, load_config, parse_config

VALID = {
    "model": {"provider": "mock", "name": "mock-model", "temperature": 0, "max_tokens": 256},
    "benchmark": {"k_values": [0, 1, 3, "full"], "repetitions": 2, "random_seed": 7},
    "tasks": [{"name": "factual_qa", "dataset": "data/examples/factual_qa.jsonl"}],
    "scoring": {"weights": {"correctness": 0.5}},
}


def test_parse_valid_config():
    cfg = parse_config(VALID)
    assert cfg.model.provider == "mock"
    assert cfg.benchmark.k_values == (0, 1, 3, "full")
    assert cfg.benchmark.repetitions == 2
    # unspecified weights fall back to defaults
    assert cfg.scoring.weights["correctness"] == 0.5
    assert cfg.scoring.weights["grounding"] == 0.20


def test_load_config_from_file(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            model: {provider: mock, name: m}
            benchmark: {k_values: [0, 1, "full"]}
            tasks: [{name: factual_qa, dataset: d.jsonl}]
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.benchmark.k_values == (0, 1, "full")
    assert cfg.benchmark.repetitions == 1  # default


def test_unknown_provider_rejected():
    bad = {**VALID, "model": {"provider": "magic", "name": "x"}}
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_unknown_weight_rejected():
    bad = {**VALID, "scoring": {"weights": {"nonsense": 1.0}}}
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_negative_and_bool_k_values_rejected():
    with pytest.raises(ConfigError):
        parse_config({**VALID, "benchmark": {"k_values": [-1]}})
    with pytest.raises(ConfigError):
        parse_config({**VALID, "benchmark": {"k_values": [True]}})


def test_missing_required_section():
    with pytest.raises(ConfigError):
        parse_config({"model": {"provider": "mock", "name": "m"}})


def test_resolved_yaml_roundtrips():
    cfg = parse_config(VALID)
    text = dump_resolved_yaml(cfg)
    assert "provider: mock" in text
    assert "k_values" in text
