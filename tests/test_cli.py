"""CLI smoke test: init -> run -> summarize -> export-profile end to end."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evidence_k.cli import main


@pytest.fixture()
def in_tmp(tmp_path: Path):
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


def test_full_cli_flow(in_tmp: Path):
    # init scaffolds config + datasets
    assert main(["init"]) == 0
    assert (in_tmp / "configs" / "example.yaml").exists()
    assert (in_tmp / "data" / "examples" / "factual_qa.jsonl").exists()

    # run the sweep
    assert main(["run", "--config", "configs/example.yaml"]) == 0
    runs = list((in_tmp / "runs").glob("*/"))
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "k_profile.json").exists()

    # summarize the saved run
    assert main(["summarize", "--run-dir", str(run_dir)]) == 0

    # export a fresh profile to a custom location
    out = in_tmp / "exported_profile.json"
    assert main(["export-profile", "--run-dir", str(run_dir), "--out", str(out)]) == 0
    profile = json.loads(out.read_text())
    assert profile["provider"] == "mock"
    assert "factual_qa" in profile["tasks"]


def test_bad_config_returns_error_code(in_tmp: Path):
    (in_tmp / "bad.yaml").write_text("model: {provider: nope, name: x}\n", encoding="utf-8")
    # missing benchmark/tasks + bad provider -> non-zero exit, no traceback
    assert main(["run", "--config", "bad.yaml"]) != 0
