"""Scoring primitives and JSONL loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_k.runners import scorer
from evidence_k.utils.jsonl import read_jsonl, write_jsonl


def test_correctness_normalized_match():
    assert scorer.score_correctness("Paris", "paris") == 1.0
    assert scorer.score_correctness("The answer is Paris.", "Paris") == 1.0
    assert scorer.score_correctness("Berlin", "Paris") == 0.0


def test_correctness_digit_word_equivalence():
    # gold word, model answers digit (the gemini failure mode) and vice versa
    assert scorer.score_correctness("The practice lasts 10 minutes.", "ten") == 1.0
    assert scorer.score_correctness("ten minutes each morning", "10") == 1.0
    assert scorer.score_correctness("There are 12 people.", "twelve") == 1.0
    # a different number must still be wrong
    assert scorer.score_correctness("8 minutes", "ten") == 0.0


def test_grounding_counts_evidence_overlap():
    # answer token present in evidence
    assert scorer.score_grounding("Paris", ["The tower is in Paris."]) == 1.0
    # no evidence at all -> ungrounded
    assert scorer.score_grounding("Paris", []) == 0.0


def test_hallucination_rate():
    # answer fully covered by expected -> no hallucination
    assert scorer.hallucination_rate("Paris", [], "Paris") == 0.0
    # invented token, not in evidence or expected
    assert scorer.hallucination_rate("Atlantis", ["Paris is a city."], "Paris") == 1.0


def test_constraint_checks():
    adh, detail = scorer.score_constraints("Mars", ["Answer in exactly one word."])
    assert adh == 1.0 and detail["Answer in exactly one word."] is True

    adh, _ = scorer.score_constraints("the red planet", ["Answer in exactly one word."])
    assert adh == 0.0

    adh, _ = scorer.score_constraints("7", ["Answer with a number only."])
    assert adh == 1.0

    adh, _ = scorer.score_constraints("Berlin", ["Do not use the word 'Berlin'."])
    assert adh == 0.0

    # unverifiable constraint -> treated as satisfied (1.0), not a silent failure
    adh, detail = scorer.score_constraints("anything", ["Respond politely."])
    assert adh == 1.0 and detail == {}


def test_reliability_clamped_and_weighted():
    weights = {
        "correctness": 0.5,
        "grounding": 0.5,
        "hallucination_penalty": 1.0,
        "cost_penalty": 0.0,
    }
    dims = {"correctness": 1.0, "grounding": 1.0, "hallucination_rate": 0.0}
    assert scorer.compute_reliability(dims, weights) == 1.0
    # heavy hallucination penalty cannot push below 0
    dims2 = {"correctness": 0.0, "grounding": 0.0, "hallucination_rate": 1.0}
    assert scorer.compute_reliability(dims2, weights) == 0.0


def test_jsonl_roundtrip_and_error(tmp_path: Path):
    rows = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    p = tmp_path / "x.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(bad)
