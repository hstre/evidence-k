"""Canonical example config + datasets.

This module is the single source of truth for the bundled example assets. Both the files
checked into the repository (``configs/example.yaml``, ``data/examples/*.jsonl``) and the
``evidence-k init`` command are produced from here, so they can never drift apart.

The datasets are crafted so the offline :class:`~evidence_k.models.mock.MockModel`
exhibits a realistic evidence-saturation curve: each case has exactly one *supporting*
fragment (rank 1) followed by *distractors*, so reliability rises as the supporting
fragment enters the window and then declines as distractors accumulate.
"""

from __future__ import annotations

from typing import Any

import yaml

EXAMPLE_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "mock",
        "name": "mock-model",
        "temperature": 0,
        "max_tokens": 512,
    },
    "benchmark": {
        "k_values": [0, 1, 2, 3, 5, 8, 13, "full"],
        "repetitions": 3,
        "random_seed": 42,
    },
    "tasks": [
        {"name": "factual_qa", "dataset": "data/examples/factual_qa.jsonl"},
        {"name": "state_consistency", "dataset": "data/examples/state_consistency.jsonl"},
        {"name": "conflict_resolution", "dataset": "data/examples/conflict_resolution.jsonl"},
        {"name": "constraint_following", "dataset": "data/examples/constraint_following.jsonl"},
    ],
    "scoring": {
        "weights": {
            "correctness": 0.35,
            "grounding": 0.20,
            "constraint_adherence": 0.20,
            "state_consistency": 0.15,
            "hallucination_penalty": 0.10,
            "cost_penalty": 0.00,
        }
    },
}

# Generic, true-but-irrelevant facts used as distractors.
_DISTRACTOR_POOL = [
    "The Great Wall of China stretches for thousands of miles.",
    "At sea level water boils at one hundred degrees Celsius.",
    "The mitochondrion is often called the powerhouse of the cell.",
    "Honey can remain edible for a very long time when sealed.",
    "The Pacific is the largest ocean on the planet.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Amazon discharges more water than any other river.",
    "A standard calendar adds an extra day every four years.",
    "Sound travels faster through water than through air.",
    "The adult human skeleton contains a couple hundred bones.",
    "Lightning is hotter than the surface of the sun.",
    "Octopuses have three hearts and blue blood.",
    "The Sahara is the largest hot desert in the world.",
    "Bamboo is among the fastest growing plants on Earth.",
    "A bolt of silk was once worth its weight in trade.",
    "Venus rotates in the opposite direction to most planets.",
]

_TARGET_EVIDENCE = 14  # 2 supporting + 12 distractors → "full" differs from k=13
_N_DISTRACTORS = 12


def _tokens(text: str) -> set[str]:
    return {w.strip(".,';").lower() for w in text.split()}


def _distractors_excluding(answer: str, lead: list[str] | None = None) -> list[str]:
    """Pick distractors that don't accidentally contain the answer's words."""
    avoid = _tokens(answer)
    chosen: list[str] = list(lead or [])
    for fact in _DISTRACTOR_POOL:
        if len(chosen) >= _N_DISTRACTORS:
            break
        if _tokens(fact) & avoid:
            continue
        if fact in chosen:
            continue
        chosen.append(fact)
    return chosen[:_N_DISTRACTORS]


def _case(
    case_id: str,
    task: str,
    question: str,
    supporting: str,
    expected: str,
    *,
    constraints: list[str],
    corroboration: str | None = None,
    lead_distractors: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a case with two supporting fragments (ranks 1 and 3) interleaved with
    distractors. The corroborating fragment at rank 3 is what pushes the optimal k past 1:
    a single distractor at rank 2 makes k=2 dip before corroboration recovers it at k=3."""
    if corroboration is None:
        corroboration = f"This is independently corroborated: the answer is {expected}."
    distractors = _distractors_excluding(expected, lead_distractors)
    evidence = [
        {"rank": 1, "text": supporting, "label": "supporting"},
        {"rank": 2, "text": distractors[0], "label": "distractor"},
        {"rank": 3, "text": corroboration, "label": "supporting"},
    ]
    for i, text in enumerate(distractors[1:], start=4):
        evidence.append({"rank": i, "text": text, "label": "distractor"})
    row: dict[str, Any] = {
        "id": case_id,
        "task": task,
        "question": question,
        "evidence": evidence,
        "expected_answer": expected,
        "constraints": constraints,
    }
    if extra:
        row.update(extra)
    return row


def _factual_qa() -> list[dict[str, Any]]:
    three_words = ["Answer in at most 3 words."]
    return [
        _case("fqa_001", "factual_qa", "Which city is the Eiffel Tower in?",
              "The Eiffel Tower is located in Paris, France.", "Paris",
              constraints=three_words),
        _case("fqa_002", "factual_qa", "What is the capital of Japan?",
              "Tokyo is the capital and largest city of Japan.", "Tokyo",
              constraints=three_words),
        _case("fqa_003", "factual_qa", "Which planet is the largest in the solar system?",
              "Jupiter is the largest planet in the solar system.", "Jupiter",
              constraints=three_words),
        _case("fqa_004", "factual_qa", "What is the chemical symbol for gold?",
              "Gold has the chemical symbol Au on the periodic table.", "Au",
              constraints=three_words),
        _case("fqa_005", "factual_qa", "What is the tallest mountain above sea level?",
              "Mount Everest is the tallest mountain above sea level.", "Everest",
              constraints=three_words),
        _case("fqa_006", "factual_qa", "Who painted the Mona Lisa?",
              "The Mona Lisa was painted by Leonardo da Vinci.", "Leonardo da Vinci",
              constraints=three_words),
    ]


def _state_consistency() -> list[dict[str, Any]]:
    three_words = ["Answer in at most 3 words."]
    specs = [
        ("sc_001", "What is the patient's current medication?",
         "Update: as of today the patient's medication was switched to Amoxicillin.",
         "Amoxicillin", "The patient was previously prescribed Penicillin.", "Penicillin"),
        ("sc_002", "What is the project's current deadline?",
         "The deadline has now been moved to March 15.",
         "March 15", "Originally the deadline was March 1.", "March 1"),
        ("sc_003", "Where is the meeting now scheduled?",
         "The meeting has been relocated and is now in Room 204.",
         "Room 204", "The meeting used to be held in Room 101.", "Room 101"),
        ("sc_004", "What is the user's current subscription tier?",
         "The account was upgraded; the current tier is Premium.",
         "Premium", "The user signed up on the Basic tier.", "Basic"),
        ("sc_005", "What is the current primary server region?",
         "Traffic was migrated; the primary region is now eu-central.",
         "eu-central", "The service originally ran in us-east.", "us-east"),
    ]
    return [
        _case(cid, "state_consistency", q, supp, exp,
              constraints=three_words,
              lead_distractors=[stale_text],
              extra={"stale_values": [stale_val]})
        for cid, q, supp, exp, stale_text, stale_val in specs
    ]


def _conflict_resolution() -> list[dict[str, Any]]:
    three_words = ["Answer in at most 3 words."]
    specs = [
        ("cr_001", "What year did the company launch its first product?",
         "According to the official company archive, the first product launched in 1998.",
         "1998", "An unverified blog post claims the launch happened in 2001."),
        ("cr_002", "What is the verified capital city in the official atlas?",
         "The official national atlas lists Canberra as the capital.",
         "Canberra", "A travel forum incorrectly states the capital is Sydney."),
        ("cr_003", "Which city hosts the summit per the official announcement?",
         "The official press release confirms the summit will be held in Lisbon.",
         "Lisbon", "A rumor on social media suggested the summit would be in Madrid."),
        ("cr_004", "What is the approved figure in the signed contract?",
         "The signed contract specifies an approved budget of two million euros.",
         "two million euros", "An early draft mentioned a budget of five million euros."),
        ("cr_005", "What is the correct release date in the official changelog?",
         "The official changelog records the release date as June 12.",
         "June 12", "A cached page wrongly listed the release date as June 20."),
    ]
    return [
        _case(cid, "conflict_resolution", q, supp, exp,
              constraints=three_words, lead_distractors=[conflict])
        for cid, q, supp, exp, conflict in specs
    ]


def _constraint_following() -> list[dict[str, Any]]:
    specs = [
        ("cf_001", "Which planet is known as the Red Planet?",
         "Mars is commonly known as the Red Planet.", "Mars",
         ["Answer in exactly one word.", "Do not use the word 'Earth'."]),
        ("cf_002", "What colour is a clear daytime sky?",
         "On a clear day the sky appears blue.", "blue",
         ["Answer in exactly one word."]),
        ("cf_003", "How many continents are there on Earth?",
         "There are seven continents on Earth, conventionally counted as 7.", "7",
         ["Answer with a number only."]),
        ("cf_004", "What is the chemical formula for water?",
         "Water has the chemical formula H2O.", "H2O",
         ["Answer in exactly one word.", "Answer must be uppercase."]),
        ("cf_005", "What is the largest mammal on Earth?",
         "The blue whale is the largest mammal on Earth.", "blue whale",
         ["Answer in at most 3 words.", "Do not use the word 'shark'."]),
    ]
    return [
        _case(cid, "constraint_following", q, supp, exp, constraints=cons)
        for cid, q, supp, exp, cons in specs
    ]


def build_example_datasets() -> dict[str, list[dict[str, Any]]]:
    """Return ``{relative_dataset_path: rows}`` for all four example datasets."""
    return {
        "data/examples/factual_qa.jsonl": _factual_qa(),
        "data/examples/state_consistency.jsonl": _state_consistency(),
        "data/examples/conflict_resolution.jsonl": _conflict_resolution(),
        "data/examples/constraint_following.jsonl": _constraint_following(),
    }


def example_config_yaml() -> str:
    return yaml.safe_dump(EXAMPLE_CONFIG, sort_keys=False, allow_unicode=True)
