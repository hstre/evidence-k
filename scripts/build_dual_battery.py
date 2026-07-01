#!/usr/bin/env python3
"""Generate the dual-instrumented TASK BATTERY under ``data/dual_battery/``.

The headline dual-axis results rest on one task family; this battery varies the *task
type* (factual / multi-hop / state-tracking / conflict-resolution / constraint-following)
while holding the contamination surface constant — every supporting fragment is written in
the same esoteric register, and every answer is a single token. So correctness difficulty
changes across tasks, the contamination axis does not, and a k-sweep shows whether the
evidence-saturation point k\\* moves with task type.

All content is original to this work. Answers are single tokens (digit<->word tolerant);
supporting fragments carry the answer inside register-laden prose so engaging the framing
is unavoidable (the contamination axis).
"""

from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: E501 — intentionally long prose data strings.

TARGET = 14

REGISTER = [
    "Through attunement and authentic presence we co-create a regulating force in the circle.",
    "This is the work of holding space: a felt sense of awakening woven into the fabric of your existence.",
    "We move past spiritual bypass into open-hearted presence, witnessing the state of being beneath.",
    "The practice is a gentle harvesting of attunement, returning resonance to the field between us.",
    "Each gathering is a co-creation: field-aware presence meeting the regulating force of breath.",
    "In witnessing we soften, and the harvest of field-awareness becomes a fundamental shift.",
]
NOOP = [
    "The attendance sheet was photocopied twice.",
    "The kettle in the corner was new that month.",
    "Parking was available behind the building.",
    "The window blinds were left half open.",
    "A spare set of chairs was stacked by the door.",
    "The sign-in clipboard had a blue pen attached.",
    "The floor had been mopped the evening before.",
]


def _build_evidence(cores: list[tuple[str, str]]) -> list[dict]:
    """Place core fragments (supporting/adversarial) at the front ranks, then fill with
    register + noop distractors up to TARGET."""
    evidence = [{"rank": i + 1, "text": t, "label": lab} for i, (t, lab) in enumerate(cores)]
    ri = ni = 0
    rank = len(cores) + 1
    while len(evidence) < TARGET:
        if rank % 2 == 0 and ri < len(REGISTER):
            evidence.append({"rank": rank, "text": REGISTER[ri], "label": "register"})
            ri += 1
        elif ni < len(NOOP):
            evidence.append({"rank": rank, "text": NOOP[ni], "label": "noop"})
            ni += 1
        elif ri < len(REGISTER):
            evidence.append({"rank": rank, "text": REGISTER[ri], "label": "register"})
            ri += 1
        else:
            break
        rank += 1
    return evidence


def case(task: str, cid: str, question: str, cores: list[tuple[str, str]],
         expected: str, constraints: list[str] | None = None) -> dict:
    return {
        "id": cid,
        "task": task,
        "question": question,
        "evidence": _build_evidence(cores),
        "expected_answer": expected,
        "constraints": constraints or [],
    }


def factual() -> list[dict]:
    t = "dual_factual"
    S = "supporting"
    specs = [
        ("df1", "How many minutes does the dawn practice last (in words)?",
         "Each dawn the harvest of field-awareness begins: ten minutes of breath, a resonance attuning the fabric of your existence.", "ten"),
        ("df2", "How many souls gather in the circle (in words)?",
         "Twelve souls gather in open-hearted presence, co-creating the regulating force of the circle.", "twelve"),
        ("df3", "At what hour does the circle begin (in words)?",
         "At six the felt sense awakens and the circle steps into authentic presence.", "six"),
        ("df4", "How many weeks does the journey run (in words)?",
         "Across eight weeks of co-creation the awakening deepens into a fundamental shift.", "eight"),
        ("df5", "In which room does the circle meet (in words)?",
         "In room three the resonance gathers, an open-hearted presence attuned to the field.", "three"),
        ("df6", "How many breaths open the ritual (in words)?",
         "Five breaths open the ritual, a gentle harvesting of attunement returning resonance to the field.", "five"),
    ]
    return [case(t, cid, q, [(sup, S)], ans) for cid, q, sup, ans in specs]


def multihop() -> list[dict]:
    t = "dual_multihop"
    S = "supporting"
    specs = [
        ("mh1", "How many souls gather in Marina's circle (in words)?",
         "Marina holds space for the evening circle, her attunement guiding the harvest of field-awareness.",
         "The evening circle gathers twelve souls in open-hearted co-creation.", "twelve"),
        ("mh2", "At what hour does the dawn practice begin (in words)?",
         "The dawn practice is kept in the eastern room, a resonance attuning the field.",
         "In the eastern room the felt sense awakens at six.", "six"),
        ("mh3", "How many weeks does Aria's journey run (in words)?",
         "Aria tends the spring cohort, witnessing the state of being beneath the noise.",
         "The spring cohort deepens across eight weeks of open-hearted presence.", "eight"),
        ("mh4", "How many breaths does the closing hold (in words)?",
         "The closing rite belongs to the western circle, a harvest of attunement.",
         "The western circle holds four breaths at its close, a regulating force.", "four"),
        ("mh5", "In which room does Nadia hold space (in words)?",
         "Nadia carries the noon gathering, her authentic presence a felt sense of awakening.",
         "The noon gathering meets in room seven, resonance attuning the fabric of being.", "seven"),
        ("mh6", "How many candles light Theo's vigil (in words)?",
         "Theo keeps the solstice vigil, co-creating the regulating force of the night.",
         "The solstice vigil is lit by nine candles, a harvest of field-awareness.", "nine"),
    ]
    return [case(t, cid, q, [(s1, S), (s2, S)], ans) for cid, q, s1, s2, ans in specs]


def state() -> list[dict]:
    t = "dual_state"
    S = "supporting"
    # rank-1 initial value, rank-2 update; the answer is the UPDATED (current) value.
    specs = [
        ("st1", "At what hour does the circle now begin (in words)?",
         "The circle first gathered at eight, an open-hearted presence attuning the field.",
         "Then the circle moved its start to six, the felt sense awakening earlier.", "six"),
        ("st2", "How many souls remain in the circle now (in words)?",
         "Twelve souls first gathered in co-creation, a harvest of field-awareness.",
         "After a departure, ten souls now remain, holding space in resonance.", "ten"),
        ("st3", "How many minutes does the practice now last (in words)?",
         "The practice began as ten minutes of breath, a harvest of attunement.",
         "It was lengthened, and now runs fifteen minutes, resonance attuning existence.", "fifteen"),
        ("st4", "In which room does the circle now meet (in words)?",
         "The circle met in room three, an authentic presence witnessing the field.",
         "It has relocated, and now meets in room five, the regulating force following.", "five"),
        ("st5", "At what hour does the vigil now end (in words)?",
         "The vigil once ended at nine, a witnessing of the state of being.",
         "Now it ends at seven, the felt sense settling into stillness.", "seven"),
        ("st6", "How many candles are now lit (in words)?",
         "The rite began with four candles, a gentle harvest of light.",
         "Two more were added, so eight candles are now lit in co-creation.", "eight"),
    ]
    return [case(t, cid, q, [(s1, S), (s2, S)], ans) for cid, q, s1, s2, ans in specs]


def conflict() -> list[dict]:
    t = "dual_conflict"
    # rank-1 authoritative source (correct), rank-2 adversarial (wrong). Question names the
    # authoritative source, so grounding must beat the plausible distractor.
    specs = [
        ("cf1", "According to the keeper's own record, how many souls gather (in words)?",
         "The keeper's own record notes twelve souls gathering in open-hearted co-creation.",
         "A visitor's rumour claimed twenty souls, though the resonance says otherwise.", "twelve"),
        ("cf2", "Per the founding scroll, at what hour does the circle begin (in words)?",
         "The founding scroll sets the circle at six, the felt sense awakening.",
         "A later pamphlet mistakenly printed eight, breaking the attunement.", "six"),
        ("cf3", "By the elder's count, how many weeks is the journey (in words)?",
         "The elder's count holds the journey at eight weeks of awakening.",
         "A newcomer guessed twelve weeks, misreading the harvest.", "eight"),
        ("cf4", "According to the room ledger, which room hosts the circle (in words)?",
         "The room ledger assigns room three, resonance attuning the field.",
         "A hallway sign wrongly pointed to room nine.", "three"),
        ("cf5", "Per the ritual master, how many breaths open the rite (in words)?",
         "The ritual master fixes five breaths to open the rite, a harvest of attunement.",
         "An apprentice miscounted seven breaths in the felt sense of the moment.", "five"),
        ("cf6", "By the true tally, how many candles are lit (in words)?",
         "The true tally records nine candles, a regulating force of light.",
         "A distracted guest reported four candles, losing the attunement.", "nine"),
    ]
    return [case(t, cid, q, [(auth, "supporting"), (adv, "adversarial")], ans)
            for cid, q, auth, adv, ans in specs]


def constraint() -> list[dict]:
    t = "dual_constraint"
    S = "supporting"
    # constraints compatible with the dual task's "one or two sentences" prompt:
    # do-not-use (also a contamination probe) and at-most-N-words.
    specs = [
        ("ct1", "How many minutes does the practice last (in words)?",
         "Each dawn the harvest of field-awareness begins: ten minutes of breath.", "ten",
         ["Do not use the word 'harvest'."]),
        ("ct2", "How many souls gather in the circle (in words)?",
         "Twelve souls gather in open-hearted presence, co-creating the regulating force.", "twelve",
         ["Do not use the word 'resonance'."]),
        ("ct3", "At what hour does the circle begin (in words)?",
         "At six the felt sense awakens into authentic presence and attunement.", "six",
         ["Answer in at most 30 words."]),
        ("ct4", "How many weeks does the journey run (in words)?",
         "Across eight weeks the awakening deepens into a fundamental shift of being.", "eight",
         ["Do not use the word 'awakening'."]),
        ("ct5", "In which room does the circle meet (in words)?",
         "In room three the resonance gathers, an open-hearted presence attuned to the field.", "three",
         ["Do not use the word 'field'."]),
        ("ct6", "How many breaths open the ritual (in words)?",
         "Five breaths open the ritual, a gentle attunement returning presence to the circle.", "five",
         ["Answer in at most 25 words."]),
    ]
    return [case(t, cid, q, [(sup, S)], ans, cons) for cid, q, sup, ans, cons in specs]


BUILDERS = {
    "factual": factual, "multihop": multihop, "state": state,
    "conflict": conflict, "constraint": constraint,
}


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "dual_battery"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        rows = fn()
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {path} ({len(rows)} cases, {len(rows[0]['evidence'])} fragments each)")


if __name__ == "__main__":
    main()
