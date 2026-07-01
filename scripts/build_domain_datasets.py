#!/usr/bin/env python3
"""Generate a small cross-DOMAIN probe under ``data/domains/`` (factual_qa).

Goal: hold the *task structure* fixed (single-token factual answer + plausible-but-wrong
adversarial distractors + neutral noise, ~14 fragments) and vary only the **domain**
(technical / medical / legal / finance), so a k-sweep shows whether the evidence-saturation
point k\\* is stable across domains or moves with them. Entities are fictional so the model
must rely on the supplied evidence, and every answer is a number so scoring is unambiguous.

This is a deliberately rough probe (few cases, small models): enough to see whether the
domain axis moves k\\* at all, not a calibrated per-domain estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: E501 — intentionally long prose data strings.

TARGET = 14  # fragments per case (matches the hard factual_qa datasets)

# domain -> list of (id, question, answer, supporting, [adversarial, ...])
CASES: dict[str, list[tuple]] = {
    "technical": [
        ("t1", "On which port does the Kesterly gateway listen (number only)?", "8730",
         "The Kesterly gateway listens on port 8730.", ["8703", "8370", "8720"]),
        ("t2", "How many retries does the Varnex client attempt (number only)?", "5",
         "The Varnex client attempts 5 retries before failing.", ["3", "7", "4"]),
        ("t3", "What is the Orlon cache TTL in seconds (number only)?", "240",
         "The Orlon cache TTL is 240 seconds.", ["120", "360", "200"]),
        ("t4", "How many shards does the Pellucid index use (number only)?", "16",
         "The Pellucid index is split across 16 shards.", ["8", "32", "12"]),
        ("t5", "At which firmware major version was the Talvane sensor certified (number only)?", "7",
         "The Talvane sensor was certified at firmware major version 7.", ["5", "9", "6"]),
        ("t6", "What is the Cindral request timeout in milliseconds (number only)?", "500",
         "The Cindral request timeout is 500 milliseconds.", ["250", "750", "400"]),
        ("t7", "How many CPU cores does the Marrow build node have (number only)?", "12",
         "The Marrow build node has 12 CPU cores.", ["8", "16", "10"]),
    ],
    "medical": [
        ("m1", "What dose of Veltricine was administered in milligrams (number only)?", "40",
         "A 40 milligram dose of Veltricine was administered.", ["20", "60", "45"]),
        ("m2", "How many days was the Harrowmoor protocol prescribed (number only)?", "14",
         "The Harrowmoor protocol was prescribed for 14 days.", ["7", "21", "10"]),
        ("m3", "At what heart rate was the Ostry index recorded in bpm (number only)?", "88",
         "The Ostry index was recorded at a heart rate of 88 bpm.", ["80", "96", "84"]),
        ("m4", "What was the Calvex marker in milligrams per litre (number only)?", "6",
         "The Calvex marker measured 6 milligrams per litre.", ["3", "9", "5"]),
        ("m5", "How many hours were there between Ferndale infusions (number only)?", "8",
         "Ferndale infusions were spaced 8 hours apart.", ["6", "12", "4"]),
        ("m6", "What was the Threndle platelet count in thousands (number only)?", "150",
         "The Threndle platelet count was 150 thousand per microlitre.", ["90", "210", "120"]),
        ("m7", "How many weeks gestation at the Marlowe scan (number only)?", "32",
         "The Marlowe scan was performed at 32 weeks gestation.", ["28", "36", "30"]),
    ],
    "legal": [
        ("l1", "In which article of the Brenmark Accord is arbitration defined (number only)?", "12",
         "Arbitration is defined in Article 12 of the Brenmark Accord.", ["9", "15", "11"]),
        ("l2", "How many days notice does the Fenwick clause require (number only)?", "30",
         "The Fenwick clause requires 30 days notice.", ["14", "60", "21"]),
        ("l3", "What is the Aldworth penalty in thousands of dollars (number only)?", "50",
         "The Aldworth penalty is set at 50 thousand dollars.", ["25", "75", "40"]),
        ("l4", "How many signatories ratified the Corvane Treaty (number only)?", "7",
         "The Corvane Treaty was ratified by 7 signatories.", ["5", "9", "6"]),
        ("l5", "In which section is the Pelham indemnity found (number only)?", "4",
         "The Pelham indemnity is found in Section 4.", ["3", "6", "5"]),
        ("l6", "How many years is the Wracombe non-compete (number only)?", "2",
         "The Wracombe non-compete runs for 2 years.", ["1", "3", "5"]),
        ("l7", "What is the Danforth filing fee in dollars (number only)?", "300",
         "The Danforth filing fee is 300 dollars.", ["150", "450", "250"]),
    ],
    "finance": [
        ("f1", "What was Halvern Ltd's 2024 revenue in millions (number only)?", "42",
         "The audited 2024 revenue of Halvern Ltd was 42 million.", ["48", "39", "45"]),
        ("f2", "How many basis points did the Corliss note yield (number only)?", "320",
         "The Corliss note yielded 320 basis points.", ["280", "360", "300"]),
        ("f3", "What is the Trenmoor dividend in cents (number only)?", "18",
         "The Trenmoor dividend is 18 cents per share.", ["12", "24", "15"]),
        ("f4", "How many shares in the Velmont buyback in millions (number only)?", "5",
         "The Velmont buyback covered 5 million shares.", ["3", "8", "4"]),
        ("f5", "What was the Ashbury default rate in percent (number only)?", "6",
         "The Ashbury portfolio default rate was 6 percent.", ["3", "9", "5"]),
        ("f6", "At what price did Grenwold IPO in dollars (number only)?", "27",
         "Grenwold priced its IPO at 27 dollars per share.", ["22", "32", "25"]),
        ("f7", "How many months is the Palverton bond term (number only)?", "36",
         "The Palverton bond has a term of 36 months.", ["24", "48", "30"]),
    ],
}

NOOPS: dict[str, list[str]] = {
    "technical": [
        "The runbook was last edited on a Tuesday.", "The staging cluster is in a different region.",
        "The config file uses two-space indentation.", "The dashboard theme was set to dark mode.",
        "The repository has a green build badge.", "The on-call rotation is weekly.",
        "The logo was updated last quarter.", "The changelog is written in Markdown.",
        "The office wifi was rebooted overnight.", "The mascot is a small blue fox.",
    ],
    "medical": [
        "The ward was repainted last spring.", "The chart was filed in a blue folder.",
        "The clinic closes early on Fridays.", "The waiting room had new chairs.",
        "The nurse's badge was laminated.", "The corridor lights were replaced.",
        "The pharmacy is on the second floor.", "The kettle in the break room was new.",
        "The parking lot was resurfaced.", "The noticeboard had a fire-drill poster.",
    ],
    "legal": [
        "The deed was bound in green ribbon.", "The clerk used a fountain pen.",
        "The hearing room had oak panelling.", "The archive is kept in the basement.",
        "The seal was pressed in red wax.", "The transcript was double-spaced.",
        "The library subscribes to two journals.", "The corridor carpet was burgundy.",
        "The filing cabinet was locked at night.", "The reception desk had a brass bell.",
    ],
    "finance": [
        "The quarterly report used a blue cover.", "The trading desk faces the window.",
        "The auditor travelled by train.", "The spreadsheet had frozen header rows.",
        "The boardroom clock ran two minutes fast.", "The coffee machine was replaced.",
        "The annual meeting was held offsite.", "The lobby displayed a stock ticker.",
        "The stapler on the desk was red.", "The elevator music was instrumental.",
    ],
}


def build_case(domain: str, cid: str, question: str, answer: str,
               supporting: str, adversarials: list[str]) -> dict:
    evidence = [{"rank": 1, "text": supporting, "label": "supporting"}]
    noops = NOOPS[domain]
    ai = ni = 0
    rank = 2
    # interleave adversarial then noise, adversarials first so they sit in early k
    while len(evidence) < TARGET:
        if ai < len(adversarials) and rank % 2 == 0:
            evidence.append({"rank": rank, "text": adversarials[ai], "label": "adversarial"})
            ai += 1
        elif ni < len(noops):
            evidence.append({"rank": rank, "text": noops[ni], "label": "noop"})
            ni += 1
        elif ai < len(adversarials):
            evidence.append({"rank": rank, "text": adversarials[ai], "label": "adversarial"})
            ai += 1
        else:
            break
        rank += 1
    return {
        "id": f"{domain}_{cid}",
        "task": "factual_qa",
        "question": question,
        "evidence": evidence,
        "expected_answer": answer,
        "constraints": ["Answer with a number only."],
    }


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "domains"
    out.mkdir(parents=True, exist_ok=True)
    for domain, specs in CASES.items():
        rows = [build_case(domain, *s) for s in specs]
        path = out / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {path} ({len(rows)} cases, {len(rows[0]['evidence'])} fragments each)")


if __name__ == "__main__":
    main()
