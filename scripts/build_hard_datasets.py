#!/usr/bin/env python3
"""Generate the *hard* Evidence-k datasets under ``data/hard/``.

Design goal: induce a real evidence-saturation drop (k* > 1) even for a capable model.
Two ideas, borrowed from DESi's GSM-Symbolic-P2 clause taxonomy (noop / load_bearing /
adversarial):

* **Obscure / fictional entities** so the model cannot fall back on parametric knowledge
  and must rely on the supplied evidence.
* **Adversarial distractors** — fragments that state a *plausible but wrong* answer to the
  same question in an authoritative tone (conflicting years, near-duplicate entities,
  superseded values). As more of these enter the window, the model is increasingly pulled
  off the correct answer. ``noop`` distractors are harmless filler.

Layout per case: rank 1 = supporting (correct), then adversarial and noop fragments
interleaved up to 14 items, so k=13 and "full" still differ.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET = 14

# Harmless filler facts about nothing in particular (noop distractors).
NOOP_POOL = [
    "The document was archived in triplicate.",
    "The record was transcribed by a junior clerk.",
    "The file was stored on the third shelf.",
    "The seal used green wax that season.",
    "The ledger was bound in blue cloth.",
    "The margin notes were written in pencil.",
    "The cover page bore a faded watermark.",
    "The appendix listed unrelated correspondence.",
    "The index was compiled two years later.",
    "The footnotes referenced a lost volume.",
]


def _interleave(supporting: str, adversarial: list[str], noop: list[str]) -> list[dict]:
    evidence = [{"rank": 1, "text": supporting, "label": "supporting"}]
    pool_noop = list(noop) + [n for n in NOOP_POOL if n not in noop]
    ai = ni = 0
    rank = 2
    # Alternate adversarial / noop, leading with adversarial so the trap appears early.
    while len(evidence) < TARGET:
        if ai < len(adversarial) and (len(evidence) % 2 == 1):
            evidence.append({"rank": rank, "text": adversarial[ai], "label": "adversarial"})
            ai += 1
        elif ni < len(pool_noop):
            evidence.append({"rank": rank, "text": pool_noop[ni], "label": "noop"})
            ni += 1
        elif ai < len(adversarial):
            evidence.append({"rank": rank, "text": adversarial[ai], "label": "adversarial"})
            ai += 1
        else:
            break
        rank += 1
    return evidence


def case(cid, task, question, supporting, expected, adversarial, constraints,
         noop=None, extra=None):
    row = {
        "id": cid,
        "task": task,
        "question": question,
        "evidence": _interleave(supporting, adversarial, noop or []),
        "expected_answer": expected,
        "constraints": constraints,
    }
    if extra:
        row.update(extra)
    return row


N3 = ["Answer in at most 3 words."]
NUM = ["Answer with a number only."]


def factual_qa():
    return [
        case("hfqa_001", "factual_qa",
             "In what year was the Zephyrian Concord ratified?",
             "The Zephyrian Concord was ratified in 1387.", "1387",
             ["Some chronicles date the Zephyrian Concord to 1359.",
              "The Zephyrian Concord was first drafted in 1402.",
              "The Second Zephyrian Concord was ratified in 1421.",
              "A later reprint mistakenly listed the Concord as 1378."], NUM),
        case("hfqa_002", "factual_qa",
             "What is the capital of the kingdom of Valdoria?",
             "The capital of Valdoria is Tessmar.", "Tessmar",
             ["The largest city of Valdoria is Brunholt.",
              "Valdoria's former capital was Aldreth.",
              "The neighbouring kingdom Valdaria has capital Tessen.",
              "The Valdorian court often met in Brunholt."], N3),
        case("hfqa_003", "factual_qa",
             "How many moons does the planet Orvex have?",
             "The planet Orvex has 7 moons.", "7",
             ["The planet Orvix has 9 moons.",
              "Orvex's largest moon hosts 4 research stations.",
              "Early surveys of Orvex counted 5 moons.",
              "The gas giant Orvex-B has 12 moons."], NUM),
        case("hfqa_004", "factual_qa",
             "Who composed the opera 'Marenval'?",
             "The opera 'Marenval' was composed by Edda Lorne.", "Edda Lorne",
             ["The libretto of 'Marenval' was written by Pier Vasch.",
              "The opera 'Marenvale' was composed by Otto Lerne.",
              "Edda Lorne conducted but did not write 'Sarenval'.",
              "'Marenval' premiered under conductor Hals Krume."], N3),
        case("hfqa_005", "factual_qa",
             "What is the boiling point of the fictional liquid quellar, in degrees?",
             "Quellar boils at 63 degrees.", "63",
             ["Quellar freezes at 18 degrees.",
              "The related liquid quellan boils at 81 degrees.",
              "Under pressure quellar boils at 92 degrees.",
              "Impure quellar was once measured at 57 degrees."], NUM),
        case("hfqa_006", "factual_qa",
             "Which river runs through the city of Karneth?",
             "The river Vos runs through Karneth.", "Vos",
             ["The river Vossa runs through Karnith, not Karneth.",
              "The canal Drel borders the city of Karneth.",
              "Karneth's old harbour sat on the river Tamm.",
              "The Vos delta lies far south of Karneth."], N3),
        case("hfqa_007", "factual_qa",
             "What is the population of the town of Eldsmoor, in thousands?",
             "Eldsmoor has a population of 24 thousand.", "24",
             ["Nearby Eldsmere has a population of 42 thousand.",
              "Eldsmoor's population was 19 thousand a decade ago.",
              "The Eldsmoor district totals 31 thousand.",
              "A tourist estimate put Eldsmoor at 28 thousand."], NUM),
        case("hfqa_008", "factual_qa",
             "What mineral is mined at the Korreth deposit?",
             "The Korreth deposit yields vexite.", "vexite",
             ["The Korrath deposit yields vexine.",
              "Korreth also has traces of dolomite and quartz.",
              "Historically Korreth was mined for tellurite.",
              "The vexite refinery sits near the Korrath deposit."], N3),
    ]


def conflict_resolution():
    return [
        case("hcr_001", "conflict_resolution",
             "What was the verified annual revenue of Halcyon Ltd in 2024?",
             "The official 2024 audit confirms Halcyon Ltd revenue of 4.2 million.",
             "4.2 million",
             ["A widely-shared news article reported Halcyon Ltd revenue of 4.8 million.",
              "An analyst note estimated Halcyon Ltd revenue at 3.9 million.",
              "A leaked draft mentioned 5.1 million for Halcyon Ltd.",
              "Social media claimed Halcyon Ltd made 4.5 million."], N3),
        case("hcr_002", "conflict_resolution",
             "Per the official ruling, who won the 2023 Merrowford election?",
             "The official electoral commission declared Pell the winner of Merrowford 2023.",
             "Pell",
             ["Early exit polls projected Quist as the Merrowford winner.",
              "A newspaper initially called the race for Rensohn.",
              "A recount rumour suggested Quist had actually won.",
              "Quist's campaign claimed victory on election night."], N3),
        case("hcr_003", "conflict_resolution",
             "What is the certified summit height of Mount Aubrec, in metres?",
             "The national survey certifies Mount Aubrec at 3120 metres.", "3120",
             ["An old almanac lists Mount Aubrec at 3080 metres.",
              "A hiking blog states Mount Aubrec is 3210 metres.",
              "GPS hobbyists reported 3095 metres for Aubrec.",
              "The nearby peak Aubrac is 3120... no, 2980 metres."], NUM),
        case("hcr_004", "conflict_resolution",
             "According to the signed contract, what is the delivery date?",
             "The signed contract states the delivery date is March 9.", "March 9",
             ["An email thread suggested delivery by March 2.",
              "The original proposal listed March 15 as delivery.",
              "A vendor reminder mentioned March 6 informally.",
              "A spreadsheet cell showed March 19 in error."], N3),
        case("hcr_005", "conflict_resolution",
             "What is the officially recorded founding year of Greycastle Abbey?",
             "The abbey's own charter records Greycastle Abbey's founding in 1146.", "1146",
             ["A guidebook claims Greycastle Abbey was founded in 1162.",
              "Local legend dates Greycastle Abbey to 1109.",
              "A plaque on site reads 1151 for the founding.",
              "An encyclopedia entry says circa 1140."], NUM),
        case("hcr_006", "conflict_resolution",
             "Per the lab's certified result, what is the sample's pH?",
             "The certified laboratory result records the sample pH as 6.4.", "6.4",
             ["A field test strip suggested a pH of 5.9.",
              "An intern's notebook recorded pH 7.1.",
              "A second uncalibrated meter read 6.8.",
              "The shipping label guessed pH around 6.0."], NUM),
    ]


def state_consistency():
    return [
        case("hsc_001", "state_consistency",
             "What is the order's current status right now?",
             "Latest update (today, 14:00): the order status is now DELIVERED.",
             "DELIVERED",
             ["Yesterday the order status was OUT_FOR_DELIVERY.",
              "Last week the order status was PROCESSING.",
              "At order creation the status was PENDING.",
              "A cached page still showed SHIPPED this morning."], N3,
             extra={"stale_values": ["OUT_FOR_DELIVERY", "PROCESSING", "PENDING", "SHIPPED"]}),
        case("hsc_002", "state_consistency",
             "What is the patient's current prescribed dose?",
             "Most recent note (today): the dose was changed to 50 mg.", "50 mg",
             ["Two days ago the dose was 75 mg.",
              "On admission the dose was 100 mg.",
              "An older chart entry lists 25 mg.",
              "A draft order had suggested 60 mg."], N3,
             extra={"stale_values": ["75 mg", "100 mg", "25 mg", "60 mg"]}),
        case("hsc_003", "state_consistency",
             "Which room is the meeting in now, after the latest change?",
             "Final update: the meeting has been moved to Room C3.", "C3",
             ["Earlier today it was set for Room A1.",
              "The original booking was Room B2.",
              "A calendar invite still shows Room A1.",
              "The lobby screen lists Room D4 by mistake."], N3,
             extra={"stale_values": ["A1", "B2", "D4"]}),
        case("hsc_004", "state_consistency",
             "What is the feature flag's current value in production?",
             "Deploy log (latest): the flag is now set to ENABLED in production.",
             "ENABLED",
             ["The previous release had the flag DISABLED.",
              "Staging still has the flag set to DISABLED.",
              "An old runbook says the flag defaults to OFF.",
              "A rollback last month set it to DISABLED."], N3,
             extra={"stale_values": ["DISABLED", "OFF"]}),
        case("hsc_005", "state_consistency",
             "What is the account's current tier after the most recent change?",
             "Most recent billing event: the account was upgraded to ENTERPRISE.",
             "ENTERPRISE",
             ["Last quarter the account was on PRO.",
              "The signup tier was FREE.",
              "A trial briefly placed it on PRO_TRIAL.",
              "An invoice draft still referenced PRO."], N3,
             extra={"stale_values": ["PRO", "FREE", "PRO_TRIAL"]}),
        case("hsc_006", "state_consistency",
             "Which warehouse currently holds the inventory after the transfer?",
             "Transfer completed today: inventory now sits in Warehouse North.",
             "Warehouse North",
             ["Before the transfer it was in Warehouse South.",
              "The manifest was printed for Warehouse South.",
              "An older record lists Warehouse East.",
              "A label still reads Warehouse South."], N3,
             extra={"stale_values": ["Warehouse South", "Warehouse East"]}),
    ]


def constraint_following():
    return [
        case("hcf_001", "constraint_following",
             "Which fictional element has atomic number 26 in the Orvex table?",
             "In the Orvex table, atomic number 26 is the element vexium.", "vexium",
             ["Vexium is described as a lustrous metal used in alloys.",
              "Atomic number 26 corresponds to the metal vexium (Vx).",
              "Vexium is often confused with the metal vexine."],
             ["Answer in exactly one word.", "Do not use the word 'metal'."]),
        case("hcf_002", "constraint_following",
             "What is the access code for vault 7, as a number?",
             "The access code for vault 7 is 4821.", "4821",
             ["The access code for vault 9 is 4812.",
              "An old code for vault 7 was 4281.",
              "The backup code is written as four-eight-two-one."], NUM),
        case("hcf_003", "constraint_following",
             "Name the capital of the realm of Sunmar.",
             "The capital of Sunmar is Halvoth.", "Halvoth",
             ["The capital of Sunmer is Halvoth, with an extra letter.",
              "Sunmar's largest port city is Drennan.",
              "The royal seat once stood at Old Halvoth."],
             ["Answer in exactly one word.", "Do not use the word 'city'."]),
        case("hcf_004", "constraint_following",
             "What is the chemical-style code for the compound 'brak'?",
             "The compound brak has the code KR2.", "KR2",
             ["The compound brek has the code KR3.",
              "Brak is sometimes written lowercase as kr2 in old texts.",
              "The related compound brak-oxide is KRO."],
             ["Answer in exactly one word.", "Answer must be uppercase."]),
        case("hcf_005", "constraint_following",
             "How many districts are in the city of Lumen? Give the count.",
             "The city of Lumen has 5 districts.", "5",
             ["The city of Lumin has 8 districts.",
              "Lumen had 3 districts before the last reform.",
              "The greater Lumen metro spans 11 districts."], NUM),
        case("hcf_006", "constraint_following",
             "Give the surname of the founder of Vexline Corp.",
             "Vexline Corp was founded by Mara Dolch.", "Dolch",
             ["Vexine Corp was founded by Mara Dölch with an umlaut.",
              "Mara Dolch later founded the rival firm Dolchworks.",
              "The co-founder of Vexline Corp was Ren Dolph."],
             ["Answer in exactly one word.", "Do not use the word 'founder'."]),
    ]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "hard"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "factual_qa.jsonl": factual_qa(),
        "conflict_resolution.jsonl": conflict_resolution(),
        "state_consistency.jsonl": state_consistency(),
        "constraint_following.jsonl": constraint_following(),
    }
    for fname, rows in datasets.items():
        path = out_dir / fname
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {path} ({len(rows)} cases, {len(rows[0]['evidence'])} evidence each)")


if __name__ == "__main__":
    main()
