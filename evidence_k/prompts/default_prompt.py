"""The default, provider-neutral prompt template.

The format is intentionally simple and stable so that (a) real models get a clean,
unambiguous instruction and (b) the offline simulator can rely on the structure.
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PREAMBLE = (
    "You are a careful assistant. Answer the question using ONLY the evidence provided. "
    "If the evidence does not contain the answer, say you do not know. "
    "Be concise and do not invent facts."
)


def build_system(constraints: Sequence[str]) -> str:
    parts = [SYSTEM_PREAMBLE]
    if constraints:
        parts.append("\nYou must obey these constraints:")
        for c in constraints:
            parts.append(f"- {c}")
    return "\n".join(parts)


def build_user(question: str, evidence_texts: Sequence[str]) -> str:
    lines = [f"Question: {question}", "", "Evidence:"]
    if evidence_texts:
        for i, text in enumerate(evidence_texts, start=1):
            lines.append(f"[{i}] {text}")
    else:
        lines.append("(no evidence provided)")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)
