"""LLM-free scoring primitives.

Every dimension returns a float in ``[0, 1]`` where higher is better, except the
hallucination *rate* (higher is worse) which is applied as a penalty by
:func:`compute_reliability`. The defaults are deliberately simple and deterministic:
normalized matching, token overlap and rule-based constraint checks. An LLM judge can be
added later, but it is explicitly NOT a default dependency.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import ScoringConfig

_TOKEN = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and",
    "or", "for", "with", "it", "this", "that", "as", "by", "be", "from", "i", "do",
    "not", "no", "yes", "answer", "city", "name", "located", "only",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_text(text))


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning (stopwords removed)."""
    return [t for t in tokenize(text) if t not in _STOPWORDS]


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_DIGIT_WORDS = {v: k for k, v in _NUMBER_WORDS.items()}


def _expected_variants(ne: str) -> set[str]:
    """Acceptable surface forms of the expected answer (digit <-> word equivalence).

    Reasoning/flagship models often answer "10" where the gold is "ten" (or vice versa).
    Treating those as equivalent stops a pure word-match from systematically marking such
    models wrong — the gemini-2.5-pro failure mode in the dual-axis run.
    """
    variants = {ne}
    if ne in _NUMBER_WORDS:
        variants.add(_NUMBER_WORDS[ne])
    if ne in _DIGIT_WORDS:
        variants.add(_DIGIT_WORDS[ne])
    return variants


def score_correctness(answer: str, expected: str | None) -> float:
    """1.0 if any accepted form of the expected answer is present in the answer.

    Tolerates (a) the answer appearing anywhere in longer prose (token-boundary match) and
    (b) digit<->word equivalence for numbers, so verbose/reasoning models are scored fairly.
    """
    if not expected:
        return 1.0  # nothing to check against
    na = normalize_text(answer)
    ne = normalize_text(expected)
    if not ne:
        return 1.0
    for variant in _expected_variants(ne):
        if na == variant:
            return 1.0
        if re.search(rf"(?:^| ){re.escape(variant)}(?: |$)", na):
            return 1.0
    return 0.0


def score_grounding(answer: str, evidence_texts: Sequence[str]) -> float:
    """Fraction of the answer's content tokens that appear in the provided evidence."""
    ans_tokens = content_tokens(answer)
    if not ans_tokens:
        return 1.0  # an empty / pure-stopword answer makes no ungrounded claim
    evidence_blob = set()
    for t in evidence_texts:
        evidence_blob.update(content_tokens(t))
    if not evidence_blob:
        return 0.0  # there is no evidence to be grounded in
    hits = sum(1 for t in ans_tokens if t in evidence_blob)
    return hits / len(ans_tokens)


def hallucination_rate(
    answer: str, evidence_texts: Sequence[str], expected: str | None
) -> float:
    """Fraction of answer content tokens found in NEITHER evidence NOR expected answer."""
    ans_tokens = content_tokens(answer)
    if not ans_tokens:
        return 0.0
    known = set()
    for t in evidence_texts:
        known.update(content_tokens(t))
    if expected:
        known.update(content_tokens(expected))
    unknown = sum(1 for t in ans_tokens if t not in known)
    return unknown / len(ans_tokens)


_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")


def _check_one_constraint(answer: str, constraint: str) -> bool | None:
    """Return True/False if the constraint is machine-checkable, else None."""
    c = constraint.lower()
    na = normalize_text(answer)
    words = na.split()

    if any(p in c for p in ("one word", "single word", "exactly one word")):
        return len(words) == 1
    m = re.search(r"at most (\d+) words", c)
    if m:
        return len(words) <= int(m.group(1))
    if "uppercase" in c:
        letters = [ch for ch in answer if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters)
    if "lowercase" in c:
        letters = [ch for ch in answer if ch.isalpha()]
        return bool(letters) and all(ch.islower() for ch in letters)
    if any(p in c for p in ("a number", "number only", "digits only", "numeric")):
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", answer.strip()))
    if "yes or no" in c or "yes/no" in c:
        return na in {"yes", "no"}
    if any(p in c for p in ("do not use", "do not mention", "without using", "don't use")):
        targets = _QUOTED.findall(constraint)
        if not targets:
            # fall back to the last token of the constraint phrase
            tail = re.sub(r"[^\w\s]", "", c).split()
            targets = tail[-1:] if tail else []
        return all(normalize_text(t) not in na for t in targets if t)
    if "only the city name" in c or "city name only" in c:
        return len(words) <= 2
    return None


def score_constraints(answer: str, constraints: Sequence[str]) -> tuple[float, dict[str, bool]]:
    """Adherence = satisfied / checkable. Unverifiable constraints are ignored.

    Returns the adherence score and a per-constraint pass/fail map (only for those that
    could be checked).
    """
    details: dict[str, bool] = {}
    for c in constraints:
        result = _check_one_constraint(answer, c)
        if result is not None:
            details[c] = result
    if not details:
        return 1.0, details
    satisfied = sum(1 for v in details.values() if v)
    return satisfied / len(details), details


def compute_reliability(
    dims: dict[str, float],
    weights: dict[str, float],
    normalized_cost: float = 0.0,
) -> float:
    """Combine per-dimension scores into a single reliability value in ``[0, 1]``.

    ``reliability = Σ w_positive · dim  −  w_hallucination · halluc_rate  −  w_cost · cost``
    clamped to ``[0, 1]``.
    """
    score = 0.0
    for key in ScoringConfig.POSITIVE:
        score += weights.get(key, 0.0) * dims.get(key, 0.0)
    score -= weights.get("hallucination_penalty", 0.0) * dims.get("hallucination_rate", 0.0)
    score -= weights.get("cost_penalty", 0.0) * normalized_cost
    return max(0.0, min(1.0, score))
