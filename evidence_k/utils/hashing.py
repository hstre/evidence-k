"""Stable, salt-free hashing for reproducible runs.

Python's built-in ``hash()`` is randomised per process, so we use SHA-256 for any
value that must stay constant across runs (seeds, run ids).
"""

from __future__ import annotations

import hashlib
import random


def stable_hash(*parts: object) -> int:
    """Deterministic non-negative integer hash of the given parts."""
    joined = "␟".join(repr(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def seeded_random(*parts: object) -> random.Random:
    """A ``random.Random`` seeded deterministically from ``parts``."""
    return random.Random(stable_hash(*parts))


def short_hash(*parts: object, length: int = 10) -> str:
    """A short hex digest, handy for run ids and content fingerprints."""
    joined = "␟".join(repr(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
