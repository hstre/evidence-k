"""Epistemic-contamination metrics — the *second axis* of Evidence-k.

`markers.py` and `metrics.py` are **vendored verbatim** from the upstream
DESi `context_contamination` benchmark (`src/desi/context_contamination/`) — treat them as
vendored, do not refactor casually. They are deterministic, offline, closed-marker
heuristics (framing leakage, register drift, attribution collapse, role adoption, loops),
which is exactly why they fit Evidence-k's LLM-free scoring philosophy.

Evidence-k uses these to score the *contamination axis* of a model response, in parallel
with the correctness axis (`runners/scorer.py`). The whole point of the dual-instrumented
benchmark is to run both axes on the same input and the same k, and contrast where each
puts the optimum — see `tasks/dual_instrumented.py` and `outputs/dual_report.py`.

Provenance: vendored from hstre/DESi context_contamination (markers.py, metrics.py).
"""

from . import markers, metrics
from .metrics import score_response, score_run

__all__ = ["markers", "metrics", "score_response", "score_run"]
