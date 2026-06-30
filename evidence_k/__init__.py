"""Evidence-k: measure how much evidence an LLM can use before context turns into noise.

The package estimates ``k*`` — the practical evidence-saturation point for a given
``(model, task, evidence_format)`` triple — and exports a ``k_profile.json`` that
routers, RAG systems and governance layers can load.

Evidence-k is intentionally standalone: it has no dependency on DESi or any other
in-house system. It only *produces* a profile those systems may later consume.
"""

__version__ = "0.1.0"

PROFILE_VERSION = "0.1"

__all__ = ["__version__", "PROFILE_VERSION"]
