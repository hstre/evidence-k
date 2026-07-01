"""Cross-domain probe tasks.

Thin aliases of :class:`FactualQATask` with distinct names, one per domain, so a single
config can sweep several domains and keep their results in *separate* buckets (results are
keyed by task name). Behaviour is identical to ``factual_qa`` — only the label differs — so
the domain is the only varied factor when comparing k\\* across these tasks.
"""

from __future__ import annotations

from .factual_qa import FactualQATask


class TechnicalQATask(FactualQATask):
    name = "domain_technical"


class MedicalQATask(FactualQATask):
    name = "domain_medical"


class LegalQATask(FactualQATask):
    name = "domain_legal"


class FinanceQATask(FactualQATask):
    name = "domain_finance"
