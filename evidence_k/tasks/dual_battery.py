"""Dual-instrumented task-battery aliases.

Thin aliases of :class:`DualInstrumentedTask` with distinct names, one per *task type*, so a
single config can sweep several task types and keep their results in separate buckets while
scoring both axes (correctness + contamination) identically. Only the label and the dataset
differ; the task type is the varied factor.
"""

from __future__ import annotations

from .dual_instrumented import DualInstrumentedTask


class DualFactualTask(DualInstrumentedTask):
    name = "dual_factual"


class DualMultihopTask(DualInstrumentedTask):
    name = "dual_multihop"


class DualStateTask(DualInstrumentedTask):
    name = "dual_state"


class DualConflictTask(DualInstrumentedTask):
    name = "dual_conflict"


class DualConstraintTask(DualInstrumentedTask):
    name = "dual_constraint"
