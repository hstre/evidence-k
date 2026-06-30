"""Conflict resolution: evidence fragments disagree; pick the authoritative value.

The supporting fragment carries the correct (more recent / higher-authority) value; the
distractors carry conflicting values. The expected answer is the resolved value.
"""

from __future__ import annotations

from .base import Task


class ConflictResolutionTask(Task):
    name = "conflict_resolution"
