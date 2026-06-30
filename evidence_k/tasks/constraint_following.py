"""Constraint following: answer correctly *and* obey machine-checkable constraints."""

from __future__ import annotations

from .base import Task


class ConstraintFollowingTask(Task):
    name = "constraint_following"
