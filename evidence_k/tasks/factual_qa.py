"""Factual question answering: one supporting fact, optional distractors."""

from __future__ import annotations

from .base import Task


class FactualQATask(Task):
    name = "factual_qa"
