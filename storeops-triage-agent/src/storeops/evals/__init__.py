"""Evaluation utilities for StoreOps triage scenarios."""

from storeops.evals.deterministic import DeterministicEvaluator, EvalCaseResult
from storeops.evals.datasets import GoldenCase, load_golden_cases

__all__ = [
    "DeterministicEvaluator",
    "EvalCaseResult",
    "GoldenCase",
    "load_golden_cases",
]

