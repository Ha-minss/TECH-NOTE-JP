"""Metric helpers for evaluation summaries."""


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = ["ratio"]

