"""Cost helpers for deterministic mode."""

from storeops.observability.trace import EstimatedCost


def zero_cost() -> EstimatedCost:
    return EstimatedCost()


__all__ = ["zero_cost"]
