from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class ResamplingResult:
    observed_diff: float
    ci_low: float
    ci_high: float
    p_value: float | None
    samples: np.ndarray


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict[str, float]:
    p_a = success_a / n_a
    p_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    unpooled_se = sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_b - p_a
    return {
        "rate_a": p_a,
        "rate_b": p_b,
        "diff": diff,
        "relative_diff": p_b / p_a - 1,
        "z": z,
        "p_value": p_value,
        "ci_low": diff - 1.96 * unpooled_se,
        "ci_high": diff + 1.96 * unpooled_se,
    }


def bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 5000,
    seed: int = 42,
    chunk_size: int = 100,
) -> ResamplingResult:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    samples = np.empty(iterations, dtype=float)
    position = 0
    while position < iterations:
        chunk = min(chunk_size, iterations - position)
        a_idx = rng.integers(0, len(a), size=(chunk, len(a)))
        b_idx = rng.integers(0, len(b), size=(chunk, len(b)))
        samples[position : position + chunk] = b[b_idx].mean(axis=1) - a[a_idx].mean(axis=1)
        position += chunk
    observed = float(b.mean() - a.mean())
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    p_value = float(2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return ResamplingResult(observed, float(ci_low), float(ci_high), min(p_value, 1.0), samples)


def bootstrap_relative_lift(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 5000,
    seed: int = 42,
    chunk_size: int = 100,
) -> ResamplingResult:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    samples = np.empty(iterations, dtype=float)
    position = 0
    while position < iterations:
        chunk = min(chunk_size, iterations - position)
        a_idx = rng.integers(0, len(a), size=(chunk, len(a)))
        b_idx = rng.integers(0, len(b), size=(chunk, len(b)))
        a_means = a[a_idx].mean(axis=1)
        b_means = b[b_idx].mean(axis=1)
        samples[position : position + chunk] = b_means / a_means - 1
        position += chunk
    observed = float(b.mean() / a.mean() - 1)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    p_value = float(2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return ResamplingResult(observed, float(ci_low), float(ci_high), min(p_value, 1.0), samples)


def permutation_test_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 5000,
    seed: int = 42,
) -> ResamplingResult:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float(b.mean() - a.mean())
    combined = np.concatenate([a, b])
    n_a = len(a)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        permuted = rng.permutation(combined)
        samples[i] = permuted[n_a:].mean() - permuted[:n_a].mean()
    p_value = float((np.abs(samples) >= abs(observed)).mean())
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return ResamplingResult(observed, float(ci_low), float(ci_high), p_value, samples)
