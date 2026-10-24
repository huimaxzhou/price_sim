"""Fixed-seed statistical evidence for the independent Gaussian sampler."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

pytestmark = pytest.mark.statistical

PATHS = 2_000
STEPS = 32


def _assert_covariance_within_five_se(samples: np.ndarray, target: np.ndarray) -> None:
    observed = np.cov(samples, rowvar=False, ddof=1)
    count = samples.shape[0]
    for row in range(target.shape[0]):
        for column in range(target.shape[1]):
            variance = (
                target[row, column] ** 2 + target[row, row] * target[column, column]
            ) / (count - 1)
            assert abs(observed[row, column] - target[row, column]) <= (
                5.0 * math.sqrt(variance) + 1e-12
            )


def test_independent_gaussian_moments_cross_covariance_and_serial_policy(
    sample_factory: Callable[..., object],
) -> None:
    factor_target = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_target = np.diag([0.02, 0.03, 0.01])
    batch = sample_factory(
        path_count=PATHS,
        step_count=STEPS,
        factor_covariance=factor_target,
        specific_variances=np.diag(specific_target),
        seed=8_149,
    )
    factor = batch.factor_innovations.reshape(-1, 2)
    specific = batch.specific_innovations.reshape(-1, 3)
    count = factor.shape[0]

    for column, variance in enumerate(np.diag(factor_target)):
        assert abs(float(factor[:, column].mean())) <= (
            5.0 * math.sqrt(variance / count) + 1e-12
        )
    for column, variance in enumerate(np.diag(specific_target)):
        assert abs(float(specific[:, column].mean())) <= (
            5.0 * math.sqrt(variance / count) + 1e-12
        )
    _assert_covariance_within_five_se(factor, factor_target)
    _assert_covariance_within_five_se(specific, specific_target)

    cross = np.cov(factor.T, specific.T, ddof=1)[:2, 2:]
    cross_se = np.sqrt(
        np.diag(factor_target)[:, None]
        * np.diag(specific_target)[None, :]
        / (count - 1)
    )
    assert np.all(np.abs(cross) <= 5.0 * cross_se + 1e-12)

    lag = np.corrcoef(
        batch.factor_innovations[:, :-1, 0].ravel(),
        batch.factor_innovations[:, 1:, 0].ravel(),
    )[0, 1]
    assert abs(lag) <= 5.0 / math.sqrt(PATHS * (STEPS - 1)) + 0.005
