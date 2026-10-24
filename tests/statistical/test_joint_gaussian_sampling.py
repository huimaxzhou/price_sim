"""Fixed-seed statistical evidence for the joint Gaussian sampler."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytestmark = pytest.mark.statistical

PATHS = 2_000
STEPS = 32


def test_joint_gaussian_preserves_marginals_and_zero_cross_covariance(
    sample_factory: Callable[..., object],
) -> None:
    factor_target = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_target = np.diag([0.02, 0.03, 0.01])
    batch = sample_factory(
        joint=True,
        path_count=PATHS,
        step_count=STEPS,
        factor_covariance=factor_target,
        specific_variances=np.diag(specific_target),
        seed=2_717,
    )
    factor = batch.factor_innovations.reshape(-1, 2)
    specific = batch.specific_innovations.reshape(-1, 3)
    count = factor.shape[0]

    np.testing.assert_allclose(
        np.cov(factor, rowvar=False), factor_target, rtol=0.025, atol=0.001
    )
    np.testing.assert_allclose(
        np.cov(specific, rowvar=False), specific_target, rtol=0.025, atol=0.001
    )
    cross = np.cov(factor.T, specific.T, ddof=1)[:2, 2:]
    cross_se = np.sqrt(
        np.diag(factor_target)[:, None]
        * np.diag(specific_target)[None, :]
        / (count - 1)
    )
    assert np.all(np.abs(cross) <= 5.0 * cross_se + 1e-12)


def test_joint_and_independent_gaussian_policies_are_distributionally_consistent(
    sample_factory: Callable[..., object],
) -> None:
    independent = sample_factory(path_count=PATHS, step_count=STEPS, seed=4_669)
    joint = sample_factory(joint=True, path_count=PATHS, step_count=STEPS, seed=4_669)
    independent_all = np.concatenate(
        (
            independent.factor_innovations.reshape(-1, 2),
            independent.specific_innovations.reshape(-1, 3),
        ),
        axis=1,
    )
    joint_all = np.concatenate(
        (
            joint.factor_innovations.reshape(-1, 2),
            joint.specific_innovations.reshape(-1, 3),
        ),
        axis=1,
    )
    count = independent_all.shape[0]
    target_variances = np.array([0.04, 0.09, 0.02, 0.03, 0.0])
    mean_standard_error = np.sqrt(2.0 * target_variances / count)
    assert np.all(
        np.abs(independent_all.mean(axis=0) - joint_all.mean(axis=0))
        <= 5.0 * mean_standard_error + 1e-12
    )
    np.testing.assert_allclose(
        np.cov(independent_all, rowvar=False),
        np.cov(joint_all, rowvar=False),
        rtol=0.05,
        atol=0.0015,
    )
    assert (
        independent.sampler_metadata["stream_policy"]
        != joint.sampler_metadata["stream_policy"]
    )
