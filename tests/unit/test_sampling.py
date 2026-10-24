"""Independent Gaussian sampler tests for implementation-plan Step 6."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from factor_sim.config import JointDistributionPolicy
from factor_sim.errors import ConfigurationError
from factor_sim.sampling import build_random_streams, sample_innovations

pytestmark = pytest.mark.unit


def test_independent_sampler_shapes_zero_variance_and_input_ownership(
    sample_factory: Callable[..., object],
) -> None:
    factor_covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
    specific_variances = np.array([0.02, 0.03, 0.0])
    factor_snapshot = factor_covariance.copy()
    specific_snapshot = specific_variances.copy()

    batch = sample_factory(
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
    )

    assert batch.factor_innovations.shape == (8, 4, 2)
    assert batch.specific_innovations.shape == (8, 4, 3)
    assert batch.factor_innovations.dtype == np.float64
    assert batch.specific_innovations.dtype == np.float64
    assert batch.factor_innovations.flags.owndata
    assert batch.specific_innovations.flags.owndata
    assert not batch.factor_innovations.flags.writeable
    assert not batch.specific_innovations.flags.writeable
    np.testing.assert_array_equal(batch.specific_innovations[..., 2], 0.0)
    np.testing.assert_array_equal(factor_covariance, factor_snapshot)
    np.testing.assert_array_equal(specific_variances, specific_snapshot)
    assert batch.sampler_metadata["sampler_name"] == "independent_gaussian"


def test_identical_seed_reproduces_complete_independent_batch(
    sample_factory: Callable[..., object],
) -> None:
    first = sample_factory(seed=19)
    second = sample_factory(seed=19)
    np.testing.assert_array_equal(first.factor_innovations, second.factor_innovations)
    np.testing.assert_array_equal(
        first.specific_innovations, second.specific_innovations
    )


def test_dispatch_rejects_unsupported_distribution_before_drawing(
    policy_factory: Callable[..., object],
) -> None:
    streams = build_random_streams(
        seed=5,
        path_count=1,
        joint_policy=JointDistributionPolicy.INDEPENDENT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    before = copy.deepcopy(streams.factor_generators[0].bit_generator.state)
    unsupported = dataclasses.replace(
        policy_factory(), factor_return_distribution="unsupported"
    )

    with pytest.raises(ConfigurationError) as captured:
        sample_innovations(
            resolved_policy=unsupported,
            factor_covariance=np.eye(1),
            specific_variances=np.ones(1),
            step_count=1,
            rng_streams=streams,
            symmetry_tolerance=1e-12,
            positive_semidefinite_tolerance=1e-12,
        )

    assert captured.value.issues[0].code == "unsupported_policy_value"
    assert streams.factor_generators[0].bit_generator.state == before


def test_typed_dispatch_selects_independent_registry_entry(
    policy_factory: Callable[..., object],
) -> None:
    streams = build_random_streams(
        seed=5,
        path_count=2,
        joint_policy=JointDistributionPolicy.INDEPENDENT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    batch = sample_innovations(
        resolved_policy=policy_factory(),
        factor_covariance=np.array([[0.04]]),
        specific_variances=np.array([0.01]),
        step_count=3,
        rng_streams=streams,
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )
    assert batch.sampler_metadata["sampler_name"] == "independent_gaussian"
