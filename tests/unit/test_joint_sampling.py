"""Joint Gaussian sampler tests for implementation-plan Step 7."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from factor_sim.config import JointDistributionPolicy
from factor_sim.sampling import build_random_streams, sample_innovations

pytestmark = pytest.mark.unit


def test_joint_sampler_records_block_identity_and_owned_outputs(
    sample_factory: Callable[..., object],
) -> None:
    batch = sample_factory(joint=True)

    assert batch.factor_innovations.shape == (8, 4, 2)
    assert batch.specific_innovations.shape == (8, 4, 3)
    assert batch.factor_innovations.flags.owndata
    assert batch.specific_innovations.flags.owndata
    assert batch.sampler_metadata["sampler_name"] == "block_diagonal_gaussian"
    assert batch.sampler_metadata["sampler_version"] == "1"
    assert batch.sampler_metadata["block_dimensions"] == {
        "factor": 2,
        "specific": 3,
        "joint": 5,
    }
    assert batch.sampler_metadata["stream_policy"] == ("SeedSequence -> path -> joint")


def test_typed_dispatch_selects_registered_joint_sampler(
    policy_factory: Callable[..., object],
) -> None:
    streams = build_random_streams(
        seed=5,
        path_count=2,
        joint_policy=JointDistributionPolicy.JOINT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    batch = sample_innovations(
        resolved_policy=policy_factory(joint=True),
        factor_covariance=np.array([[0.04]]),
        specific_variances=np.array([0.01]),
        step_count=3,
        rng_streams=streams,
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )
    assert batch.sampler_metadata["sampler_name"] == "block_diagonal_gaussian"


def test_joint_sampler_supports_singular_blocks_and_zero_specific_variance(
    sample_factory: Callable[..., object],
) -> None:
    batch = sample_factory(
        joint=True,
        factor_covariance=np.array([[0.04, 0.04], [0.04, 0.04]]),
        specific_variances=np.array([0.02, 0.0, 0.01]),
    )
    np.testing.assert_array_equal(batch.specific_innovations[..., 1], 0.0)
