"""RNG derivation tests for implementation-plan Step 5."""

from __future__ import annotations

import numpy as np
import pytest

from factor_sim.config import JointDistributionPolicy
from factor_sim.sampling import build_random_streams

pytestmark = pytest.mark.unit


def test_independent_stream_assignment_is_reproducible_and_separate() -> None:
    arguments = {
        "seed": 42,
        "path_count": 3,
        "joint_policy": JointDistributionPolicy.INDEPENDENT,
        "bit_generator_name": "PCG64",
        "stream_policy": "per_path",
    }
    first = build_random_streams(**arguments)
    second = build_random_streams(**arguments)

    first_factor = [rng.standard_normal(8) for rng in first.factor_generators]
    second_factor = [rng.standard_normal(8) for rng in second.factor_generators]
    for left, right in zip(first_factor, second_factor, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first_factor[0], first_factor[1])

    rebuilt = build_random_streams(**arguments)
    assert not np.array_equal(
        rebuilt.factor_generators[0].standard_normal(8),
        rebuilt.specific_generators[0].standard_normal(8),
    )
    assert rebuilt.metadata["stream_policy"] == (
        "SeedSequence -> path -> factor,specific"
    )
    assert rebuilt.metadata["bit_generator"] == "PCG64"


def test_joint_streams_use_one_stable_child_per_path() -> None:
    streams = build_random_streams(
        seed=42,
        path_count=3,
        joint_policy=JointDistributionPolicy.JOINT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )

    assert len(streams.joint_generators) == 3
    assert not streams.factor_generators
    assert not streams.specific_generators
    assert streams.metadata["stream_policy"] == "SeedSequence -> path -> joint"
    assert len({repr(item) for item in streams.metadata["path_streams"]}) == 3


def test_rng_construction_does_not_use_legacy_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def legacy_used(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("legacy global NumPy random state was used")

    monkeypatch.setattr(np.random, "seed", legacy_used)
    monkeypatch.setattr(np.random, "normal", legacy_used)
    streams = build_random_streams(
        seed=7,
        path_count=1,
        joint_policy=JointDistributionPolicy.INDEPENDENT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    assert streams.path_count == 1
