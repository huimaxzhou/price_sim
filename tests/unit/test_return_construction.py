"""Focused tests for Packet C return construction."""

import numpy as np
import pytest

from factor_sim.engine import (
    construct_factor_returns,
    construct_modeled_returns,
    construct_specific_returns,
)
from factor_sim.errors import NumericalError


def test_return_construction_matches_factor_equation() -> None:
    exposures = np.array([[1.0, 0.5], [-0.25, 2.0]])
    factor = construct_factor_returns(
        np.array([[0.01, -0.02]]),
        np.array([[[0.03, 0.04]], [[-0.01, 0.02]]]),
    )
    specific = construct_specific_returns(
        np.array([[0.001, 0.002]]),
        np.zeros((2, 1, 2)),
    )

    modeled = construct_modeled_returns(exposures, factor, specific)

    np.testing.assert_array_equal(
        modeled,
        np.einsum("mhk,nk->mhn", factor, exposures) + specific,
    )


def test_return_helpers_do_not_alias_inputs() -> None:
    means = np.zeros((2, 1))
    innovations = np.ones((3, 2, 1))

    result = construct_factor_returns(means, innovations)

    assert result.flags.owndata
    assert result is not innovations
    np.testing.assert_array_equal(innovations, 1.0)


def test_return_shape_mismatch_is_numerical_error() -> None:
    with pytest.raises(NumericalError, match="shape"):
        construct_specific_returns(np.zeros((2, 1)), np.zeros((1, 3, 1)))
