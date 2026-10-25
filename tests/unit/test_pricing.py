"""Focused tests for Packet C arithmetic pricing."""

import numpy as np
import pytest

from factor_sim.config import InvalidReturnPolicy
from factor_sim.errors import NumericalError, PolicyExecutionError
from factor_sim.pricing import evolve_arithmetic_prices


def _evolve(returns: np.ndarray) -> np.ndarray:
    return evolve_arithmetic_prices(
        np.array([100.0, 50.0]),
        returns,
        asset_ids=("a", "b"),
        future_dates=("2026-10-26", "2026-10-27"),
        invalid_return_policy=InvalidReturnPolicy.FAIL,
    )


def test_prices_compound_one_date_at_a_time() -> None:
    result = _evolve(np.array([[[0.1, -0.1], [0.2, 0.0]]]))

    np.testing.assert_allclose(
        result,
        np.array([[[100.0, 50.0], [110.0, 45.0], [132.0, 45.0]]]),
    )


def test_invalid_return_reports_stable_location() -> None:
    returns = np.zeros((2, 2, 2))
    returns[1, 0, 0] = -1.0

    with pytest.raises(PolicyExecutionError) as captured:
        _evolve(returns)

    assert captured.value.code == "invalid_arithmetic_return"
    assert captured.value.context["path_id"] == 1
    assert captured.value.context["asset_id"] == "a"


def test_price_overflow_is_contextual_numerical_error() -> None:
    with pytest.raises(NumericalError) as captured:
        evolve_arithmetic_prices(
            np.array([np.finfo(np.float64).max]),
            np.array([[[1.0]]]),
            asset_ids=("a",),
            future_dates=("2026-10-26",),
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )

    assert captured.value.code == "nonfinite_price"
    assert captured.value.context["future_date"] == "2026-10-26"
