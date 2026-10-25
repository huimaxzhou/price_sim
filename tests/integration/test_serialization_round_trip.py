"""End-to-end persistence round-trip test."""

from collections.abc import Callable

import numpy as np

from factor_sim.engine import simulate
from factor_sim.io import load_result, save_result


def test_saved_result_round_trips(
    request_factory: Callable[..., object], tmp_path: object
) -> None:
    original = simulate(request_factory())
    restored = load_result(save_result(original, tmp_path / "run"))

    np.testing.assert_array_equal(restored.prices, original.prices)
    np.testing.assert_array_equal(
        restored.risk_state.exposures, original.risk_state.exposures
    )
    np.testing.assert_array_equal(restored.factor_means, original.factor_means)
    assert restored.validation_report == original.validation_report
