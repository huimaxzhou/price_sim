"""Routine no-serial-correlation evidence."""

import dataclasses
from collections.abc import Callable

from factor_sim.engine import simulate


def test_no_serial_policy_diagnostics_pass(
    request_factory: Callable[..., object],
) -> None:
    request = request_factory(path_count=400)
    request = dataclasses.replace(
        request,
        risk_state=dataclasses.replace(
            request.risk_state,
            factor_covariance=request.risk_state.factor_covariance * 0.001,
            specific_variances=request.risk_state.specific_variances * 0.001,
        ),
    )
    checks = {
        item["name"]: item
        for item in simulate(request).validation_report.summaries["checks"]
    }

    assert checks["factor_lag_one_correlation"]["status"] == "pass"
    assert checks["specific_lag_one_correlation"]["status"] == "pass"
