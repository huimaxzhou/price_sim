"""Routine 400-path covariance smoke evidence."""

import dataclasses
from collections.abc import Callable

from factor_sim.engine import simulate


def test_complete_covariance_diagnostics_pass_at_routine_scale(
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

    assert checks["factor_innovation_covariance"]["status"] == "pass"
    assert checks["specific_innovation_covariance"]["status"] == "pass"
    assert checks["modeled_asset_covariance"]["status"] == "pass"
