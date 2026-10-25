"""Focused diagnostics tests independent of the acceptance contract."""

import dataclasses
from collections.abc import Callable

import numpy as np

from factor_sim.diagnostics import evaluate_result
from factor_sim.engine import simulate


def _safe_request(request_factory: Callable[..., object], *, paths: int = 2) -> object:
    request = request_factory(path_count=paths)
    return dataclasses.replace(
        request,
        risk_state=dataclasses.replace(
            request.risk_state,
            factor_covariance=request.risk_state.factor_covariance * 0.001,
            specific_variances=np.array([0.00002, 0.00003, 0.0]),
        ),
    )


def test_simulate_attaches_complete_diagnostics(
    request_factory: Callable[..., object],
) -> None:
    result = simulate(_safe_request(request_factory))
    checks = {
        item["name"]: item for item in result.validation_report.summaries["checks"]
    }

    assert result.validation_report.summaries["diagnostic_scope"] == "complete"
    assert checks["modeled_return_equation"]["status"] == "pass"
    assert checks["specific_correlation"]["status"] == "not_applicable"


def test_corrupted_return_is_reported_without_mutation(
    request_factory: Callable[..., object],
) -> None:
    result = simulate(_safe_request(request_factory))
    corrupted = np.array(result.modeled_returns, copy=True)
    corrupted[0, 0, 0] += 0.5
    altered = dataclasses.replace(result, modeled_returns=corrupted)

    report = evaluate_result(altered)
    checks = {item["name"]: item for item in report.summaries["checks"]}

    assert checks["modeled_return_equation"]["status"] == "fail"
    assert result.modeled_returns[0, 0, 0] != altered.modeled_returns[0, 0, 0]
