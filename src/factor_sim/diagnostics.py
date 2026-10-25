"""Structured deterministic and Monte Carlo diagnostics for simulation results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import JointDistributionPolicy
from .errors import NumericalError
from .models import SimulationResult, ValidationReport

FloatArray = NDArray[np.float64]


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check(
    name: str,
    category: str,
    *,
    target: object,
    estimate: object,
    sample_size: int | None,
    rule: Mapping[str, object],
    status: str,
    message: str,
) -> dict[str, object]:
    return {
        "name": name,
        "category": category,
        "target": _json_value(target),
        "estimate": _json_value(estimate),
        "sample_size": sample_size,
        "rule": _json_value(rule),
        "status": status,
        "message": message,
    }


def _equality_check(
    name: str,
    passed: bool,
    *,
    target: object,
    estimate: object,
    message: str,
) -> dict[str, object]:
    return _check(
        name,
        "deterministic",
        target=target,
        estimate=estimate,
        sample_size=None,
        rule={"method": "exact_equality"},
        status="pass" if passed else "fail",
        message=message,
    )


def _allclose_check(
    name: str,
    actual: FloatArray,
    expected: FloatArray,
    tolerance: float,
    message: str,
) -> dict[str, object]:
    compatible = actual.shape == expected.shape
    maximum_error = (
        float(np.max(np.abs(actual - expected))) if compatible and actual.size else 0.0
    )
    passed = (
        compatible and bool(np.isfinite(actual).all()) and maximum_error <= tolerance
    )
    return _check(
        name,
        "deterministic",
        target={"shape": list(expected.shape), "maximum_absolute_error": tolerance},
        estimate={"shape": list(actual.shape), "maximum_absolute_error": maximum_error},
        sample_size=None,
        rule={"method": "absolute_error", "maximum_absolute_error": tolerance},
        status="pass" if passed else "fail",
        message=message,
    )


def _sample_covariance(samples: FloatArray) -> FloatArray:
    if samples.ndim != 2:
        raise NumericalError("statistical samples must be a two-dimensional matrix")
    if samples.shape[0] < 2:
        raise NumericalError("at least two observations are required for covariance")
    return np.atleast_2d(
        np.asarray(np.cov(samples, rowvar=False, ddof=1), dtype=np.float64)
    )


def _mean_check(
    name: str, samples: FloatArray, target_covariance: FloatArray
) -> dict[str, object]:
    sample_size = samples.shape[0]
    estimate = np.mean(samples, axis=0)
    threshold = 5.0 * np.sqrt(np.diag(target_covariance) / sample_size) + 1e-12
    passed = bool(np.all(np.abs(estimate) <= threshold))
    return _check(
        name,
        "statistical",
        target=np.zeros_like(estimate),
        estimate=estimate,
        sample_size=sample_size,
        rule={"method": "five_standard_errors", "maximum_absolute_error": threshold},
        status="pass" if passed else "fail",
        message="Empirical innovation means are compared with their zero targets.",
    )


def _covariance_check(
    name: str, samples: FloatArray, target: FloatArray
) -> dict[str, object]:
    sample_size = samples.shape[0]
    if sample_size < 2:
        return _check(
            name,
            "statistical",
            target=target,
            estimate=None,
            sample_size=sample_size,
            rule={"method": "five_standard_errors"},
            status="not_applicable",
            message="Covariance requires at least two observations.",
        )
    estimate = _sample_covariance(samples)
    variance_products = np.outer(np.diag(target), np.diag(target))
    threshold = 5.0 * np.sqrt((target**2 + variance_products) / (sample_size - 1))
    threshold += 1e-12
    passed = bool(np.all(np.abs(estimate - target) <= threshold))
    return _check(
        name,
        "statistical",
        target=target,
        estimate=estimate,
        sample_size=sample_size,
        rule={"method": "five_standard_errors", "maximum_absolute_error": threshold},
        status="pass" if passed else "fail",
        message="Empirical covariance is compared with the configured target.",
    )


def _cross_covariance_check(result: SimulationResult) -> dict[str, object]:
    factors = result.factor_innovations.reshape(-1, result.factor_innovations.shape[2])
    specifics = result.specific_innovations.reshape(
        -1, result.specific_innovations.shape[2]
    )
    sample_size = factors.shape[0]
    if sample_size < 2:
        return _check(
            "factor_specific_cross_covariance",
            "statistical",
            target=0.0,
            estimate=None,
            sample_size=sample_size,
            rule={"method": "five_standard_errors"},
            status="not_applicable",
            message="Cross-covariance requires at least two observations.",
        )
    centered_factors = factors - np.mean(factors, axis=0)
    centered_specifics = specifics - np.mean(specifics, axis=0)
    estimate = centered_factors.T @ centered_specifics / (sample_size - 1)
    factor_variances = np.diag(result.risk_state.factor_covariance)
    threshold = 5.0 * np.sqrt(
        np.outer(factor_variances, result.risk_state.specific_variances)
        / (sample_size - 1)
    )
    threshold += 1e-12
    passed = bool(np.all(np.abs(estimate) <= threshold))
    return _check(
        "factor_specific_cross_covariance",
        "statistical",
        target=0.0,
        estimate=estimate,
        sample_size=sample_size,
        rule={"method": "five_standard_errors", "maximum_absolute_error": threshold},
        status="pass" if passed else "fail",
        message="Factor-specific cross-covariance is compared with zero.",
    )


def _correlation_check(
    name: str, samples: FloatArray, target_covariance: FloatArray
) -> dict[str, object]:
    sample_size, width = samples.shape
    if sample_size < 2:
        return _check(
            name,
            "statistical",
            target=None,
            estimate=None,
            sample_size=sample_size,
            rule={"method": "five_standard_errors_plus_margin"},
            status="not_applicable",
            message="Correlation requires at least two observations.",
        )
    covariance = _sample_covariance(samples)
    sample_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    target_deviations = np.sqrt(np.maximum(np.diag(target_covariance), 0.0))
    estimate: list[list[float | None]] = []
    target: list[list[float | None]] = []
    undefined = False
    defined_errors: list[float] = []
    for row in range(width):
        values: list[float | None] = []
        targets: list[float | None] = []
        for column in range(width):
            target_denominator = target_deviations[row] * target_deviations[column]
            sample_denominator = sample_deviations[row] * sample_deviations[column]
            if target_denominator == 0.0 or sample_denominator == 0.0:
                values.append(None)
                targets.append(None)
                undefined = True
            else:
                sample_value = float(covariance[row, column] / sample_denominator)
                target_value = float(
                    target_covariance[row, column] / target_denominator
                )
                values.append(sample_value)
                targets.append(target_value)
                defined_errors.append(abs(sample_value - target_value))
        estimate.append(values)
        target.append(targets)
    threshold = 5.0 / np.sqrt(sample_size) + 0.01
    passed = all(error <= threshold for error in defined_errors)
    status = "fail" if not passed else "not_applicable" if undefined else "pass"
    return _check(
        name,
        "statistical",
        target=target,
        estimate=estimate,
        sample_size=sample_size,
        rule={
            "method": "five_standard_errors_plus_margin",
            "maximum_absolute_error": threshold,
            "undefined_when_zero_variance": True,
        },
        status=status,
        message=(
            "Correlation contains undefined entries because a component has "
            "zero variance."
            if undefined
            else "All correlation entries are mathematically defined."
        ),
    )


def _volatility_check(
    name: str, samples: FloatArray, target_covariance: FloatArray
) -> dict[str, object]:
    sample_size = samples.shape[0]
    if sample_size < 2:
        return _check(
            name,
            "statistical",
            target=np.sqrt(np.diag(target_covariance)),
            estimate=None,
            sample_size=sample_size,
            rule={"method": "five_standard_errors_on_variance"},
            status="not_applicable",
            message="Marginal volatility requires at least two observations.",
        )
    sample_variances = np.diag(_sample_covariance(samples))
    target_variances = np.diag(target_covariance)
    threshold = 5.0 * np.sqrt(2.0 * target_variances**2 / (sample_size - 1)) + 1e-12
    passed = bool(np.all(np.abs(sample_variances - target_variances) <= threshold))
    return _check(
        name,
        "statistical",
        target=np.sqrt(target_variances),
        estimate=np.sqrt(np.maximum(sample_variances, 0.0)),
        sample_size=sample_size,
        rule={
            "method": "five_standard_errors_on_variance",
            "maximum_variance_error": threshold,
        },
        status="pass" if passed else "fail",
        message="Empirical marginal volatility is checked through sample variance.",
    )


def _lag_one_check(name: str, values: FloatArray) -> dict[str, object]:
    path_count, step_count, width = values.shape
    if path_count < 2 or step_count < 2:
        return _check(
            name,
            "statistical",
            target=0.0,
            estimate=None,
            sample_size=path_count,
            rule={"method": "five_standard_errors_plus_margin"},
            status="not_applicable",
            message="Lag-one correlation requires at least two paths and dates.",
        )
    threshold = 5.0 / np.sqrt(path_count) + 0.01
    estimates: list[float | None] = []
    for index in range(width):
        left = values[:, 0, index]
        right = values[:, 1, index]
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            estimates.append(None)
        else:
            estimates.append(float(np.corrcoef(left, right)[0, 1]))
    defined = [value for value in estimates if value is not None]
    passed = all(abs(value) <= threshold for value in defined)
    return _check(
        name,
        "statistical",
        target=0.0,
        estimate=estimates,
        sample_size=path_count,
        rule={
            "method": "five_standard_errors_plus_margin",
            "maximum_absolute_error": threshold,
        },
        status="pass" if passed else "fail",
        message="Defined lag-one correlations are compared with the no-serial target.",
    )


def _independence_check(result: SimulationResult) -> dict[str, object]:
    factors = result.factor_innovations.reshape(-1, result.factor_innovations.shape[2])
    specifics = result.specific_innovations.reshape(
        -1, result.specific_innovations.shape[2]
    )
    factor_energy = np.sum(factors**2, axis=1)
    specific_energy = np.sum(specifics**2, axis=1)
    if np.std(factor_energy) == 0.0 or np.std(specific_energy) == 0.0:
        squared_dependence = None
    else:
        squared_dependence = float(np.corrcoef(factor_energy, specific_energy)[0, 1])
    construction = (
        "separate_rng_streams"
        if result.resolved_policy.return_joint_distribution
        == JointDistributionPolicy.INDEPENDENT
        else "block_diagonal_joint_generator"
    )
    threshold = 5.0 / np.sqrt(factors.shape[0]) + 0.01
    passed = squared_dependence is None or abs(squared_dependence) <= threshold
    return _check(
        "independent_block_evidence",
        "statistical",
        target={"factor_specific_cross_covariance": 0.0},
        estimate={
            "generator_construction": construction,
            "squared_return_dependence": squared_dependence,
        },
        sample_size=factors.shape[0],
        rule={
            "method": "construction_plus_distribution_sensitive_check",
            "maximum_absolute_error": threshold,
        },
        status="pass" if passed else "fail",
        message=(
            "Generator construction is primary evidence; zero covariance alone is "
            "not proof of independence."
        ),
    )


def evaluate_result(result: SimulationResult) -> ValidationReport:
    """Return complete diagnostics without modifying ``result``."""

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if result.initial_prices is None:
        raise NumericalError("complete diagnostics require retained initial prices")
    path_count, step_count, factor_count = result.factor_returns.shape
    asset_count = result.risk_state.exposures.shape[0]
    equation_tolerance = float(result.manifest["numerical_tolerances"]["equation"])
    expected_modeled = (
        np.einsum("mhk,nk->mhn", result.factor_returns, result.risk_state.exposures)
        + result.specific_returns
    )
    checks = [
        _equality_check(
            "array_shapes",
            result.factor_innovations.shape == (path_count, step_count, factor_count)
            and result.specific_innovations.shape
            == (path_count, step_count, asset_count)
            and result.modeled_returns.shape == (path_count, step_count, asset_count)
            and result.prices.shape == (path_count, step_count + 1, asset_count),
            target="canonical M,H,N,K axis shapes",
            estimate={
                "factor_innovations": list(result.factor_innovations.shape),
                "specific_innovations": list(result.specific_innovations.shape),
                "modeled_returns": list(result.modeled_returns.shape),
                "prices": list(result.prices.shape),
            },
            message="Result arrays use the declared axis order.",
        ),
        _allclose_check(
            "modeled_return_equation",
            result.modeled_returns,
            expected_modeled,
            equation_tolerance,
            "Modeled returns equal X f plus specific returns.",
        ),
        _allclose_check(
            "raw_identity_mapping",
            result.price_link_returns,
            result.modeled_returns,
            equation_tolerance,
            "Raw price-link returns equal modeled returns.",
        ),
        _equality_check(
            "live_price_validity",
            bool(np.isfinite(result.prices).all() and np.all(result.prices > 0.0)),
            target="all finite and strictly positive",
            estimate=bool(
                np.isfinite(result.prices).all() and np.all(result.prices > 0.0)
            ),
            message="All version-one assets remain live and require positive prices.",
        ),
        _allclose_check(
            "initial_price_preservation",
            result.prices[:, 0, :],
            np.broadcast_to(result.initial_prices, (path_count, asset_count)),
            0.0,
            "The first price slice preserves supplied initial prices.",
        ),
        _equality_check(
            "none_handler_asset_state",
            not result.asset_state_events
            and result.resolved_policy.corporate_action_adjustment is None
            and result.resolved_policy.new_listing is None,
            target={"events": 0, "handlers": None},
            estimate={"events": len(result.asset_state_events)},
            message="None handlers emit no asset-state events.",
        ),
        _equality_check(
            "identifier_alignment",
            result.asset_ids == result.risk_state.asset_ids
            and result.factor_ids == result.risk_state.factor_ids,
            target={
                "assets": result.risk_state.asset_ids,
                "factors": result.risk_state.factor_ids,
            },
            estimate={"assets": result.asset_ids, "factors": result.factor_ids},
            message="Result identifiers remain aligned with the frozen risk state.",
        ),
        _equality_check(
            "currency_alignment",
            len(result.asset_currencies) == asset_count
            and tuple(result.manifest["asset_currencies"]) == result.asset_currencies,
            target=result.manifest["asset_currencies"],
            estimate=result.asset_currencies,
            message="Currencies remain aligned with ordered asset identifiers.",
        ),
        _equality_check(
            "date_alignment",
            len(result.future_dates) == step_count
            and tuple(result.manifest["schedule"]["resolved_dates"])
            == result.future_dates,
            target=result.manifest["schedule"]["resolved_dates"],
            estimate=result.future_dates,
            message="Future dates remain aligned with the date axis.",
        ),
        _equality_check(
            "finite_returns",
            all(
                np.isfinite(array).all()
                for array in (
                    result.factor_returns,
                    result.specific_returns,
                    result.modeled_returns,
                    result.price_link_returns,
                )
            ),
            target=True,
            estimate=True,
            message="All emitted return arrays are finite.",
        ),
        _equality_check(
            "risk_state_and_policy_metadata",
            bool(result.manifest.get("risk_state_frozen"))
            and "resolved_policy" in result.manifest
            and "sampler_diagnostics" in result.manifest,
            target=True,
            estimate=bool(result.manifest.get("risk_state_frozen")),
            message=(
                "Frozen-state, policy, sampler, and canonicalization metadata "
                "are recorded."
            ),
        ),
    ]
    factor_samples = result.factor_innovations.reshape(-1, factor_count)
    specific_samples = result.specific_innovations.reshape(-1, asset_count)
    asset_innovations = (
        np.einsum("mhk,nk->mhn", result.factor_innovations, result.risk_state.exposures)
        + result.specific_innovations
    ).reshape(-1, asset_count)
    checks.extend(
        [
            _mean_check(
                "factor_innovation_mean",
                factor_samples,
                result.risk_state.factor_covariance,
            ),
            _covariance_check(
                "factor_innovation_covariance",
                factor_samples,
                result.risk_state.factor_covariance,
            ),
            _volatility_check(
                "factor_marginal_volatility",
                factor_samples,
                result.risk_state.factor_covariance,
            ),
            _mean_check(
                "specific_innovation_mean",
                specific_samples,
                np.diag(result.risk_state.specific_variances),
            ),
            _covariance_check(
                "specific_innovation_covariance",
                specific_samples,
                np.diag(result.risk_state.specific_variances),
            ),
            _volatility_check(
                "specific_marginal_volatility",
                specific_samples,
                np.diag(result.risk_state.specific_variances),
            ),
            _cross_covariance_check(result),
            _covariance_check(
                "modeled_asset_covariance",
                asset_innovations,
                result.target_asset_covariance,
            ),
            _volatility_check(
                "modeled_asset_marginal_volatility",
                asset_innovations,
                result.target_asset_covariance,
            ),
            _correlation_check(
                "specific_correlation",
                specific_samples,
                np.diag(result.risk_state.specific_variances),
            ),
            _correlation_check(
                "modeled_asset_correlation",
                asset_innovations,
                result.target_asset_covariance,
            ),
            _lag_one_check("factor_lag_one_correlation", result.factor_innovations),
            _lag_one_check("specific_lag_one_correlation", result.specific_innovations),
            _independence_check(result),
        ]
    )
    overall_status = (
        "fail" if any(item["status"] == "fail" for item in checks) else "pass"
    )
    summaries: dict[str, Any] = dict(result.validation_report.summaries)
    summaries.update(
        {
            "diagnostic_scope": "complete",
            "statistical_diagnostics_included": True,
            "overall_status": overall_status,
            "checks": tuple(checks),
        }
    )
    return ValidationReport(issues=(), summaries=summaries)
