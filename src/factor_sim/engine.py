"""Readable return construction and top-level simulation orchestration."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import resolve_policy
from .errors import NumericalError
from .models import (
    JointGeneratorConfig,
    ResolvedSimulationPolicy,
    SimulationRequest,
    SimulationResult,
    ValidationReport,
)
from .pricing import (
    apply_corporate_action_policy,
    apply_currency_policy,
    apply_new_listing_policy,
    evolve_arithmetic_prices,
    map_modeled_returns,
)
from .sampling import build_random_streams, sample_innovations
from .validation import ValidatedContext, validate_request

FloatArray = NDArray[np.float64]


def _construct_returns(
    means: FloatArray, innovations: FloatArray, name: str
) -> FloatArray:
    if means.ndim != 2 or innovations.ndim != 3:
        raise NumericalError(
            f"{name} means and innovations must have shapes (H, W) and (M, H, W)"
        )
    if means.shape != innovations.shape[1:]:
        raise NumericalError(f"{name} means shape must match innovation date and width")
    result = np.array(innovations + means, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise NumericalError(f"{name} return construction produced nonfinite values")
    return result


def construct_factor_returns(
    factor_means: FloatArray, factor_innovations: FloatArray
) -> FloatArray:
    """Add ``(H,K)`` factor means to ``(M,H,K)`` innovations."""

    return _construct_returns(factor_means, factor_innovations, "factor")


def construct_specific_returns(
    specific_means: FloatArray, specific_innovations: FloatArray
) -> FloatArray:
    """Add ``(H,N)`` specific means to ``(M,H,N)`` innovations."""

    return _construct_returns(specific_means, specific_innovations, "specific")


def construct_modeled_returns(
    exposures: FloatArray,
    factor_returns: FloatArray,
    specific_returns: FloatArray,
) -> FloatArray:
    """Construct ``r[m,h,n] = X[n,k] f[m,h,k] + epsilon[m,h,n]``."""

    if exposures.ndim != 2 or factor_returns.ndim != 3 or specific_returns.ndim != 3:
        raise NumericalError(
            "modeled-return inputs must have shapes (N,K), (M,H,K), and (M,H,N)"
        )
    path_count, step_count, factor_count = factor_returns.shape
    asset_count, exposure_factor_count = exposures.shape
    if exposure_factor_count != factor_count:
        raise NumericalError("exposures shape must match the factor-return factor axis")
    if specific_returns.shape != (path_count, step_count, asset_count):
        raise NumericalError(
            "specific_returns shape must match factor paths/dates and exposure assets"
        )
    # m=path, h=future date, k=factor, n=asset.
    result = np.einsum("mhk,nk->mhn", factor_returns, exposures)
    result = np.array(result + specific_returns, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise NumericalError("modeled-return construction produced nonfinite values")
    return result


def _policy_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, JointGeneratorConfig):
        return {
            "name": value.name,
            "version": value.version,
            "parameters": value.parameters,
            "marginal_contract": value.marginal_contract,
            "random_stream_requirements": value.random_stream_requirements,
        }
    return value


def _resolved_policy_manifest(
    policy: ResolvedSimulationPolicy,
) -> dict[str, Mapping[str, object]]:
    names = (
        "factor_exposure_dynamics",
        "factor_volatility_dynamics",
        "factor_correlation_dynamics",
        "specific_risk_dynamics",
        "factor_return_distribution",
        "specific_return_distribution",
        "return_serial_correlation",
        "return_joint_distribution",
        "price_return_convention",
        "invalid_return_policy",
        "currency_handling",
        "corporate_action_adjustment",
        "new_listing",
        "joint_generator",
    )
    return {
        name: {
            "value": _policy_value(getattr(policy, name)),
            "value_source": policy.value_sources[name],
        }
        for name in names
    }


def _schedule_manifest(
    request: SimulationRequest, context: ValidatedContext
) -> dict[str, object]:
    schedule = request.schedule
    if schedule.explicit_future_dates is not None:
        method = "explicit_future_dates"
        supplied_value: object = schedule.explicit_future_dates
    elif schedule.trading_day_count is not None:
        method = "trading_day_count"
        supplied_value = schedule.trading_day_count
    else:
        method = "simulation_end_date"
        supplied_value = schedule.simulation_end_date
    return {
        "method": method,
        "supplied_value": supplied_value,
        "resolved_dates": context.resolved_future_dates,
    }


def _build_manifest(
    *,
    request: SimulationRequest,
    context: ValidatedContext,
    policy: ResolvedSimulationPolicy,
    sampler_metadata: Mapping[str, Any],
    mapping_metadata: Mapping[str, Any],
    currency_metadata: Mapping[str, Any],
    corporate_action_metadata: Mapping[str, Any],
    listing_metadata: Mapping[str, Any],
    report: ValidationReport,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run": {
            "run_id": request.run_id,
            "risk_model_name": request.risk_model_name,
            "risk_model_version": request.risk_model_version,
        },
        "dimensions": dict(context.dimensions),
        "axis_order": {
            "factor_arrays": ("path", "future_date", "factor"),
            "asset_return_arrays": ("path", "future_date", "asset"),
            "prices": ("path", "date_including_origin", "asset"),
            "target_asset_covariance": ("asset", "asset"),
        },
        "schedule": _schedule_manifest(request, context),
        "calendar": {
            "calendar_id": context.calendar_id,
            "time_zone": context.calendar_time_zone,
            "provenance": context.calendar_provenance,
        },
        "returns": {
            "modeled": {
                "return_type": request.risk_state.return_convention,
                "return_basis": request.risk_state.return_basis,
            },
            "price_link": {
                "return_type": mapping_metadata["return_type"],
                "return_basis": mapping_metadata["return_basis"],
                "mapping": mapping_metadata["mapping"],
                "adjustment_basis": mapping_metadata["adjustment"],
            },
        },
        "asset_currencies": currency_metadata["price_currencies"],
        "resolved_policy": _resolved_policy_manifest(policy),
        "rng": {
            "seed": sampler_metadata["seed"],
            "bit_generator": sampler_metadata["bit_generator"],
            "stream_policy": sampler_metadata["stream_policy"],
            "path_streams": sampler_metadata["path_streams"],
            "sampler_name": sampler_metadata["sampler_name"],
            "sampler_version": sampler_metadata["sampler_version"],
            "numpy_version": sampler_metadata["numpy_version"],
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "numerical_tolerances": {
            "symmetry": request.numerical_tolerances.symmetry,
            "positive_semidefinite": (
                request.numerical_tolerances.positive_semidefinite
            ),
            "equation": request.numerical_tolerances.equation,
        },
        "numerical_summaries": report.summaries,
        "sampler_diagnostics": {
            key: value
            for key, value in sampler_metadata.items()
            if key.endswith("covariance_root") or key == "zero_specific_variance_count"
        },
        "handlers": {
            "currency_handling": dict(currency_metadata),
            "corporate_action_adjustment": dict(corporate_action_metadata),
            "new_listing": dict(listing_metadata),
        },
        "provenance": {
            "request": request.provenance,
            "risk_state": request.risk_state.provenance,
        },
        "risk_state_frozen": True,
    }


def simulate(request: SimulationRequest) -> SimulationResult:
    """Run one complete in-memory version-one simulation."""

    resolved_policy = resolve_policy(request.policy)
    context = validate_request(request, resolved_policy)

    streams = build_random_streams(
        seed=request.random_seed,
        path_count=request.path_count,
        joint_policy=resolved_policy.return_joint_distribution,
        bit_generator_name=request.rng_bit_generator,
        stream_policy=request.rng_stream_policy,
    )
    innovations = sample_innovations(
        resolved_policy=resolved_policy,
        factor_covariance=request.risk_state.factor_covariance,
        specific_variances=request.risk_state.specific_variances,
        step_count=context.dimensions["H"],
        rng_streams=streams,
        symmetry_tolerance=request.numerical_tolerances.symmetry,
        positive_semidefinite_tolerance=(
            request.numerical_tolerances.positive_semidefinite
        ),
    )
    factor_returns = construct_factor_returns(
        request.factor_means, innovations.factor_innovations
    )
    specific_returns = construct_specific_returns(
        request.specific_means, innovations.specific_innovations
    )
    modeled_returns = construct_modeled_returns(
        request.risk_state.exposures, factor_returns, specific_returns
    )
    price_link_returns, mapping_metadata = map_modeled_returns(
        modeled_returns,
        risk_state=request.risk_state,
        resolved_policy=resolved_policy,
    )
    asset_currencies = tuple(item.asset_currency for item in request.asset_metadata)
    price_link_returns, currency_metadata = apply_currency_policy(
        price_link_returns,
        resolved_policy=resolved_policy,
        asset_currencies=asset_currencies,
    )
    price_link_returns, asset_state_events, action_metadata = (
        apply_corporate_action_policy(
            price_link_returns,
            resolved_policy=resolved_policy,
        )
    )
    asset_ids, listing_metadata = apply_new_listing_policy(
        context.asset_ids,
        resolved_policy=resolved_policy,
    )
    prices = evolve_arithmetic_prices(
        request.initial_prices,
        price_link_returns,
        asset_ids=asset_ids,
        future_dates=context.resolved_future_dates,
        invalid_return_policy=resolved_policy.invalid_return_policy,
    )

    summaries = dict(context.validation_summaries)
    summaries.update(
        {
            "diagnostic_scope": "preflight_only",
            "statistical_diagnostics_included": False,
        }
    )
    report = ValidationReport(issues=(), summaries=summaries)
    manifest = _build_manifest(
        request=request,
        context=context,
        policy=resolved_policy,
        sampler_metadata=innovations.sampler_metadata,
        mapping_metadata=mapping_metadata,
        currency_metadata=currency_metadata,
        corporate_action_metadata=action_metadata,
        listing_metadata=listing_metadata,
        report=report,
    )
    return SimulationResult(
        factor_innovations=innovations.factor_innovations,
        specific_innovations=innovations.specific_innovations,
        factor_returns=factor_returns,
        specific_returns=specific_returns,
        modeled_returns=modeled_returns,
        price_link_returns=price_link_returns,
        prices=prices,
        asset_state_events=asset_state_events,
        target_asset_covariance=context.target_asset_covariance,
        resolved_policy=resolved_policy,
        manifest=manifest,
        validation_report=report,
        asset_ids=asset_ids,
        factor_ids=context.factor_ids,
        future_dates=context.resolved_future_dates,
        asset_currencies=asset_currencies,
        risk_state=request.risk_state,
        factor_metadata=request.factor_metadata,
    )
