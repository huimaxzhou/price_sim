"""Deterministic preflight validation for simulation requests.

Validation resolves the schedule, checks the complete frozen input contract,
constructs the target asset covariance once, and estimates peak memory before
any random-number generator is created.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import (
    FactorReturnDistribution,
    InvalidReturnPolicy,
    JointDistributionPolicy,
    PriceReturnConvention,
    SerialCorrelationPolicy,
    SpecificReturnDistribution,
)
from .errors import ValidationError, ValidationIssue
from .models import DateLike, ResolvedSimulationPolicy, SimulationRequest

FloatArray = NDArray[np.float64]
SUPPORTED_BIT_GENERATORS = frozenset({"PCG64"})
SUPPORTED_STREAM_POLICIES = frozenset({"per_path"})
ONE_DAY_COVARIANCE_UNITS = frozenset(
    {"one_trading_day_return_squared", "one_trading_day_return_variance"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedContext:
    """Reusable immutable output of one complete deterministic preflight."""

    dimensions: Mapping[str, int]
    asset_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    resolved_future_dates: tuple[DateLike, ...]
    calendar_id: str
    calendar_time_zone: str
    calendar_provenance: Mapping[str, Any]
    target_asset_covariance: FloatArray
    validation_summaries: Mapping[str, Any]
    estimated_peak_memory_bytes: int
    memory_limit_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(
            self,
            "calendar_provenance",
            MappingProxyType(dict(self.calendar_provenance)),
        )
        object.__setattr__(
            self,
            "validation_summaries",
            MappingProxyType(dict(self.validation_summaries)),
        )
        covariance = self.target_asset_covariance
        if covariance.dtype != np.dtype(np.float64) or not covariance.flags.owndata:
            raise TypeError("target covariance must be an owned float64 array")
        covariance.setflags(write=False)


def _issue(
    issues: list[ValidationIssue],
    code: str,
    field_path: str,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            code=code,
            field_path=field_path,
            message=message,
            context=context,
        )
    )


def _is_missing_text(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _parse_date(
    value: DateLike,
    *,
    field_path: str,
    issues: list[ValidationIssue],
) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            _issue(
                issues,
                "invalid_date",
                field_path,
                "the value must be an ISO 8601 date or timestamp",
                {"value": value},
            )
            return None
    _issue(
        issues,
        "invalid_date",
        field_path,
        "the value must be a date, datetime, or ISO 8601 string",
        {"type": type(value).__name__},
    )
    return None


def _parse_datetime(
    value: DateLike,
    *,
    field_path: str,
    issues: list[ValidationIssue],
) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            _issue(
                issues,
                "invalid_date",
                field_path,
                "the value must be an ISO 8601 date or timestamp",
                {"value": value},
            )
            return None
    _issue(
        issues,
        "invalid_date",
        field_path,
        "the value must be a date, datetime, or ISO 8601 string",
        {"type": type(value).__name__},
    )
    return None


def _resolve_schedule(
    request: SimulationRequest,
    issues: list[ValidationIssue],
) -> tuple[DateLike, ...]:
    schedule = request.schedule
    methods = (
        schedule.explicit_future_dates,
        schedule.trading_day_count,
        schedule.simulation_end_date,
    )
    if sum(value is not None for value in methods) != 1:
        _issue(
            issues,
            "invalid_schedule_selection",
            "schedule",
            "exactly one schedule method must be supplied",
        )
        return ()

    calendar = request.trading_calendar
    session_dates: list[date] = []
    calendar_valid = True
    for index, value in enumerate(calendar.sessions):
        parsed = _parse_date(
            value,
            field_path=f"trading_calendar.sessions[{index}]",
            issues=issues,
        )
        if parsed is None:
            calendar_valid = False
        else:
            session_dates.append(parsed)

    if not session_dates:
        _issue(
            issues,
            "empty_trading_calendar",
            "trading_calendar.sessions",
            "at least one deterministic trading session is required",
        )
        calendar_valid = False
    elif any(
        later <= earlier for earlier, later in zip(session_dates, session_dates[1:])
    ):
        _issue(
            issues,
            "non_increasing_calendar_sessions",
            "trading_calendar.sessions",
            "calendar sessions must be unique and strictly increasing",
        )
        calendar_valid = False

    origin = _parse_date(
        request.simulation_origin,
        field_path="simulation_origin",
        issues=issues,
    )
    if not calendar_valid or origin is None:
        return ()

    future_calendar = [
        (value, parsed)
        for value, parsed in zip(calendar.sessions, session_dates, strict=True)
        if parsed > origin
    ]

    if schedule.explicit_future_dates is not None:
        explicit = schedule.explicit_future_dates
        if not explicit:
            _issue(
                issues,
                "empty_resolved_schedule",
                "schedule.explicit_future_dates",
                "the explicit schedule must contain at least one future date",
            )
            return ()
        explicit_dates: list[date] = []
        valid = True
        for index, value in enumerate(explicit):
            parsed = _parse_date(
                value,
                field_path=f"schedule.explicit_future_dates[{index}]",
                issues=issues,
            )
            if parsed is None:
                valid = False
                continue
            explicit_dates.append(parsed)
            if parsed <= origin:
                _issue(
                    issues,
                    "date_not_after_simulation_origin",
                    f"schedule.explicit_future_dates[{index}]",
                    "every resolved date must be after the simulation origin",
                )
                valid = False
            if parsed not in session_dates:
                _issue(
                    issues,
                    "date_not_in_calendar",
                    f"schedule.explicit_future_dates[{index}]",
                    "the explicit date is not a supplied trading session",
                )
                valid = False
        if len(explicit_dates) > 1 and any(
            later <= earlier
            for earlier, later in zip(explicit_dates, explicit_dates[1:])
        ):
            _issue(
                issues,
                "non_increasing_resolved_dates",
                "schedule.explicit_future_dates",
                "resolved dates must be unique and strictly increasing",
            )
            valid = False
        return tuple(explicit) if valid else ()

    if schedule.trading_day_count is not None:
        count = schedule.trading_day_count
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            _issue(
                issues,
                "invalid_trading_day_count",
                "schedule.trading_day_count",
                "the trading-day count must be a positive integer",
            )
            return ()
        if len(future_calendar) < count:
            _issue(
                issues,
                "insufficient_calendar_coverage",
                "trading_calendar.sessions",
                "the calendar does not cover the requested trading-day count",
                {"requested": count, "available": len(future_calendar)},
            )
            return ()
        return tuple(value for value, _ in future_calendar[:count])

    end_date = _parse_date(
        schedule.simulation_end_date,  # type: ignore[arg-type]
        field_path="schedule.simulation_end_date",
        issues=issues,
    )
    if end_date is None:
        return ()
    resolved = tuple(value for value, parsed in future_calendar if parsed <= end_date)
    if not resolved:
        _issue(
            issues,
            "empty_resolved_schedule",
            "schedule.simulation_end_date",
            "the end-date schedule resolves to zero future trading sessions",
        )
    return resolved


def _check_required_text(
    request: SimulationRequest, issues: list[ValidationIssue]
) -> None:
    values = {
        "run_id": request.run_id,
        "risk_model_name": request.risk_model_name,
        "risk_model_version": request.risk_model_version,
        "trading_calendar.calendar_id": request.trading_calendar.calendar_id,
        "trading_calendar.time_zone": request.trading_calendar.time_zone,
        "risk_state.observation_frequency": request.risk_state.observation_frequency,
    }
    for field_path, value in values.items():
        if _is_missing_text(value):
            _issue(
                issues,
                "missing_required_value",
                field_path,
                "a nonempty string is required",
            )


def _check_unique_identifiers(
    request: SimulationRequest, issues: list[ValidationIssue]
) -> None:
    for field_path, values in (
        ("risk_state.asset_ids", request.risk_state.asset_ids),
        ("risk_state.factor_ids", request.risk_state.factor_ids),
    ):
        if len(values) != len(set(values)):
            _issue(
                issues,
                "duplicate_identifier",
                field_path,
                "identifiers must be unique and retain canonical order",
            )


def _check_shape(
    array: np.ndarray,
    expected: tuple[int, ...],
    field_path: str,
    issues: list[ValidationIssue],
) -> bool:
    if array.shape == expected:
        return True
    _issue(
        issues,
        "invalid_shape",
        field_path,
        "the array shape does not match the ordered request dimensions",
        {"expected": expected, "actual": array.shape},
    )
    return False


def _check_metadata_alignment(
    request: SimulationRequest,
    issues: list[ValidationIssue],
) -> None:
    asset_metadata_ids = tuple(item.asset_id for item in request.asset_metadata)
    if asset_metadata_ids != request.risk_state.asset_ids:
        _issue(
            issues,
            "identifier_misalignment",
            "asset_metadata",
            "asset metadata must follow risk-state asset order exactly",
        )
    if request.factor_metadata is None:
        return
    factor_metadata_ids = tuple(item.factor_id for item in request.factor_metadata)
    if factor_metadata_ids != request.risk_state.factor_ids:
        _issue(
            issues,
            "identifier_misalignment",
            "factor_metadata",
            "factor metadata must follow risk-state factor order exactly",
        )
    for index, item in enumerate(request.factor_metadata):
        if (
            item.factor_class is not None
            and item.exposure_class is not None
            and item.factor_class != item.exposure_class
        ):
            _issue(
                issues,
                "conflicting_factor_classification",
                f"factor_metadata[{index}]",
                "factor_class and exposure_class must agree",
            )


def _check_finite_arrays(
    request: SimulationRequest, issues: list[ValidationIssue]
) -> None:
    arrays = {
        "initial_prices": request.initial_prices,
        "factor_means": request.factor_means,
        "specific_means": request.specific_means,
        "risk_state.exposures": request.risk_state.exposures,
        "risk_state.factor_covariance": request.risk_state.factor_covariance,
        "risk_state.specific_variances": request.risk_state.specific_variances,
    }
    for field_path, array in arrays.items():
        if not np.isfinite(array).all():
            _issue(
                issues,
                "nonfinite_value",
                field_path,
                "all numerical inputs must be finite",
            )


def _check_risk_metadata(
    request: SimulationRequest,
    resolved_policy: ResolvedSimulationPolicy,
    issues: list[ValidationIssue],
) -> None:
    state = request.risk_state
    expected_values = (
        ("return_horizon", state.return_horizon, "one_trading_day"),
        ("return_convention", state.return_convention, "arithmetic"),
        ("return_basis", state.return_basis, "raw_price"),
        ("annualization_state", state.annualization_state, "deannualized_daily"),
    )
    for field_name, actual, expected in expected_values:
        if actual != expected:
            _issue(
                issues,
                f"incompatible_{field_name}",
                f"risk_state.{field_name}",
                f"version one requires {expected}",
                {"value": actual},
            )
    if state.covariance_units not in ONE_DAY_COVARIANCE_UNITS:
        _issue(
            issues,
            "incompatible_covariance_units",
            "risk_state.covariance_units",
            "covariance inputs must already be in one-trading-day return units",
            {"value": state.covariance_units},
        )

    known_at = _parse_datetime(
        state.known_at,
        field_path="risk_state.known_at",
        issues=issues,
    )
    origin = _parse_datetime(
        request.simulation_origin,
        field_path="simulation_origin",
        issues=issues,
    )
    if known_at is not None and origin is not None:
        if (known_at.tzinfo is None) != (origin.tzinfo is None):
            _issue(
                issues,
                "incompatible_timestamp_awareness",
                "risk_state.known_at",
                "known_at and simulation_origin must use compatible time zones",
            )
        elif known_at > origin:
            _issue(
                issues,
                "risk_state_lookahead",
                "risk_state.known_at",
                "risk inputs must be known by the simulation origin",
            )

    if resolved_policy.price_return_convention is PriceReturnConvention.RAW and (
        state.return_convention != "arithmetic" or state.return_basis != "raw_price"
    ):
        _issue(
            issues,
            "incompatible_raw_price_return",
            "policy.price_return_convention",
            "raw price linking requires raw arithmetic modeled returns",
        )


def _check_policy_contract(
    resolved_policy: ResolvedSimulationPolicy,
    *,
    factor_count: int,
    asset_count: int,
    issues: list[ValidationIssue],
) -> None:
    if resolved_policy.invalid_return_policy is not InvalidReturnPolicy.FAIL:
        _issue(
            issues,
            "unsupported_policy_value",
            "policy.invalid_return_policy",
            "version one supports only the explicit fail policy",
        )
    if (
        resolved_policy.factor_return_distribution
        is not FactorReturnDistribution.GAUSSIAN
    ):
        _issue(
            issues,
            "unsupported_policy_value",
            "policy.factor_return_distribution",
            "version one supports only Gaussian factor innovations",
        )
    if (
        resolved_policy.specific_return_distribution
        is not SpecificReturnDistribution.GAUSSIAN
    ):
        _issue(
            issues,
            "unsupported_policy_value",
            "policy.specific_return_distribution",
            "version one supports only Gaussian specific innovations",
        )
    if resolved_policy.return_serial_correlation is not SerialCorrelationPolicy.NONE:
        _issue(
            issues,
            "unsupported_policy_value",
            "policy.return_serial_correlation",
            "version one supports only serially independent innovations",
        )

    generator = resolved_policy.joint_generator
    if resolved_policy.return_joint_distribution is JointDistributionPolicy.JOINT:
        if generator is None:
            _issue(
                issues,
                "missing_joint_generator",
                "policy.joint_generator",
                "joint generation requires its grouped configuration",
            )
            return
        contract = generator.marginal_contract
        if (
            contract.get("factor") != "Gaussian"
            or contract.get("specific") != "Gaussian"
        ):
            _issue(
                issues,
                "incompatible_joint_marginal_contract",
                "policy.joint_generator.marginal_contract",
                "the baseline joint generator requires Gaussian marginals",
            )
        declared_cross = contract.get("factor_specific_cross_covariance", "zero")
        if declared_cross != "zero":
            _issue(
                issues,
                "nonzero_factor_specific_cross_covariance",
                "policy.joint_generator.marginal_contract.factor_specific_cross_covariance",
                "the baseline identity requires zero factor-specific cross-covariance",
            )
        configured_cross = generator.parameters.get("factor_specific_cross_covariance")
        if configured_cross is not None:
            cross = np.asarray(configured_cross, dtype=np.float64)
            field_path = (
                "policy.joint_generator.parameters." "factor_specific_cross_covariance"
            )
            if cross.shape != (factor_count, asset_count):
                _issue(
                    issues,
                    "invalid_shape",
                    field_path,
                    "factor-specific cross-covariance has the wrong shape",
                    {
                        "expected": (factor_count, asset_count),
                        "actual": cross.shape,
                    },
                )
            elif not np.isfinite(cross).all() or np.any(cross != 0.0):
                _issue(
                    issues,
                    "nonzero_factor_specific_cross_covariance",
                    field_path,
                    "the baseline identity requires an all-zero cross block",
                )


def _check_runtime_controls(
    request: SimulationRequest, issues: list[ValidationIssue]
) -> None:
    if request.rng_bit_generator not in SUPPORTED_BIT_GENERATORS:
        _issue(
            issues,
            "unsupported_rng_bit_generator",
            "rng_bit_generator",
            "the bit generator is not registered for version one",
            {"value": request.rng_bit_generator},
        )
    if request.rng_stream_policy not in SUPPORTED_STREAM_POLICIES:
        _issue(
            issues,
            "unsupported_rng_stream_policy",
            "rng_stream_policy",
            "the stream-allocation policy is not registered for version one",
            {"value": request.rng_stream_policy},
        )
    if (
        isinstance(request.random_seed, bool)
        or not isinstance(request.random_seed, int)
        or request.random_seed < 0
    ):
        _issue(
            issues,
            "invalid_random_seed",
            "random_seed",
            "the seed must be a nonnegative integer",
        )
    if (
        isinstance(request.memory_limit_bytes, bool)
        or not isinstance(request.memory_limit_bytes, int)
        or request.memory_limit_bytes <= 0
    ):
        _issue(
            issues,
            "nonpositive_memory_limit",
            "memory_limit_bytes",
            "the peak-memory limit must be a positive integer number of bytes",
        )
    tolerances = request.numerical_tolerances
    for field_name in ("symmetry", "positive_semidefinite", "equation"):
        value = getattr(tolerances, field_name)
        if not np.isfinite(value) or value < 0:
            _issue(
                issues,
                "negative_tolerance",
                f"numerical_tolerances.{field_name}",
                "numerical tolerances must be finite and nonnegative",
                {"value": value},
            )


def estimate_peak_memory_bytes(
    *,
    path_count: int,
    step_count: int,
    asset_count: int,
    factor_count: int,
    joint: bool,
) -> int:
    """Estimate peak bytes for version-one in-memory float64 execution.

    The estimate covers final factor arrays, asset-shaped return arrays, price
    paths, covariance targets and roots, plus the largest sampler workspace.
    """

    itemsize = np.dtype(np.float64).itemsize
    factor_arrays = 2 * path_count * step_count * factor_count
    asset_return_arrays = 4 * path_count * step_count * asset_count
    price_array = path_count * (step_count + 1) * asset_count
    covariance_arrays = 2 * factor_count**2 + asset_count**2
    if joint:
        joint_width = factor_count + asset_count
        sampler_workspace = joint_width**2 + step_count * joint_width
    else:
        sampler_workspace = factor_count**2 + step_count * (factor_count + asset_count)
    return itemsize * (
        factor_arrays
        + asset_return_arrays
        + price_array
        + covariance_arrays
        + sampler_workspace
    )


def validate_request(
    request: SimulationRequest,
    resolved_policy: ResolvedSimulationPolicy,
) -> ValidatedContext:
    """Validate ``request`` completely without consuming random state."""

    if not isinstance(request, SimulationRequest):
        raise TypeError("request must be a SimulationRequest")
    if not isinstance(resolved_policy, ResolvedSimulationPolicy):
        raise TypeError("resolved_policy must be a ResolvedSimulationPolicy")

    issues: list[ValidationIssue] = []
    resolved_dates = _resolve_schedule(request, issues)
    _check_required_text(request, issues)

    path_count = request.path_count
    asset_count = len(request.risk_state.asset_ids)
    factor_count = len(request.risk_state.factor_ids)
    step_count = len(resolved_dates)
    for field_path, value in (
        ("path_count", path_count),
        ("risk_state.asset_ids", asset_count),
        ("risk_state.factor_ids", factor_count),
        ("schedule", step_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _issue(
                issues,
                "nonpositive_dimension",
                field_path,
                "M, H, N, and K must each be positive",
                {"value": value},
            )

    _check_unique_identifiers(request, issues)
    shapes_valid = all(
        (
            _check_shape(
                request.risk_state.exposures,
                (asset_count, factor_count),
                "risk_state.exposures",
                issues,
            ),
            _check_shape(
                request.risk_state.factor_covariance,
                (factor_count, factor_count),
                "risk_state.factor_covariance",
                issues,
            ),
            _check_shape(
                request.risk_state.specific_variances,
                (asset_count,),
                "risk_state.specific_variances",
                issues,
            ),
            _check_shape(
                request.initial_prices,
                (asset_count,),
                "initial_prices",
                issues,
            ),
            _check_shape(
                request.factor_means,
                (step_count, factor_count),
                "factor_means",
                issues,
            ),
            _check_shape(
                request.specific_means,
                (step_count, asset_count),
                "specific_means",
                issues,
            ),
        )
    )
    _check_metadata_alignment(request, issues)
    _check_finite_arrays(request, issues)

    finite_prices = np.isfinite(request.initial_prices)
    if np.any(request.initial_prices[finite_prices] <= 0.0):
        _issue(
            issues,
            "nonpositive_initial_price",
            "initial_prices",
            "every initial price must be strictly positive",
        )
    finite_specific_variances = np.isfinite(request.risk_state.specific_variances)
    if np.any(
        request.risk_state.specific_variances[finite_specific_variances] < 0.0
    ):
        _issue(
            issues,
            "negative_specific_variance",
            "risk_state.specific_variances",
            "specific variances must be nonnegative",
        )

    _check_risk_metadata(request, resolved_policy, issues)
    _check_policy_contract(
        resolved_policy,
        factor_count=factor_count,
        asset_count=asset_count,
        issues=issues,
    )
    _check_runtime_controls(request, issues)
    for index, metadata in enumerate(request.asset_metadata):
        if _is_missing_text(metadata.asset_currency):
            _issue(
                issues,
                "missing_asset_currency",
                f"asset_metadata[{index}].asset_currency",
                "every asset requires a quote currency",
            )

    tolerance = request.numerical_tolerances
    factor_asymmetry = float("nan")
    factor_minimum_eigenvalue = float("nan")
    target_asymmetry = float("nan")
    target_minimum_eigenvalue = float("nan")
    target_covariance: FloatArray | None = None

    factor_covariance = request.risk_state.factor_covariance
    factor_covariance_usable = (
        factor_covariance.shape == (factor_count, factor_count)
        and np.isfinite(factor_covariance).all()
        and np.isfinite(tolerance.symmetry)
        and np.isfinite(tolerance.positive_semidefinite)
        and tolerance.symmetry >= 0
        and tolerance.positive_semidefinite >= 0
    )
    if factor_covariance_usable:
        factor_asymmetry = float(
            np.max(np.abs(factor_covariance - factor_covariance.T), initial=0.0)
        )
        if factor_asymmetry > tolerance.symmetry:
            _issue(
                issues,
                "asymmetric_covariance",
                "risk_state.factor_covariance",
                "factor covariance exceeds the declared symmetry tolerance",
                {"maximum_asymmetry": factor_asymmetry},
            )
        else:
            canonical_factor = (factor_covariance + factor_covariance.T) / 2.0
            factor_minimum_eigenvalue = float(
                np.linalg.eigvalsh(canonical_factor).min(initial=np.inf)
            )
            if factor_minimum_eigenvalue < -tolerance.positive_semidefinite:
                _issue(
                    issues,
                    "non_psd_covariance",
                    "risk_state.factor_covariance",
                    "factor covariance is materially indefinite",
                    {"minimum_eigenvalue": factor_minimum_eigenvalue},
                )

    numerical_inputs_usable = (
        shapes_valid
        and request.risk_state.exposures.shape == (asset_count, factor_count)
        and request.risk_state.specific_variances.shape == (asset_count,)
        and np.isfinite(request.risk_state.exposures).all()
        and np.isfinite(factor_covariance).all()
        and np.isfinite(request.risk_state.specific_variances).all()
    )
    if numerical_inputs_usable:
        target_covariance = np.asarray(
            request.risk_state.exposures
            @ factor_covariance
            @ request.risk_state.exposures.T,
            dtype=np.float64,
        )
        target_covariance = np.array(target_covariance, copy=True, order="C")
        diagonal = np.diag_indices(asset_count)
        target_covariance[diagonal] += request.risk_state.specific_variances
        target_asymmetry = float(
            np.max(np.abs(target_covariance - target_covariance.T), initial=0.0)
        )
        if target_asymmetry > tolerance.symmetry:
            _issue(
                issues,
                "asymmetric_covariance",
                "target_asset_covariance",
                "target asset covariance exceeds the symmetry tolerance",
                {"maximum_asymmetry": target_asymmetry},
            )
        else:
            canonical_target = (target_covariance + target_covariance.T) / 2.0
            target_minimum_eigenvalue = float(
                np.linalg.eigvalsh(canonical_target).min(initial=np.inf)
            )
            if target_minimum_eigenvalue < -tolerance.positive_semidefinite:
                _issue(
                    issues,
                    "non_psd_covariance",
                    "target_asset_covariance",
                    "target asset covariance is materially indefinite",
                    {"minimum_eigenvalue": target_minimum_eigenvalue},
                )

    dimensions_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (path_count, step_count, asset_count, factor_count)
    )
    estimated_peak_memory_bytes = 0
    if dimensions_valid:
        estimated_peak_memory_bytes = estimate_peak_memory_bytes(
            path_count=path_count,
            step_count=step_count,
            asset_count=asset_count,
            factor_count=factor_count,
            joint=(
                resolved_policy.return_joint_distribution
                is JointDistributionPolicy.JOINT
            ),
        )
        if (
            isinstance(request.memory_limit_bytes, int)
            and not isinstance(request.memory_limit_bytes, bool)
            and request.memory_limit_bytes > 0
            and estimated_peak_memory_bytes > request.memory_limit_bytes
        ):
            _issue(
                issues,
                "memory_limit_exceeded",
                "memory_limit_bytes",
                "estimated peak memory exceeds the configured hard limit",
                {
                    "memory_limit_bytes": request.memory_limit_bytes,
                    "estimated_peak_memory_bytes": estimated_peak_memory_bytes,
                },
            )

    if issues:
        raise ValidationError(issues)
    if target_covariance is None:
        raise RuntimeError("validated request did not produce target covariance")

    summaries = {
        "factor_covariance_maximum_asymmetry": factor_asymmetry,
        "factor_covariance_minimum_eigenvalue": factor_minimum_eigenvalue,
        "factor_covariance_is_singular": (
            abs(factor_minimum_eigenvalue) <= tolerance.positive_semidefinite
        ),
        "zero_specific_variance_count": int(
            np.count_nonzero(request.risk_state.specific_variances == 0.0)
        ),
        "target_covariance_maximum_asymmetry": target_asymmetry,
        "target_covariance_minimum_eigenvalue": target_minimum_eigenvalue,
    }
    return ValidatedContext(
        dimensions={
            "M": path_count,
            "H": step_count,
            "N": asset_count,
            "K": factor_count,
        },
        asset_ids=request.risk_state.asset_ids,
        factor_ids=request.risk_state.factor_ids,
        resolved_future_dates=resolved_dates,
        calendar_id=request.trading_calendar.calendar_id,
        calendar_time_zone=request.trading_calendar.time_zone,
        calendar_provenance=request.trading_calendar.provenance,
        target_asset_covariance=target_covariance,
        validation_summaries=summaries,
        estimated_peak_memory_bytes=estimated_peak_memory_bytes,
        memory_limit_bytes=request.memory_limit_bytes,
    )
