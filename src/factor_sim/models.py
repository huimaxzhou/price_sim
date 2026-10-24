"""Typed, immutable domain models for the factor-risk simulator.

Array docstrings use the canonical dimensions from the v4 specification:
``N`` assets, ``K`` factors, ``M`` paths, and ``H`` future trading days.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationIssue

if TYPE_CHECKING:
    from .config import (
        AssetStatus,
        ExposureDynamics,
        FactorCorrelationDynamics,
        FactorReturnDistribution,
        FactorVolatilityDynamics,
        InvalidReturnPolicy,
        JointDistributionPolicy,
        PriceReturnConvention,
        SerialCorrelationPolicy,
        SpecificReturnDistribution,
        SpecificRiskDynamics,
    )


FloatArray = NDArray[np.float64]
DateLike = date | datetime | str


class _UnsetType:
    """Private sentinel that distinguishes omission from explicit ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __copy__(self) -> _UnsetType:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _UnsetType:
        del memo
        return self


UNSET = _UnsetType()


def _is_unset(value: object) -> bool:
    return value is UNSET


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _copy_readonly_float_input(value: object) -> FloatArray:
    """Copy one external numerical input and make the owned copy read-only."""

    array = np.array(value, dtype=np.float64, copy=True, order="C")
    array.setflags(write=False)
    return array


def _take_readonly_float_output(value: object) -> FloatArray:
    """Mark one engine-owned float64 array read-only without copying it."""

    if not isinstance(value, np.ndarray):
        raise TypeError("engine output must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise TypeError("engine output must have dtype float64")
    if not value.flags.owndata:
        raise ValueError("engine output must own its data")
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskState:
    """Frozen risk-model snapshot.

    ``exposures`` has shape ``(N, K)``, ``factor_covariance`` has shape
    ``(K, K)``, and ``specific_variances`` has shape ``(N,)``. Shape and
    financial-semantic validation belongs to the validation component.
    """

    exposures: FloatArray
    factor_covariance: FloatArray
    specific_variances: FloatArray
    asset_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    as_of_date: DateLike
    known_at: DateLike
    return_horizon: str
    observation_frequency: str
    return_convention: str
    return_basis: str
    covariance_units: str
    annualization_state: str
    provenance: Mapping[str, Any]
    exposure_units: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exposures", _copy_readonly_float_input(self.exposures)
        )
        object.__setattr__(
            self,
            "factor_covariance",
            _copy_readonly_float_input(self.factor_covariance),
        )
        object.__setattr__(
            self,
            "specific_variances",
            _copy_readonly_float_input(self.specific_variances),
        )
        object.__setattr__(self, "asset_ids", tuple(self.asset_ids))
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))


@dataclass(frozen=True, slots=True, kw_only=True)
class AssetMetadata:
    """Model-independent metadata for one asset."""

    asset_id: str
    asset_currency: str
    display_name: str | None = None
    market_or_exchange: str | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.provenance is not None:
            object.__setattr__(
                self, "provenance", _frozen_mapping(self.provenance)
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorMetadata:
    """Optional descriptive metadata for one factor."""

    factor_id: str
    factor_name: str | None = None
    factor_description: str | None = None
    factor_class: str | None = None
    exposure_class: str | None = None
    attributes: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.attributes is not None:
            object.__setattr__(
                self, "attributes", _frozen_mapping(self.attributes)
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationSchedule:
    """One caller-supplied schedule method; preflight enforces the one-of rule."""

    explicit_future_dates: tuple[DateLike, ...] | None = None
    trading_day_count: int | None = None
    simulation_end_date: DateLike | None = None

    def __post_init__(self) -> None:
        if self.explicit_future_dates is not None:
            object.__setattr__(
                self, "explicit_future_dates", tuple(self.explicit_future_dates)
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TradingCalendar:
    """Deterministic version-one trading sessions and their provenance."""

    calendar_id: str
    time_zone: str
    sessions: tuple[DateLike, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))


@dataclass(frozen=True, slots=True, kw_only=True)
class JointGeneratorConfig:
    """Complete identity and contract for one registered joint generator."""

    name: str
    version: str
    parameters: Mapping[str, Any]
    marginal_contract: Mapping[str, Any]
    random_stream_requirements: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "parameters",
            "marginal_contract",
            "random_stream_requirements",
        ):
            object.__setattr__(
                self, field_name, _frozen_mapping(getattr(self, field_name))
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationPolicy:
    """Caller-supplied policy values before default resolution.

    ``UNSET`` is private implementation state used to distinguish omission from
    an explicitly supplied ``None`` handler.
    """

    factor_exposure_dynamics: ExposureDynamics | _UnsetType = UNSET
    factor_volatility_dynamics: FactorVolatilityDynamics | _UnsetType = UNSET
    factor_correlation_dynamics: FactorCorrelationDynamics | _UnsetType = UNSET
    specific_risk_dynamics: SpecificRiskDynamics | _UnsetType = UNSET
    factor_return_distribution: FactorReturnDistribution | _UnsetType = UNSET
    specific_return_distribution: SpecificReturnDistribution | _UnsetType = UNSET
    return_serial_correlation: SerialCorrelationPolicy | _UnsetType = UNSET
    return_joint_distribution: JointDistributionPolicy | _UnsetType = UNSET
    price_return_convention: PriceReturnConvention | _UnsetType = UNSET
    invalid_return_policy: InvalidReturnPolicy | _UnsetType = UNSET
    currency_handling: None | _UnsetType = UNSET
    corporate_action_adjustment: None | _UnsetType = UNSET
    new_listing: None | _UnsetType = UNSET
    joint_generator: JointGeneratorConfig | None | _UnsetType = UNSET


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSimulationPolicy:
    """Complete, validated policy plus explicit/default provenance."""

    factor_exposure_dynamics: ExposureDynamics
    factor_volatility_dynamics: FactorVolatilityDynamics
    factor_correlation_dynamics: FactorCorrelationDynamics
    specific_risk_dynamics: SpecificRiskDynamics
    factor_return_distribution: FactorReturnDistribution
    specific_return_distribution: SpecificReturnDistribution
    return_serial_correlation: SerialCorrelationPolicy
    return_joint_distribution: JointDistributionPolicy
    price_return_convention: PriceReturnConvention
    invalid_return_policy: InvalidReturnPolicy
    currency_handling: None
    corporate_action_adjustment: None
    new_listing: None
    joint_generator: JointGeneratorConfig | None
    value_sources: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value_sources", _frozen_mapping(self.value_sources)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericalTolerances:
    """Caller-approved numerical tolerances; no implicit thresholds are defined."""

    symmetry: float
    positive_semidefinite: float
    equation: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationRequest:
    """Complete request presented to the simulation boundary.

    ``initial_prices`` has shape ``(N,)``, ``factor_means`` has shape ``(H, K)``,
    and ``specific_means`` has shape ``(H, N)``.
    """

    run_id: str
    risk_model_name: str
    risk_model_version: str
    simulation_origin: DateLike
    schedule: SimulationSchedule
    trading_calendar: TradingCalendar
    path_count: int
    initial_prices: FloatArray
    factor_means: FloatArray
    specific_means: FloatArray
    risk_state: RiskState
    asset_metadata: tuple[AssetMetadata, ...]
    policy: SimulationPolicy
    random_seed: int
    rng_bit_generator: str
    rng_stream_policy: str
    numerical_tolerances: NumericalTolerances
    provenance: Mapping[str, Any]
    memory_limit_bytes: int = 1_000_000_000
    factor_metadata: tuple[FactorMetadata, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_prices", _copy_readonly_float_input(self.initial_prices)
        )
        object.__setattr__(
            self, "factor_means", _copy_readonly_float_input(self.factor_means)
        )
        object.__setattr__(
            self, "specific_means", _copy_readonly_float_input(self.specific_means)
        )
        object.__setattr__(self, "asset_metadata", tuple(self.asset_metadata))
        if self.factor_metadata is not None:
            object.__setattr__(self, "factor_metadata", tuple(self.factor_metadata))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))


@dataclass(frozen=True, slots=True, kw_only=True)
class InnovationBatch:
    """Sampled innovations.

    ``factor_innovations`` has shape ``(M, H, K)`` and
    ``specific_innovations`` has shape ``(M, H, N)``.
    """

    factor_innovations: FloatArray
    specific_innovations: FloatArray
    sampler_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "factor_innovations",
            _take_readonly_float_output(self.factor_innovations),
        )
        object.__setattr__(
            self,
            "specific_innovations",
            _take_readonly_float_output(self.specific_innovations),
        )
        object.__setattr__(
            self, "sampler_metadata", _frozen_mapping(self.sampler_metadata)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AssetStateEvent:
    """One path-specific change in an asset's simulated state."""

    asset_id: str
    path_id: int
    effective_date: DateLike
    status: AssetStatus
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    """Structured deterministic or statistical validation evidence."""

    issues: tuple[ValidationIssue, ...]
    summaries: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "summaries", _frozen_mapping(self.summaries))


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationResult:
    """Complete in-memory simulation output.

    Factor arrays have shape ``(M, H, K)``; specific and asset-return arrays have
    shape ``(M, H, N)``; ``prices`` has shape ``(M, H + 1, N)``; and
    ``target_asset_covariance`` has shape ``(N, N)``.
    """

    factor_innovations: FloatArray
    specific_innovations: FloatArray
    factor_returns: FloatArray
    specific_returns: FloatArray
    modeled_returns: FloatArray
    price_link_returns: FloatArray
    prices: FloatArray
    asset_state_events: tuple[AssetStateEvent, ...]
    target_asset_covariance: FloatArray
    resolved_policy: ResolvedSimulationPolicy
    manifest: Mapping[str, Any]
    validation_report: ValidationReport
    asset_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    future_dates: tuple[DateLike, ...]
    asset_currencies: tuple[str, ...]
    risk_state: RiskState
    factor_metadata: tuple[FactorMetadata, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "factor_innovations",
            "specific_innovations",
            "factor_returns",
            "specific_returns",
            "modeled_returns",
            "price_link_returns",
            "prices",
            "target_asset_covariance",
        ):
            object.__setattr__(
                self,
                field_name,
                _take_readonly_float_output(getattr(self, field_name)),
            )
        object.__setattr__(
            self, "asset_state_events", tuple(self.asset_state_events)
        )
        object.__setattr__(self, "manifest", _frozen_mapping(self.manifest))
        object.__setattr__(self, "asset_ids", tuple(self.asset_ids))
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        object.__setattr__(self, "future_dates", tuple(self.future_dates))
        object.__setattr__(
            self, "asset_currencies", tuple(self.asset_currencies)
        )
        if self.factor_metadata is not None:
            object.__setattr__(self, "factor_metadata", tuple(self.factor_metadata))
