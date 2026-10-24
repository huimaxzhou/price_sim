"""Approved Packet B acceptance contract for implementation-plan Steps 4–7.

The user approved Packet B and Review Gate B on 2026-10-24. These tests are now
the protected executable behavioral contract described by ``AGENTS.md``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import factor_sim.sampling as sampling
import factor_sim.validation as validation
from factor_sim.config import (
    InvalidReturnPolicy,
    JointDistributionPolicy,
    resolve_policy,
)
from factor_sim.errors import NumericalError, ValidationError
from factor_sim.models import (
    AssetMetadata,
    FactorMetadata,
    InnovationBatch,
    JointGeneratorConfig,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
)

pytestmark = pytest.mark.acceptance

DEFAULT_MEMORY_LIMIT_BYTES = 1_000_000_000
STATISTICAL_PATHS = 40_000
SEED = 20_261_022


def _require(module: ModuleType, name: str) -> Any:
    """Return a proposed Packet B symbol or fail with a focused message."""

    assert hasattr(module, name), f"Packet B requires {module.__name__}.{name}"
    return getattr(module, name)


def _supplied_policy(
    *, joint: bool = False, cross_covariance: object = None
) -> SimulationPolicy:
    values: dict[str, object] = {
        "invalid_return_policy": InvalidReturnPolicy.FAIL,
    }
    if joint:
        parameters: dict[str, object] = {}
        if cross_covariance is not None:
            parameters["factor_specific_cross_covariance"] = cross_covariance
        values.update(
            return_joint_distribution=JointDistributionPolicy.JOINT,
            joint_generator=JointGeneratorConfig(
                name="block_diagonal_gaussian",
                version="1",
                parameters=parameters,
                marginal_contract={
                    "factor": "Gaussian",
                    "specific": "Gaussian",
                    "factor_specific_cross_covariance": "zero",
                },
                random_stream_requirements={"stream": "joint"},
            ),
        )
    return SimulationPolicy(**values)


def _policy(*, joint: bool = False, cross_covariance: object = None) -> object:
    return resolve_policy(
        _supplied_policy(joint=joint, cross_covariance=cross_covariance)
    )


def _request(
    *,
    risk_state: RiskState | None = None,
    factor_metadata: tuple[FactorMetadata, ...] | None = None,
    asset_metadata: tuple[AssetMetadata, ...] | None = None,
    **overrides: object,
) -> SimulationRequest:
    """Return the architecture's small, manually inspectable 3x2x2x4 fixture."""

    state = risk_state or RiskState(
        exposures=np.array(
            [
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, -0.5],
            ]
        ),
        factor_covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
        specific_variances=np.array([0.02, 0.03, 0.01]),
        asset_ids=("asset-a", "asset-b", "asset-c"),
        factor_ids=("market", "value"),
        as_of_date="2026-10-21",
        known_at="2026-10-21T18:00:00-04:00",
        return_horizon="one_trading_day",
        observation_frequency="daily",
        return_convention="arithmetic",
        return_basis="raw_price",
        covariance_units="one_trading_day_return_variance",
        annualization_state="deannualized_daily",
        provenance={
            "risk_model_name": "contract-fixture",
            "risk_model_version": "1",
            "model_run_id": "fixture-risk-001",
        },
    )
    metadata = (
        tuple(
            AssetMetadata(asset_id=asset_id, asset_currency=currency)
            for asset_id, currency in zip(
                state.asset_ids, ("USD", "USD", "CAD"), strict=True
            )
        )
        if asset_metadata is None
        else asset_metadata
    )
    values: dict[str, object] = {
        "run_id": "packet-b-fixture",
        "risk_model_name": "contract-fixture",
        "risk_model_version": "1",
        "simulation_origin": "2026-10-22T09:30:00-04:00",
        "schedule": SimulationSchedule(
            explicit_future_dates=(
                "2026-10-23",
                "2026-10-26",
                "2026-10-27",
                "2026-10-28",
            )
        ),
        "trading_calendar": TradingCalendar(
            calendar_id="XNYS-fixture",
            time_zone="America/New_York",
            sessions=(
                "2026-10-21",
                "2026-10-22",
                "2026-10-23",
                "2026-10-26",
                "2026-10-27",
                "2026-10-28",
                "2026-10-29",
            ),
            provenance={"source": "Packet B deterministic fixture"},
        ),
        "path_count": 2,
        "initial_prices": np.array([100.0, 50.0, 75.0]),
        "factor_means": np.zeros((4, 2)),
        "specific_means": np.zeros((4, 3)),
        "risk_state": state,
        "asset_metadata": metadata,
        "factor_metadata": factor_metadata,
        "policy": _supplied_policy(),
        "random_seed": SEED,
        "rng_bit_generator": "PCG64",
        "rng_stream_policy": "per_path",
        "numerical_tolerances": NumericalTolerances(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        "provenance": {"source": "Packet B acceptance fixture"},
        "memory_limit_bytes": DEFAULT_MEMORY_LIMIT_BYTES,
    }
    values.update(overrides)
    return SimulationRequest(**values)


def _validate(
    request: SimulationRequest,
    *,
    resolved_policy: object | None = None,
) -> object:
    validate_request = _require(validation, "validate_request")
    return validate_request(request, resolved_policy or _policy())


def _issues(error: ValidationError) -> tuple[tuple[str, str], ...]:
    return tuple((issue.code, issue.field_path) for issue in error.issues)


def _replace_state(request: SimulationRequest, **changes: object) -> SimulationRequest:
    return dataclasses.replace(
        request, risk_state=dataclasses.replace(request.risk_state, **changes)
    )


def _sample(
    name: str,
    *,
    path_count: int = 8,
    step_count: int = 4,
    factor_covariance: np.ndarray | None = None,
    specific_variances: np.ndarray | None = None,
    seed: int = SEED,
) -> InnovationBatch:
    sampler = _require(sampling, name)
    build_random_streams = _require(sampling, "build_random_streams")
    joint_policy = (
        JointDistributionPolicy.JOINT
        if name == "sample_joint_gaussian"
        else JointDistributionPolicy.INDEPENDENT
    )
    streams = build_random_streams(
        seed=seed,
        path_count=path_count,
        joint_policy=joint_policy,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    return sampler(
        factor_covariance=(
            np.array([[0.04, 0.01], [0.01, 0.09]])
            if factor_covariance is None
            else factor_covariance
        ),
        specific_variances=(
            np.array([0.02, 0.03, 0.0])
            if specific_variances is None
            else specific_variances
        ),
        step_count=step_count,
        rng_streams=streams,
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )


def _assert_mean_within_five_se(samples: np.ndarray, target_variance: float) -> None:
    count = samples.shape[0]
    bound = 5.0 * math.sqrt(target_variance / count) + 1e-12
    assert abs(float(np.mean(samples))) <= bound


def _assert_covariance_within_five_se(
    samples: np.ndarray, target: np.ndarray
) -> None:
    observed = np.cov(samples, rowvar=False, ddof=1)
    count = samples.shape[0]
    for row in range(target.shape[0]):
        for column in range(target.shape[1]):
            variance = (
                target[row, column] ** 2
                + target[row, row] * target[column, column]
            ) / (count - 1)
            bound = 5.0 * math.sqrt(max(variance, 0.0)) + 1e-12
            assert abs(observed[row, column] - target[row, column]) <= bound


def _assert_cross_covariance_within_five_se(
    left: np.ndarray, right: np.ndarray, left_cov: np.ndarray, right_cov: np.ndarray
) -> None:
    left_centered = left - np.mean(left, axis=0)
    right_centered = right - np.mean(right, axis=0)
    observed = left_centered.T @ right_centered / (left.shape[0] - 1)
    for row in range(left.shape[1]):
        for column in range(right.shape[1]):
            standard_error = math.sqrt(
                left_cov[row, row]
                * right_cov[column, column]
                / (left.shape[0] - 1)
            )
            assert abs(observed[row, column]) <= 5.0 * standard_error + 1e-12


def test_b_val_001_valid_request_returns_aligned_frozen_context() -> None:
    """B-VAL-001: valid preflight exposes dimensions and exact Sigma_0."""

    request = _request()
    context = _validate(request)
    expected = (
        request.risk_state.exposures
        @ request.risk_state.factor_covariance
        @ request.risk_state.exposures.T
        + np.diag(request.risk_state.specific_variances)
    )

    assert dataclasses.is_dataclass(context)
    assert context.__dataclass_params__.frozen is True
    assert context.dimensions == {"M": 2, "H": 4, "N": 3, "K": 2}
    assert context.asset_ids == request.risk_state.asset_ids
    assert context.factor_ids == request.risk_state.factor_ids
    assert context.resolved_future_dates == (
        "2026-10-23",
        "2026-10-26",
        "2026-10-27",
        "2026-10-28",
    )
    assert context.calendar_id == "XNYS-fixture"
    assert context.calendar_time_zone == "America/New_York"
    assert context.calendar_provenance == request.trading_calendar.provenance
    np.testing.assert_array_equal(context.target_asset_covariance, expected)
    assert context.target_asset_covariance.flags.owndata is True
    assert context.target_asset_covariance.flags.writeable is False
    assert (
        0
        < context.estimated_peak_memory_bytes
        <= request.memory_limit_bytes
        == context.memory_limit_bytes
    )


@pytest.mark.parametrize(
    "schedule",
    [
        SimulationSchedule(trading_day_count=4),
        SimulationSchedule(simulation_end_date="2026-10-28"),
    ],
)
def test_b_cal_001_count_and_end_date_resolve_to_calendar_sessions(
    schedule: SimulationSchedule,
) -> None:
    """B-CAL-001: count and end-date schedules resolve once and identically."""

    context = _validate(_request(schedule=schedule))

    assert context.resolved_future_dates == (
        "2026-10-23",
        "2026-10-26",
        "2026-10-27",
        "2026-10-28",
    )


def test_b_cal_002_non_session_end_date_uses_last_preceding_session() -> None:
    """B-CAL-002: a weekend end date does not invent a trading session."""

    request = _request(
        schedule=SimulationSchedule(simulation_end_date="2026-10-25"),
        factor_means=np.zeros((1, 2)),
        specific_means=np.zeros((1, 3)),
    )

    context = _validate(request)

    assert context.resolved_future_dates == ("2026-10-23",)


@pytest.mark.parametrize(
    ("mutate", "expected_code", "expected_path"),
    [
        (
            lambda request: dataclasses.replace(
                request, schedule=SimulationSchedule()
            ),
            "invalid_schedule_selection",
            "schedule",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(
                    explicit_future_dates=("2026-10-23",),
                    trading_day_count=1,
                ),
            ),
            "invalid_schedule_selection",
            "schedule",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                trading_calendar=dataclasses.replace(
                    request.trading_calendar,
                    sessions=("2026-10-23", "2026-10-23"),
                ),
            ),
            "non_increasing_calendar_sessions",
            "trading_calendar.sessions",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(
                    explicit_future_dates=("2026-10-24",)
                ),
                factor_means=np.zeros((1, 2)),
                specific_means=np.zeros((1, 3)),
            ),
            "date_not_in_calendar",
            "schedule.explicit_future_dates[0]",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(
                    explicit_future_dates=("2026-10-23", "2026-10-23")
                ),
                factor_means=np.zeros((2, 2)),
                specific_means=np.zeros((2, 3)),
            ),
            "non_increasing_resolved_dates",
            "schedule.explicit_future_dates",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(
                    explicit_future_dates=("2026-10-22",)
                ),
                factor_means=np.zeros((1, 2)),
                specific_means=np.zeros((1, 3)),
            ),
            "date_not_after_simulation_origin",
            "schedule.explicit_future_dates[0]",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(simulation_end_date="2026-10-22"),
                factor_means=np.zeros((0, 2)),
                specific_means=np.zeros((0, 3)),
            ),
            "empty_resolved_schedule",
            "schedule.simulation_end_date",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                schedule=SimulationSchedule(trading_day_count=100),
            ),
            "insufficient_calendar_coverage",
            "trading_calendar.sessions",
        ),
        (
            lambda request: dataclasses.replace(request, run_id=""),
            "missing_required_value",
            "run_id",
        ),
        (
            lambda request: dataclasses.replace(request, risk_model_name=""),
            "missing_required_value",
            "risk_model_name",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                trading_calendar=dataclasses.replace(
                    request.trading_calendar, time_zone=""
                ),
            ),
            "missing_required_value",
            "trading_calendar.time_zone",
        ),
        (
            lambda request: dataclasses.replace(request, path_count=0),
            "nonpositive_dimension",
            "path_count",
        ),
        (
            lambda request: _replace_state(
                request, asset_ids=("asset-a", "asset-a", "asset-c")
            ),
            "duplicate_identifier",
            "risk_state.asset_ids",
        ),
        (
            lambda request: _replace_state(
                request, factor_ids=("market", "market")
            ),
            "duplicate_identifier",
            "risk_state.factor_ids",
        ),
        (
            lambda request: _replace_state(
                request, exposures=np.ones((3, 3))
            ),
            "invalid_shape",
            "risk_state.exposures",
        ),
        (
            lambda request: _replace_state(
                request, factor_covariance=np.ones((2, 3))
            ),
            "invalid_shape",
            "risk_state.factor_covariance",
        ),
        (
            lambda request: _replace_state(
                request, specific_variances=np.ones(2)
            ),
            "invalid_shape",
            "risk_state.specific_variances",
        ),
        (
            lambda request: dataclasses.replace(
                request, factor_means=np.zeros((3, 2))
            ),
            "invalid_shape",
            "factor_means",
        ),
        (
            lambda request: dataclasses.replace(
                request, specific_means=np.zeros((4, 2))
            ),
            "invalid_shape",
            "specific_means",
        ),
        (
            lambda request: dataclasses.replace(
                request, initial_prices=np.ones(2)
            ),
            "invalid_shape",
            "initial_prices",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                asset_metadata=tuple(reversed(request.asset_metadata)),
            ),
            "identifier_misalignment",
            "asset_metadata",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                factor_metadata=(
                    FactorMetadata(factor_id="value"),
                    FactorMetadata(factor_id="market"),
                ),
            ),
            "identifier_misalignment",
            "factor_metadata",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                initial_prices=np.array([100.0, np.nan, 75.0]),
            ),
            "nonfinite_value",
            "initial_prices",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                factor_means=np.array(
                    [[0.0, 0.0], [0.0, np.inf], [0.0, 0.0], [0.0, 0.0]]
                ),
            ),
            "nonfinite_value",
            "factor_means",
        ),
        (
            lambda request: _replace_state(
                request,
                factor_covariance=np.array([[0.04, 0.01], [0.01, np.nan]]),
            ),
            "nonfinite_value",
            "risk_state.factor_covariance",
        ),
        (
            lambda request: dataclasses.replace(
                request, initial_prices=np.array([100.0, 0.0, 75.0])
            ),
            "nonpositive_initial_price",
            "initial_prices",
        ),
        (
            lambda request: _replace_state(
                request, specific_variances=np.array([0.02, -0.01, 0.01])
            ),
            "negative_specific_variance",
            "risk_state.specific_variances",
        ),
        (
            lambda request: _replace_state(
                request,
                factor_covariance=np.array([[0.04, 0.02], [0.01, 0.09]]),
            ),
            "asymmetric_covariance",
            "risk_state.factor_covariance",
        ),
        (
            lambda request: _replace_state(
                request,
                factor_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            ),
            "non_psd_covariance",
            "risk_state.factor_covariance",
        ),
        (
            lambda request: _replace_state(request, return_horizon="monthly"),
            "incompatible_return_horizon",
            "risk_state.return_horizon",
        ),
        (
            lambda request: _replace_state(
                request, covariance_units="annualized_return_variance"
            ),
            "incompatible_covariance_units",
            "risk_state.covariance_units",
        ),
        (
            lambda request: _replace_state(
                request, annualization_state="annualized"
            ),
            "incompatible_annualization_state",
            "risk_state.annualization_state",
        ),
        (
            lambda request: _replace_state(request, return_basis="total"),
            "incompatible_return_basis",
            "risk_state.return_basis",
        ),
        (
            lambda request: _replace_state(request, return_convention="log"),
            "incompatible_return_convention",
            "risk_state.return_convention",
        ),
        (
            lambda request: _replace_state(request, observation_frequency=""),
            "missing_required_value",
            "risk_state.observation_frequency",
        ),
        (
            lambda request: _replace_state(
                request, known_at="2026-10-22T10:00:00-04:00"
            ),
            "risk_state_lookahead",
            "risk_state.known_at",
        ),
        (
            lambda request: dataclasses.replace(
                request, rng_bit_generator="unregistered-generator"
            ),
            "unsupported_rng_bit_generator",
            "rng_bit_generator",
        ),
        (
            lambda request: dataclasses.replace(
                request, rng_stream_policy="shared-global-stream"
            ),
            "unsupported_rng_stream_policy",
            "rng_stream_policy",
        ),
        (
            lambda request: dataclasses.replace(
                request,
                numerical_tolerances=NumericalTolerances(
                    symmetry=-1.0,
                    positive_semidefinite=1e-12,
                    equation=1e-12,
                ),
            ),
            "negative_tolerance",
            "numerical_tolerances.symmetry",
        ),
        (
            lambda request: dataclasses.replace(request, memory_limit_bytes=0),
            "nonpositive_memory_limit",
            "memory_limit_bytes",
        ),
    ],
)
def test_b_val_002_each_preflight_rule_has_a_field_addressed_failure(
    mutate: Callable[[SimulationRequest], SimulationRequest],
    expected_code: str,
    expected_path: str,
) -> None:
    """B-VAL-002: invalid deterministic inputs have stable, useful issues."""

    with pytest.raises(ValidationError) as captured:
        _validate(mutate(_request()))

    assert (expected_code, expected_path) in _issues(captured.value)


def test_b_val_003_collects_related_issues_without_initializing_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-VAL-003: preflight aggregation is deterministic and draw-free."""

    def unexpected_rng(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("validation initialized random state")

    monkeypatch.setattr(np.random, "SeedSequence", unexpected_rng)
    request = dataclasses.replace(
        _request(),
        path_count=0,
        initial_prices=np.array([0.0, np.nan, 75.0]),
    )

    with pytest.raises(ValidationError) as captured:
        _validate(request)

    codes = {code for code, _ in _issues(captured.value)}
    assert {
        "nonpositive_dimension",
        "nonpositive_initial_price",
        "nonfinite_value",
    } <= codes


def test_b_val_004_zero_variance_and_singular_psd_are_valid() -> None:
    """B-VAL-004: boundary PSD inputs are supported rather than repaired."""

    request = _replace_state(
        _request(),
        factor_covariance=np.array([[0.04, 0.04], [0.04, 0.04]]),
        specific_variances=np.array([0.02, 0.0, 0.01]),
    )

    context = _validate(request)

    assert context.validation_summaries["factor_covariance_is_singular"] is True
    assert context.validation_summaries["zero_specific_variance_count"] == 1


def test_b_val_005_optional_metadata_is_optional_but_conflicts_fail() -> None:
    """B-VAL-005: metadata absence is valid and semantic conflict is not."""

    assert _validate(_request(factor_metadata=None)).factor_ids == (
        "market",
        "value",
    )
    conflicting = (
        FactorMetadata(
            factor_id="market", factor_class="market", exposure_class="style"
        ),
        FactorMetadata(factor_id="value", factor_class="style"),
    )

    with pytest.raises(ValidationError) as captured:
        _validate(_request(factor_metadata=conflicting))

    assert (
        "conflicting_factor_classification",
        "factor_metadata[0]",
    ) in _issues(captured.value)


def test_b_val_006_currency_factor_does_not_replace_asset_currency() -> None:
    """B-VAL-006: every asset needs quote-currency metadata."""

    factors = (
        FactorMetadata(factor_id="market", factor_class="market"),
        FactorMetadata(factor_id="value", factor_class="currency"),
    )
    assets = (
        AssetMetadata(asset_id="asset-a", asset_currency="USD"),
        AssetMetadata(asset_id="asset-b", asset_currency=""),
        AssetMetadata(asset_id="asset-c", asset_currency="CAD"),
    )

    with pytest.raises(ValidationError) as captured:
        _validate(_request(factor_metadata=factors, asset_metadata=assets))

    assert ("missing_asset_currency", "asset_metadata[1].asset_currency") in _issues(
        captured.value
    )


def test_b_val_007_nonzero_joint_cross_covariance_is_rejected() -> None:
    """B-VAL-007: baseline joint generation cannot change Sigma_0."""

    cross_covariance = np.zeros((2, 3))
    cross_covariance[0, 0] = 0.001
    supplied_policy = _supplied_policy(
        joint=True, cross_covariance=cross_covariance.tolist()
    )

    with pytest.raises(ValidationError) as captured:
        _validate(
            _request(policy=supplied_policy),
            resolved_policy=resolve_policy(supplied_policy),
        )

    assert (
        "nonzero_factor_specific_cross_covariance",
        "policy.joint_generator.parameters.factor_specific_cross_covariance",
    ) in _issues(captured.value)


def test_b_val_008_memory_limit_fails_before_rng_and_reports_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-VAL-008: the approved memory limit is a pre-RNG hard gate."""

    def unexpected_rng(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("memory validation initialized random state")

    monkeypatch.setattr(np.random, "SeedSequence", unexpected_rng)

    with pytest.raises(ValidationError) as captured:
        _validate(_request(path_count=100, memory_limit_bytes=1))

    issue = next(
        issue
        for issue in captured.value.issues
        if issue.code == "memory_limit_exceeded"
    )
    assert issue.field_path == "memory_limit_bytes"
    assert issue.context is not None
    assert issue.context["memory_limit_bytes"] == 1
    assert issue.context["estimated_peak_memory_bytes"] > 1


def test_b_mem_001_memory_estimate_scales_and_covers_final_innovations() -> None:
    """B-MEM-001: peak-memory evidence is monotone and nontrivial."""

    small = _validate(_request(path_count=2))
    large_path_count = 100
    large = _validate(_request(path_count=large_path_count))
    minimum_final_innovation_bytes = 8 * large_path_count * 4 * (2 + 3)

    assert large.estimated_peak_memory_bytes > small.estimated_peak_memory_bytes
    assert large.estimated_peak_memory_bytes >= minimum_final_innovation_bytes
    assert large.memory_limit_bytes == DEFAULT_MEMORY_LIMIT_BYTES


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.04, 0.01], [0.01, 0.09]]),
        np.array([[0.04, 0.04], [0.04, 0.04]]),
    ],
)
def test_b_num_001_covariance_root_reconstructs_psd_matrix(
    matrix: np.ndarray,
) -> None:
    """B-NUM-001: eigendecomposition supports positive and singular PSD."""

    covariance_root = _require(sampling, "covariance_root")
    original = matrix.copy()
    root, diagnostics = covariance_root(
        matrix,
        matrix_name="factor_covariance",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, matrix, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(matrix, original)
    assert diagnostics["matrix_name"] == "factor_covariance"
    assert diagnostics["minimum_eigenvalue"] >= -1e-12


def test_b_num_002_tolerated_negative_eigenvalue_is_zeroed_and_reported() -> None:
    """B-NUM-002: tolerance-bound zeroing is visible, not silent repair."""

    covariance_root = _require(sampling, "covariance_root")
    matrix = np.diag([0.04, -5e-13])

    root, diagnostics = covariance_root(
        matrix,
        matrix_name="near_psd",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, np.diag([0.04, 0.0]), atol=1e-15)
    assert diagnostics["minimum_eigenvalue"] == pytest.approx(-5e-13)
    assert diagnostics["zeroed_eigenvalue_count"] == 1
    assert diagnostics["zeroed_eigenvalue_minimum"] == pytest.approx(-5e-13)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "indefinite"),
        (np.array([[1.0, 1e-3], [0.0, 1.0]]), "asymmetric"),
    ],
)
def test_b_num_003_invalid_covariance_fails_without_mutation(
    matrix: np.ndarray, message: str
) -> None:
    """B-NUM-003: numerical failures identify the matrix and preserve input."""

    covariance_root = _require(sampling, "covariance_root")
    original = matrix.copy()

    with pytest.raises(NumericalError) as captured:
        covariance_root(
            matrix,
            matrix_name="factor_covariance",
            symmetry_tolerance=1e-12,
            positive_semidefinite_tolerance=1e-12,
        )

    np.testing.assert_array_equal(matrix, original)
    assert "factor_covariance" in str(captured.value)
    assert message in str(captured.value).lower()


def test_b_num_004_within_tolerance_asymmetry_is_canonicalized_and_reported() -> None:
    """B-NUM-004: permitted canonicalization is explicit and reproducible."""

    covariance_root = _require(sampling, "covariance_root")
    matrix = np.array([[0.04, 0.01 + 2e-13], [0.01, 0.09]])
    canonical = (matrix + matrix.T) / 2.0
    original = matrix.copy()

    root, diagnostics = covariance_root(
        matrix,
        matrix_name="near_symmetric",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, canonical, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(matrix, original)
    assert diagnostics["canonicalized"] is True
    assert diagnostics["maximum_asymmetry"] == pytest.approx(2e-13)


def test_b_rng_001_seed_reproduction_and_stream_separation() -> None:
    """B-RNG-001: repeated construction is stable without stream reuse."""

    first = _sample("sample_independent_gaussian")
    second = _sample("sample_independent_gaussian")

    np.testing.assert_array_equal(first.factor_innovations, second.factor_innovations)
    np.testing.assert_array_equal(
        first.specific_innovations, second.specific_innovations
    )
    assert not np.array_equal(
        first.factor_innovations[0], first.factor_innovations[1]
    )
    assert first.sampler_metadata["stream_policy"] == (
        "SeedSequence -> path -> factor,specific"
    )
    path_streams = first.sampler_metadata["path_streams"]
    assert len(path_streams) == 8
    assert len({repr(item) for item in path_streams}) == 8
    for item in path_streams:
        assert item["factor_spawn_key"] != item["specific_spawn_key"]


def test_b_rng_002_legacy_global_numpy_rng_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-RNG-002: samplers use Generator objects and record their identity."""

    def legacy_rng_used(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("legacy global NumPy random state was used")

    monkeypatch.setattr(np.random, "seed", legacy_rng_used)
    monkeypatch.setattr(np.random, "normal", legacy_rng_used)
    monkeypatch.setattr(np.random, "multivariate_normal", legacy_rng_used)

    batch = _sample("sample_independent_gaussian")

    assert batch.sampler_metadata["bit_generator"] == "PCG64"
    assert batch.sampler_metadata["seed"] == SEED
    assert batch.sampler_metadata["numpy_version"] == np.__version__


def test_b_sample_001_independent_shapes_zero_variance_and_input_ownership() -> None:
    """B-SAMPLE-001: deterministic independent-sampler behavior is explicit."""

    factor_covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
    specific_variances = np.array([0.02, 0.03, 0.0])
    factor_snapshot = factor_covariance.copy()
    specific_snapshot = specific_variances.copy()

    batch = _sample(
        "sample_independent_gaussian",
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
    )

    assert batch.factor_innovations.shape == (8, 4, 2)
    assert batch.specific_innovations.shape == (8, 4, 3)
    assert batch.factor_innovations.dtype == np.float64
    assert batch.specific_innovations.dtype == np.float64
    assert batch.factor_innovations.flags.owndata is True
    assert batch.specific_innovations.flags.owndata is True
    assert batch.factor_innovations.flags.writeable is False
    assert batch.specific_innovations.flags.writeable is False
    np.testing.assert_array_equal(batch.specific_innovations[..., 2], 0.0)
    np.testing.assert_array_equal(factor_covariance, factor_snapshot)
    np.testing.assert_array_equal(specific_variances, specific_snapshot)


def test_b_sample_002_independent_gaussian_statistical_contract() -> None:
    """B-SAMPLE-002: fixed-seed moments satisfy sample-size-aware bounds."""

    factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_covariance = np.diag([0.02, 0.03, 0.01])
    batch = _sample(
        "sample_independent_gaussian",
        path_count=STATISTICAL_PATHS,
        step_count=2,
        factor_covariance=factor_covariance,
        specific_variances=np.diag(specific_covariance),
    )
    factor = batch.factor_innovations[:, 0, :]
    specific = batch.specific_innovations[:, 0, :]

    for column, variance in enumerate(np.diag(factor_covariance)):
        _assert_mean_within_five_se(factor[:, column], float(variance))
    for column, variance in enumerate(np.diag(specific_covariance)):
        _assert_mean_within_five_se(specific[:, column], float(variance))
    _assert_covariance_within_five_se(factor, factor_covariance)
    _assert_covariance_within_five_se(specific, specific_covariance)
    _assert_cross_covariance_within_five_se(
        factor, specific, factor_covariance, specific_covariance
    )
    squared_cross_correlation = np.corrcoef(
        factor[:, 0] ** 2, specific[:, 0] ** 2
    )[0, 1]
    assert abs(squared_cross_correlation) <= (
        5.0 / math.sqrt(STATISTICAL_PATHS) + 0.01
    )

    serial_samples = batch.factor_innovations[:, :, 0]
    lag_correlation = np.corrcoef(
        serial_samples[:, 0], serial_samples[:, 1]
    )[0, 1]
    assert abs(lag_correlation) <= 5.0 / math.sqrt(STATISTICAL_PATHS) + 0.01


def test_b_joint_001_joint_sampler_records_one_stream_and_block_dimensions() -> None:
    """B-JOINT-001: joint mode is operationally explicit and auditable."""

    batch = _sample("sample_joint_gaussian")

    assert batch.factor_innovations.shape == (8, 4, 2)
    assert batch.specific_innovations.shape == (8, 4, 3)
    assert batch.sampler_metadata["sampler_name"] == "block_diagonal_gaussian"
    assert batch.sampler_metadata["block_dimensions"] == {
        "factor": 2,
        "specific": 3,
        "joint": 5,
    }
    assert batch.sampler_metadata["stream_policy"] == (
        "SeedSequence -> path -> joint"
    )


def test_b_joint_002_joint_gaussian_statistical_contract() -> None:
    """B-JOINT-002: joint mode preserves marginals and zero cross-covariance."""

    factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_covariance = np.diag([0.02, 0.03, 0.01])
    batch = _sample(
        "sample_joint_gaussian",
        path_count=STATISTICAL_PATHS,
        step_count=1,
        factor_covariance=factor_covariance,
        specific_variances=np.diag(specific_covariance),
    )
    factor = batch.factor_innovations[:, 0, :]
    specific = batch.specific_innovations[:, 0, :]

    for column, variance in enumerate(np.diag(factor_covariance)):
        _assert_mean_within_five_se(factor[:, column], float(variance))
    for column, variance in enumerate(np.diag(specific_covariance)):
        _assert_mean_within_five_se(specific[:, column], float(variance))
    _assert_covariance_within_five_se(factor, factor_covariance)
    _assert_covariance_within_five_se(specific, specific_covariance)
    _assert_cross_covariance_within_five_se(
        factor, specific, factor_covariance, specific_covariance
    )
    squared_cross_correlation = np.corrcoef(
        factor[:, 0] ** 2, specific[:, 0] ** 2
    )[0, 1]
    assert abs(squared_cross_correlation) <= (
        5.0 / math.sqrt(STATISTICAL_PATHS) + 0.01
    )


def test_b_dispatch_001_registry_dispatch_follows_resolved_typed_policy() -> None:
    """B-DISPATCH-001: dispatch follows the approved resolved-policy objects."""

    dispatch = _require(sampling, "sample_innovations")
    build_random_streams = _require(sampling, "build_random_streams")
    factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_variances = np.array([0.02, 0.03, 0.01])
    independent_policy = _policy()
    independent_streams = build_random_streams(
        seed=SEED,
        path_count=2,
        joint_policy=JointDistributionPolicy.INDEPENDENT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )

    independent = dispatch(
        resolved_policy=independent_policy,
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        step_count=4,
        rng_streams=independent_streams,
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )
    assert independent.sampler_metadata["sampler_name"] == "independent_gaussian"

    joint_policy = _policy(joint=True)
    joint_streams = build_random_streams(
        seed=SEED,
        path_count=2,
        joint_policy=JointDistributionPolicy.JOINT,
        bit_generator_name="PCG64",
        stream_policy="per_path",
    )
    joint = dispatch(
        resolved_policy=joint_policy,
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        step_count=4,
        rng_streams=joint_streams,
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )
    assert joint.sampler_metadata["sampler_name"] == "block_diagonal_gaussian"
    assert joint.sampler_metadata["sampler_version"] == "1"
