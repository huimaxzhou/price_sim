"""Focused deterministic preflight tests for implementation-plan Step 4."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from factor_sim.config import (
    InvalidReturnPolicy,
    JointDistributionPolicy,
    resolve_policy,
)
from factor_sim.errors import ValidationError
from factor_sim.models import (
    AssetMetadata,
    FactorMetadata,
    JointGeneratorConfig,
    NumericalTolerances,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
)
from factor_sim.validation import validate_request

pytestmark = pytest.mark.unit


def _issue_pairs(error: ValidationError) -> set[tuple[str, str]]:
    return {(issue.code, issue.field_path) for issue in error.issues}


def _replace_state(request: SimulationRequest, **changes: object) -> SimulationRequest:
    return dataclasses.replace(
        request, risk_state=dataclasses.replace(request.risk_state, **changes)
    )


def test_valid_request_returns_reusable_immutable_context(
    request_factory: Callable[..., SimulationRequest],
    policy_factory: Callable[..., object],
) -> None:
    request = request_factory()
    context = validate_request(request, policy_factory())
    expected = (
        request.risk_state.exposures
        @ request.risk_state.factor_covariance
        @ request.risk_state.exposures.T
    )
    expected[np.diag_indices(3)] += request.risk_state.specific_variances

    assert context.dimensions == {"M": 2, "H": 4, "N": 3, "K": 2}
    assert context.resolved_future_dates == request.schedule.explicit_future_dates
    assert context.calendar_id == "XNYS-fixture"
    np.testing.assert_allclose(context.target_asset_covariance, expected)
    assert context.target_asset_covariance.flags.owndata
    assert not context.target_asset_covariance.flags.writeable
    assert context.estimated_peak_memory_bytes > 0


@pytest.mark.parametrize(
    ("case", "code", "field_path"),
    [
        ("empty_run_id", "missing_required_value", "run_id"),
        ("nonpositive_path_count", "nonpositive_dimension", "path_count"),
        ("duplicate_assets", "duplicate_identifier", "risk_state.asset_ids"),
        ("bad_exposure_shape", "invalid_shape", "risk_state.exposures"),
        ("metadata_order", "identifier_misalignment", "asset_metadata"),
        ("nonfinite", "nonfinite_value", "risk_state.exposures"),
        ("nonpositive_price", "nonpositive_initial_price", "initial_prices"),
        (
            "negative_specific",
            "negative_specific_variance",
            "risk_state.specific_variances",
        ),
        ("asymmetric_factor", "asymmetric_covariance", "risk_state.factor_covariance"),
        ("indefinite_factor", "non_psd_covariance", "risk_state.factor_covariance"),
        (
            "wrong_horizon",
            "incompatible_return_horizon",
            "risk_state.return_horizon",
        ),
        (
            "wrong_units",
            "incompatible_covariance_units",
            "risk_state.covariance_units",
        ),
        ("lookahead", "risk_state_lookahead", "risk_state.known_at"),
        (
            "missing_currency",
            "missing_asset_currency",
            "asset_metadata[1].asset_currency",
        ),
        ("unsupported_rng", "unsupported_rng_bit_generator", "rng_bit_generator"),
        ("negative_tolerance", "negative_tolerance", "numerical_tolerances.symmetry"),
        ("memory", "memory_limit_exceeded", "memory_limit_bytes"),
    ],
)
def test_preflight_rules_report_stable_field_addressed_issues(
    case: str,
    code: str,
    field_path: str,
    request_factory: Callable[..., SimulationRequest],
    policy_factory: Callable[..., object],
) -> None:
    request = request_factory()
    if case == "empty_run_id":
        request = dataclasses.replace(request, run_id="")
    elif case == "nonpositive_path_count":
        request = dataclasses.replace(request, path_count=0)
    elif case == "duplicate_assets":
        request = _replace_state(request, asset_ids=("asset-a", "asset-a", "asset-c"))
    elif case == "bad_exposure_shape":
        request = _replace_state(request, exposures=np.ones((3, 1)))
    elif case == "metadata_order":
        request = dataclasses.replace(
            request, asset_metadata=tuple(reversed(request.asset_metadata))
        )
    elif case == "nonfinite":
        exposures = np.array(request.risk_state.exposures, copy=True)
        exposures[0, 0] = np.nan
        request = _replace_state(request, exposures=exposures)
    elif case == "nonpositive_price":
        request = dataclasses.replace(request, initial_prices=np.array([0.0, 1.0, 1.0]))
    elif case == "negative_specific":
        request = _replace_state(
            request, specific_variances=np.array([0.02, -0.01, 0.01])
        )
    elif case == "asymmetric_factor":
        request = _replace_state(
            request, factor_covariance=np.array([[0.04, 0.02], [0.01, 0.09]])
        )
    elif case == "indefinite_factor":
        request = _replace_state(
            request, factor_covariance=np.array([[1.0, 2.0], [2.0, 1.0]])
        )
    elif case == "wrong_horizon":
        request = _replace_state(request, return_horizon="monthly")
    elif case == "wrong_units":
        request = _replace_state(request, covariance_units="annualized")
    elif case == "lookahead":
        request = _replace_state(request, known_at="2026-10-22T10:00:00-04:00")
    elif case == "missing_currency":
        metadata = list(request.asset_metadata)
        metadata[1] = AssetMetadata(asset_id="asset-b", asset_currency="")
        request = dataclasses.replace(request, asset_metadata=tuple(metadata))
    elif case == "unsupported_rng":
        request = dataclasses.replace(request, rng_bit_generator="MT19937")
    elif case == "negative_tolerance":
        request = dataclasses.replace(
            request,
            numerical_tolerances=NumericalTolerances(
                symmetry=-1.0,
                positive_semidefinite=1e-12,
                equation=1e-12,
            ),
        )
    elif case == "memory":
        request = dataclasses.replace(request, memory_limit_bytes=1)

    with pytest.raises(ValidationError) as captured:
        validate_request(request, policy_factory())

    assert (code, field_path) in _issue_pairs(captured.value)


def test_schedule_methods_resolve_against_supplied_calendar(
    request_factory: Callable[..., SimulationRequest],
    policy_factory: Callable[..., object],
) -> None:
    count_request = request_factory(
        schedule=SimulationSchedule(trading_day_count=3),
        factor_means=np.zeros((3, 2)),
        specific_means=np.zeros((3, 3)),
    )
    end_request = request_factory(
        schedule=SimulationSchedule(simulation_end_date="2026-10-27"),
        factor_means=np.zeros((3, 2)),
        specific_means=np.zeros((3, 3)),
    )

    expected = ("2026-10-23", "2026-10-26", "2026-10-27")
    assert (
        validate_request(count_request, policy_factory()).resolved_future_dates
        == expected
    )
    assert (
        validate_request(end_request, policy_factory()).resolved_future_dates
        == expected
    )


def test_invalid_schedule_and_multiple_errors_are_aggregated_before_rng(
    monkeypatch: pytest.MonkeyPatch,
    request_factory: Callable[..., SimulationRequest],
    policy_factory: Callable[..., object],
) -> None:
    def unexpected_rng(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("validation initialized random state")

    monkeypatch.setattr(np.random, "SeedSequence", unexpected_rng)
    request = request_factory(
        schedule=SimulationSchedule(
            explicit_future_dates=("2026-10-23",), trading_day_count=1
        ),
        path_count=0,
        initial_prices=np.array([0.0, np.nan, 75.0]),
    )

    with pytest.raises(ValidationError) as captured:
        validate_request(request, policy_factory())

    codes = {issue.code for issue in captured.value.issues}
    assert {
        "invalid_schedule_selection",
        "nonpositive_dimension",
        "nonfinite_value",
        "nonpositive_initial_price",
    } <= codes


def test_boundary_psd_optional_metadata_and_factor_class_conflict(
    request_factory: Callable[..., SimulationRequest],
    policy_factory: Callable[..., object],
) -> None:
    request = request_factory()
    boundary = _replace_state(
        request,
        factor_covariance=np.array([[0.04, 0.04], [0.04, 0.04]]),
        specific_variances=np.array([0.02, 0.0, 0.01]),
    )
    context = validate_request(boundary, policy_factory())
    assert context.validation_summaries["factor_covariance_is_singular"] is True
    assert context.validation_summaries["zero_specific_variance_count"] == 1

    conflict = (
        FactorMetadata(
            factor_id="market", factor_class="market", exposure_class="style"
        ),
        FactorMetadata(factor_id="value", factor_class="style"),
    )
    with pytest.raises(ValidationError) as captured:
        validate_request(request_factory(factor_metadata=conflict), policy_factory())
    assert (
        "conflicting_factor_classification",
        "factor_metadata[0]",
    ) in _issue_pairs(captured.value)


def test_joint_policy_rejects_nonzero_factor_specific_cross_covariance(
    request_factory: Callable[..., SimulationRequest],
) -> None:
    cross_covariance = np.zeros((2, 3))
    cross_covariance[0, 0] = 0.001
    supplied = SimulationPolicy(
        invalid_return_policy=InvalidReturnPolicy.FAIL,
        return_joint_distribution=JointDistributionPolicy.JOINT,
        joint_generator=JointGeneratorConfig(
            name="block_diagonal_gaussian",
            version="1",
            parameters={"factor_specific_cross_covariance": cross_covariance.tolist()},
            marginal_contract={
                "factor": "Gaussian",
                "specific": "Gaussian",
                "factor_specific_cross_covariance": "zero",
            },
            random_stream_requirements={"stream": "joint"},
        ),
    )

    with pytest.raises(ValidationError) as captured:
        validate_request(request_factory(policy=supplied), resolve_policy(supplied))

    assert (
        "nonzero_factor_specific_cross_covariance",
        "policy.joint_generator.parameters.factor_specific_cross_covariance",
    ) in _issue_pairs(captured.value)
