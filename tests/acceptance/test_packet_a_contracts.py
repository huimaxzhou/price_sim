"""Approved executable contract for implementation-plan Steps 1 through 3."""

from __future__ import annotations

import copy
import dataclasses
import enum
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import factor_sim.config as config
import factor_sim.errors as errors
import factor_sim.models as models

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]


def _require(module: ModuleType, name: str) -> Any:
    """Return a required symbol or fail with a focused contract message."""

    assert hasattr(module, name), f"Packet A requires {module.__name__}.{name}"
    return getattr(module, name)


def _enum_value(value: object) -> object:
    """Expose an enum's serialized value without requiring a specific enum base."""

    return value.value if isinstance(value, enum.Enum) else value


def _issue_codes(exc: BaseException) -> tuple[str, ...]:
    """Read stable issue codes from a structured configuration error."""

    issues = getattr(exc, "issues", ())
    return tuple(getattr(issue, "code", "") for issue in issues)


def _minimal_policy(**overrides: object) -> object:
    """Construct the minimally valid policy plus explicit test overrides."""

    simulation_policy = _require(models, "SimulationPolicy")
    invalid_policy = _require(config, "InvalidReturnPolicy")
    values: dict[str, object] = {"invalid_return_policy": invalid_policy.FAIL}
    values.update(overrides)
    return simulation_policy(**values)


def _resolve_policy() -> Callable[[object], object]:
    return _require(config, "resolve_policy")


def _default_expectations() -> dict[str, object]:
    """Return the v4 defaults using the proposed Packet A enum names."""

    return {
        "factor_exposure_dynamics": _require(config, "ExposureDynamics").FROZEN,
        "factor_volatility_dynamics": _require(
            config, "FactorVolatilityDynamics"
        ).FROZEN,
        "factor_correlation_dynamics": _require(
            config, "FactorCorrelationDynamics"
        ).FROZEN,
        "specific_risk_dynamics": _require(config, "SpecificRiskDynamics").FROZEN,
        "factor_return_distribution": _require(
            config, "FactorReturnDistribution"
        ).GAUSSIAN,
        "specific_return_distribution": _require(
            config, "SpecificReturnDistribution"
        ).GAUSSIAN,
        "return_serial_correlation": _require(
            config, "SerialCorrelationPolicy"
        ).NONE,
        "return_joint_distribution": _require(
            config, "JointDistributionPolicy"
        ).INDEPENDENT,
        "price_return_convention": _require(
            config, "PriceReturnConvention"
        ).RAW,
        "currency_handling": None,
        "corporate_action_adjustment": None,
        "new_listing": None,
    }


def _risk_state() -> object:
    risk_state_type = _require(models, "RiskState")
    return risk_state_type(
        exposures=np.array([[1.0]]),
        factor_covariance=np.array([[0.04]]),
        specific_variances=np.array([0.01]),
        asset_ids=("asset-a",),
        factor_ids=("factor-a",),
        as_of_date="2026-10-22",
        known_at="2026-10-22T16:00:00-04:00",
        return_horizon="one_trading_day",
        observation_frequency="daily",
        return_convention="arithmetic",
        return_basis="raw_price",
        covariance_units="one_trading_day_return_squared",
        annualization_state="deannualized_daily",
        provenance={"model": "fixture"},
    )


def _joint_generator(*, version: str = "1") -> object:
    joint_type = _require(models, "JointGeneratorConfig")
    return joint_type(
        name="block_diagonal_gaussian",
        version=version,
        parameters={},
        marginal_contract={"factor": "Gaussian", "specific": "Gaussian"},
        random_stream_requirements={"stream": "joint"},
    )


def test_a_boot_001_package_metadata_declares_approved_toolchain() -> None:
    """A-BOOT-001: package metadata contains only the approved dependencies."""

    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    def dependency_name(item: str) -> str:
        return re.split(r"[<>=!~;\[]", item, maxsplit=1)[0].lower()

    production_names = {
        dependency_name(item) for item in project["project"]["dependencies"]
    }
    development_names = {
        dependency_name(item)
        for item in project["project"]["optional-dependencies"]["dev"]
    }

    assert project["project"]["requires-python"] == ">=3.11"
    assert production_names == {"numpy"}
    assert development_names == {"pytest", "ruff"}
    assert sys.version_info >= (3, 11)


def test_a_boot_002_planned_modules_import() -> None:
    """A-BOOT-002: all shallow architecture modules are importable."""

    module_names = (
        "config",
        "diagnostics",
        "engine",
        "errors",
        "io",
        "models",
        "pricing",
        "sampling",
        "validation",
    )
    for name in module_names:
        __import__(f"factor_sim.{name}")


@pytest.mark.parametrize(
    "name",
    [
        "SimulationError",
        "ConfigurationError",
        "ValidationError",
        "NumericalError",
        "PolicyExecutionError",
    ],
)
def test_a_err_001_required_exception_hierarchy(name: str) -> None:
    """A-ERR-001: every public simulator exception derives from one root."""

    simulation_error = _require(errors, "SimulationError")
    exception_type = _require(errors, name)
    assert issubclass(simulation_error, Exception)
    assert issubclass(exception_type, simulation_error)


def test_a_err_002_validation_error_preserves_structured_issues() -> None:
    """A-ERR-002: one exception carries multiple safe validation issues."""

    issue_type = _require(errors, "ValidationIssue")
    validation_error = _require(errors, "ValidationError")
    issues = (
        issue_type(
            code="invalid_shape",
            field_path="risk_state.exposures",
            message="expected an N by K matrix",
            context={"actual_shape": (2, 3)},
        ),
        issue_type(
            code="missing_policy",
            field_path="policy.invalid_return_policy",
            message="an explicit policy is required",
            context=None,
        ),
    )

    exc = validation_error(issues)

    assert tuple(exc.issues) == issues
    assert exc.issues[0].code == "invalid_shape"
    assert exc.issues[0].field_path == "risk_state.exposures"
    assert exc.issues[0].message == "expected an N by K matrix"
    assert exc.issues[0].context == {"actual_shape": (2, 3)}


@pytest.mark.parametrize(
    ("enum_name", "members"),
    [
        ("ExposureDynamics", {"FROZEN": "frozen"}),
        ("FactorVolatilityDynamics", {"FROZEN": "frozen"}),
        ("FactorCorrelationDynamics", {"FROZEN": "frozen"}),
        ("SpecificRiskDynamics", {"FROZEN": "frozen"}),
        ("FactorReturnDistribution", {"GAUSSIAN": "Gaussian"}),
        ("SpecificReturnDistribution", {"GAUSSIAN": "Gaussian"}),
        ("SerialCorrelationPolicy", {"NONE": "none"}),
        (
            "JointDistributionPolicy",
            {"INDEPENDENT": "independent", "JOINT": "joint"},
        ),
        ("PriceReturnConvention", {"RAW": "raw"}),
        ("InvalidReturnPolicy", {"FAIL": "fail"}),
        (
            "AssetStatus",
            {"LIVE": "live", "TERMINAL": "terminal", "DELISTED": "delisted"},
        ),
    ],
)
def test_a_enum_001_enums_have_stable_string_values(
    enum_name: str, members: dict[str, str]
) -> None:
    """A-ENUM-001: enum values are stable strings suitable for manifests."""

    enum_type = _require(config, enum_name)
    assert issubclass(enum_type, str)
    assert issubclass(enum_type, enum.Enum)
    actual = {
        name: member.value for name, member in enum_type.__members__.items()
    }
    assert actual == members


@pytest.mark.parametrize(
    "name",
    [
        "RiskState",
        "AssetMetadata",
        "FactorMetadata",
        "SimulationSchedule",
        "TradingCalendar",
        "JointGeneratorConfig",
        "SimulationPolicy",
        "ResolvedSimulationPolicy",
        "NumericalTolerances",
        "SimulationRequest",
        "InnovationBatch",
        "AssetStateEvent",
        "ValidationReport",
        "SimulationResult",
    ],
)
def test_a_model_001_required_models_are_frozen_dataclasses(name: str) -> None:
    """A-MODEL-001: public domain objects are immutable typed dataclasses."""

    model_type = _require(models, name)
    assert dataclasses.is_dataclass(model_type)
    parameters = getattr(model_type, "__dataclass_params__")
    assert parameters.frozen is True
    assert model_type.__annotations__, f"{name} must declare typed fields"


def test_a_model_002_risk_state_preserves_order_and_owns_arrays() -> None:
    """A-MODEL-002: RiskState is isolated from caller array mutation."""

    risk_state_type = _require(models, "RiskState")
    exposures = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    specific_variances = np.array([0.16, 0.25])

    state = risk_state_type(
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        asset_ids=("asset-b", "asset-a"),
        factor_ids=("factor-2", "factor-1"),
        as_of_date="2026-10-22",
        known_at="2026-10-22T16:00:00-04:00",
        return_horizon="one_trading_day",
        observation_frequency="daily",
        return_convention="arithmetic",
        return_basis="raw_price",
        covariance_units="one_trading_day_return_squared",
        annualization_state="deannualized_daily",
        provenance={"model": "fixture"},
    )

    exposures[0, 0] = 999.0
    factor_covariance[0, 0] = 999.0
    specific_variances[0] = 999.0

    assert state.asset_ids == ("asset-b", "asset-a")
    assert state.factor_ids == ("factor-2", "factor-1")
    assert state.exposures[0, 0] == 1.0
    assert state.factor_covariance[0, 0] == 0.04
    assert state.specific_variances[0] == 0.16
    assert state.exposures.flags.writeable is False
    assert state.factor_covariance.flags.writeable is False
    assert state.specific_variances.flags.writeable is False


def test_a_model_003_factor_metadata_is_optional() -> None:
    """A-MODEL-003: request construction accepts absent factor metadata."""

    request_type = _require(models, "SimulationRequest")
    field = next(
        candidate
        for candidate in dataclasses.fields(request_type)
        if candidate.name == "factor_metadata"
    )
    assert field.default is None


def test_a_model_004_asset_state_event_has_required_fields() -> None:
    """A-MODEL-004: state events carry the complete v4 identity and reason."""

    event_type = _require(models, "AssetStateEvent")
    asset_status = _require(config, "AssetStatus")
    event = event_type(
        asset_id="asset-a",
        path_id=7,
        effective_date="2026-10-23",
        status=asset_status.TERMINAL,
        reason="fixture terminal event",
    )

    assert event.asset_id == "asset-a"
    assert event.path_id == 7
    assert event.effective_date == "2026-10-23"
    assert event.status == asset_status.TERMINAL
    assert event.reason == "fixture terminal event"


def test_a_model_005_schedule_alternatives_are_representable() -> None:
    """A-MODEL-005: each supported schedule method preserves its value."""

    schedule_type = _require(models, "SimulationSchedule")
    explicit = schedule_type(
        explicit_future_dates=("2026-10-23", "2026-10-26")
    )
    count = schedule_type(trading_day_count=100)
    end = schedule_type(simulation_end_date="2027-03-31")

    assert explicit.explicit_future_dates == ("2026-10-23", "2026-10-26")
    assert count.trading_day_count == 100
    assert end.simulation_end_date == "2027-03-31"


def test_a_model_006_request_owns_external_numerical_inputs() -> None:
    """A-MODEL-006: accepted request arrays are copied once and protected."""

    request_type = _require(models, "SimulationRequest")
    schedule_type = _require(models, "SimulationSchedule")
    calendar_type = _require(models, "TradingCalendar")
    asset_metadata_type = _require(models, "AssetMetadata")
    tolerances_type = _require(models, "NumericalTolerances")
    initial_prices = np.array([100.0])
    factor_means = np.array([[0.01]])
    specific_means = np.array([[0.02]])

    request = request_type(
        run_id="run-a",
        risk_model_name="fixture-model",
        risk_model_version="1",
        simulation_origin="2026-10-22",
        schedule=schedule_type(trading_day_count=1),
        trading_calendar=calendar_type(
            calendar_id="fixture-calendar",
            time_zone="America/New_York",
            sessions=("2026-10-23",),
            provenance={"source": "fixture"},
        ),
        path_count=1,
        initial_prices=initial_prices,
        factor_means=factor_means,
        specific_means=specific_means,
        risk_state=_risk_state(),
        asset_metadata=(
            asset_metadata_type(asset_id="asset-a", asset_currency="USD"),
        ),
        policy=_minimal_policy(),
        random_seed=7,
        rng_bit_generator="PCG64",
        rng_stream_policy="per_path",
        numerical_tolerances=tolerances_type(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        provenance={"source": "fixture"},
    )

    initial_prices[0] = 999.0
    factor_means[0, 0] = 999.0
    specific_means[0, 0] = 999.0

    for actual, expected in (
        (request.initial_prices, np.array([100.0])),
        (request.factor_means, np.array([[0.01]])),
        (request.specific_means, np.array([[0.02]])),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
        assert actual.flags.writeable is False

    assert request.memory_limit_bytes == 1_000_000_000


def test_a_model_007_engine_owned_outputs_are_not_recopied() -> None:
    """A-MODEL-007: output objects take ownership without full-array copies."""

    batch_type = _require(models, "InnovationBatch")
    result_type = _require(models, "SimulationResult")
    report_type = _require(models, "ValidationReport")
    factor_innovations = np.zeros((1, 1, 1), dtype=np.float64)
    specific_innovations = np.zeros((1, 1, 1), dtype=np.float64)
    batch = batch_type(
        factor_innovations=factor_innovations,
        specific_innovations=specific_innovations,
        sampler_metadata={},
    )

    assert batch.factor_innovations is factor_innovations
    assert batch.specific_innovations is specific_innovations
    assert batch.factor_innovations.flags.writeable is False
    assert batch.specific_innovations.flags.writeable is False

    arrays = {
        "factor_innovations": np.zeros((1, 1, 1), dtype=np.float64),
        "specific_innovations": np.zeros((1, 1, 1), dtype=np.float64),
        "factor_returns": np.zeros((1, 1, 1), dtype=np.float64),
        "specific_returns": np.zeros((1, 1, 1), dtype=np.float64),
        "modeled_returns": np.zeros((1, 1, 1), dtype=np.float64),
        "price_link_returns": np.zeros((1, 1, 1), dtype=np.float64),
        "prices": np.ones((1, 2, 1), dtype=np.float64),
        "target_asset_covariance": np.eye(1, dtype=np.float64),
    }
    result = result_type(
        **arrays,
        asset_state_events=(),
        resolved_policy=_resolve_policy()(_minimal_policy()),
        manifest={},
        validation_report=report_type(issues=(), summaries={}),
        asset_ids=("asset-a",),
        factor_ids=("factor-a",),
        future_dates=("2026-10-23",),
        asset_currencies=("USD",),
        risk_state=_risk_state(),
    )

    for field_name, supplied_array in arrays.items():
        actual = getattr(result, field_name)
        assert actual is supplied_array
        assert actual.flags.writeable is False

    with pytest.raises(TypeError):
        batch_type(
            factor_innovations=[[[0.0]]],
            specific_innovations=np.zeros((1, 1, 1), dtype=np.float64),
            sampler_metadata={},
        )
    with pytest.raises(TypeError):
        batch_type(
            factor_innovations=np.zeros((1, 1, 1), dtype=np.float32),
            specific_innovations=np.zeros((1, 1, 1), dtype=np.float64),
            sampler_metadata={},
        )


def test_a_policy_001_minimal_policy_resolves_documented_defaults() -> None:
    """A-POLICY-001: only fail is mandatory; every v4 default is recorded."""

    resolved = _resolve_policy()(_minimal_policy())

    for field_name, expected in _default_expectations().items():
        assert _enum_value(getattr(resolved, field_name)) == _enum_value(expected)
        assert resolved.value_sources[field_name] == "default"
    assert _enum_value(resolved.invalid_return_policy) == "fail"
    assert resolved.value_sources["invalid_return_policy"] == "explicit"


def test_a_policy_002_explicit_defaults_remain_explicit() -> None:
    """A-POLICY-002: explicit default values are not reported as omissions."""

    defaults = _default_expectations()
    supplied = _minimal_policy(**defaults)
    resolved = _resolve_policy()(supplied)

    expected_fields = set(defaults) | {"invalid_return_policy"}
    assert all(resolved.value_sources[name] == "explicit" for name in expected_fields)


def test_a_policy_003_missing_invalid_return_policy_is_rejected() -> None:
    """A-POLICY-003: the invalid-return policy has no implicit default."""

    simulation_policy = _require(models, "SimulationPolicy")
    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(simulation_policy())

    assert "missing_required_policy" in _issue_codes(captured.value)


def test_a_policy_004_typed_boundary_rejects_wrong_runtime_type() -> None:
    """A-POLICY-004: core enum fields do not parse arbitrary strings."""

    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(_minimal_policy(factor_exposure_dynamics="frozen"))

    issue = captured.value.issues[0]
    assert issue.code == "invalid_policy_type"
    assert issue.field_path == "policy.factor_exposure_dynamics"


def test_a_policy_005_joint_policy_requires_grouped_generator() -> None:
    """A-POLICY-005: joint mode cannot infer an omitted generator."""

    joint_policy = _require(config, "JointDistributionPolicy")
    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(
            _minimal_policy(return_joint_distribution=joint_policy.JOINT)
        )

    assert "missing_joint_generator" in _issue_codes(captured.value)


def test_a_policy_006_registered_joint_configuration_resolves() -> None:
    """A-POLICY-006: one registered grouped joint configuration is preserved."""

    joint_policy = _require(config, "JointDistributionPolicy")
    joint_generator = _joint_generator()
    resolved = _resolve_policy()(
        _minimal_policy(
            return_joint_distribution=joint_policy.JOINT,
            joint_generator=joint_generator,
        )
    )

    assert resolved.joint_generator is joint_generator
    assert resolved.value_sources["joint_generator"] == "explicit"


def test_a_policy_007_joint_generator_version_must_match() -> None:
    """A-POLICY-007: a registered name does not authorize another version."""

    joint_policy = _require(config, "JointDistributionPolicy")
    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(
            _minimal_policy(
                return_joint_distribution=joint_policy.JOINT,
                joint_generator=_joint_generator(version="999"),
            )
        )

    issue = captured.value.issues[0]
    assert issue.code == "unsupported_joint_generator_version"
    assert issue.field_path == "policy.joint_generator.version"


def test_a_policy_008_independent_mode_rejects_joint_configuration() -> None:
    """A-POLICY-008: joint metadata is incompatible with independent mode."""

    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(_minimal_policy(joint_generator=_joint_generator()))

    issue = captured.value.issues[0]
    assert issue.code == "incompatible_policy_parameter"
    assert issue.field_path == "policy.joint_generator"


@pytest.mark.parametrize(
    "field_name",
    ["currency_handling", "corporate_action_adjustment", "new_listing"],
)
def test_a_policy_009_non_none_handlers_are_unsupported(field_name: str) -> None:
    """A-POLICY-009: version one supports no active optional handlers."""

    configuration_error = _require(errors, "ConfigurationError")

    with pytest.raises(configuration_error) as captured:
        _resolve_policy()(_minimal_policy(**{field_name: "custom"}))

    issue = captured.value.issues[0]
    assert issue.code == "unsupported_policy_value"
    assert issue.field_path == f"policy.{field_name}"


def test_a_policy_010_resolution_is_pure_and_deterministic() -> None:
    """A-POLICY-010: repeated resolution does not mutate supplied configuration."""

    supplied = _minimal_policy(currency_handling=None)
    snapshot = copy.deepcopy(supplied)

    first = _resolve_policy()(supplied)
    second = _resolve_policy()(supplied)

    assert supplied == snapshot
    assert first == second
    assert first is not second
    assert first.value_sources["currency_handling"] == "explicit"
