"""Focused tests for Packet A policy resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from factor_sim.config import (
    ExposureDynamics,
    InvalidReturnPolicy,
    JointDistributionPolicy,
    resolve_policy,
)
from factor_sim.errors import ConfigurationError
from factor_sim.models import JointGeneratorConfig, SimulationPolicy


def _minimal_policy(**overrides: object) -> SimulationPolicy:
    values: dict[str, object] = {
        "invalid_return_policy": InvalidReturnPolicy.FAIL
    }
    values.update(overrides)
    return SimulationPolicy(**values)


def _codes(error: ConfigurationError) -> set[str]:
    return {issue.code for issue in error.issues}


def test_minimal_policy_resolves_defaults_and_provenance() -> None:
    resolved = resolve_policy(_minimal_policy())

    assert resolved.factor_exposure_dynamics == "frozen"
    assert resolved.factor_return_distribution == "Gaussian"
    assert resolved.return_joint_distribution == "independent"
    assert resolved.price_return_convention == "raw"
    assert resolved.currency_handling is None
    assert resolved.value_sources["factor_exposure_dynamics"] == "default"
    assert resolved.value_sources["invalid_return_policy"] == "explicit"


def test_explicit_none_is_distinct_from_omission() -> None:
    resolved = resolve_policy(_minimal_policy(currency_handling=None))

    assert resolved.currency_handling is None
    assert resolved.value_sources["currency_handling"] == "explicit"
    assert resolved.value_sources["corporate_action_adjustment"] == "default"


def test_explicit_enum_values_are_preserved() -> None:
    resolved = resolve_policy(
        SimulationPolicy(
            factor_exposure_dynamics=ExposureDynamics.FROZEN,
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )
    )

    assert resolved.factor_exposure_dynamics is ExposureDynamics.FROZEN
    assert resolved.invalid_return_policy is InvalidReturnPolicy.FAIL


def test_resolution_is_pure_and_returns_distinct_objects() -> None:
    supplied = _minimal_policy(currency_handling=None)

    first = resolve_policy(supplied)
    second = resolve_policy(supplied)

    assert first == second
    assert first is not second
    assert supplied.currency_handling is None
    with pytest.raises(FrozenInstanceError):
        supplied.currency_handling = "fx"  # type: ignore[misc]


def test_missing_and_wrong_type_fields_are_reported_together() -> None:
    with pytest.raises(ConfigurationError) as captured:
        resolve_policy(
            SimulationPolicy(
                factor_exposure_dynamics="frozen-ish",
                price_return_convention="log",
            )
        )

    assert _codes(captured.value) == {
        "missing_required_policy",
        "invalid_policy_type",
    }


def test_joint_policy_requires_grouped_generator_configuration() -> None:
    with pytest.raises(ConfigurationError) as captured:
        resolve_policy(
            _minimal_policy(return_joint_distribution=JointDistributionPolicy.JOINT)
        )

    assert "missing_joint_generator" in _codes(captured.value)


def test_registered_joint_policy_resolves_complete_metadata() -> None:
    joint_generator = JointGeneratorConfig(
        name="block_diagonal_gaussian",
        version="1",
        parameters={},
        marginal_contract={"factor": "Gaussian", "specific": "Gaussian"},
        random_stream_requirements={"stream_count": 1},
    )
    resolved = resolve_policy(
        _minimal_policy(
            return_joint_distribution=JointDistributionPolicy.JOINT,
            joint_generator=joint_generator,
        )
    )

    assert resolved.return_joint_distribution is JointDistributionPolicy.JOINT
    assert resolved.joint_generator is joint_generator
    assert resolved.value_sources["joint_generator"] == "explicit"


def test_joint_metadata_is_rejected_under_independent_policy() -> None:
    joint_generator = JointGeneratorConfig(
        name="block_diagonal_gaussian",
        version="1",
        parameters={},
        marginal_contract={},
        random_stream_requirements={},
    )
    with pytest.raises(ConfigurationError) as captured:
        resolve_policy(_minimal_policy(joint_generator=joint_generator))

    assert "incompatible_policy_parameter" in _codes(captured.value)


@pytest.mark.parametrize(
    "field_name",
    ("currency_handling", "corporate_action_adjustment", "new_listing"),
)
def test_non_none_handlers_are_rejected(field_name: str) -> None:
    with pytest.raises(ConfigurationError) as captured:
        resolve_policy(_minimal_policy(**{field_name: "unsupported-handler"}))

    assert "unsupported_policy_value" in _codes(captured.value)
