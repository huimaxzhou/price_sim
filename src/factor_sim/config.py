"""Typed policy vocabulary and pure default resolution.

The core accepts enum members and typed configuration objects. Parsing strings
or serialized configuration belongs to a future adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from .errors import ConfigurationError, ValidationIssue
from .models import (
    JointGeneratorConfig,
    ResolvedSimulationPolicy,
    SimulationPolicy,
    _is_unset,
)


class ExposureDynamics(StrEnum):
    FROZEN = "frozen"


class FactorVolatilityDynamics(StrEnum):
    FROZEN = "frozen"


class FactorCorrelationDynamics(StrEnum):
    FROZEN = "frozen"


class SpecificRiskDynamics(StrEnum):
    FROZEN = "frozen"


class FactorReturnDistribution(StrEnum):
    GAUSSIAN = "Gaussian"


class SpecificReturnDistribution(StrEnum):
    GAUSSIAN = "Gaussian"


class SerialCorrelationPolicy(StrEnum):
    NONE = "none"


class JointDistributionPolicy(StrEnum):
    INDEPENDENT = "independent"
    JOINT = "joint"


class PriceReturnConvention(StrEnum):
    RAW = "raw"


class InvalidReturnPolicy(StrEnum):
    FAIL = "fail"


class AssetStatus(StrEnum):
    LIVE = "live"
    TERMINAL = "terminal"
    DELISTED = "delisted"


SUPPORTED_JOINT_GENERATORS: Mapping[str, str] = MappingProxyType(
    {"block_diagonal_gaussian": "1"}
)

_ENUM_DEFAULTS: Mapping[str, StrEnum] = MappingProxyType(
    {
        "factor_exposure_dynamics": ExposureDynamics.FROZEN,
        "factor_volatility_dynamics": FactorVolatilityDynamics.FROZEN,
        "factor_correlation_dynamics": FactorCorrelationDynamics.FROZEN,
        "specific_risk_dynamics": SpecificRiskDynamics.FROZEN,
        "factor_return_distribution": FactorReturnDistribution.GAUSSIAN,
        "specific_return_distribution": SpecificReturnDistribution.GAUSSIAN,
        "return_serial_correlation": SerialCorrelationPolicy.NONE,
        "return_joint_distribution": JointDistributionPolicy.INDEPENDENT,
        "price_return_convention": PriceReturnConvention.RAW,
    }
)

_ENUM_TYPES: Mapping[str, type[StrEnum]] = MappingProxyType(
    {
        "factor_exposure_dynamics": ExposureDynamics,
        "factor_volatility_dynamics": FactorVolatilityDynamics,
        "factor_correlation_dynamics": FactorCorrelationDynamics,
        "specific_risk_dynamics": SpecificRiskDynamics,
        "factor_return_distribution": FactorReturnDistribution,
        "specific_return_distribution": SpecificReturnDistribution,
        "return_serial_correlation": SerialCorrelationPolicy,
        "return_joint_distribution": JointDistributionPolicy,
        "price_return_convention": PriceReturnConvention,
    }
)

_NONE_DEFAULT_FIELDS = (
    "currency_handling",
    "corporate_action_adjustment",
    "new_listing",
)


def _invalid_type_issue(field_name: str, value: object) -> ValidationIssue:
    return ValidationIssue(
        code="invalid_policy_type",
        field_path=f"policy.{field_name}",
        message="the direct Python API requires the declared enum type",
        context={"type": type(value).__name__},
    )


def _resolve_defaulted_enum(
    supplied: SimulationPolicy,
    field_name: str,
    issues: list[ValidationIssue],
    value_sources: dict[str, str],
) -> StrEnum | None:
    supplied_value = getattr(supplied, field_name)
    if _is_unset(supplied_value):
        value_sources[field_name] = "default"
        return _ENUM_DEFAULTS[field_name]

    value_sources[field_name] = "explicit"
    if type(supplied_value) is not _ENUM_TYPES[field_name]:
        issues.append(_invalid_type_issue(field_name, supplied_value))
        return None
    return supplied_value


def _resolve_none_handler(
    supplied: SimulationPolicy,
    field_name: str,
    issues: list[ValidationIssue],
    value_sources: dict[str, str],
) -> None:
    supplied_value = getattr(supplied, field_name)
    if _is_unset(supplied_value):
        value_sources[field_name] = "default"
        return None

    value_sources[field_name] = "explicit"
    if supplied_value is not None:
        issues.append(
            ValidationIssue(
                code="unsupported_policy_value",
                field_path=f"policy.{field_name}",
                message="only the None handler is supported in version one",
                context={"type": type(supplied_value).__name__},
            )
        )
    return None


def _resolve_required_invalid_return_policy(
    supplied: SimulationPolicy,
    issues: list[ValidationIssue],
    value_sources: dict[str, str],
) -> InvalidReturnPolicy | None:
    field_name = "invalid_return_policy"
    supplied_value = supplied.invalid_return_policy
    if _is_unset(supplied_value):
        value_sources[field_name] = "missing"
        issues.append(
            ValidationIssue(
                code="missing_required_policy",
                field_path="policy.invalid_return_policy",
                message="the fail policy must be supplied explicitly",
            )
        )
        return None

    value_sources[field_name] = "explicit"
    if type(supplied_value) is not InvalidReturnPolicy:
        issues.append(_invalid_type_issue(field_name, supplied_value))
        return None
    return supplied_value


def _resolve_joint_generator(
    supplied: SimulationPolicy,
    joint_policy: StrEnum | None,
    issues: list[ValidationIssue],
    value_sources: dict[str, str],
) -> JointGeneratorConfig | None:
    supplied_value = supplied.joint_generator
    value_sources["joint_generator"] = (
        "default" if _is_unset(supplied_value) else "explicit"
    )
    joint_generator = None if _is_unset(supplied_value) else supplied_value

    if joint_generator is not None and not isinstance(
        joint_generator, JointGeneratorConfig
    ):
        issues.append(_invalid_type_issue("joint_generator", joint_generator))
        return None

    if joint_policy == JointDistributionPolicy.INDEPENDENT:
        if joint_generator is not None:
            issues.append(
                ValidationIssue(
                    code="incompatible_policy_parameter",
                    field_path="policy.joint_generator",
                    message="joint-generator configuration requires joint mode",
                )
            )
        return None

    if joint_policy != JointDistributionPolicy.JOINT:
        return None

    if joint_generator is None:
        issues.append(
            ValidationIssue(
                code="missing_joint_generator",
                field_path="policy.joint_generator",
                message="joint mode requires a grouped generator configuration",
            )
        )
        return None

    registered_version = SUPPORTED_JOINT_GENERATORS.get(joint_generator.name)
    if registered_version is None:
        issues.append(
            ValidationIssue(
                code="unknown_joint_generator",
                field_path="policy.joint_generator.name",
                message="the named joint generator is not registered",
                context={"value": joint_generator.name},
            )
        )
    elif joint_generator.version != registered_version:
        issues.append(
            ValidationIssue(
                code="unsupported_joint_generator_version",
                field_path="policy.joint_generator.version",
                message="the joint-generator version is not registered",
                context={
                    "value": joint_generator.version,
                    "registered_version": registered_version,
                },
            )
        )
    return joint_generator


def resolve_policy(supplied_policy: SimulationPolicy) -> ResolvedSimulationPolicy:
    """Resolve documented defaults without mutating ``supplied_policy``."""

    if not isinstance(supplied_policy, SimulationPolicy):
        raise TypeError("supplied_policy must be a SimulationPolicy")

    issues: list[ValidationIssue] = []
    value_sources: dict[str, str] = {}
    resolved_enums = {
        name: _resolve_defaulted_enum(
            supplied_policy, name, issues, value_sources
        )
        for name in _ENUM_DEFAULTS
    }
    resolved_handlers = {
        name: _resolve_none_handler(
            supplied_policy, name, issues, value_sources
        )
        for name in _NONE_DEFAULT_FIELDS
    }
    invalid_return_policy = _resolve_required_invalid_return_policy(
        supplied_policy, issues, value_sources
    )
    joint_generator = _resolve_joint_generator(
        supplied_policy,
        resolved_enums["return_joint_distribution"],
        issues,
        value_sources,
    )

    if issues:
        raise ConfigurationError(issues)

    return ResolvedSimulationPolicy(
        **resolved_enums,
        invalid_return_policy=invalid_return_policy,
        **resolved_handlers,
        joint_generator=joint_generator,
        value_sources=value_sources,
    )
