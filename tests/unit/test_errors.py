"""Focused tests for structured simulator errors."""

from types import MappingProxyType

import pytest

from factor_sim.errors import (
    ConfigurationError,
    NumericalError,
    PolicyExecutionError,
    SimulationError,
    ValidationError,
    ValidationIssue,
)


def test_issue_copies_and_freezes_safe_context() -> None:
    context = {"actual_shape": (2, 3)}
    issue = ValidationIssue(
        code="invalid_shape",
        field_path="risk_state.exposures",
        message="unexpected shape",
        context=context,
    )

    context["actual_shape"] = (9, 9)

    assert issue.context == {"actual_shape": (2, 3)}
    assert isinstance(issue.context, MappingProxyType)
    with pytest.raises(TypeError):
        issue.context["extra"] = True  # type: ignore[index]


def test_issue_collection_error_preserves_order_and_formats_message() -> None:
    issues = (
        ValidationIssue("first", "field.one", "first problem"),
        ValidationIssue("second", "field.two", "second problem"),
    )

    error = ValidationError(issues)

    assert error.issues == issues
    assert str(error) == "field.one: first problem; field.two: second problem"


def test_issue_collection_error_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ConfigurationError(())


@pytest.mark.parametrize(
    "error_type",
    (ConfigurationError, ValidationError, NumericalError, PolicyExecutionError),
)
def test_public_errors_share_one_root(error_type: type[Exception]) -> None:
    assert issubclass(error_type, SimulationError)
