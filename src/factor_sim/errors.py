"""Structured exception types used by the simulator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable, field-addressed validation or configuration issue."""

    code: str
    field_path: str
    message: str
    context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.context is not None:
            object.__setattr__(
                self,
                "context",
                MappingProxyType(dict(self.context)),
            )


class SimulationError(Exception):
    """Base class for expected simulator failures."""


class _IssueCollectionError(SimulationError):
    """Base class for failures that report one or more structured issues."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("at least one validation issue is required")
        message = "; ".join(
            f"{issue.field_path}: {issue.message}" for issue in self.issues
        )
        super().__init__(message)


class ConfigurationError(_IssueCollectionError):
    """The supplied or resolved simulation configuration is invalid."""


class ValidationError(_IssueCollectionError):
    """One or more deterministic simulation-input checks failed."""


class NumericalError(SimulationError):
    """A numerical operation could not satisfy its declared contract."""


class PolicyExecutionError(SimulationError):
    """A supported policy failed while being applied."""
