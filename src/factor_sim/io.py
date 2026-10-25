"""Atomic, validated persistence for complete simulation results."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
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
from .errors import ValidationIssue
from .models import (
    FactorMetadata,
    JointGeneratorConfig,
    ResolvedSimulationPolicy,
    RiskState,
    SimulationResult,
    ValidationReport,
)

SERIALIZATION_SCHEMA_VERSION = "1"
RESULT_ARRAY_NAMES = (
    "factor_innovations",
    "specific_innovations",
    "factor_returns",
    "specific_returns",
    "modeled_returns",
    "price_link_returns",
    "prices",
    "target_asset_covariance",
)
INPUT_ARRAY_NAMES = (
    "exposures",
    "factor_covariance",
    "specific_variances",
    "initial_prices",
    "factor_means",
    "specific_means",
)


def _portable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_portable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _portable(asdict(value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            _portable(value),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"present": True, "bytes": path.stat().st_size, "sha256": digest}


def _factor_metadata(result: SimulationResult) -> object:
    if result.factor_metadata is None:
        return None
    return [_portable(item) for item in result.factor_metadata]


def _saved_manifest(result: SimulationResult) -> dict[str, object]:
    risk = result.risk_state
    return {
        "serialization_schema_version": SERIALIZATION_SCHEMA_VERSION,
        "engine_manifest": _portable(result.manifest),
        "resolved_policy": _portable(result.manifest["resolved_policy"]),
        "asset_ids": list(result.asset_ids),
        "factor_ids": list(result.factor_ids),
        "future_dates": [_portable(value) for value in result.future_dates],
        "asset_currencies": list(result.asset_currencies),
        "factor_metadata": _factor_metadata(result),
        "risk_state": {
            "asset_ids": list(risk.asset_ids),
            "factor_ids": list(risk.factor_ids),
            "as_of_date": _portable(risk.as_of_date),
            "known_at": _portable(risk.known_at),
            "return_horizon": risk.return_horizon,
            "observation_frequency": risk.observation_frequency,
            "return_convention": risk.return_convention,
            "return_basis": risk.return_basis,
            "covariance_units": risk.covariance_units,
            "annualization_state": risk.annualization_state,
            "provenance": _portable(risk.provenance),
            "exposure_units": risk.exposure_units,
        },
        "files": {},
    }


def _array_payload(result: SimulationResult) -> dict[str, np.ndarray]:
    payload = {name: getattr(result, name) for name in RESULT_ARRAY_NAMES}
    payload.update(
        {
            "exposures": result.risk_state.exposures,
            "factor_covariance": result.risk_state.factor_covariance,
            "specific_variances": result.risk_state.specific_variances,
            "initial_prices": result.initial_prices,
            "factor_means": result.factor_means,
            "specific_means": result.specific_means,
        }
    )
    return payload


def save_result(result: SimulationResult, output_dir: Path) -> Path:
    """Save ``result`` atomically into a new directory and return its path."""

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    if result.asset_state_events:
        raise ValueError(
            "asset-state events are inconsistent with version-one None handlers"
        )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        arrays_path = temporary / "arrays.npz"
        diagnostics_path = temporary / "diagnostics.json"
        np.savez(arrays_path, **_array_payload(result))
        _write_json(
            diagnostics_path,
            {
                "issues": result.validation_report.issues,
                "summaries": result.validation_report.summaries,
            },
        )
        manifest = _saved_manifest(result)
        manifest["files"] = {
            "arrays.npz": _file_record(arrays_path),
            "diagnostics.json": _file_record(diagnostics_path),
            "asset_state_events.jsonl": {
                "present": False,
                "count": 0,
                "reason": "none_handlers",
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        _read_and_validate_bundle(temporary)
        temporary.rename(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path.name}") from error


def _verify_file(path: Path, record: Mapping[str, object]) -> None:
    if not path.is_file():
        raise ValueError(f"missing saved file: {path.name}")
    if path.stat().st_size != record.get("bytes"):
        raise ValueError(f"byte-count mismatch for {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise ValueError(f"hash mismatch for {path.name}")


def _read_and_validate_bundle(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing saved file: manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("serialization_schema_version") != SERIALIZATION_SCHEMA_VERSION:
        raise ValueError("unknown serialization schema version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files inventory is missing or malformed")
    expected = {"manifest.json"}
    for name in ("arrays.npz", "diagnostics.json"):
        record = files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"manifest record is missing for {name}")
        _verify_file(output_dir / name, record)
        expected.add(name)
    event_record = files.get("asset_state_events.jsonl")
    if event_record != {"present": False, "count": 0, "reason": "none_handlers"}:
        raise ValueError("version-one event-file record is inconsistent")
    actual = {path.name for path in output_dir.iterdir()}
    if actual != expected:
        raise ValueError("saved bundle contains missing or unexpected files")
    diagnostics = _read_json(output_dir / "diagnostics.json")
    return manifest, diagnostics


def _resolved_policy(
    records: Mapping[str, Mapping[str, object]],
) -> ResolvedSimulationPolicy:
    enum_types = {
        "factor_exposure_dynamics": ExposureDynamics,
        "factor_volatility_dynamics": FactorVolatilityDynamics,
        "factor_correlation_dynamics": FactorCorrelationDynamics,
        "specific_risk_dynamics": SpecificRiskDynamics,
        "factor_return_distribution": FactorReturnDistribution,
        "specific_return_distribution": SpecificReturnDistribution,
        "return_serial_correlation": SerialCorrelationPolicy,
        "return_joint_distribution": JointDistributionPolicy,
        "price_return_convention": PriceReturnConvention,
        "invalid_return_policy": InvalidReturnPolicy,
    }
    values: dict[str, object] = {
        name: enum_type(records[name]["value"])
        for name, enum_type in enum_types.items()
    }
    values.update(
        {
            "currency_handling": None,
            "corporate_action_adjustment": None,
            "new_listing": None,
        }
    )
    joint_value = records["joint_generator"]["value"]
    values["joint_generator"] = (
        None if joint_value is None else JointGeneratorConfig(**joint_value)
    )
    values["value_sources"] = {
        name: record["value_source"] for name, record in records.items()
    }
    return ResolvedSimulationPolicy(**values)


def _validation_report(data: Mapping[str, object]) -> ValidationReport:
    issues = tuple(ValidationIssue(**item) for item in data.get("issues", []))
    summaries = dict(data["summaries"])
    if "checks" in summaries:
        summaries["checks"] = tuple(summaries["checks"])
    return ValidationReport(issues=issues, summaries=summaries)


def _factor_metadata_from_json(value: object) -> tuple[FactorMetadata, ...] | None:
    if value is None:
        return None
    return tuple(FactorMetadata(**item) for item in value)


def _owned_array(archive: Any, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"arrays.npz is missing member: {name}")
    array = np.array(archive[name], dtype=np.float64, copy=True, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"arrays.npz member is nonfinite: {name}")
    return array


def load_result(output_dir: Path) -> SimulationResult:
    """Load and validate one saved simulation result."""

    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"saved result directory does not exist: {root}")
    manifest, diagnostics = _read_and_validate_bundle(root)
    expected_names = set(RESULT_ARRAY_NAMES) | set(INPUT_ARRAY_NAMES)
    try:
        with np.load(root / "arrays.npz", allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise ValueError("arrays.npz members do not match schema version 1")
            arrays = {name: _owned_array(archive, name) for name in expected_names}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("arrays.npz"):
            raise
        raise ValueError("arrays.npz is malformed") from error

    engine_manifest = manifest["engine_manifest"]
    dimensions = engine_manifest["dimensions"]
    path_count = int(dimensions["M"])
    step_count = int(dimensions["H"])
    asset_count = int(dimensions["N"])
    factor_count = int(dimensions["K"])
    expected_shapes = {
        "factor_innovations": (path_count, step_count, factor_count),
        "specific_innovations": (path_count, step_count, asset_count),
        "factor_returns": (path_count, step_count, factor_count),
        "specific_returns": (path_count, step_count, asset_count),
        "modeled_returns": (path_count, step_count, asset_count),
        "price_link_returns": (path_count, step_count, asset_count),
        "prices": (path_count, step_count + 1, asset_count),
        "target_asset_covariance": (asset_count, asset_count),
        "exposures": (asset_count, factor_count),
        "factor_covariance": (factor_count, factor_count),
        "specific_variances": (asset_count,),
        "initial_prices": (asset_count,),
        "factor_means": (step_count, factor_count),
        "specific_means": (step_count, asset_count),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("saved array dimensions are inconsistent with the manifest")

    risk_data = manifest["risk_state"]
    risk_state = RiskState(
        exposures=arrays["exposures"],
        factor_covariance=arrays["factor_covariance"],
        specific_variances=arrays["specific_variances"],
        **risk_data,
    )
    return SimulationResult(
        **{name: arrays[name] for name in RESULT_ARRAY_NAMES},
        initial_prices=arrays["initial_prices"],
        factor_means=arrays["factor_means"],
        specific_means=arrays["specific_means"],
        asset_state_events=(),
        resolved_policy=_resolved_policy(manifest["resolved_policy"]),
        manifest=engine_manifest,
        validation_report=_validation_report(diagnostics),
        asset_ids=tuple(manifest["asset_ids"]),
        factor_ids=tuple(manifest["factor_ids"]),
        future_dates=tuple(manifest["future_dates"]),
        asset_currencies=tuple(manifest["asset_currencies"]),
        risk_state=risk_state,
        factor_metadata=_factor_metadata_from_json(manifest["factor_metadata"]),
    )
