"""Approved Packet D acceptance contract for implementation-plan Steps 11–15."""

from __future__ import annotations

import builtins
import dataclasses
import importlib.util
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import factor_sim.diagnostics as diagnostics
import factor_sim.io as result_io
from factor_sim.config import AssetStatus, InvalidReturnPolicy
from factor_sim.engine import simulate
from factor_sim.errors import NumericalError
from factor_sim.models import (
    AssetMetadata,
    AssetStateEvent,
    FactorMetadata,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
)

pytestmark = pytest.mark.acceptance

RESULT_ARRAY_MEMBERS = {
    "factor_innovations",
    "specific_innovations",
    "factor_returns",
    "specific_returns",
    "modeled_returns",
    "price_link_returns",
    "prices",
    "target_asset_covariance",
}
INPUT_ARRAY_MEMBERS = {
    "exposures",
    "factor_covariance",
    "specific_variances",
    "initial_prices",
    "factor_means",
    "specific_means",
}
EXPECTED_ARRAY_MEMBERS = RESULT_ARRAY_MEMBERS | INPUT_ARRAY_MEMBERS
EXPECTED_CHECK_FIELDS = {
    "name",
    "category",
    "target",
    "estimate",
    "sample_size",
    "rule",
    "status",
    "message",
}
V4_CRITERIA = {str(index) for index in range(1, 11)}


def _require(module: ModuleType, name: str) -> Any:
    assert hasattr(module, name), f"Packet D requires {module.__name__}.{name}"
    return getattr(module, name)


def _request(
    *,
    path_count: int = 2,
    step_count: int = 4,
    seed: int = 20_261_025,
    singular: bool = False,
) -> SimulationRequest:
    dates = tuple(f"2026-10-{26 + index:02d}" for index in range(step_count))
    factor_covariance = (
        np.array([[0.0001, 0.0001], [0.0001, 0.0001]])
        if singular
        else np.array([[0.0001, 0.00002], [0.00002, 0.0002]])
    )
    factor_means = np.zeros((step_count, 2))
    factor_means[:, 0] = np.linspace(0.0001, 0.0004, step_count)
    specific_means = np.zeros((step_count, 3))
    specific_means[:, 1] = np.linspace(-0.0002, 0.0001, step_count)
    return SimulationRequest(
        run_id="packet-d-fixture",
        risk_model_name="packet-d-risk",
        risk_model_version="1",
        simulation_origin="2026-10-25",
        schedule=SimulationSchedule(explicit_future_dates=dates),
        trading_calendar=TradingCalendar(
            calendar_id="packet-d-calendar",
            time_zone="America/New_York",
            sessions=dates,
            provenance={"source": "Packet D fixture"},
        ),
        path_count=path_count,
        initial_prices=np.array([100.0, 80.0, 50.0]),
        factor_means=factor_means,
        specific_means=specific_means,
        risk_state=RiskState(
            exposures=np.array([[1.0, 0.25], [0.5, -0.75], [-0.2, 1.25]]),
            factor_covariance=factor_covariance,
            specific_variances=np.array([0.0001, 0.00015, 0.0]),
            asset_ids=("asset-a", "asset-b", "asset-c"),
            factor_ids=("market", "style"),
            as_of_date="2026-10-25",
            known_at="2026-10-25",
            return_horizon="one_trading_day",
            observation_frequency="daily",
            return_convention="arithmetic",
            return_basis="raw_price",
            covariance_units="one_trading_day_return_variance",
            annualization_state="deannualized_daily",
            provenance={"source": "Packet D fixture"},
        ),
        asset_metadata=(
            AssetMetadata(asset_id="asset-a", asset_currency="USD"),
            AssetMetadata(asset_id="asset-b", asset_currency="CAD"),
            AssetMetadata(asset_id="asset-c", asset_currency="JPY"),
        ),
        factor_metadata=(
            FactorMetadata(factor_id="market", factor_class="market"),
            FactorMetadata(factor_id="style", factor_class="style"),
        ),
        policy=SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL),
        random_seed=seed,
        rng_bit_generator="PCG64",
        rng_stream_policy="per_path",
        numerical_tolerances=NumericalTolerances(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        provenance={"source": "Packet D fixture"},
        memory_limit_bytes=1_000_000_000,
    )


@pytest.fixture(scope="module")
def statistical_result() -> object:
    return simulate(_request(path_count=400, step_count=2))


def _checks(report: object) -> dict[str, Mapping[str, Any]]:
    summaries = report.summaries
    return {check["name"]: check for check in summaries["checks"]}


def _owned_copy(array: np.ndarray) -> np.ndarray:
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _numerical_arrays(result: object) -> dict[str, np.ndarray]:
    arrays = {name: getattr(result, name) for name in RESULT_ARRAY_MEMBERS}
    arrays.update(
        {
            f"risk_state.{name}": getattr(result.risk_state, name)
            for name in ("exposures", "factor_covariance", "specific_variances")
        }
    )
    arrays.update(
        {
            name: getattr(result, name)
            for name in ("initial_prices", "factor_means", "specific_means")
        }
    )
    return arrays


def _load_local_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_result_arrays_equal(left: object, right: object) -> None:
    for field_name in RESULT_ARRAY_MEMBERS:
        np.testing.assert_array_equal(
            getattr(left, field_name), getattr(right, field_name)
        )
    for field_name in ("exposures", "factor_covariance", "specific_variances"):
        np.testing.assert_array_equal(
            getattr(left.risk_state, field_name),
            getattr(right.risk_state, field_name),
        )
    for field_name in ("initial_prices", "factor_means", "specific_means"):
        np.testing.assert_array_equal(
            getattr(left, field_name), getattr(right, field_name)
        )


def test_d_diag_001_simulate_returns_complete_structured_diagnostics() -> None:
    result = simulate(_request())
    report = result.validation_report

    assert report.summaries["diagnostic_scope"] == "complete"
    assert report.summaries["overall_status"] in {"pass", "fail"}
    assert isinstance(report.summaries["checks"], tuple)
    assert report.summaries["checks"]
    for check in report.summaries["checks"]:
        assert set(check) == EXPECTED_CHECK_FIELDS
        assert check["category"] in {"deterministic", "statistical"}
        assert check["status"] in {"pass", "fail", "not_applicable"}


def test_d_diag_002_exact_equation_and_raw_mapping_pass() -> None:
    result = simulate(_request())
    checks = _checks(result.validation_report)

    assert checks["modeled_return_equation"]["status"] == "pass"
    assert checks["raw_identity_mapping"]["status"] == "pass"


def test_d_diag_002_corrupted_modeled_returns_fail_equation() -> None:
    evaluate = _require(diagnostics, "evaluate_result")
    result = simulate(_request())
    corrupted = _owned_copy(result.modeled_returns)
    corrupted[0, 0, 0] += 0.1
    altered = dataclasses.replace(result, modeled_returns=corrupted)

    report = evaluate(altered)

    assert _checks(report)["modeled_return_equation"]["status"] == "fail"
    assert report.summaries["overall_status"] == "fail"


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        ("zero_price", "live_price_validity"),
        ("wrong_initial_slice", "initial_price_preservation"),
        ("unexpected_event", "none_handler_asset_state"),
    ],
)
def test_d_diag_003_price_and_asset_state_corruption_is_visible(
    mutation: str, check_name: str
) -> None:
    evaluate = _require(diagnostics, "evaluate_result")
    result = simulate(_request())
    if mutation == "unexpected_event":
        altered = dataclasses.replace(
            result,
            asset_state_events=(
                AssetStateEvent(
                    asset_id="asset-a",
                    path_id=0,
                    effective_date=result.future_dates[0],
                    status=AssetStatus.TERMINAL,
                    reason="corruption fixture",
                ),
            ),
        )
    else:
        prices = _owned_copy(result.prices)
        if mutation == "zero_price":
            prices[0, 1, 0] = 0.0
        else:
            prices[0, 0, 0] += 1.0
        altered = dataclasses.replace(result, prices=prices)

    assert _checks(evaluate(altered))[check_name]["status"] == "fail"


@pytest.mark.parametrize(
    ("field_name", "value", "check_name"),
    [
        ("asset_ids", ("asset-b", "asset-a", "asset-c"), "identifier_alignment"),
        ("asset_currencies", ("CAD", "USD", "JPY"), "currency_alignment"),
        ("future_dates", ("2026-10-27",), "date_alignment"),
    ],
)
def test_d_diag_004_metadata_corruption_is_visible(
    field_name: str, value: object, check_name: str
) -> None:
    evaluate = _require(diagnostics, "evaluate_result")
    altered = dataclasses.replace(simulate(_request()), **{field_name: value})

    assert _checks(evaluate(altered))[check_name]["status"] == "fail"


def test_d_diag_005_diagnostics_do_not_mutate_result() -> None:
    evaluate = _require(diagnostics, "evaluate_result")
    result = simulate(_request())
    snapshots = {
        name: array.copy() for name, array in _numerical_arrays(result).items()
    }

    evaluate(result)

    current = _numerical_arrays(result)
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(current[name], snapshot)


@pytest.mark.parametrize(
    "check_name",
    ["factor_innovation_mean", "factor_innovation_covariance"],
)
def test_d_stat_001_factor_moments_pass(
    statistical_result: object, check_name: str
) -> None:
    assert _checks(statistical_result.validation_report)[check_name]["status"] == "pass"


@pytest.mark.parametrize(
    "check_name",
    ["specific_innovation_mean", "specific_innovation_covariance"],
)
def test_d_stat_002_specific_moments_pass(
    statistical_result: object, check_name: str
) -> None:
    assert _checks(statistical_result.validation_report)[check_name]["status"] == "pass"


def test_d_stat_003_factor_specific_cross_covariance_passes(
    statistical_result: object,
) -> None:
    check = _checks(statistical_result.validation_report)[
        "factor_specific_cross_covariance"
    ]
    assert check["status"] == "pass"
    assert check["target"] == 0.0
    assert check["sample_size"] == 800


def test_d_stat_004_modeled_asset_covariance_passes(
    statistical_result: object,
) -> None:
    check = _checks(statistical_result.validation_report)["modeled_asset_covariance"]
    assert check["status"] == "pass"
    np.testing.assert_allclose(
        np.asarray(check["target"]),
        statistical_result.target_asset_covariance,
        atol=0.0,
        rtol=0.0,
    )


def test_d_stat_005_zero_variance_correlations_are_not_fabricated(
    statistical_result: object,
) -> None:
    check = _checks(statistical_result.validation_report)["specific_correlation"]
    assert check["status"] == "not_applicable"
    assert check["estimate"][2][0] is None
    assert "zero variance" in check["message"].lower()


@pytest.mark.parametrize(
    "check_name",
    ["factor_lag_one_correlation", "specific_lag_one_correlation"],
)
def test_d_stat_006_no_serial_policy_passes(
    statistical_result: object, check_name: str
) -> None:
    check = _checks(statistical_result.validation_report)[check_name]
    assert check["status"] == "pass"
    assert check["sample_size"] == 400


def test_d_stat_007_tolerance_tightens_with_sample_size() -> None:
    evaluate = _require(diagnostics, "evaluate_result")
    small = evaluate(simulate(_request(path_count=100, step_count=2)))
    large = evaluate(simulate(_request(path_count=400, step_count=2)))
    small_threshold = _checks(small)["factor_innovation_mean"]["rule"][
        "maximum_absolute_error"
    ]
    large_threshold = _checks(large)["factor_innovation_mean"]["rule"][
        "maximum_absolute_error"
    ]

    assert np.max(large_threshold) < np.max(small_threshold)


def test_d_stat_008_independence_evidence_is_not_covariance_only(
    statistical_result: object,
) -> None:
    check = _checks(statistical_result.validation_report)["independent_block_evidence"]
    assert check["status"] == "pass"
    assert check["estimate"]["generator_construction"] == "separate_rng_streams"
    assert "squared_return_dependence" in check["estimate"]
    assert "zero covariance alone" in check["message"].lower()


def test_d_io_001_saved_layout_and_npz_members_are_exact(tmp_path: Path) -> None:
    save = _require(result_io, "save_result")
    output = save(simulate(_request()), tmp_path / "run")

    assert {path.name for path in output.iterdir()} == {
        "arrays.npz",
        "manifest.json",
        "diagnostics.json",
    }
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        assert set(archive.files) == EXPECTED_ARRAY_MEMBERS
        request = _request()
        np.testing.assert_array_equal(
            archive["exposures"], request.risk_state.exposures
        )
        np.testing.assert_array_equal(
            archive["factor_covariance"], request.risk_state.factor_covariance
        )
        np.testing.assert_array_equal(
            archive["specific_variances"], request.risk_state.specific_variances
        )
        np.testing.assert_array_equal(archive["initial_prices"], request.initial_prices)
        np.testing.assert_array_equal(archive["factor_means"], request.factor_means)
        np.testing.assert_array_equal(archive["specific_means"], request.specific_means)
    with zipfile.ZipFile(output / "arrays.npz") as archive:
        assert all(
            item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
        )


def test_d_io_002_json_is_portable_finite_and_versioned(tmp_path: Path) -> None:
    output = _require(result_io, "save_result")(simulate(_request()), tmp_path / "run")
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    diagnostics_text = (output / "diagnostics.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["serialization_schema_version"] == "1"
    assert "NaN" not in manifest_text + diagnostics_text
    assert "Infinity" not in manifest_text + diagnostics_text
    assert manifest["resolved_policy"]["invalid_return_policy"]["value"] == "fail"
    assert all(isinstance(value, str) for value in manifest["future_dates"])


def test_d_io_003_empty_event_file_is_omitted_and_disclosed(tmp_path: Path) -> None:
    output = _require(result_io, "save_result")(simulate(_request()), tmp_path / "run")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert not (output / "asset_state_events.jsonl").exists()
    assert manifest["files"]["asset_state_events.jsonl"] == {
        "present": False,
        "count": 0,
        "reason": "none_handlers",
    }


def test_d_io_004_events_are_rejected_under_none_handlers(tmp_path: Path) -> None:
    save = _require(result_io, "save_result")
    result = simulate(_request())
    event = AssetStateEvent(
        asset_id="asset-a",
        path_id=0,
        effective_date=result.future_dates[0],
        status=AssetStatus.TERMINAL,
        reason="event serialization fixture",
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="event|handler|policy"):
        save(dataclasses.replace(result, asset_state_events=(event,)), output)

    assert not output.exists()


def test_d_io_005_round_trip_preserves_arrays_and_metadata(tmp_path: Path) -> None:
    save = _require(result_io, "save_result")
    load = _require(result_io, "load_result")
    original = simulate(_request())
    restored = load(save(original, tmp_path / "run"))

    _assert_result_arrays_equal(original, restored)
    assert restored.asset_ids == original.asset_ids
    assert restored.factor_ids == original.factor_ids
    assert restored.future_dates == original.future_dates
    assert restored.asset_currencies == original.asset_currencies
    assert restored.factor_metadata == original.factor_metadata
    assert restored.resolved_policy == original.resolved_policy
    assert restored.validation_report == original.validation_report
    assert all(
        not getattr(restored, name).flags.writeable for name in RESULT_ARRAY_MEMBERS
    )
    assert all(
        not getattr(restored, name).flags.writeable
        for name in ("initial_prices", "factor_means", "specific_means")
    )
    assert all(
        not getattr(restored.risk_state, name).flags.writeable
        for name in ("exposures", "factor_covariance", "specific_variances")
    )


def test_d_io_006_hashes_and_byte_counts_match_files(tmp_path: Path) -> None:
    output = _require(result_io, "save_result")(simulate(_request()), tmp_path / "run")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    for name in ("arrays.npz", "diagnostics.json"):
        record = manifest["files"][name]
        assert len(record["sha256"]) == 64
        assert record["bytes"] == (output / name).stat().st_size


@pytest.mark.parametrize(
    "failure",
    [
        "missing_arrays",
        "tampered_arrays",
        "malformed_diagnostics",
        "unknown_schema",
        "extra_file",
    ],
)
def test_d_io_007_invalid_saved_bundle_fails_closed(
    tmp_path: Path, failure: str
) -> None:
    save = _require(result_io, "save_result")
    load = _require(result_io, "load_result")
    output = save(simulate(_request()), tmp_path / "run")
    if failure == "missing_arrays":
        (output / "arrays.npz").unlink()
    elif failure == "tampered_arrays":
        with (output / "arrays.npz").open("ab") as stream:
            stream.write(b"tamper")
    elif failure == "malformed_diagnostics":
        (output / "diagnostics.json").write_text("{", encoding="utf-8")
    elif failure == "unknown_schema":
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["serialization_schema_version"] = "999"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises((ValueError, NumericalError)):
        load(output)


def test_d_io_008_failed_save_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save = _require(result_io, "save_result")
    target = tmp_path / "run"

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez", fail_write)
    with pytest.raises(OSError, match="injected"):
        save(simulate(_request()), target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_d_io_008_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    save = _require(result_io, "save_result")
    target = tmp_path / "run"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save(simulate(_request()), target)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_d_io_009_serialization_does_not_mutate_result(tmp_path: Path) -> None:
    save = _require(result_io, "save_result")
    result = simulate(_request())
    snapshots = {
        name: array.copy() for name, array in _numerical_arrays(result).items()
    }

    save(result, tmp_path / "run")

    current = _numerical_arrays(result)
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(current[name], snapshot)


def test_d_io_009_simulate_remains_file_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("simulate attempted file I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert simulate(_request()).prices.shape == (2, 5, 3)


@pytest.mark.parametrize("singular", [False, True])
def test_d_acc_001_canonical_fixtures_are_traceable(singular: bool) -> None:
    request = _request(singular=singular)
    result = simulate(request)
    reconstructed = (
        np.einsum("mhk,nk->mhn", result.factor_returns, request.risk_state.exposures)
        + result.specific_returns
    )

    np.testing.assert_allclose(
        result.modeled_returns,
        reconstructed,
        atol=request.numerical_tolerances.equation,
        rtol=0.0,
    )
    assert result.validation_report.summaries["diagnostic_scope"] == "complete"


def _passing_evidence() -> dict[str, Mapping[str, object]]:
    return {
        str(index): {
            "test_ids": (f"criterion-{index:02d}",),
            "evidence_summary": f"criterion {index} evidence",
            "modules": ("factor_sim",),
            "status": "pass",
        }
        for index in range(1, 11)
    }


def test_d_acc_002_report_covers_every_v4_criterion() -> None:
    reporting = _load_local_module(
        Path(__file__).with_name("acceptance_reporting.py"),
        "packet_d_acceptance_reporting",
    )
    report = reporting.build_acceptance_report(
        _passing_evidence(), manual_review_status="approved"
    )

    assert set(report["criteria"]) == V4_CRITERIA
    assert report["overall_status"] == "pass"
    assert all(record["test_ids"] for record in report["criteria"].values())


@pytest.mark.parametrize("failure", ["missing", "unknown", "failed", "skipped"])
def test_d_acc_003_incomplete_evidence_prevents_acceptance(failure: str) -> None:
    reporting = _load_local_module(
        Path(__file__).with_name("acceptance_reporting.py"),
        f"packet_d_acceptance_reporting_{failure}",
    )
    evidence = _passing_evidence()
    if failure == "missing":
        evidence.pop("10")
    elif failure == "unknown":
        evidence["11"] = evidence["10"]
    else:
        evidence["10"] = {**evidence["10"], "status": failure}

    report = reporting.build_acceptance_report(
        evidence, manual_review_status="approved"
    )
    assert report["overall_status"] == "fail"


def test_d_arch_001_dependency_and_module_boundaries_remain_narrow() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    engine_source = (root / "src/factor_sim/engine.py").read_text(encoding="utf-8")
    pricing_source = (root / "src/factor_sim/pricing.py").read_text(encoding="utf-8")
    sampling_source = (root / "src/factor_sim/sampling.py").read_text(encoding="utf-8")

    assert 'dependencies = [\n    "numpy>=1.26",\n]' in pyproject
    assert "from .io import" not in engine_source
    assert "factor_sim.io" not in engine_source
    assert "sampling" not in pricing_source
    assert "pricing" not in sampling_source


def test_d_perf_001_benchmark_reports_stages_without_thresholds() -> None:
    root = Path(__file__).resolve().parents[2]
    benchmark_module = _load_local_module(
        root / "tests/performance/benchmark_packet_d.py",
        "packet_d_benchmark",
    )
    result = benchmark_module.benchmark("canonical")

    assert set(result["seconds"]) == {
        "validation",
        "sampling",
        "return_construction",
        "pricing",
        "diagnostics",
        "serialization",
        "read_back",
        "end_to_end",
    }
    assert "threshold" not in result
    assert result["estimated_peak_memory_bytes"] > 0
    assert result["observed_result_bytes"] > 0


@pytest.mark.parametrize(
    "required_text",
    [
        "Installation",
        "Quick start",
        "Array conventions",
        "Supported policies",
        "Unsupported policies",
        "Reproducibility",
        "Saved output",
        "Running tests",
    ],
)
def test_d_doc_001_readme_documents_actual_version_one(required_text: str) -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )
    assert required_text.lower() in readme.lower()


@pytest.mark.parametrize("manual_status", ["pending", "rejected"])
def test_d_final_001_manual_review_is_required(manual_status: str) -> None:
    reporting = _load_local_module(
        Path(__file__).with_name("acceptance_reporting.py"),
        f"packet_d_final_reporting_{manual_status}",
    )
    report = reporting.build_acceptance_report(
        _passing_evidence(), manual_review_status=manual_status
    )

    assert report["overall_status"] == "fail"
    assert report["manual_review_status"] == manual_status
