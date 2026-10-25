"""Approved Packet C acceptance contract for implementation-plan Steps 8–10."""

from __future__ import annotations

import builtins
import dataclasses
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import factor_sim
import factor_sim.engine as engine
import factor_sim.pricing as pricing
from factor_sim.config import InvalidReturnPolicy, resolve_policy
from factor_sim.errors import NumericalError, PolicyExecutionError, ValidationError
from factor_sim.models import (
    AssetMetadata,
    FactorMetadata,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
)
from factor_sim.validation import validate_request

pytestmark = pytest.mark.acceptance

PUBLIC_NAMES = {
    "simulate",
    "SimulationRequest",
    "SimulationResult",
    "SimulationPolicy",
    "ResolvedSimulationPolicy",
    "AssetStateEvent",
    "ValidationReport",
    "RiskState",
    "AssetMetadata",
    "FactorMetadata",
    "SimulationSchedule",
    "TradingCalendar",
    "JointGeneratorConfig",
    "NumericalTolerances",
}


def _require(module: ModuleType, name: str) -> Any:
    assert hasattr(module, name), f"Packet C requires {module.__name__}.{name}"
    return getattr(module, name)


def _policy() -> object:
    return resolve_policy(
        SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL)
    )


def _request(
    *,
    zero_risk: bool = False,
    seed: int = 20_261_024,
    path_count: int = 2,
    **overrides: object,
) -> SimulationRequest:
    factor_covariance = (
        np.zeros((2, 2))
        if zero_risk
        else np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    )
    specific_variances = np.zeros(2) if zero_risk else np.array([0.0002, 0.0003])
    state = RiskState(
        exposures=np.array([[1.0, 0.5], [0.25, -1.0]]),
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        asset_ids=("asset-a", "asset-b"),
        factor_ids=("market", "value"),
        as_of_date="2026-10-23",
        known_at="2026-10-23T16:00:00-04:00",
        return_horizon="one_trading_day",
        observation_frequency="daily",
        return_convention="arithmetic",
        return_basis="raw_price",
        covariance_units="one_trading_day_return_variance",
        annualization_state="deannualized_daily",
        provenance={"model_run_id": "packet-c-risk-001"},
    )
    values: dict[str, object] = {
        "run_id": "packet-c-fixture",
        "risk_model_name": "fixture-model",
        "risk_model_version": "1",
        "simulation_origin": "2026-10-23T16:00:00-04:00",
        "schedule": SimulationSchedule(
            explicit_future_dates=("2026-10-26", "2026-10-27", "2026-10-28")
        ),
        "trading_calendar": TradingCalendar(
            calendar_id="XNYS-fixture",
            time_zone="America/New_York",
            sessions=("2026-10-26", "2026-10-27", "2026-10-28"),
            provenance={"source": "Packet C deterministic fixture"},
        ),
        "path_count": path_count,
        "initial_prices": np.array([100.0, 50.0]),
        "factor_means": np.array([[0.01, 0.00], [0.00, 0.02], [-0.01, 0.01]]),
        "specific_means": np.array([[0.001, -0.002], [0.003, 0.004], [-0.001, 0.002]]),
        "risk_state": state,
        "asset_metadata": (
            AssetMetadata(asset_id="asset-a", asset_currency="USD"),
            AssetMetadata(asset_id="asset-b", asset_currency="CAD"),
        ),
        "factor_metadata": (
            FactorMetadata(factor_id="market", factor_class="market"),
            FactorMetadata(factor_id="value", factor_class="style"),
        ),
        "policy": SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL),
        "random_seed": seed,
        "rng_bit_generator": "PCG64",
        "rng_stream_policy": "per_path",
        "numerical_tolerances": NumericalTolerances(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        "provenance": {"source": "Packet C acceptance fixture"},
        "memory_limit_bytes": 1_000_000_000,
    }
    values.update(overrides)
    return SimulationRequest(**values)


def _expected_zero_risk_outputs(
    request: SimulationRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    factor = np.broadcast_to(
        request.factor_means, (request.path_count, *request.factor_means.shape)
    ).copy()
    specific = np.broadcast_to(
        request.specific_means, (request.path_count, *request.specific_means.shape)
    ).copy()
    modeled = np.einsum("mhk,nk->mhn", factor, request.risk_state.exposures) + specific
    prices = np.empty((request.path_count, 4, 2))
    prices[:, 0, :] = request.initial_prices
    for step in range(3):
        prices[:, step + 1, :] = prices[:, step, :] * (1.0 + modeled[:, step, :])
    return factor, specific, modeled, prices


def test_c_ret_001_means_broadcast_exactly_over_paths() -> None:
    construct_factor = _require(engine, "construct_factor_returns")
    construct_specific = _require(engine, "construct_specific_returns")
    factor_means = np.array([[0.1, -0.2], [0.3, 0.4]])
    specific_means = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    factor_innovations = np.arange(12, dtype=np.float64).reshape(3, 2, 2) / 100
    specific_innovations = np.arange(18, dtype=np.float64).reshape(3, 2, 3) / 100

    factor = construct_factor(factor_means, factor_innovations)
    specific = construct_specific(specific_means, specific_innovations)

    np.testing.assert_array_equal(factor, factor_innovations + factor_means)
    np.testing.assert_array_equal(specific, specific_innovations + specific_means)
    assert factor.shape == (3, 2, 2)
    assert specific.shape == (3, 2, 3)


def test_c_ret_002_modeled_return_equation_is_exact() -> None:
    construct_modeled = _require(engine, "construct_modeled_returns")
    exposures = np.array([[1.0, 0.5], [0.25, -1.0], [0.0, 2.0]])
    factor = np.array(
        [
            [[[0.01, 0.02], [0.03, -0.01]]],
            [[[-0.02, 0.04], [0.00, 0.05]]],
        ]
    ).reshape(2, 2, 2)
    specific = np.array(
        [
            [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]],
            [[-0.001, -0.002, -0.003], [0.007, 0.008, 0.009]],
        ]
    )
    expected = np.einsum("mhk,nk->mhn", factor, exposures) + specific

    actual = construct_modeled(exposures, factor, specific)

    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_c_ret_003_constant_and_time_varying_means_are_preserved() -> None:
    construct_factor = _require(engine, "construct_factor_returns")
    innovations = np.zeros((2, 3, 2))
    time_varying = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    constant = np.broadcast_to(np.array([[0.01, 0.02]]), (3, 2)).copy()

    np.testing.assert_array_equal(
        construct_factor(time_varying, innovations),
        np.broadcast_to(time_varying, innovations.shape),
    )
    np.testing.assert_array_equal(
        construct_factor(constant, innovations),
        np.broadcast_to(constant, innovations.shape),
    )


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        ("construct_factor_returns", (np.zeros((2, 2)), np.zeros((1, 3, 2)))),
        ("construct_factor_returns", (np.zeros((2, 3)), np.zeros((1, 2, 2)))),
        ("construct_specific_returns", (np.zeros((2, 2)), np.zeros((1, 3, 2)))),
        (
            "construct_modeled_returns",
            (np.zeros((3, 2)), np.zeros((1, 2, 3)), np.zeros((1, 2, 3))),
        ),
        (
            "construct_modeled_returns",
            (np.zeros((3, 2)), np.zeros((1, 2, 2)), np.zeros((1, 3, 3))),
        ),
    ],
)
def test_c_ret_004_shape_mismatches_fail_clearly(
    function_name: str, arguments: tuple[np.ndarray, ...]
) -> None:
    function = _require(engine, function_name)
    with pytest.raises(NumericalError, match="shape"):
        function(*arguments)


def test_c_ret_005_helpers_preserve_inputs_and_return_owned_float64() -> None:
    construct_factor = _require(engine, "construct_factor_returns")
    means = np.array([[0.01, 0.02]])
    innovations = np.array([[[0.03, 0.04]], [[0.05, 0.06]]])
    means_snapshot = means.copy()
    innovations_snapshot = innovations.copy()

    result = construct_factor(means, innovations)

    np.testing.assert_array_equal(means, means_snapshot)
    np.testing.assert_array_equal(innovations, innovations_snapshot)
    assert result.dtype == np.float64
    assert result.flags.owndata
    assert result is not innovations


def test_c_map_001_raw_mapping_is_distinct_tagged_identity() -> None:
    map_returns = _require(pricing, "map_modeled_returns")
    request = _request()
    modeled = np.array([[[0.01, -0.02], [0.03, 0.04]]])

    mapped, metadata = map_returns(
        modeled,
        risk_state=request.risk_state,
        resolved_policy=_policy(),
    )

    np.testing.assert_array_equal(mapped, modeled)
    assert mapped is not modeled
    assert mapped.flags.owndata
    assert metadata == {
        "mapping": "identity",
        "return_type": "arithmetic",
        "return_basis": "raw_price",
        "adjustment": "unadjusted",
    }


@pytest.mark.parametrize(
    ("return_convention", "return_basis", "issue_code"),
    [
        ("log", "raw_price", "incompatible_return_convention"),
        ("arithmetic", "total", "incompatible_return_basis"),
    ],
)
def test_c_map_002_incompatible_metadata_fails_preflight_before_rng(
    monkeypatch: pytest.MonkeyPatch,
    return_convention: str,
    return_basis: str,
    issue_code: str,
) -> None:
    request = _request()
    state = dataclasses.replace(
        request.risk_state,
        return_convention=return_convention,
        return_basis=return_basis,
    )
    incompatible_request = dataclasses.replace(request, risk_state=state)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("incompatible return metadata reached RNG creation")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    simulate = _require(engine, "simulate")
    with pytest.raises(ValidationError) as captured:
        simulate(incompatible_request)

    assert any(issue.code == issue_code for issue in captured.value.issues)


def test_c_handler_001_none_currency_policy_reuses_array_and_records_currencies() -> (
    None
):
    apply_currency = _require(pricing, "apply_currency_policy")
    returns = np.zeros((2, 3, 2))

    actual, metadata = apply_currency(
        returns,
        resolved_policy=_policy(),
        asset_currencies=("USD", "CAD"),
    )

    assert actual is returns
    assert metadata == {
        "handler": None,
        "converted": False,
        "price_currencies": ("USD", "CAD"),
    }


def test_c_handler_002_none_event_handlers_preserve_universe_and_emit_no_events() -> (
    None
):
    apply_actions = _require(pricing, "apply_corporate_action_policy")
    apply_listings = _require(pricing, "apply_new_listing_policy")
    returns = np.zeros((2, 3, 2))
    asset_ids = ("asset-a", "asset-b")

    adjusted, events, action_metadata = apply_actions(
        returns, resolved_policy=_policy()
    )
    retained_ids, listing_metadata = apply_listings(
        asset_ids, resolved_policy=_policy()
    )

    assert adjusted is returns
    assert events == ()
    assert action_metadata == {"handler": None, "adjusted": False}
    assert retained_ids is asset_ids
    assert listing_metadata == {"handler": None, "universe_changed": False}


def test_c_price_001_arithmetic_prices_compound_over_dates_paths_and_assets() -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    initial = np.array([100.0, 50.0])
    returns = np.array(
        [
            [[0.10, 0.00], [-0.10, 0.20]],
            [[0.00, 0.10], [0.05, -0.05]],
        ]
    )
    expected = np.array(
        [
            [[100.0, 50.0], [110.0, 50.0], [99.0, 60.0]],
            [[100.0, 50.0], [100.0, 55.0], [105.0, 52.25]],
        ]
    )

    actual = evolve(
        initial,
        returns,
        asset_ids=("asset-a", "asset-b"),
        future_dates=("2026-10-26", "2026-10-27"),
        invalid_return_policy=InvalidReturnPolicy.FAIL,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_c_price_002_zero_and_positive_returns_preserve_positive_prices() -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    prices = evolve(
        np.array([100.0, 50.0]),
        np.array([[[0.0, 1.0], [0.0, 0.5]]]),
        asset_ids=("asset-a", "asset-b"),
        future_dates=("2026-10-26", "2026-10-27"),
        invalid_return_policy=InvalidReturnPolicy.FAIL,
    )
    assert np.all(prices > 0.0)
    np.testing.assert_array_equal(prices[0, :, 0], np.array([100.0, 100.0, 100.0]))


@pytest.mark.parametrize("invalid_return", [-1.0, -1.01])
def test_c_price_003_invalid_arithmetic_returns_fail_without_repair(
    invalid_return: float,
) -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    returns = np.zeros((2, 2, 2))
    returns[1, 0, 1] = invalid_return

    with pytest.raises(PolicyExecutionError) as captured:
        evolve(
            np.array([100.0, 50.0]),
            returns,
            asset_ids=("asset-a", "asset-b"),
            future_dates=("2026-10-26", "2026-10-27"),
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )

    error = captured.value
    assert error.code == "invalid_arithmetic_return"
    assert error.context == {
        "path_id": 1,
        "step_index": 0,
        "future_date": "2026-10-26",
        "asset_id": "asset-b",
        "return_value": invalid_return,
    }
    np.testing.assert_array_equal(returns[1, 0, 1], invalid_return)


def test_c_price_004_first_failure_is_deterministic() -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    returns = np.zeros((2, 2, 2))
    returns[1, 0, 0] = -1.2
    returns[0, 0, 1] = -1.1
    returns[0, 1, 0] = -2.0

    with pytest.raises(PolicyExecutionError) as captured:
        evolve(
            np.array([100.0, 50.0]),
            returns,
            asset_ids=("asset-a", "asset-b"),
            future_dates=("2026-10-26", "2026-10-27"),
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )

    assert captured.value.context["path_id"] == 0
    assert captured.value.context["step_index"] == 0
    assert captured.value.context["asset_id"] == "asset-b"


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_c_price_005_nonfinite_price_link_returns_fail(value: float) -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    with pytest.raises(NumericalError):
        evolve(
            np.array([100.0]),
            np.array([[[value]]]),
            asset_ids=("asset-a",),
            future_dates=("2026-10-26",),
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )


def test_c_price_005_finite_input_overflow_identifies_location() -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    with pytest.raises(NumericalError) as captured:
        evolve(
            np.array([np.finfo(np.float64).max]),
            np.array([[[1.0]]]),
            asset_ids=("asset-a",),
            future_dates=("2026-10-26",),
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )

    error = captured.value
    assert error.code == "nonfinite_price"
    assert error.context["path_id"] == 0
    assert error.context["step_index"] == 0
    assert error.context["future_date"] == "2026-10-26"
    assert error.context["asset_id"] == "asset-a"


def test_c_price_006_output_owns_data_and_contains_initial_slice() -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    initial = np.array([100.0, 50.0])
    prices = evolve(
        initial,
        np.zeros((3, 2, 2)),
        asset_ids=("asset-a", "asset-b"),
        future_dates=("2026-10-26", "2026-10-27"),
        invalid_return_policy=InvalidReturnPolicy.FAIL,
    )

    assert prices.shape == (3, 3, 2)
    assert prices.dtype == np.float64
    assert prices.flags.owndata
    np.testing.assert_array_equal(prices[:, 0, :], np.broadcast_to(initial, (3, 2)))


@pytest.mark.parametrize(
    ("initial", "returns", "asset_ids", "future_dates"),
    [
        (
            np.ones(3),
            np.zeros((1, 2, 2)),
            ("asset-a", "asset-b"),
            ("2026-10-26", "2026-10-27"),
        ),
        (
            np.ones(2),
            np.zeros((1, 2, 2)),
            ("asset-a",),
            ("2026-10-26", "2026-10-27"),
        ),
        (
            np.ones(2),
            np.zeros((1, 2, 2)),
            ("asset-a", "asset-b"),
            ("2026-10-26",),
        ),
    ],
)
def test_c_price_007_price_dimensions_ids_and_dates_must_align(
    initial: np.ndarray,
    returns: np.ndarray,
    asset_ids: tuple[str, ...],
    future_dates: tuple[str, ...],
) -> None:
    evolve = _require(pricing, "evolve_arithmetic_prices")
    with pytest.raises(NumericalError, match="shape|length"):
        evolve(
            initial,
            returns,
            asset_ids=asset_ids,
            future_dates=future_dates,
            invalid_return_policy=InvalidReturnPolicy.FAIL,
        )


def test_c_engine_001_zero_risk_simulation_matches_hand_calculation() -> None:
    simulate = _require(engine, "simulate")
    request = _request(zero_risk=True)
    expected_factor, expected_specific, expected_modeled, expected_prices = (
        _expected_zero_risk_outputs(request)
    )

    result = simulate(request)

    np.testing.assert_allclose(result.factor_returns, expected_factor, atol=1e-15)
    np.testing.assert_allclose(result.specific_returns, expected_specific, atol=1e-15)
    np.testing.assert_allclose(result.modeled_returns, expected_modeled, atol=1e-15)
    np.testing.assert_array_equal(result.price_link_returns, result.modeled_returns)
    np.testing.assert_allclose(result.prices, expected_prices, atol=1e-12)


def test_c_engine_002_result_shapes_dtype_axis_order_and_initial_prices() -> None:
    simulate = _require(engine, "simulate")
    request = _request()
    result = simulate(request)

    expected_shapes = {
        "factor_innovations": (2, 3, 2),
        "specific_innovations": (2, 3, 2),
        "factor_returns": (2, 3, 2),
        "specific_returns": (2, 3, 2),
        "modeled_returns": (2, 3, 2),
        "price_link_returns": (2, 3, 2),
        "prices": (2, 4, 2),
        "target_asset_covariance": (2, 2),
    }
    for field_name, shape in expected_shapes.items():
        array = getattr(result, field_name)
        assert array.shape == shape
        assert array.dtype == np.float64
    assert result.manifest["axis_order"] == {
        "factor_arrays": ("path", "future_date", "factor"),
        "asset_return_arrays": ("path", "future_date", "asset"),
        "prices": ("path", "date_including_origin", "asset"),
        "target_asset_covariance": ("asset", "asset"),
    }
    np.testing.assert_array_equal(
        result.prices[:, 0, :], np.broadcast_to(request.initial_prices, (2, 2))
    )


def test_c_engine_003_seed_reproduction_and_seed_sensitivity() -> None:
    simulate = _require(engine, "simulate")
    first = simulate(_request(seed=7))
    repeated = simulate(_request(seed=7))
    changed = simulate(_request(seed=8))

    for field_name in (
        "factor_innovations",
        "specific_innovations",
        "factor_returns",
        "specific_returns",
        "modeled_returns",
        "price_link_returns",
        "prices",
    ):
        np.testing.assert_array_equal(
            getattr(first, field_name), getattr(repeated, field_name)
        )
    assert not np.array_equal(first.factor_innovations, changed.factor_innovations)


def test_c_engine_004_request_and_risk_state_remain_unchanged() -> None:
    simulate = _require(engine, "simulate")
    request = _request()
    snapshots = {
        "initial_prices": request.initial_prices.copy(),
        "factor_means": request.factor_means.copy(),
        "specific_means": request.specific_means.copy(),
        "exposures": request.risk_state.exposures.copy(),
        "factor_covariance": request.risk_state.factor_covariance.copy(),
        "specific_variances": request.risk_state.specific_variances.copy(),
    }

    simulate(request)

    np.testing.assert_array_equal(request.initial_prices, snapshots["initial_prices"])
    np.testing.assert_array_equal(request.factor_means, snapshots["factor_means"])
    np.testing.assert_array_equal(request.specific_means, snapshots["specific_means"])
    np.testing.assert_array_equal(request.risk_state.exposures, snapshots["exposures"])
    np.testing.assert_array_equal(
        request.risk_state.factor_covariance, snapshots["factor_covariance"]
    )
    np.testing.assert_array_equal(
        request.risk_state.specific_variances, snapshots["specific_variances"]
    )


def test_c_engine_005_identity_and_manifest_metadata_survive() -> None:
    simulate = _require(engine, "simulate")
    request = _request()
    result = simulate(request)
    manifest = result.manifest

    assert result.asset_ids == request.risk_state.asset_ids
    assert result.factor_ids == request.risk_state.factor_ids
    assert result.future_dates == request.schedule.explicit_future_dates
    assert result.asset_currencies == ("USD", "CAD")
    assert result.factor_metadata == request.factor_metadata
    assert result.risk_state is request.risk_state
    assert manifest["schema_version"] == "1"
    assert manifest["run"] == {
        "run_id": request.run_id,
        "risk_model_name": request.risk_model_name,
        "risk_model_version": request.risk_model_version,
    }
    assert manifest["dimensions"] == {"M": 2, "H": 3, "N": 2, "K": 2}
    assert manifest["schedule"]["method"] == "explicit_future_dates"
    assert manifest["schedule"]["supplied_value"] == (
        "2026-10-26",
        "2026-10-27",
        "2026-10-28",
    )
    assert manifest["schedule"]["resolved_dates"] == result.future_dates
    assert manifest["calendar"]["calendar_id"] == "XNYS-fixture"
    assert manifest["returns"]["modeled"]["return_type"] == "arithmetic"
    assert manifest["returns"]["modeled"]["return_basis"] == "raw_price"
    assert manifest["returns"]["price_link"]["return_type"] == "arithmetic"
    assert manifest["returns"]["price_link"]["return_basis"] == "raw_price"
    assert manifest["returns"]["price_link"]["mapping"] == "identity"
    assert manifest["returns"]["price_link"]["adjustment_basis"] == "unadjusted"
    assert manifest["rng"]["seed"] == request.random_seed
    assert manifest["rng"]["bit_generator"] == "PCG64"
    assert manifest["rng"]["stream_policy"] == (
        "SeedSequence -> path -> factor,specific"
    )
    assert manifest["rng"]["sampler_name"] == "independent_gaussian"
    assert manifest["resolved_policy"]["price_return_convention"] == {
        "value": "raw",
        "value_source": "default",
    }
    assert manifest["resolved_policy"]["invalid_return_policy"] == {
        "value": "fail",
        "value_source": "explicit",
    }
    assert manifest["numerical_tolerances"] == {
        "symmetry": 1e-12,
        "positive_semidefinite": 1e-12,
        "equation": 1e-12,
    }
    assert manifest["numerical_summaries"] == result.validation_report.summaries
    assert manifest["provenance"]["request"] == request.provenance
    assert manifest["provenance"]["risk_state"] == request.risk_state.provenance
    assert manifest["risk_state_frozen"] is True


def test_c_engine_006_default_handlers_are_explicit_and_emit_no_events() -> None:
    simulate = _require(engine, "simulate")
    result = simulate(_request())

    assert result.asset_state_events == ()
    assert result.manifest["handlers"] == {
        "currency_handling": {
            "handler": None,
            "converted": False,
            "price_currencies": ("USD", "CAD"),
        },
        "corporate_action_adjustment": {"handler": None, "adjusted": False},
        "new_listing": {"handler": None, "universe_changed": False},
    }


def test_c_engine_007_preflight_failure_consumes_no_rng_and_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulate = _require(engine, "simulate")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("failed simulation consumed RNG state or wrote a file")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "savez", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)

    with pytest.raises(ValidationError):
        simulate(_request(path_count=0))


def test_c_engine_007_successful_simulation_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulate = _require(engine, "simulate")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("core simulation attempted file I/O")

    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "savez", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)

    result = simulate(_request(zero_risk=True))
    assert result.prices.shape == (2, 4, 2)


def test_c_engine_008_preserves_preflight_summaries_with_complete_diagnostics() -> None:
    simulate = _require(engine, "simulate")
    request = _request()
    context = validate_request(request, _policy())
    result = simulate(request)

    assert result.validation_report.issues == ()
    for key, value in context.validation_summaries.items():
        assert result.validation_report.summaries[key] == value
    assert result.validation_report.summaries["diagnostic_scope"] == "complete"
    assert (
        result.validation_report.summaries["statistical_diagnostics_included"] is True
    )
    assert result.validation_report.summaries["checks"]


def test_c_engine_009_result_arrays_are_owned_and_read_only() -> None:
    simulate = _require(engine, "simulate")
    result = simulate(_request())
    arrays = (
        result.factor_innovations,
        result.specific_innovations,
        result.factor_returns,
        result.specific_returns,
        result.modeled_returns,
        result.price_link_returns,
        result.prices,
        result.target_asset_covariance,
    )
    assert all(array.flags.owndata for array in arrays)
    assert all(not array.flags.writeable for array in arrays)
    assert result.modeled_returns is not result.price_link_returns


def test_c_engine_010_every_modeled_return_satisfies_the_equation() -> None:
    simulate = _require(engine, "simulate")
    request = _request()
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


def test_c_api_001_package_root_exports_only_approved_names() -> None:
    assert set(factor_sim.__all__) == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert hasattr(factor_sim, name)
    for private_name in (
        "ValidatedContext",
        "covariance_root",
        "build_random_streams",
        "construct_factor_returns",
        "evolve_arithmetic_prices",
    ):
        assert not hasattr(factor_sim, private_name)
