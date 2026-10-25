"""Three-asset, two-factor integration test for the public API."""

import numpy as np

from factor_sim import (
    AssetMetadata,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
    simulate,
)
from factor_sim.config import InvalidReturnPolicy


def test_zero_risk_public_simulation_is_exact() -> None:
    request = SimulationRequest(
        run_id="integration-zero-risk",
        risk_model_name="fixture",
        risk_model_version="1",
        simulation_origin="2026-10-23",
        schedule=SimulationSchedule(explicit_future_dates=("2026-10-26", "2026-10-27")),
        trading_calendar=TradingCalendar(
            calendar_id="fixture",
            time_zone="UTC",
            sessions=("2026-10-26", "2026-10-27"),
            provenance={"source": "integration fixture"},
        ),
        path_count=2,
        initial_prices=np.array([100.0, 80.0, 50.0]),
        factor_means=np.array([[0.01, 0.02], [0.0, -0.01]]),
        specific_means=np.zeros((2, 3)),
        risk_state=RiskState(
            exposures=np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.25]]),
            factor_covariance=np.zeros((2, 2)),
            specific_variances=np.zeros(3),
            asset_ids=("a", "b", "c"),
            factor_ids=("market", "style"),
            as_of_date="2026-10-23",
            known_at="2026-10-23",
            return_horizon="one_trading_day",
            observation_frequency="daily",
            return_convention="arithmetic",
            return_basis="raw_price",
            covariance_units="one_trading_day_return_variance",
            annualization_state="deannualized_daily",
            provenance={"source": "integration fixture"},
        ),
        asset_metadata=tuple(
            AssetMetadata(asset_id=asset_id, asset_currency="USD")
            for asset_id in ("a", "b", "c")
        ),
        policy=SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL),
        random_seed=7,
        rng_bit_generator="PCG64",
        rng_stream_policy="per_path",
        numerical_tolerances=NumericalTolerances(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        provenance={"source": "integration fixture"},
    )

    result = simulate(request)

    expected = np.einsum(
        "hk,nk->hn", request.factor_means, request.risk_state.exposures
    )
    np.testing.assert_array_equal(result.modeled_returns[0], expected)
    np.testing.assert_array_equal(result.modeled_returns[1], expected)
    assert result.prices.shape == (2, 3, 3)
    assert result.asset_state_events == ()
