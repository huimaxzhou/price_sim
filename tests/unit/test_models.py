"""Focused tests for immutable Packet A domain objects."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from factor_sim.models import FactorMetadata, RiskState


def _risk_state() -> RiskState:
    return RiskState(
        exposures=np.array([[1.0, 0.5], [0.0, 1.0]]),
        factor_covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
        specific_variances=np.array([0.02, 0.03]),
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
        provenance={"source": "unit-fixture"},
    )


def test_risk_state_owns_float64_read_only_arrays() -> None:
    exposures = np.array([[1, 2]], dtype=np.int64)
    state = RiskState(
        exposures=exposures,
        factor_covariance=np.array([[0.1, 0.0], [0.0, 0.2]]),
        specific_variances=np.array([0.3]),
        asset_ids=("asset-a",),
        factor_ids=("factor-a", "factor-b"),
        as_of_date="2026-10-22",
        known_at="2026-10-22T16:00:00-04:00",
        return_horizon="one_trading_day",
        observation_frequency="daily",
        return_convention="arithmetic",
        return_basis="raw_price",
        covariance_units="one_trading_day_return_squared",
        annualization_state="deannualized_daily",
        provenance={},
    )

    exposures[0, 0] = 99

    assert state.exposures.dtype == np.float64
    assert state.exposures.tolist() == [[1.0, 2.0]]
    assert state.exposures.flags.writeable is False
    with pytest.raises(ValueError):
        state.exposures[0, 0] = 3.0


def test_risk_state_preserves_identifier_order_and_is_frozen() -> None:
    state = _risk_state()

    assert state.asset_ids == ("asset-b", "asset-a")
    assert state.factor_ids == ("factor-2", "factor-1")
    with pytest.raises(FrozenInstanceError):
        state.return_basis = "total"  # type: ignore[misc]


def test_risk_state_copies_and_freezes_provenance() -> None:
    provenance = {"source": "first"}
    state = _risk_state()
    copied_state = RiskState(
        exposures=state.exposures,
        factor_covariance=state.factor_covariance,
        specific_variances=state.specific_variances,
        asset_ids=state.asset_ids,
        factor_ids=state.factor_ids,
        as_of_date=state.as_of_date,
        known_at=state.known_at,
        return_horizon=state.return_horizon,
        observation_frequency=state.observation_frequency,
        return_convention=state.return_convention,
        return_basis=state.return_basis,
        covariance_units=state.covariance_units,
        annualization_state=state.annualization_state,
        provenance=provenance,
    )

    provenance["source"] = "changed"

    assert copied_state.provenance == {"source": "first"}
    with pytest.raises(TypeError):
        copied_state.provenance["source"] = "changed"  # type: ignore[index]


def test_factor_metadata_copies_optional_attributes() -> None:
    attributes = {"vendor_code": "STYLE"}
    metadata = FactorMetadata(factor_id="value", attributes=attributes)

    attributes["vendor_code"] = "changed"

    assert metadata.attributes == {"vendor_code": "STYLE"}
