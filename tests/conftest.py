"""Shared deterministic fixtures for Packet B implementation tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from factor_sim.config import (
    InvalidReturnPolicy,
    JointDistributionPolicy,
    resolve_policy,
)
from factor_sim.models import (
    AssetMetadata,
    JointGeneratorConfig,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
)
from factor_sim.sampling import (
    build_random_streams,
    sample_independent_gaussian,
    sample_joint_gaussian,
)


def make_policy(*, joint: bool = False) -> SimulationPolicy:
    values: dict[str, object] = {"invalid_return_policy": InvalidReturnPolicy.FAIL}
    if joint:
        values.update(
            return_joint_distribution=JointDistributionPolicy.JOINT,
            joint_generator=JointGeneratorConfig(
                name="block_diagonal_gaussian",
                version="1",
                parameters={},
                marginal_contract={
                    "factor": "Gaussian",
                    "specific": "Gaussian",
                    "factor_specific_cross_covariance": "zero",
                },
                random_stream_requirements={"stream": "joint"},
            ),
        )
    return SimulationPolicy(**values)


@pytest.fixture
def policy_factory() -> Callable[..., object]:
    return lambda *, joint=False: resolve_policy(make_policy(joint=joint))


@pytest.fixture
def request_factory() -> Callable[..., SimulationRequest]:
    def make_request(**overrides: object) -> SimulationRequest:
        state = RiskState(
            exposures=np.array([[1.0, 0.5], [0.0, 1.0], [1.0, -0.5]]),
            factor_covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
            specific_variances=np.array([0.02, 0.03, 0.01]),
            asset_ids=("asset-a", "asset-b", "asset-c"),
            factor_ids=("market", "value"),
            as_of_date="2026-10-21",
            known_at="2026-10-21T18:00:00-04:00",
            return_horizon="one_trading_day",
            observation_frequency="daily",
            return_convention="arithmetic",
            return_basis="raw_price",
            covariance_units="one_trading_day_return_variance",
            annualization_state="deannualized_daily",
            provenance={"source": "Packet B unit fixture"},
        )
        values: dict[str, object] = {
            "run_id": "packet-b-unit",
            "risk_model_name": "fixture-model",
            "risk_model_version": "1",
            "simulation_origin": "2026-10-22T09:30:00-04:00",
            "schedule": SimulationSchedule(
                explicit_future_dates=(
                    "2026-10-23",
                    "2026-10-26",
                    "2026-10-27",
                    "2026-10-28",
                )
            ),
            "trading_calendar": TradingCalendar(
                calendar_id="XNYS-fixture",
                time_zone="America/New_York",
                sessions=(
                    "2026-10-21",
                    "2026-10-22",
                    "2026-10-23",
                    "2026-10-26",
                    "2026-10-27",
                    "2026-10-28",
                    "2026-10-29",
                ),
                provenance={"source": "Packet B unit fixture"},
            ),
            "path_count": 2,
            "initial_prices": np.array([100.0, 50.0, 75.0]),
            "factor_means": np.zeros((4, 2)),
            "specific_means": np.zeros((4, 3)),
            "risk_state": state,
            "asset_metadata": tuple(
                AssetMetadata(asset_id=asset_id, asset_currency=currency)
                for asset_id, currency in zip(
                    state.asset_ids, ("USD", "USD", "CAD"), strict=True
                )
            ),
            "policy": make_policy(),
            "random_seed": 20_261_023,
            "rng_bit_generator": "PCG64",
            "rng_stream_policy": "per_path",
            "numerical_tolerances": NumericalTolerances(
                symmetry=1e-12,
                positive_semidefinite=1e-12,
                equation=1e-12,
            ),
            "provenance": {"source": "Packet B unit fixture"},
            "memory_limit_bytes": 1_000_000_000,
        }
        values.update(overrides)
        return SimulationRequest(**values)

    return make_request


@pytest.fixture
def sample_factory() -> Callable[..., object]:
    def sample(
        *,
        joint: bool = False,
        path_count: int = 8,
        step_count: int = 4,
        seed: int = 20_261_023,
        factor_covariance: np.ndarray | None = None,
        specific_variances: np.ndarray | None = None,
    ) -> object:
        joint_policy = (
            JointDistributionPolicy.JOINT
            if joint
            else JointDistributionPolicy.INDEPENDENT
        )
        streams = build_random_streams(
            seed=seed,
            path_count=path_count,
            joint_policy=joint_policy,
            bit_generator_name="PCG64",
            stream_policy="per_path",
        )
        sampler = sample_joint_gaussian if joint else sample_independent_gaussian
        return sampler(
            factor_covariance=(
                np.array([[0.04, 0.01], [0.01, 0.09]])
                if factor_covariance is None
                else factor_covariance
            ),
            specific_variances=(
                np.array([0.02, 0.03, 0.0])
                if specific_variances is None
                else specific_variances
            ),
            step_count=step_count,
            rng_streams=streams,
            symmetry_tolerance=1e-12,
            positive_semidefinite_tolerance=1e-12,
        )

    return sample
