"""Non-gating Packet B timing harness for representative supported scales.

Run ``python tests/performance/benchmark_packet_b.py --case all``. Timings are
observations, not acceptance thresholds, and vary with hardware and BLAS setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_sim.config import (  # noqa: E402
    InvalidReturnPolicy,
    JointDistributionPolicy,
    resolve_policy,
)
from factor_sim.models import (  # noqa: E402
    AssetMetadata,
    NumericalTolerances,
    RiskState,
    SimulationPolicy,
    SimulationRequest,
    SimulationSchedule,
    TradingCalendar,
)
from factor_sim.sampling import (  # noqa: E402
    build_random_streams,
    covariance_root,
    sample_independent_gaussian,
    sample_joint_gaussian,
)
from factor_sim.validation import validate_request  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    paths: int
    steps: int
    assets: int
    factors: int


CASES = {
    "canonical": Case("canonical", paths=20, steps=63, assets=50, factors=10),
    "max": Case("max", paths=100, steps=252, assets=250, factors=100),
}


def _time(callable_: Callable[[], object]) -> tuple[object, float]:
    start = perf_counter()
    result = callable_()
    return result, perf_counter() - start


def _policy() -> object:
    return resolve_policy(
        SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL)
    )


def _request(case: Case) -> SimulationRequest:
    rng = np.random.default_rng(91)
    exposures = rng.normal(scale=0.25, size=(case.assets, case.factors))
    factor_scales = np.linspace(0.01, 0.04, case.factors)
    factor_covariance = np.diag(factor_scales)
    specific_variances = np.linspace(0.005, 0.02, case.assets)
    asset_ids = tuple(f"asset-{index:03d}" for index in range(case.assets))
    factor_ids = tuple(f"factor-{index:03d}" for index in range(case.factors))
    origin = date(2026, 1, 1)
    sessions = tuple(origin + timedelta(days=index + 1) for index in range(case.steps))
    return SimulationRequest(
        run_id=f"packet-b-benchmark-{case.name}",
        risk_model_name="benchmark-model",
        risk_model_version="1",
        simulation_origin=origin,
        schedule=SimulationSchedule(explicit_future_dates=sessions),
        trading_calendar=TradingCalendar(
            calendar_id="benchmark-calendar",
            time_zone="UTC",
            sessions=sessions,
            provenance={"source": "synthetic deterministic benchmark"},
        ),
        path_count=case.paths,
        initial_prices=np.full(case.assets, 100.0),
        factor_means=np.zeros((case.steps, case.factors)),
        specific_means=np.zeros((case.steps, case.assets)),
        risk_state=RiskState(
            exposures=exposures,
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            asset_ids=asset_ids,
            factor_ids=factor_ids,
            as_of_date=origin,
            known_at=origin,
            return_horizon="one_trading_day",
            observation_frequency="daily",
            return_convention="arithmetic",
            return_basis="raw_price",
            covariance_units="one_trading_day_return_variance",
            annualization_state="deannualized_daily",
            provenance={"source": "synthetic deterministic benchmark"},
        ),
        asset_metadata=tuple(
            AssetMetadata(asset_id=asset_id, asset_currency="USD")
            for asset_id in asset_ids
        ),
        policy=SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL),
        random_seed=91,
        rng_bit_generator="PCG64",
        rng_stream_policy="per_path",
        numerical_tolerances=NumericalTolerances(
            symmetry=1e-12,
            positive_semidefinite=1e-12,
            equation=1e-12,
        ),
        provenance={"source": "synthetic deterministic benchmark"},
        memory_limit_bytes=1_000_000_000,
    )


def benchmark(case: Case) -> dict[str, object]:
    request = _request(case)
    context, validation_seconds = _time(lambda: validate_request(request, _policy()))
    _, factor_root_seconds = _time(
        lambda: covariance_root(
            request.risk_state.factor_covariance,
            matrix_name="factor_covariance",
            symmetry_tolerance=request.numerical_tolerances.symmetry,
            positive_semidefinite_tolerance=(
                request.numerical_tolerances.positive_semidefinite
            ),
        )
    )
    independent_streams, independent_stream_seconds = _time(
        lambda: build_random_streams(
            seed=request.random_seed,
            path_count=case.paths,
            joint_policy=JointDistributionPolicy.INDEPENDENT,
            bit_generator_name=request.rng_bit_generator,
            stream_policy=request.rng_stream_policy,
        )
    )
    independent, independent_seconds = _time(
        lambda: sample_independent_gaussian(
            factor_covariance=request.risk_state.factor_covariance,
            specific_variances=request.risk_state.specific_variances,
            step_count=case.steps,
            rng_streams=independent_streams,
            symmetry_tolerance=request.numerical_tolerances.symmetry,
            positive_semidefinite_tolerance=(
                request.numerical_tolerances.positive_semidefinite
            ),
        )
    )
    joint_streams, joint_stream_seconds = _time(
        lambda: build_random_streams(
            seed=request.random_seed,
            path_count=case.paths,
            joint_policy=JointDistributionPolicy.JOINT,
            bit_generator_name=request.rng_bit_generator,
            stream_policy=request.rng_stream_policy,
        )
    )
    joint, joint_seconds = _time(
        lambda: sample_joint_gaussian(
            factor_covariance=request.risk_state.factor_covariance,
            specific_variances=request.risk_state.specific_variances,
            step_count=case.steps,
            rng_streams=joint_streams,
            symmetry_tolerance=request.numerical_tolerances.symmetry,
            positive_semidefinite_tolerance=(
                request.numerical_tolerances.positive_semidefinite
            ),
        )
    )
    output_bytes = sum(
        array.nbytes
        for array in (
            independent.factor_innovations,
            independent.specific_innovations,
            joint.factor_innovations,
            joint.specific_innovations,
        )
    )
    return {
        "dimensions": {
            "M": case.paths,
            "H": case.steps,
            "N": case.assets,
            "K": case.factors,
        },
        "seconds": {
            "validation": validation_seconds,
            "factor_covariance_root": factor_root_seconds,
            "independent_streams": independent_stream_seconds,
            "independent_sampling": independent_seconds,
            "joint_streams": joint_stream_seconds,
            "joint_sampling": joint_seconds,
        },
        "estimated_peak_memory_bytes": context.estimated_peak_memory_bytes,
        "two_sampler_output_bytes": output_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("canonical", "max", "all"), default="canonical"
    )
    arguments = parser.parse_args()
    selected = CASES.values() if arguments.case == "all" else (CASES[arguments.case],)
    results = {case.name: benchmark(case) for case in selected}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
