"""Non-gating Packet C timings at the approved representative scales.

Run ``python tests/performance/benchmark_packet_c.py --case all``. Results are
observations only; Packet C has no hardware-dependent performance threshold.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_packet_b import CASES, _request  # noqa: E402

from factor_sim.config import resolve_policy  # noqa: E402
from factor_sim.engine import (  # noqa: E402
    construct_factor_returns,
    construct_modeled_returns,
    construct_specific_returns,
    simulate,
)
from factor_sim.pricing import evolve_arithmetic_prices  # noqa: E402
from factor_sim.sampling import build_random_streams, sample_innovations  # noqa: E402
from factor_sim.validation import validate_request  # noqa: E402


def _seconds(callable_: object) -> tuple[object, float]:
    start = perf_counter()
    result = callable_()  # type: ignore[operator]
    return result, perf_counter() - start


def benchmark(case_name: str) -> dict[str, object]:
    case = CASES[case_name]
    base_request = _request(case)
    request = dataclasses.replace(
        base_request,
        risk_state=dataclasses.replace(
            base_request.risk_state,
            factor_covariance=base_request.risk_state.factor_covariance * 0.01,
            specific_variances=base_request.risk_state.specific_variances * 0.01,
        ),
    )
    policy = resolve_policy(request.policy)
    context = validate_request(request, policy)
    streams = build_random_streams(
        seed=request.random_seed,
        path_count=request.path_count,
        joint_policy=policy.return_joint_distribution,
        bit_generator_name=request.rng_bit_generator,
        stream_policy=request.rng_stream_policy,
    )
    innovations = sample_innovations(
        resolved_policy=policy,
        factor_covariance=request.risk_state.factor_covariance,
        specific_variances=request.risk_state.specific_variances,
        step_count=case.steps,
        rng_streams=streams,
        symmetry_tolerance=request.numerical_tolerances.symmetry,
        positive_semidefinite_tolerance=(
            request.numerical_tolerances.positive_semidefinite
        ),
    )

    def construct_returns() -> object:
        factor = construct_factor_returns(
            request.factor_means, innovations.factor_innovations
        )
        specific = construct_specific_returns(
            request.specific_means, innovations.specific_innovations
        )
        return construct_modeled_returns(request.risk_state.exposures, factor, specific)

    modeled, return_seconds = _seconds(construct_returns)
    _, pricing_seconds = _seconds(
        lambda: evolve_arithmetic_prices(
            request.initial_prices,
            modeled,
            asset_ids=context.asset_ids,
            future_dates=context.resolved_future_dates,
            invalid_return_policy=policy.invalid_return_policy,
        )
    )
    result, end_to_end_seconds = _seconds(lambda: simulate(request))
    return {
        "dimensions": dict(context.dimensions),
        "seconds": {
            "return_construction": return_seconds,
            "price_evolution": pricing_seconds,
            "end_to_end": end_to_end_seconds,
        },
        "result_bytes": sum(
            value.nbytes
            for value in (
                result.factor_innovations,
                result.specific_innovations,
                result.factor_returns,
                result.specific_returns,
                result.modeled_returns,
                result.price_link_returns,
                result.prices,
                result.target_asset_covariance,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("canonical", "max", "all"), default="canonical"
    )
    arguments = parser.parse_args()
    names = CASES if arguments.case == "all" else (arguments.case,)
    print(
        json.dumps(
            {name: benchmark(name) for name in names},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
