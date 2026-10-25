"""Non-gating Packet D stage timings and memory observations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PERFORMANCE_DIR))

from benchmark_packet_b import CASES, _request  # noqa: E402

from factor_sim.config import resolve_policy  # noqa: E402
from factor_sim.diagnostics import evaluate_result  # noqa: E402
from factor_sim.engine import (  # noqa: E402
    construct_factor_returns,
    construct_modeled_returns,
    construct_specific_returns,
    simulate,
)
from factor_sim.io import load_result, save_result  # noqa: E402
from factor_sim.pricing import (  # noqa: E402
    apply_corporate_action_policy,
    apply_currency_policy,
    apply_new_listing_policy,
    evolve_arithmetic_prices,
    map_modeled_returns,
)
from factor_sim.sampling import build_random_streams, sample_innovations  # noqa: E402
from factor_sim.validation import validate_request  # noqa: E402


def _time(callable_: object) -> tuple[object, float]:
    start = perf_counter()
    value = callable_()  # type: ignore[operator]
    return value, perf_counter() - start


def benchmark(case_name: str) -> dict[str, object]:
    """Measure one approved scale without applying a pass threshold."""

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
    context, validation_seconds = _time(lambda: validate_request(request, policy))
    streams = build_random_streams(
        seed=request.random_seed,
        path_count=request.path_count,
        joint_policy=policy.return_joint_distribution,
        bit_generator_name=request.rng_bit_generator,
        stream_policy=request.rng_stream_policy,
    )
    innovations, sampling_seconds = _time(
        lambda: sample_innovations(
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
    )

    def construct_returns() -> tuple[object, object, object]:
        factor = construct_factor_returns(
            request.factor_means, innovations.factor_innovations
        )
        specific = construct_specific_returns(
            request.specific_means, innovations.specific_innovations
        )
        modeled = construct_modeled_returns(
            request.risk_state.exposures, factor, specific
        )
        return factor, specific, modeled

    (_, _, modeled), return_seconds = _time(construct_returns)

    def construct_prices() -> object:
        mapped, _ = map_modeled_returns(
            modeled,
            risk_state=request.risk_state,
            resolved_policy=policy,
        )
        currencies = tuple(item.asset_currency for item in request.asset_metadata)
        mapped, _ = apply_currency_policy(
            mapped,
            resolved_policy=policy,
            asset_currencies=currencies,
        )
        mapped, _, _ = apply_corporate_action_policy(
            mapped,
            resolved_policy=policy,
        )
        asset_ids, _ = apply_new_listing_policy(
            context.asset_ids,
            resolved_policy=policy,
        )
        return evolve_arithmetic_prices(
            request.initial_prices,
            mapped,
            asset_ids=asset_ids,
            future_dates=context.resolved_future_dates,
            invalid_return_policy=policy.invalid_return_policy,
        )

    _, pricing_seconds = _time(construct_prices)
    result, end_to_end_seconds = _time(lambda: simulate(request))
    _, diagnostics_seconds = _time(lambda: evaluate_result(result))
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        output = Path(temporary) / "saved-run"
        _, serialization_seconds = _time(lambda: save_result(result, output))
        _, read_back_seconds = _time(lambda: load_result(output))

    result_arrays = (
        result.factor_innovations,
        result.specific_innovations,
        result.factor_returns,
        result.specific_returns,
        result.modeled_returns,
        result.price_link_returns,
        result.prices,
        result.target_asset_covariance,
        result.initial_prices,
        result.factor_means,
        result.specific_means,
        result.risk_state.exposures,
        result.risk_state.factor_covariance,
        result.risk_state.specific_variances,
    )
    return {
        "case": case_name,
        "dimensions": dict(context.dimensions),
        "seconds": {
            "validation": validation_seconds,
            "sampling": sampling_seconds,
            "return_construction": return_seconds,
            "pricing": pricing_seconds,
            "diagnostics": diagnostics_seconds,
            "serialization": serialization_seconds,
            "read_back": read_back_seconds,
            "end_to_end": end_to_end_seconds,
        },
        "estimated_peak_memory_bytes": context.estimated_peak_memory_bytes,
        "observed_result_bytes": sum(array.nbytes for array in result_arrays),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("canonical", "max", "all"), default="canonical"
    )
    arguments = parser.parse_args()
    names = CASES if arguments.case == "all" else (arguments.case,)
    print(json.dumps({name: benchmark(name) for name in names}, indent=2))


if __name__ == "__main__":
    main()
