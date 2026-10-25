# Factor-Risk Equity Simulator

A readable NumPy simulator for equity returns and price paths under a supplied
factor-risk model. The risk model supplies covariance structure; the caller also
supplies means, distribution policies, a trading schedule, prices, and random
controls needed to define a complete stochastic simulation.

## Installation

Python 3.11 or later is supported on Windows and Linux.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux, use the virtual environment's `bin` directory instead of `Scripts`.
NumPy is the only production dependency.

## Quick start

Construct a typed `SimulationRequest`, then call `simulate(request)`. The
request contains a frozen `RiskState`, ordered asset metadata, initial prices,
date-specific means, a trading calendar, an explicit invalid-return policy, and
RNG controls.

```python
import numpy as np

from factor_sim import (
    AssetMetadata, NumericalTolerances, RiskState, SimulationPolicy,
    SimulationRequest, SimulationSchedule, TradingCalendar, simulate,
)
from factor_sim.config import InvalidReturnPolicy

dates = ("2026-10-26", "2026-10-27")
risk_state = RiskState(
    exposures=np.array([[1.0], [0.5]]),
    factor_covariance=np.array([[0.0001]]),
    specific_variances=np.array([0.0001, 0.0002]),
    asset_ids=("a", "b"), factor_ids=("market",),
    as_of_date="2026-10-25", known_at="2026-10-25",
    return_horizon="one_trading_day", observation_frequency="daily",
    return_convention="arithmetic", return_basis="raw_price",
    covariance_units="one_trading_day_return_variance",
    annualization_state="deannualized_daily",
    provenance={"source": "example"},
)
request = SimulationRequest(
    run_id="example", risk_model_name="example-risk", risk_model_version="1",
    simulation_origin="2026-10-25",
    schedule=SimulationSchedule(explicit_future_dates=dates),
    trading_calendar=TradingCalendar(
        calendar_id="example-calendar", time_zone="UTC", sessions=dates,
        provenance={"source": "example"},
    ),
    path_count=10, initial_prices=np.array([100.0, 50.0]),
    factor_means=np.zeros((2, 1)), specific_means=np.zeros((2, 2)),
    risk_state=risk_state,
    asset_metadata=(
        AssetMetadata(asset_id="a", asset_currency="USD"),
        AssetMetadata(asset_id="b", asset_currency="USD"),
    ),
    policy=SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL),
    random_seed=7, rng_bit_generator="PCG64", rng_stream_policy="per_path",
    numerical_tolerances=NumericalTolerances(
        symmetry=1e-12, positive_semidefinite=1e-12, equation=1e-12,
    ),
    provenance={"source": "example"},
)
result = simulate(request)
```

`result.validation_report` contains deterministic and statistical checks. A
statistical failure is evidence for the caller to inspect; it does not turn an
otherwise valid simulation into a runtime exception.

## Array conventions

- Factor arrays: `(path, future_date, factor)` or `M × H × K`.
- Specific and asset-return arrays: `(path, future_date, asset)` or `M × H × N`.
- Prices: `(path, date_including_origin, asset)` or `M × (H + 1) × N`.
- Exposures: `(asset, factor)` or `N × K`.
- Factor covariance: `(factor, factor)` or `K × K`.
- Specific variances: `(asset,)` or `N`.

All accepted numerical inputs and emitted arrays are owned `float64`, read-only
NumPy arrays.

## Supported policies

Version one supports Gaussian factor and specific innovations, independent or
registered block-diagonal joint generation, no serial correlation, frozen risk,
raw arithmetic price returns, explicit failure for returns at or below -100%,
and `None` currency, corporate-action, and new-listing handlers.

## Unsupported policies

Non-Gaussian distributions, serially correlated returns, nonzero
factor-specific cross-covariance, dynamic risk, currency conversion,
corporate-action or delisting processing, new listings, log-return pricing,
clipping, resampling, path batching, and parallel execution are deferred.

## Reproducibility

A fixed seed, PCG64 bit generator, and recorded per-path stream derivation make
repeated runs reproducible in the supported environment. The manifest records
runtime versions and numerical canonicalization. File hashes verify a saved
bundle's integrity, but NPZ bytes are not promised identical across platforms.

## Saved output

Persistence is explicit and remains outside `simulate`:

```python
from pathlib import Path
from factor_sim.io import load_result, save_result

output = save_result(result, Path("runs/example"))
restored = load_result(output)
```

The new directory contains uncompressed `arrays.npz`, `manifest.json`, and
`diagnostics.json`. The numerical archive includes complete risk inputs, means,
initial prices, innovations, returns, prices, and target covariance. Version-one
runs omit `asset_state_events.jsonl`. Saving refuses an existing target and
publishes through a same-volume temporary-directory rename.

## Running tests

```powershell
python -m pytest
python -m ruff check .
python tests/performance/benchmark_packet_d.py --case all
```

The Packet D statistical fixture uses 400 paths as a routine smoke test.
Performance results are observations, not hardware-dependent gates.

## Governing documents

Repository rules are in [`AGENTS.md`](AGENTS.md). Mathematical behavior,
architecture, implementation order, and version-one decisions are governed by
the documents under [`references/`](references/).
