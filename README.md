# Factor-Risk Equity Simulator

A readable NumPy simulator for equity returns and price paths under a supplied
factor-risk model. The risk model supplies covariance structure; the caller also
supplies means, distribution policies, a trading schedule, prices, and random
controls needed to define a complete stochastic simulation.

## Installation

Python 3.11 or later is supported on Windows and Linux.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux, use the virtual environment's `bin` directory instead of `Scripts`.
NumPy is the only production dependency.

## Quick start: manual end-to-end walkthrough

The project is a Python library; it does not provide a separate command-line
program. A complete run consists of constructing a typed `SimulationRequest`,
calling `simulate(request)`, reviewing the returned arrays and diagnostics, and
optionally saving the result with `save_result`.

Run the following commands from the repository root. First, create the
environment and install the package as shown under **Installation**. Confirm
that the package imports from that environment:

```powershell
.\.venv\Scripts\python -c "import factor_sim; print(factor_sim.__file__)"
```

On Linux, replace `.\.venv\Scripts\python` with `.venv/bin/python` in this
walkthrough.

### 1. Understand the example inputs

The example below uses these dimensions:

| Symbol | Meaning | Example value |
|---|---|---:|
| `M` | simulated paths | 400 |
| `H` | future trading dates | 2 |
| `N` | assets | 2 |
| `K` | factors | 1 |

The risk snapshot supplies exposures `X`, factor covariance `F`, and specific
variances `d`. The simulator derives the target asset covariance as
`X F X.T + diag(d)`. The caller must separately supply factor and specific
means, because a covariance risk model alone is not a complete return model.
All covariance values in this example are one-trading-day return variances and
the price link uses raw arithmetic returns.

### 2. Create a runnable simulation

Create `manual_pipeline.py` in the repository root with the following content.
The numbered comments correspond to the major pipeline stages.

```python
from collections import Counter
from pathlib import Path

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
from factor_sim.io import load_result, save_result

# 1. Supply one frozen risk-model snapshot.
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

# 2. Complete the stochastic model and run controls.
request = SimulationRequest(
    run_id="manual-example",
    risk_model_name="example-risk",
    risk_model_version="1",
    simulation_origin="2026-10-25",
    schedule=SimulationSchedule(explicit_future_dates=dates),
    trading_calendar=TradingCalendar(
        calendar_id="example-calendar", time_zone="UTC", sessions=dates,
        provenance={"source": "example"},
    ),
    path_count=400,
    initial_prices=np.array([100.0, 50.0]),
    factor_means=np.zeros((2, 1)),
    specific_means=np.zeros((2, 2)),
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

# 3. Validate, sample returns, evolve prices, and compute diagnostics.
result = simulate(request)

# 4. Inspect identifiers, dates, array shapes, one path, and diagnostics.
print("assets:", result.asset_ids)
print("future dates:", result.future_dates)
print("factor returns shape:", result.factor_returns.shape)
print("modeled returns shape:", result.modeled_returns.shape)
print("prices shape:", result.prices.shape)
print("first price path:\n", result.prices[0])

checks = result.validation_report.summaries["checks"]
status_counts = Counter(check["status"] for check in checks)
print(
    "diagnostic overall status:",
    result.validation_report.summaries["overall_status"],
)
print("diagnostic status counts:", dict(status_counts))
for check in checks:
    if check["status"] != "pass":
        print("review diagnostic:", check["name"], check["status"], check["message"])

# 5. Repeat in memory to demonstrate fixed-seed reproducibility.
repeated = simulate(request)
np.testing.assert_array_equal(result.factor_innovations, repeated.factor_innovations)
np.testing.assert_array_equal(
    result.specific_innovations,
    repeated.specific_innovations,
)
np.testing.assert_array_equal(result.prices, repeated.prices)
print("fixed-seed replay: exact array match")

# 6. Persist the result. save_result intentionally refuses overwrite.
output_dir = Path("outputs/manual-example")
if output_dir.exists():
    raise FileExistsError(
        f"{output_dir} already exists; rename it or choose a new output directory"
    )
saved_dir = save_result(result, output_dir)
print("saved bundle:", saved_dir)

# 7. Load the bundle through its integrity-checking reader.
restored = load_result(saved_dir)
np.testing.assert_array_equal(result.prices, restored.prices)
assert result.asset_ids == restored.asset_ids
assert result.future_dates == restored.future_dates
print("validated round trip: exact prices, assets, and dates")
```

### 3. Run it

```powershell
.\.venv\Scripts\python manual_pipeline.py
```

The important output is:

- factor returns shaped `(400, 2, 1)`, or `(M, H, K)`;
- modeled returns shaped `(400, 2, 2)`, or `(M, H, N)`;
- prices shaped `(400, 3, 2)`, because the price axis includes the origin;
- a diagnostic overall status and counts for individual checks;
- confirmation that the same seed reproduced identical arrays; and
- confirmation that save/load preserved the selected data exactly.

Diagnostics include both deterministic identities and finite-sample statistical
checks. A statistical `fail` is returned as evidence to inspect; it does not
turn an otherwise valid simulation into a runtime exception. A
`not_applicable` result can be legitimate, for example when correlation is
undefined because a component has zero variance. Always inspect non-`pass`
checks rather than relying only on the overall status.

### 4. Inspect the saved bundle

The example creates `outputs/manual-example/`. The repository ignores `outputs/`
so generated runs do not appear as source changes. Inspect the portable
metadata and diagnostics directly:

```powershell
Get-ChildItem outputs\manual-example
Get-Content -Raw outputs\manual-example\manifest.json
Get-Content -Raw outputs\manual-example\diagnostics.json
```

Use NumPy to list the stored numerical arrays without loading the bundle through
the simulator:

```powershell
.\.venv\Scripts\python -c "import numpy as np; z=np.load('outputs/manual-example/arrays.npz'); print('\n'.join(f'{n}: {z[n].shape}' for n in z.files))"
```

For a second manual run, change `run_id` and `output_dir`, or rename the first
output directory. Refusing an existing destination is an intentional atomicity
and no-overwrite safeguard.

### 5. Relate the code to the internal pipeline

`simulate(request)` performs the following sequence:

1. resolve explicit and default policy values;
2. validate dimensions, identifiers, dates, numerical inputs, supported
   policies, and the configured memory limit before initializing randomness;
3. derive per-path PCG64 random streams from the supplied seed;
4. sample factor and specific innovations from the supplied covariance model;
5. add means and construct asset returns as `X f + epsilon`;
6. map supported raw arithmetic returns into prices;
7. calculate deterministic and statistical diagnostics; and
8. return an immutable `SimulationResult` without writing files.

`save_result` is a separate, explicit stage. It writes into a temporary sibling
directory, validates the bundle, and then publishes it by rename. `load_result`
checks the file inventory, byte counts, SHA-256 hashes, schema, numerical data,
and cross-file consistency before reconstructing a typed result.

### 6. Run the automated end-to-end checks

After the manual exercise, run the focused integration tests and then the full
approved suite:

```powershell
.\.venv\Scripts\python -m pytest tests\integration -q
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check .
```

Passing tests complement the manual walkthrough; they do not replace reviewing
the request semantics, diagnostics, and saved manifest for a real model input.

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

output = save_result(result, Path("outputs/example"))
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
