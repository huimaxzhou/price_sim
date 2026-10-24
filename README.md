# Factor-Risk Equity Simulator

This repository will contain a readable Python implementation of the simulation
contract in
[`references/barra_simulation_specification_manual_v4.md`](references/barra_simulation_specification_manual_v4.md).

Packet A defines the typed domain vocabulary, structured errors, and auditable
policy-default resolution. Packet B Steps 4–7 now implement deterministic
preflight validation, covariance roots, reproducible per-path RNG streams, and
independent and joint Gaussian innovation sampling. Return construction, price
linking, diagnostics, and orchestration remain later-packet work.

Review Gate A passed on 2026-10-23. Review Gate B passed on 2026-10-24, and its
acceptance contract is active. Invasive performance optimization remains
deferred until later end-to-end profiling identifies a concrete bottleneck.

## Requirements

- Python 3.11 or later
- Windows or Linux

## Development setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux, activate or address the virtual environment with the platform's usual
`bin` path instead of `Scripts`.

## Verification commands

Run the package-import smoke test:

```powershell
python -m pytest tests/unit/test_import.py -q
```

Run the approved Packet A acceptance contract:

```powershell
python -m pytest tests/acceptance/test_packet_a_contracts.py -q
```

Run all tests:

```powershell
python -m pytest
```

Run the linter:

```powershell
python -m ruff check .
```

Record non-gating Packet B timings at representative and maximum supported
dimensions:

```powershell
python tests/performance/benchmark_packet_b.py --case all
```

Benchmark timings are observations for profiling and regression investigation;
they are not acceptance thresholds.

Passing Packet A tests establishes only the contracts and configuration behavior
covered by Steps 1–3. Packet B unit, statistical, and approved acceptance tests
establish the Step 4–7 validation and innovation-sampling behavior.

## Governing documents

Repository working rules are in [`AGENTS.md`](AGENTS.md). The mathematical
specification, architecture, implementation plan, and approved version-one
decisions are under [`references/`](references/).
