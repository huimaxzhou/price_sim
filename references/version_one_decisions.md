# Version-One Decision Register

**Status:** Approved
**Approval date:** 2026-10-22

## 1. Purpose and authority

This register records the product-scope decisions required by Step 0 of
[barra_simulation_implementation_plan.md](./barra_simulation_implementation_plan.md).
It resolves choices left open by the architecture; it does not override the
mathematical and behavioral contract in
[barra_simulation_specification_manual_v4.md](./barra_simulation_specification_manual_v4.md).

If this register conflicts with the v4 specification, the v4 specification
controls and the conflict must be reported. Changes to an approved decision
require explicit user approval under the repository `AGENTS.md` rules.

## 2. Approved runtime and interface

| Decision | Approved version-one choice |
|---|---|
| Python support | Python 3.11 or later |
| Operating systems | Windows and Linux |
| Required production dependencies | Python standard library and NumPy only |
| Primary caller interface | Direct Python construction of typed request objects |
| Configuration-file input adapters | Deferred until the core Python API is accepted |
| Core policy value form | The direct Python API accepts the declared enum types, not arbitrary strings. String parsing, aliases, and serialized-value diagnostics belong to a future configuration adapter. |
| Simulation schedule input | Exactly one of explicit future trading dates, a positive trading-day count, or a simulation end date. |
| Trading calendar | The caller supplies a deterministic trading-calendar source and time zone. The core does not infer holidays or equate weekdays with trading days. |
| Execution model | In-process and single-process |
| Path batching | Deferred |
| Parallel execution | Deferred |

## 3. Approved simulation policies

| Decision | Approved version-one choice |
|---|---|
| Factor-exposure dynamics | `frozen` only |
| Factor-volatility dynamics | `frozen` only |
| Factor-correlation dynamics | `frozen` only |
| Specific-risk dynamics | `frozen` only |
| Factor-return distribution | Gaussian only |
| Specific-return distribution | Gaussian only |
| Return serial correlation | `none` only |
| Default factor-specific joint policy | `independent` |
| Additional joint policy | Block-diagonal joint Gaussian with zero factor-specific cross-covariance |
| Price-return convention | Raw simple arithmetic price returns only |
| Invalid arithmetic-return policy | Explicit `fail` policy only; the caller must supply it and there is no implicit default |
| Log-return price evolution | Deferred and rejected as unsupported |
| Currency handling | `None` only |
| Corporate actions and delistings | `None` only |
| New listings | `None` only |

Unsupported nondefault policies must fail configuration validation. They must not
be ignored, approximated, or silently resolved to a supported default.

For a live asset, a simulated arithmetic price return at or below \(-100\%\)
must fail with an error identifying the path, date, and asset. Version one must
not clip, floor, truncate, or resample the return.

## 4. Approved persistence behavior

The core `simulate()` operation returns an in-memory result and performs no file
writes. Optional persistence uses:

- uncompressed NPZ for numerical arrays;
- JSON for the manifest, resolved policy, and diagnostics; and
- JSON Lines for asset-state events only when at least one event exists.

When no asset-state events exist, the JSON Lines file is omitted and the manifest
records an event count of zero. Compression is deferred.

## 5. Approved scale and memory policy

The expected version-one design maxima are:

| Dimension | Symbol | Approved expected maximum |
|---|---:|---:|
| Assets | \(N\) | 250 |
| Factors | \(K\) | 100 |
| Monte Carlo paths | \(M\) | 100 |
| Future trading days | \(H\) | 252 |

The default estimated peak-memory limit is exactly
\(1{,}000{,}000{,}000\) bytes (one decimal gigabyte, 1 GB).

Before initializing any random-number generator, validation must estimate peak
memory for the complete requested run. If the estimate exceeds the configured
limit, the request must fail rather than merely warn and continue. A caller may
explicitly configure a higher limit; the chosen limit and estimated peak usage
must be recorded in diagnostics.

The values of \(N\), \(K\), \(M\), and \(H\) are design targets, not independent
substitutes for the memory check. A request within every individual maximum may
still be rejected if its complete estimated peak memory exceeds the configured
limit.

External numerical inputs are copied once when accepted into immutable request
state. Numerical arrays created and owned by the engine are not copied again
when placed into `InnovationBatch` or `SimulationResult`; they are marked
read-only in place. Deterministic validation and normalization occur once
before random-number initialization. Later stages reuse the validated context
and must not repeat full-array validation or copying without a measured and
approved need.

## 6. Explicitly deferred behavior

Version one does not implement:

- non-frozen exposure or risk dynamics;
- non-Gaussian return distributions;
- serially correlated innovations;
- nonzero factor-specific cross-covariance;
- log-return price evolution;
- nondefault currency conversion;
- corporate actions or delistings;
- new listings;
- clipping, flooring, truncation, or resampling safeguards;
- configuration-file input adapters;
- parsing string policy values inside the simulation core;
- built-in exchange holiday inference or a third-party calendar dependency;
- compressed persistence;
- path batching;
- parallel execution;
- database storage; or
- risk-model estimation from historical data.

Adding any deferred behavior requires an approved specification and corresponding
acceptance tests before production implementation.

## 7. Step 0 completion

Step 0 is complete because:

1. every provisional version-one decision has an approved value;
2. unsupported behavior has an explicit rejection or deferral rule;
3. the scale and memory policy has an exact default limit; and
4. later implementation steps no longer need to infer product scope.
