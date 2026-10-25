# Packet C Acceptance Contract

**Status:** Active; implementation approved at Review Gate C on 2026-10-25
**Prepared:** 2026-10-24
**Scope:** Implementation-plan Steps 8–10
**Acceptance module:** `test_packet_c_contracts.py`

## Review-state boundary

Review Gate B passed on 2026-10-24. The user approved this revised Packet C
contract and authorized Steps 8–10 implementation on 2026-10-24. The approved
acceptance-test SHA-256 is
`1EFD00BE8411CFCAAEABB0A5C4707183A9DB292DDD87CFD133F2C4482D213AFA`.
The user approved the completed Packet C implementation at Review Gate C on
2026-10-25.

On 2026-10-25, the user approved the Packet D successor amendment to
`C-ENGINE-008`: all preflight summaries must remain present, while the completed
engine now reports `diagnostic_scope="complete"`, includes statistical
diagnostics, and exposes structured checks. The superseding Packet C
acceptance-test SHA-256 is
`E4CE706DB87B9D87F8D5D0F11F8E15F806951579C7DFEEA3F50DB5ABEA1D4B0B`.

## Purpose and boundary

Packet C turns the validated stochastic core into one complete in-memory
simulation:

1. add supplied means to factor and specific innovations;
2. construct modeled asset returns with the frozen exposure matrix;
3. map compatible raw arithmetic modeled returns to distinct price-link
   returns;
4. apply explicit version-one `None` handlers;
5. evolve arithmetic prices while enforcing the explicit fail policy; and
6. orchestrate the full sequence through `simulate(request)`.

It does not implement comprehensive statistical diagnostics, persistence,
alternative return mappings, log-price evolution, FX conversion, corporate
actions, delistings, new listings, path batching, or parallel execution. Those
remain deferred.

## Proposed callable surface

### Return construction in `engine.py`

- `construct_factor_returns(factor_means, factor_innovations)`;
- `construct_specific_returns(specific_means, specific_innovations)`; and
- `construct_modeled_returns(exposures, factor_returns, specific_returns)`.

Each function accepts validated `float64` NumPy arrays, verifies its direct
shape relationship, leaves every input unchanged, and returns a new owned
`float64` array. The modeled-return expression is mathematically equivalent to

```python
np.einsum("mhk,nk->mhn", factor_returns, exposures) + specific_returns
```

where `m` is path, `h` is future date, `k` is factor, and `n` is asset.
Internal shape inconsistencies raise `NumericalError`; they are not reclassified
as caller preflight errors after validation has succeeded.

### Pricing in `pricing.py`

- `map_modeled_returns(modeled_returns, *, risk_state, resolved_policy)`;
- `apply_currency_policy(price_link_returns, *, resolved_policy,
  asset_currencies)`;
- `apply_corporate_action_policy(price_link_returns, *, resolved_policy)`;
- `apply_new_listing_policy(asset_ids, *, resolved_policy)`; and
- `evolve_arithmetic_prices(initial_prices, price_link_returns, *, asset_ids,
  future_dates, invalid_return_policy)`.

`map_modeled_returns` permits identity mapping only when the frozen risk state
declares `return_convention="arithmetic"` and `return_basis="raw_price"`, and
the resolved price-return policy is `raw`. It returns a distinct owned array so
`modeled_returns` and `price_link_returns` remain separately named outputs. The
associated manifest metadata tags the entire array as arithmetic, raw-price,
identity-mapped, and unadjusted.

The three version-one handler functions are explicit no-ops. Currency handling
returns the same price-link array and records that every asset remains in its
declared quote currency. Corporate-action handling returns the same array, no
events, and an explicit `None` record. New-listing handling preserves the fixed
asset universe and records `None`. No no-op makes a full-array copy.
These functions are Packet C dispatch boundaries for the supported `None`
policies, not a frozen extension API for future nondefault handlers. Any such
handler still requires its separately approved design.

`evolve_arithmetic_prices` returns an owned array with shape `(M, H + 1, N)`.
The first price slice repeats `initial_prices` for every path. It loops only over
dates and vectorizes over paths and assets.

### Structured runtime failure

`PolicyExecutionError` gains readable `code`, `message`, and immutable `context`
attributes. An arithmetic price-link return at or below `-1` raises
`code="invalid_arithmetic_return"` and records:

- zero-based `path_id`;
- zero-based `step_index`;
- `future_date`;
- `asset_id`; and
- the offending `return_value`.

If several values fail at the same earliest date, the reported offender is the
first in deterministic path-major, asset-major order. The engine does not clip,
floor, truncate, resample, or silently mark the asset terminal. Nonfinite
constructed returns or prices raise `NumericalError` rather than entering the
result. A finite-input price overflow identifies the affected path, date, and
asset through structured numerical-error context.

Incompatible return convention or basis is a deterministic request error. The
engine rejects it during preflight with `ValidationError`, before RNG creation;
it is not deferred to `map_modeled_returns` as a runtime policy failure.

### Public orchestration

`simulate(request: SimulationRequest) -> SimulationResult` follows this visible
order:

1. resolve policy defaults;
2. validate once and retain the validated context;
3. build RNG streams;
4. sample innovations;
5. construct factor and specific returns;
6. construct modeled returns;
7. map price-link returns and apply the three `None` policies;
8. evolve arithmetic prices;
9. construct a deterministic preflight `ValidationReport`;
10. construct the manifest; and
11. transfer owned arrays into `SimulationResult` without redundant copies.

`simulate` performs no file writes. Comprehensive post-simulation diagnostics
remain Step 11 work; Packet C's `ValidationReport` contains no issues and the
already-computed deterministic preflight summaries only. It explicitly records
`diagnostic_scope="preflight_only"` and that statistical diagnostics are not yet
included.

## Proposed manifest minimum

The manifest's top-level mapping is immutable and contains at least:

- `schema_version`;
- run ID, risk-model name, and risk-model version;
- dimensions and explicit axis-order strings;
- supplied schedule plus resolved dates;
- calendar identity, time zone, and provenance;
- modeled-return and price-link-return tags;
- ordered asset currencies;
- every resolved policy value and its explicit/default source;
- seed, bit generator, stream derivation, sampler identity, and NumPy version;
- numerical tolerances and numerical-normalization summaries;
- handler identities and their `None` behavior;
- request and risk-state provenance; and
- the statement that the risk state remained frozen.

Packet C does not prescribe serialization-ready hashing or the later structured
statistical-diagnostics schema. Nested manifest immutability is intentionally
not part of this acceptance contract; the engine is expected to be called by a
careful, controlled caller.

## Proposed package exports

The package root exports only the primary function and caller-facing domain
dataclasses needed to construct a request or consume a result:

- `simulate`;
- `SimulationRequest`, `SimulationResult`, and `SimulationPolicy`;
- `ResolvedSimulationPolicy`, `AssetStateEvent`, and `ValidationReport`;
- `RiskState`, `AssetMetadata`, and `FactorMetadata`;
- `SimulationSchedule` and `TradingCalendar`;
- `JointGeneratorConfig`; and
- `NumericalTolerances`.

Enums and exception classes remain available from `factor_sim.config` and
`factor_sim.errors`; internal helpers and `ValidatedContext` are not re-exported.

## Requirement traceability

| Test ID | Governing requirement | Evidence type |
|---|---|---|
| `C-RET-001` | Means are added to innovations with exact `(M,H,K)` and `(M,H,N)` broadcasting | Hand-calculated deterministic fixture |
| `C-RET-002` | Modeled returns satisfy `r = Xf + epsilon` across paths and dates | Hand-calculated deterministic fixture |
| `C-RET-003` | Constant-by-date and time-varying means behave correctly | Deterministic fixtures |
| `C-RET-004` | Direct helper shape inconsistencies fail clearly | Parameterized negative tests |
| `C-RET-005` | Return construction preserves inputs and produces owned `float64` outputs | Identity and mutation checks |
| `C-MAP-001` | Compatible raw arithmetic identity mapping remains a distinct, fully tagged output | Deterministic mapping and manifest tests |
| `C-MAP-002` | Incompatible return convention or basis fails preflight before RNG creation | Structured validation and RNG tripwire |
| `C-HANDLER-001` | `None` currency handling preserves values and records quote currencies | Deterministic no-op test |
| `C-HANDLER-002` | `None` corporate-action and new-listing policies emit no events or universe changes | Deterministic no-op test |
| `C-PRICE-001` | Arithmetic prices compound exactly over dates, paths, and assets | Hand-calculated deterministic fixture |
| `C-PRICE-002` | Zero and positive returns preserve strict price positivity | Boundary fixture |
| `C-PRICE-003` | Returns at and below `-100%` fail without repair | Parameterized negative test |
| `C-PRICE-004` | Runtime failure identifies path, date, asset, and offending value deterministically | Structured-error inspection |
| `C-PRICE-005` | Nonfinite returns or prices do not enter a result; finite-input overflow identifies path, date, and asset | Negative numerical and structured-context tests |
| `C-PRICE-006` | Price outputs have `(M,H+1,N)`, own their data, and include the initial-price slice | Ownership and shape test |
| `C-PRICE-007` | Price helper dimensions, identifiers, and dates align exactly | Parameterized negative tests |
| `C-ENGINE-001` | A zero-risk canonical fixture produces exact hand-calculated returns and prices | End-to-end integration fixture |
| `C-ENGINE-002` | Every result array has the documented shape, dtype, and axis order | Integration inspection |
| `C-ENGINE-003` | Same request and seed reproduce the complete result; different seeds change stochastic arrays | Deterministic reproduction test |
| `C-ENGINE-004` | Frozen risk arrays and request inputs remain unchanged | Snapshot comparison |
| `C-ENGINE-005` | Dates, IDs, currencies, metadata, covariance, policies, and provenance survive into the result | Result and manifest inspection |
| `C-ENGINE-006` | Default handlers are explicit, emit no events, and do not alter arrays | End-to-end no-op evidence |
| `C-ENGINE-007` | Successful and failed simulations perform no file writes; invalid preflight consumes no RNG | RNG and file-I/O tripwires |
| `C-ENGINE-008` | The deterministic report reuses preflight summaries and explicitly identifies its preflight-only scope | Report inspection |
| `C-ENGINE-009` | Engine-created result arrays are owned and read-only without constructor recopying | Ownership inspection |
| `C-API-001` | The package root exposes the approved narrow public surface | Module namespace inspection |

## Performance posture

Packet C should extend the existing non-gating benchmark with one end-to-end
timing for the canonical and approved maximum shapes. It should report return
construction and price evolution separately. No hardware-dependent threshold is
proposed. The date loop required by price dependence is permitted; per-path and
per-asset Python loops are not.

The previously approved decision remains in force: path batching, parallelism,
compiled extensions, and other invasive optimization are deferred until later
end-to-end profiling identifies a material bottleneck.

## Questions resolved by this draft

- Whole-array return tags live in the manifest rather than a new large wrapper
  object.
- Runtime invalid-return failures use structured `PolicyExecutionError`
  context; incompatible return metadata fails preflight with `ValidationError`;
  and finite-input price overflow carries structured `NumericalError` context.
- Packet C's validation report is explicitly marked preflight-only and does not
  imply that Step 11 statistical diagnostics exist.
- Identity return mapping creates the separately retained price-link array once;
  the three `None` handlers reuse it without further full-array copies.
- Package-root exports include the caller-facing nested result dataclasses but
  stay narrow; policy enums and errors retain their module homes.
- Only the top-level manifest mapping is immutable; nested immutability is not
  required for the controlled version-one caller.

## Approved contract check

Run the active Packet C contract with:

```powershell
python -m pytest tests/acceptance/test_packet_c_contracts.py -q
```

Normal `python -m pytest` discovery collects this approved contract module.
