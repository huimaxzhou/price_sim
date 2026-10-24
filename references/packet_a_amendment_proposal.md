# Packet A Amendment Proposal

**Status:** Implemented and approved at Review Gate A
**Approval date:** 2026-10-23
**Prepared:** 2026-10-23
**Scope:** Exact proposed amendments to the governing documents and the
approved Packet A acceptance contract. No protected document or approved test
has been changed.

## 1. Purpose

This proposal incorporates the manual Review Gate A decisions made after the
initial Packet A implementation:

1. the caller may specify the simulation schedule using explicit future trading
   dates, a trading-day count, or a simulation end date;
2. the direct Python API is trusted and typed, so string parsing belongs in a
   future adapter rather than the simulation core;
3. joint-generator metadata is grouped into one typed object;
4. version one retains only `None` currency, corporate-action, and new-listing
   handlers and does not need empty handler registries;
5. external numerical inputs are defensively copied, while internally generated
   output arrays are transferred without a redundant full-array copy and then
   marked read-only; and
6. validation and normalization occur once at the preflight boundary. The
   engine must not repeat full-array checks or copies merely as defensive work.

The mathematical model, array axis order, policy defaults, explicit
`invalid_return_policy = fail` requirement, covariance identity, and version-one
feature scope do not change.

## 2. Proposed governing-document amendments

### 2.1 `references/version_one_decisions.md`

#### Add to Section 2, Approved runtime and interface

Add these rows:

| Decision | Approved version-one choice |
|---|---|
| Core policy value form | The direct Python API accepts the declared enum types, not arbitrary strings. String parsing, aliases, and serialized-value diagnostics belong to a future configuration adapter. |
| Simulation schedule input | Exactly one of explicit future trading dates, a positive trading-day count, or a simulation end date. |
| Trading calendar | The caller supplies a deterministic trading-calendar source and time zone. The core does not infer holidays or equate weekdays with trading days. |

#### Add to Section 5, Approved scale and memory policy

Add:

> External numerical inputs are copied once when accepted into immutable request
> state. Numerical arrays created and owned by the engine are not copied again
> when placed into `InnovationBatch` or `SimulationResult`; they are marked
> read-only in place. Deterministic validation and normalization occur once
> before random-number initialization. Later stages reuse the validated context
> and must not repeat full-array validation or copying without a measured and
> approved need.

#### Add to Section 6, Explicitly deferred behavior

Add:

- parsing string policy values inside the simulation core; and
- built-in exchange holiday inference or a third-party calendar dependency.

### 2.2 `references/barra_simulation_specification_manual_v4.md`

#### Replace the future-date bullet in Section 4.1

Replace:

> an ordered list of $H$ future trading dates;

with:

> exactly one schedule specification: (a) an ordered list of explicit future
> trading dates, (b) a positive number of future trading days, or (c) a
> simulation end date. The caller must also supply a deterministic trading
> calendar and time zone. Schedule resolution produces the ordered dates
> $(t_1,\ldots,t_H)$ used by the numerical engine.

Immediately after the Section 4.1 input list, add:

> Every resolved date must be a trading session strictly after the simulation
> origin $t_0$. Under a trading-day-count request, the resolved dates are the
> first $H$ calendar sessions after $t_0$. Under an end-date request, the
> resolved dates are all calendar sessions after $t_0$ and on or before the
> requested end date. A trading-day end date is included; a non-trading-day end
> date resolves through the last preceding trading session. A schedule that
> resolves to zero future dates is invalid. Supplying zero or more than one
> schedule method is invalid. The simulator must not silently use weekday-only
> arithmetic or invent holiday rules.

Replace the sentence saying the future-date list determines the simulated
periods with:

> The resolved future-date list, rather than calendar-day arithmetic, determines
> $H$ and the order of simulated return periods. The run manifest retains both
> the caller's schedule specification and the resolved ordered dates.

#### Add to Section 4.2, Frozen risk state

Add:

> External risk-state and request arrays are converted to `float64`, copied once,
> and made read-only when accepted. Arrays newly allocated and owned by the
> simulation engine are not copied merely to construct an output object; they
> are marked read-only before publication. This ownership rule is part of the
> peak-memory contract.

#### Add to Section 4.3, Simulation-policy inputs and defaults

Add after the policy table:

> The language-independent serialized values in this table remain authoritative
> for manifests and future file adapters. The version-one direct Python API uses
> the corresponding enum members. It does not parse arbitrary strings, aliases,
> or misspellings. A future configuration adapter may parse serialized strings
> and must reject unknown or deferred values before constructing the typed core
> request.

In the `joint` discussion, replace references to five parallel policy fields
with one grouped joint-generator configuration containing the same required
identity, version, parameters, marginal-distribution contract, and random-stream
requirements. No mathematical requirement changes.

#### Amend Section 5.1, Initialize and validate the frozen state

Begin the procedure with:

1. resolve exactly one supplied schedule method against the supplied calendar;
2. persist the original schedule specification and resolved ordered dates;
3. resolve policy defaults; and
4. perform deterministic preflight validation once.

Add:

> The validated context is reused by all later stages. Later stages must not
> repeat matrix scans, identifier-alignment work, schedule resolution, memory
> estimation, or defensive input copying unless a documented stage-specific
> invariant requires it.

#### Amend Section 6, outputs

Clarify that outputs contain both:

- the supplied schedule method and value; and
- the resolved ordered future trading dates.

#### Amend Section 7.1, deterministic preflight checklist

Add checks that:

- exactly one schedule method was supplied;
- the trading calendar and time zone are present;
- the resolved dates satisfy the count or end-date request;
- all dates are strictly after the simulation origin and strictly increasing;
- an end-date request resolving to zero sessions fails; and
- no random state is initialized before schedule and request validation passes.

#### Amend Section 9 pseudocode

Change the multi-period entry signature conceptually from:

```text
simulate_paths(initial_state, policy, future_dates, number_of_paths, seed)
```

to:

```text
simulate_paths(
    initial_state,
    policy,
    schedule,
    trading_calendar,
    number_of_paths,
    seed
)
```

and add before validation:

```text
future_dates = RESOLVE_SCHEDULE(schedule, trading_calendar, simulation_origin)
```

All loops continue to use the single resolved `future_dates` sequence.

### 2.3 `references/barra_simulation_architectural_design.md`

#### Amend Section 4, technology choices

Add:

> The direct Python boundary is typed. Serialized string parsing is an adapter
> responsibility and is not duplicated in `config.py`.

#### Amend Section 8.1, Enumerations

Clarify that the core `SimulationPolicy` accepts enum members. Exact strings are
serialization values, not alternative core runtime types.

#### Add `SimulationSchedule`

Add a small frozen dataclass with exactly these optional value fields:

```text
explicit_future_dates: tuple[DateLike, ...] | None
trading_day_count: int | None
simulation_end_date: DateLike | None
```

Exactly one field must be non-`None`. This one-of check occurs once during
preflight validation.

#### Add `TradingCalendar`

Add a small frozen dataclass containing:

```text
calendar_id: str
time_zone: str
sessions: tuple[DateLike, ...]
provenance: mapping
```

The sessions are the deterministic version-one calendar source. No third-party
calendar dependency or weekday inference is added. Sessions must be strictly
increasing and unique. Schedule resolution occurs once and uses a binary search
for the first session after the simulation origin, followed by a bounded slice;
it is never repeated by path or simulated date.

#### Add `JointGeneratorConfig`

Add one frozen dataclass containing:

```text
name: str
version: str
parameters: mapping
marginal_contract: mapping
random_stream_requirements: mapping
```

Replace the five parallel joint-generator fields in `SimulationPolicy` and
`ResolvedSimulationPolicy` with:

```text
joint_generator: JointGeneratorConfig | None | UNSET
```

The resolved form contains `JointGeneratorConfig | None`.

#### Amend Section 8.5, SimulationPolicy

Specify:

- enum policy fields accept only their enum type or the private omission
  sentinel;
- `None`-handler fields accept only explicit `None` or the omission sentinel;
- arbitrary strings are not accepted by the core; and
- a `joint` policy requires one registered `JointGeneratorConfig`, while an
  `independent` policy must not contain an active joint configuration.

#### Amend Section 8.6, SimulationRequest

Replace ordered future dates with:

- `SimulationSchedule`; and
- `TradingCalendar`.

Add explicit fields for behaviorally important inputs that must not be hidden
inside general provenance:

- `risk_model_name: str`;
- `risk_model_version: str`;
- `rng_bit_generator: str`;
- `rng_stream_policy: str`; and
- peak-memory limit in bytes, defaulting to the approved one-billion-byte
  limit through `memory_limit_bytes: int = 1_000_000_000`.

The resolved future dates belong to the validated context and final result.

#### Amend Sections 8.7 and 8.9, InnovationBatch and SimulationResult

Specify two array-ownership operations:

```text
copy_readonly_input(value)
    Convert an external input to an owned float64 copy and mark it read-only.

take_readonly_output(array)
    Require an engine-owned float64 NumPy array, mark that same array read-only,
    and return it without copying.
```

`RiskState` and `SimulationRequest` use the first operation.
`InnovationBatch` and `SimulationResult` use the second.

#### Amend Section 9.1, Configuration

Remove core responsibility for serialized string parsing and remove empty
currency, corporate-action, and new-listing registries. Retain only actual
version-one implementation registries. Require exact registered joint-generator
name and version.

#### Amend Section 9.3, Validation

Add schedule resolution and validation. State explicitly that validation returns
one reusable validated context and that later components do not repeat its
full-array checks.

#### Amend Section 18, Performance and memory strategy

Add these rules:

1. avoid redundant copies of engine-owned `M x H x N` and `M x H x K` arrays;
2. scan each large input matrix only as required by one named preflight check;
3. retain computed dimensions, aligned identifiers, target covariance,
   covariance summaries, resolved dates, and memory estimate in the validated
   context for reuse;
4. do not add defensive validation inside inner path/date/asset loops; and
5. add safety work only when it protects a documented invariant and its cost is
   included in the peak-memory or performance assessment.

### 2.4 `references/barra_simulation_implementation_plan.md`

#### Amend Step 2

Add `SimulationSchedule`, `TradingCalendar`, and `JointGeneratorConfig` to the
domain dataclasses. Replace the single array-freezing helper with the separate
input-copy and output-ownership operations. State that core policy fields use
enums, not arbitrary strings.

#### Amend Step 3

Replace string normalization and unknown-versus-deferred parsing with:

- enum-type validation at the typed boundary;
- default resolution and provenance;
- explicit `invalid_return_policy` enforcement;
- `None`-only handler enforcement;
- grouped joint-generator compatibility; and
- exact joint-generator name/version registration.

Remove the instruction to create empty handler registries. String parsing is
deferred with other configuration-file adapter work.

#### Amend Step 4

Add schedule resolution as the first validation operation. Validate the one-of
schedule rule, calendar coverage, time zone, count/end-date semantics, and
nonempty strictly increasing resolved sessions. Store resolved dates and
calendar provenance in the validated context.

Clarify that all deterministic validation, target-covariance construction, and
memory estimation happen once before RNG construction and are reused later.

#### Amend Steps 6, 7, and 10

Require sampling and engine code to allocate final-shape arrays deliberately and
transfer ownership into `InnovationBatch` and `SimulationResult` without a
redundant copy. No performance benchmark threshold is introduced at Packet A;
the deterministic ownership contract is tested now and measured performance
remains in Step 14.

## 3. Proposed Packet A acceptance-contract amendments

The approved files remain protected until the user approves the exact changes
below. After approval, the acceptance README and test module will receive a new
contract revision and SHA-256.

### 3.1 Acceptance README changes

1. Change stale wording from “Proposed public names” to “Approved public names.”
2. Remove the statement that names become contractual only after approval.
3. Add these public model names:
   - `SimulationSchedule`;
   - `TradingCalendar`; and
   - `JointGeneratorConfig`.
4. State that the direct Python API accepts enum members; serialized string
   parsing is outside Packet A and remains deferred.
5. Replace the `A-POLICY-004` traceability row for unknown/deferred strings with
   a typed-boundary requirement.
6. Replace the `A-POLICY-005` row with grouped joint-generator requirements.
7. Add traceability rows for schedule representation, exact joint-generator
   version, and input/output array ownership.
8. Replace pre-implementation “expected red” wording with current execution and
   approval instructions.

### 3.2 `test_packet_a_contracts.py` changes

#### Keep unchanged in substance

- `A-BOOT-001` package metadata;
- `A-BOOT-002` module imports;
- `A-ERR-001` exception hierarchy;
- `A-ERR-002` structured multiple issues;
- `A-ENUM-001` stable serialization values;
- frozen-dataclass inspection;
- risk-state identifier order and external-input ownership;
- optional factor metadata;
- asset-state event fields;
- minimal default resolution;
- explicit-default provenance;
- missing explicit invalid-return policy; and
- pure deterministic resolution.

Fixtures constructing `RiskState` will add the now-required observation
frequency and covariance-unit metadata.

#### Amend `A-MODEL-001`

Add `SimulationSchedule`, `TradingCalendar`, and `JointGeneratorConfig` to the
required frozen dataclass list.

#### Add `A-MODEL-005` — schedule alternatives are representable

Construct three immutable schedules, one for each supported method, and verify
that the selected value is preserved. This Packet A test does not resolve the
calendar; schedule one-of and date semantics are activated in Packet B.

#### Add `A-MODEL-006` — request owns external numerical inputs

Construct a minimal `SimulationRequest`, mutate the caller's initial-price,
factor-mean, and specific-mean arrays, and verify that request arrays remain
unchanged, float64, and read-only.

#### Add `A-MODEL-007` — engine-owned outputs are not recopied

Construct an `InnovationBatch` from owned float64 arrays and verify:

```python
batch.factor_innovations is factor_array
batch.specific_innovations is specific_array
batch.factor_innovations.flags.writeable is False
batch.specific_innovations.flags.writeable is False
```

Construct a minimal `SimulationResult` and make the equivalent identity and
read-only assertions for every numerical output array. Passing a non-NumPy or
non-float64 output array must fail rather than trigger an implicit full copy.

#### Remove current `A-POLICY-004`

Delete the parameterized core test requiring the simulation engine to parse
`"frozen-ish"` and `"log"`. Those cases belong to a future serialized-input
adapter and are not version-one core behavior.

#### Add replacement `A-POLICY-004` — typed boundary rejects wrong runtime type

Pass a non-enum value to one enum policy field and require a structured
`ConfigurationError` with:

```text
code = invalid_policy_type
field_path = policy.<field>
```

This is one-time boundary validation, not repeated simulation work.

#### Replace current `A-POLICY-005`

Require:

```text
return_joint_distribution = JOINT
joint_generator = None or omitted
    -> missing_joint_generator
```

The test no longer inspects five parallel joint metadata fields.

#### Add `A-POLICY-006` — registered joint configuration resolves

Supply one complete `JointGeneratorConfig` with name
`block_diagonal_gaussian` and version `1`. Verify the resolved policy preserves
the object and records it as explicit.

#### Add `A-POLICY-007` — joint generator version must match

Supply the registered name with version `999` and require:

```text
code = unsupported_joint_generator_version
field_path = policy.joint_generator.version
```

#### Add `A-POLICY-008` — independent mode rejects joint configuration

Supply a complete `JointGeneratorConfig` under `INDEPENDENT` and require:

```text
code = incompatible_policy_parameter
field_path = policy.joint_generator
```

#### Add `A-POLICY-009` — non-None handlers are unsupported

For currency handling, corporate-action adjustment, and new listing, require a
non-`None` runtime value to fail with:

```text
code = unsupported_policy_value
```

No empty handler registry is required to establish this rule.

#### Renumber current purity test

The current pure/deterministic resolution test becomes `A-POLICY-010` without a
substantive behavior change.

### 3.3 Acceptance behavior deliberately removed from the core

The amended Packet A contract will no longer require the core to:

- accept valid policy strings;
- distinguish misspelled strings from recognized-but-deferred strings; or
- maintain empty registries for handlers that do not exist.

A future serialized-configuration adapter must receive its own approved contract
before those parsing behaviors are implemented.

### 3.4 Follow-on impact not activated by this proposal

After the governing and Packet A amendments are approved and implemented, the
inactive Packet B draft must be revised to use:

- `SimulationSchedule` and `TradingCalendar`;
- the resolved future-date output;
- `JointGeneratorConfig`; and
- the no-redundant-copy ownership contract.

The Packet B draft remains inactive and is not acceptance evidence.

## 4. Approval boundary

Approval of this proposal would authorize the next work in this order:

1. amend the four governing documents exactly as described above;
2. amend the protected Packet A acceptance README and tests;
3. record the new approved acceptance-test hash;
4. update `models.py` and `config.py` to satisfy the amended contract;
5. update Packet A unit tests;
6. run all Packet A and repository tests plus Ruff; and
7. return the revised Packet A for manual Review Gate A approval.

It would not authorize implementation of Packet B production code.
