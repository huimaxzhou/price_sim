# Factor-Risk Equity Simulation — Step-by-Step Implementation Plan

## 1. Purpose and authority

This document translates the decisions in [barra_simulation_architectural_design.md](./barra_simulation_architectural_design.md) into an ordered implementation plan for the behavioral contract in [barra_simulation_specification_manual_v4.md](./barra_simulation_specification_manual_v4.md).

Authority order:

1. The v4 simulation specification controls mathematical behavior, inputs, outputs, policies, invariants, and acceptance criteria.
2. The architectural design controls module boundaries, dependency direction, domain objects, numerical strategy, extension seams, and readability constraints.
3. This implementation plan controls task order, intermediate deliverables, tests, and review gates.

If implementation work exposes a contradiction between these documents, stop that work and resolve the contradiction explicitly. Do not silently reinterpret the specification in code.

This plan does not authorize implementation by itself. It defines how implementation should proceed once work begins.

## 2. Implementation principles

Every step must follow these rules:

- Keep each change small enough for manual code review.
- Implement only behavior required by the current step.
- Add tests in the same step as production code.
- Do not add a general abstraction merely because a future feature might need it.
- Do not hide configuration defaults or numerical normalization.
- Preserve terminology and array-axis order from v4.
- Reject unsupported behavior explicitly.
- Do not mix file I/O into the simulation core.
- Do not optimize before correctness and reproducibility are demonstrated.
- Complete each review gate before starting the next group of steps.

## 3. Confirmed version-one decisions

The architecture identifies product decisions that must be made before implementation. The following decisions were approved on 2026-10-22 and are recorded in [version_one_decisions.md](./version_one_decisions.md).

| Decision | Confirmed version-one choice |
|---|---|
| Python support | Python 3.11 or later |
| Operating systems | Windows and Linux |
| Primary caller interface | Direct Python construction of typed request objects |
| Configuration-file adapters | Deferred until the core API is accepted |
| Invalid arithmetic-return policy | Explicit fail policy only; there is no implicit default |
| Log-return price evolution | Deferred; reject it as unsupported in version one |
| Nondefault currency handling | Deferred; implement and record None behavior |
| Nondefault corporate actions or delistings | Deferred; implement contracts and None behavior only |
| New listings | None only |
| Factor-risk dynamics | Frozen only |
| Distributions | Gaussian factor and specific marginals |
| Joint generation | Independent and block-diagonal joint Gaussian |
| Serial correlation | None only |
| Persistence | Uncompressed NPZ plus JSON and conditional JSON Lines |
| Execution | In-process, single process |
| Expected scale | Up to \(N=250\) assets, \(K=100\) factors, \(M=100\) paths, and \(H=252\) future trading days |
| Default memory limit | \(1{,}000{,}000{,}000\) bytes (1 decimal GB); reject before RNG initialization unless the caller explicitly configures a higher limit |

Any change to a confirmed decision requires explicit user approval before this plan or the implementation is changed.

## 4. Planned source tree

~~~text
project-root/
├── pyproject.toml
├── README.md
├── src/
│   └── factor_sim/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── errors.py
│       ├── validation.py
│       ├── sampling.py
│       ├── pricing.py
│       ├── engine.py
│       ├── diagnostics.py
│       └── io.py
└── tests/
    ├── unit/
    ├── integration/
    ├── statistical/
    └── fixtures/
~~~

No additional production module should be introduced without explaining why the existing responsibilities cannot contain the new code clearly.

## 5. Dependency sequence

~~~text
Step 0  Confirm scope and decisions
   |
Step 1  Bootstrap package and test harness
   |
Step 2  Errors, enums, and domain dataclasses
   |
Step 3  Policy resolution
   |
Step 4  Deterministic input validation
   |
Step 5  Numerical covariance roots and RNG streams
   |
Step 6  Independent Gaussian sampling
   |
Step 7  Joint Gaussian sampling
   |
Step 8  Return construction
   |
Step 9  Pricing, no-op handlers, and asset state
   |
Step 10 Engine orchestration
   |
Step 11 Diagnostics
   |
Step 12 Serialization
   |
Step 13 Integration and statistical acceptance suite
   |
Step 14 Readability, performance, and documentation review
   |
Step 15 Final acceptance gate
~~~

## 6. Step 0 — Confirm version-one decisions

**Status:** Complete — approved 2026-10-22. See [version_one_decisions.md](./version_one_decisions.md).

### Objective

Freeze the implementation boundary before source files are created.

### Inputs

- v4 specification Sections 1, 4, 8, and 10;
- architecture Sections 20 and 21; and
- provisional decisions in Section 3 of this plan.

### Actions

1. Confirm the supported Python versions and operating systems.
2. Confirm direct Python construction as the first input interface.
3. Confirm that version one implements only the explicit fail policy for arithmetic returns at or below \(-100\%\).
4. Confirm that log-return pricing is deferred.
5. Confirm that currency, corporate-action, and new-listing handlers support None only.
6. Confirm that both independent and joint block-diagonal Gaussian sampling are in version one.
7. Establish the configurable memory-limit field and its initial default.
8. Confirm uncompressed NPZ plus JSON output.
9. Record deferred features in a short decision register.

### Deliverable

A reviewed [version-one decision register](./version_one_decisions.md) referenced by this implementation plan.

### Acceptance criteria

- No provisional decision remains ambiguous.
- Every deferred feature has an explicit unsupported behavior.
- No implementation step relies on an unstated product choice.

### Manual review gate

Passed on 2026-10-22. The user approved the version-one boundary with \(M=100\) paths and a default peak-memory limit of \(1{,}000{,}000{,}000\) bytes; all other recommended decisions were approved unchanged.

## 7. Step 1 — Bootstrap the Python package

**Status:** Complete — verified 2026-10-22. The package skeleton imports, the smoke test passes, and an isolated editable build and import succeeded.

### Objective

Create the smallest runnable package and test environment.

### Files

- **pyproject.toml**
- **README.md**
- **src/factor_sim/__init__.py**
- empty module files from the planned source tree
- initial test directories

### Actions

1. Configure a standard source-layout Python package.
2. Declare NumPy as the only required production dependency.
3. Declare pytest and a lightweight formatter/linter as development dependencies.
4. Set the supported Python version.
5. Configure pytest to discover unit, integration, and statistical tests separately.
6. Export no unfinished public API.
7. Add one smoke test proving the package imports.

### Tests

- Package import succeeds.
- pytest discovers and runs the smoke test.
- The package can be installed in editable mode.

### Acceptance criteria

- No simulation behavior exists yet.
- There are no circular imports.
- Production dependencies match the architecture.
- The bootstrap commands are documented in the README.

## 8. Step 2 — Implement errors, enums, and domain dataclasses

**Status:** Complete — revised implementation approved at Review Gate A on
2026-10-23.

### Objective

Create the vocabulary and immutable data structures used by every later step.

### Files

- **src/factor_sim/errors.py**
- **src/factor_sim/config.py**
- **src/factor_sim/models.py**
- **tests/unit/test_models.py**
- **tests/unit/test_errors.py**

### Actions

#### 8.1 Error types

Implement:

- SimulationError;
- ConfigurationError;
- ValidationError;
- NumericalError; and
- PolicyExecutionError.

ValidationError must support multiple structured issues. Each issue contains:

- stable issue code;
- field path;
- readable message; and
- optional safe context that does not include entire large arrays.

#### 8.2 Enums

Implement string-valued enums for all v4 policy fields and asset status.

The direct Python core accepts members of the declared enum types, not
arbitrary strings. Parsing strings, aliases, misspellings, and recognized but
deferred serialized values belongs to a future configuration adapter.

#### 8.3 Domain dataclasses

Implement:

- RiskState;
- AssetMetadata;
- FactorMetadata;
- SimulationSchedule;
- TradingCalendar;
- JointGeneratorConfig;
- SimulationPolicy;
- ResolvedSimulationPolicy;
- NumericalTolerances;
- SimulationRequest;
- InnovationBatch;
- AssetStateEvent;
- ValidationReport;
- SimulationResult.

Document every array shape in docstrings.

#### 8.4 Immutability

- Use frozen dataclasses for input and result metadata.
- Copy external numerical inputs once at the accepted request boundary and mark
  them read-only.
- Require internally generated output arrays to be owned `float64` NumPy arrays;
  mark the same arrays read-only when transferred to `InnovationBatch` or
  `SimulationResult` without copying them again.
- Do not expose setters that mutate the frozen risk state.

### Tests

- Dataclasses retain exact field values and identifier order.
- Enum serialization values are stable strings.
- Structured validation issues preserve code, path, and message.
- Array copying prevents later caller mutation from changing accepted state.
- Optional factor metadata may be absent.
- AssetStateEvent requires asset ID, path ID, effective date, status, and reason.

### Acceptance criteria

- All public domain objects are type annotated.
- Models contain no file I/O, RNG, covariance validation, or pricing logic.
- The code uses v4 terminology consistently.
- No base-class hierarchy is introduced for future dynamics.

### Manual review checkpoint A1

Review the full contents of errors.py, config.py, and models.py before continuing. These files establish the vocabulary used by the rest of the codebase.

## 9. Step 3 — Implement policy-default resolution

**Status:** Complete — revised implementation approved at Review Gate A on
2026-10-23.

### Objective

Convert a partially supplied SimulationPolicy into one fully resolved, auditable policy.

### Files

- **src/factor_sim/config.py**
- **tests/unit/test_config.py**

### Actions

1. Implement a pure resolve_policy function.
2. Apply every default listed in v4 Section 4.3.
3. Record explicit or default for each field.
4. Validate enum types at the typed core boundary.
5. Require explicit `invalid_return_policy` enforcement.
6. Enforce `None`-only currency, corporate-action, and new-listing handlers
   directly, without empty handler registries.
7. Validate grouped `JointGeneratorConfig` compatibility and require an exact
   registered joint-generator name and version.
8. Ensure resolving one field does not change unrelated fields.

### Tests

- A supplied policy containing only the required `invalid_return_policy = fail` resolves every other omitted policy field to its documented default.
- Each field can be explicitly set to its default and is recorded as explicit.
- A wrong runtime type fails with a stable typed-boundary error code.
- Joint generation requires a registered joint sampler.
- Independent generation rejects an active joint-generator configuration.
- Non-`None` handlers fail as unsupported.
- The fail invalid-return policy must be supplied explicitly because v4 does not assign it a default.
- Repeated resolution is deterministic and does not mutate the supplied object.

### Acceptance criteria

- Downstream code never reads the unresolved policy.
- Every resolved field has value-source provenance.
- Defaults are described as simulator policies, not risk-model properties.

### Manual review checkpoint A2

Review one fully defaulted policy and one explicitly configured policy as rendered by the code. Confirm that the difference is visible in provenance.

## 10. Review Gate A — Contracts and configuration

**Status:** Passed — user approved the revised Packet A on 2026-10-23. The
approved acceptance-test SHA-256 is
`9CC9EAB2F29DD8297036488E83C99913C42F153E30251D6757919DCCB3A556C7`.

Before Step 4:

- all Step 1–3 tests pass;
- public dataclasses and policy names match v4;
- no numerical implementation has leaked into the models;
- no unsupported policy is silently accepted; and
- the user can read the request and resolved-policy definitions without following generated framework code.

The review packet should contain:

- changed files;
- a one-page object relationship summary;
- example request construction;
- example resolved policy; and
- test results.

## 11. Step 4 — Implement deterministic input validation

**Status:** Complete — approved at Review Gate B on 2026-10-24.

### Objective

Reject invalid simulation requests before any random values are consumed.

### Files

- **src/factor_sim/validation.py**
- **tests/unit/test_validation.py**
- small fixtures under **tests/fixtures/**

### Actions

Implement validation in the following order:

1. resolve exactly one schedule method against the supplied calendar and time
   zone, including count/end-date semantics and calendar coverage;
2. require nonempty, strictly increasing resolved sessions after the simulation
   origin;
3. validate required scalar values and positive \(M\), \(N\), \(K\), and \(H\);
4. unique asset and factor IDs;
5. exact array dimensions;
6. asset and factor metadata alignment;
7. finite values;
8. strictly positive initial prices;
9. nonnegative specific variances;
10. symmetry of \(\mathbf F_0\);
11. positive semidefiniteness of \(\mathbf F_0\);
12. return horizon, units, annualization state, and return basis;
13. known-at timing;
14. asset-currency presence;
15. policy compatibility;
16. joint-generation requirements;
17. raw-price-return compatibility; and
18. estimated memory use.

Compute

$$
\boldsymbol\Sigma_0
=
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\operatorname{diag}(\mathbf d_0)
$$

and validate its symmetry and positive semidefiniteness.

Validation returns a validated context containing dimensions, aligned
identifiers, resolved dates, calendar provenance, target covariance, numerical
summaries, and the memory estimate. All deterministic validation,
target-covariance construction, and memory estimation happen once before RNG
construction and are reused later. Validation must not mutate or repair the
request.

### Tests

Create one focused negative test for each validation rule. Also test:

- several simultaneous input errors are returned together;
- no RNG is initialized on failure;
- zero specific variance is accepted;
- singular positive-semidefinite covariance is accepted;
- eigenvalues below the negative tolerance are rejected;
- optional factor metadata may be missing;
- conflicting factor class and exposure class mappings are rejected;
- currency factors do not satisfy missing asset-currency metadata; and
- unsupported nondefault handlers are rejected.

### Acceptance criteria

- Every v4 deterministic preflight check has a test.
- Errors identify the offending field or matrix.
- Validation does not write files or log large arrays.
- Target covariance is computed exactly once for reuse.

## 12. Step 5 — Implement numerical covariance roots and RNG streams

**Status:** Complete — approved at Review Gate B on 2026-10-24.

### Objective

Provide explicit, reproducible numerical primitives needed by all samplers.

### Files

- **src/factor_sim/sampling.py**
- **tests/unit/test_sampling_numerics.py**
- **tests/unit/test_random_streams.py**

### Actions

#### 12.1 Symmetric canonicalization

1. Recheck that the matrix is within the symmetry tolerance.
2. Construct \((\mathbf A+\mathbf A^{\mathsf T})/2\).
3. Record the maximum asymmetry.
4. Do not overwrite the original matrix.

#### 12.2 PSD square root

1. Use NumPy symmetric eigendecomposition.
2. Reject eigenvalues below \(-\tau_{\mathrm{PSD}}\).
3. Set eigenvalues in \([-\tau_{\mathrm{PSD}},0)\) to zero for root construction.
4. Record the original minimum eigenvalue and every numerical zeroing event.
5. Return a deterministic covariance root.

Do not require Cholesky factorization.

#### 12.3 Random streams

1. Create a master NumPy SeedSequence from the run seed.
2. Spawn one child sequence per path.
3. Under independent generation, spawn separate factor and specific sequences for each path.
4. Under joint generation, spawn one joint sequence per path.
5. Create NumPy Generator objects from an explicitly selected bit generator.
6. Record seed derivation and bit-generator identity.
7. Prohibit global NumPy random state.

### Tests

- The covariance root reconstructs positive-definite and singular PSD matrices.
- Materially indefinite matrices fail.
- Small negative eigenvalues inside tolerance are zeroed and reported.
- The original matrix is unchanged.
- Identical seed and inputs reproduce identical draws.
- Different path streams do not produce identical sequences.
- Factor and specific streams are distinct under independent generation.
- Stream assignment is stable under repeated construction.

### Acceptance criteria

- Numerical normalization is observable in diagnostics.
- No silent nearest-PSD projection exists.
- RNG construction contains no global state.
- The numerical code is explained with short equation-oriented docstrings.

## 13. Step 6 — Implement independent Gaussian innovation sampling

**Status:** Complete — approved at Review Gate B on 2026-10-24.

### Objective

Implement the default stochastic policy.

### Files

- **src/factor_sim/sampling.py**
- **tests/unit/test_sampling.py**
- **tests/statistical/test_independent_gaussian_sampling.py**

### Actions

1. Implement sample_independent_gaussian.
2. Use the factor covariance root for factor innovations.
3. Use square roots of specific variances for specific innovations.
4. Use separate factor and specific RNG streams.
5. Allocate the final arrays deliberately with shapes
   \(M\times H\times K\) and \(M\times H\times N\).
6. Draw future steps independently under the no-serial-correlation policy.
7. Transfer the engine-owned arrays into `InnovationBatch` without copying them
   again, mark them read-only, and include sampler diagnostics.
8. Add a small dispatch function that selects the sampler from an explicit registry.

### Tests

Deterministic tests:

- exact shapes and dtype;
- zero-variance components emit zero innovation;
- singular factor covariance is supported;
- sampler does not mutate covariance inputs;
- unsupported distributions fail before drawing.

Statistical tests with fixed seeds:

- factor innovation means approximate zero;
- factor covariance approximates \(\mathbf F_0\);
- specific covariance approximates \(\mathbf D_0\);
- factor-specific cross-covariance approximates zero;
- sample lag-one correlations are consistent with the no-serial policy.

### Acceptance criteria

- Default policy behavior is fully implemented.
- Statistical tolerances are justified by sample size.
- Tests do not expect empirical moments to equal targets exactly.

## 14. Step 7 — Implement joint Gaussian innovation sampling

**Status:** Complete — approved at Review Gate B on 2026-10-24.

### Objective

Support the v4 joint-generation policy while preserving the baseline covariance identity.

### Files

- **src/factor_sim/sampling.py**
- **tests/unit/test_joint_sampling.py**
- **tests/statistical/test_joint_gaussian_sampling.py**

### Actions

1. Construct the \((K+N)\times(K+N)\) block-diagonal covariance:

$$
\begin{bmatrix}
\mathbf F_0 & \mathbf 0_{K\times N}\\
\mathbf 0_{N\times K} & \mathbf D_0
\end{bmatrix}.
$$

2. Construct one PSD square root for the joint covariance.
3. Draw one joint vector per path and future date.
4. Split it into factor and specific innovation arrays.
5. Reject any requested nonzero factor-specific cross-covariance.
6. Record joint-sampler identity and block dimensions.
7. Allocate final-shape outputs deliberately and transfer them into
   `InnovationBatch` without a redundant copy.

### Tests

- Joint samples have the required shapes.
- Marginal covariance targets match \(\mathbf F_0\) and \(\mathbf D_0\).
- Factor-specific cross-covariance approximates zero.
- Joint and independent Gaussian policies are distributionally consistent within statistical tolerance.
- The policies use different recorded stream derivations.
- Nonzero cross-covariance configuration is rejected.

### Acceptance criteria

- The baseline covariance identity remains unchanged.
- Joint generation is an explicit policy, not an accidental implementation detail.
- No non-Gaussian dependence claim is made.

### Manual review checkpoint B1

Review sampling.py in full. The reviewer should be able to identify:

- where covariance roots are formed;
- where path streams are derived;
- where independent and joint policies diverge; and
- where the zero cross-covariance requirement is enforced.

## 15. Review Gate B — Numerical and stochastic core

**Status:** Passed — user approved Packet B on 2026-10-24. The approved
acceptance-test SHA-256 is
`462E143E18AFFF5CFB72AD21918E7C880696480861281DAAB6510FC0AA1759A0`.
Invasive performance optimization remains deferred until later end-to-end
profiling.

Before Step 8:

- all Step 4–7 unit tests pass;
- statistical sampling tests pass with documented tolerances;
- deterministic seed reproduction is demonstrated;
- singular PSD cases are tested;
- no hidden repair or scaling exists; and
- sampling.py remains understandable without navigating a class hierarchy.

The review packet should contain:

- sampling control flow;
- numerical-normalization examples;
- RNG derivation example;
- statistical test summaries; and
- full changed-file list.

## 16. Step 8 — Implement return construction

**Status:** Complete — approved at Review Gate C on 2026-10-25.

### Objective

Convert innovations and means into factor, specific, and modeled asset returns.

### Files

- **src/factor_sim/engine.py**
- **tests/unit/test_return_construction.py**

### Actions

Implement small testable functions:

1. construct_factor_returns;
2. construct_specific_returns; and
3. construct_modeled_returns.

Use:

$$
\mathbf f_h^{(m)}
=
\boldsymbol\mu_h^f+\boldsymbol\eta_h^{f,(m)},
$$

$$
\boldsymbol\epsilon_h^{(m)}
=
\boldsymbol\mu_h^\epsilon+\boldsymbol\eta_h^{\epsilon,(m)},
$$

and

$$
\mathbf r_h^{(m)}
=
\mathbf X_0\mathbf f_h^{(m)}
+
\boldsymbol\epsilon_h^{(m)}.
$$

Use one clearly documented NumPy expression for the asset-factor multiplication. If einsum is used, document the meaning of each axis beside the expression.

### Tests

- Hand-calculated one-path and multi-path examples.
- Constant and time-varying means.
- Exact equation identity within the configured tolerance.
- Correct broadcasting.
- Inputs remain unchanged.
- Shape mismatches fail clearly.

### Acceptance criteria

- The formulas are immediately recognizable in the implementation.
- No pricing or file I/O appears in return construction.
- The exact return equation is tested independently of random sampling.

## 17. Step 9 — Implement baseline pricing and asset-state contracts

**Status:** Complete — approved at Review Gate C on 2026-10-25.

### Objective

Map modeled returns to price-link returns and construct price paths under version-one policies.

### Files

- **src/factor_sim/pricing.py**
- **tests/unit/test_pricing.py**
- **tests/unit/test_asset_state.py**

### Actions

#### 17.1 Raw return mapping

- Accept identity mapping only when modeled-return metadata states raw arithmetic price returns.
- Preserve modeled returns separately from price-link returns.
- Reject incompatible return bases.

#### 17.2 Arithmetic price evolution

Implement:

$$
P_{i,h}^{(m)}
=
P_{i,h-1}^{(m)}
\left(1+q_{i,h}^{(m)}\right).
$$

Loop over dates and vectorize within a date over paths and assets.

#### 17.3 Invalid returns

- Require the explicit fail policy.
- Fail if any live asset has a return at or below \(-100\%\).
- Include path, date, and asset ID in the error.
- Do not clip, floor, truncate, or resample.

#### 17.4 None handlers

Implement explicit no-op functions for:

- currency handling;
- corporate-action adjustment; and
- new listings.

The currency no-op records that each price remains in its asset currency.

#### 17.5 Asset state

- Initialize every asset as live.
- Define the AssetStateEvent contract.
- Under None handlers, emit no state-transition events.
- Provide the callable boundary a future corporate-action handler must satisfy without implementing such a handler.

### Tests

- Hand-calculated compounding across several dates.
- Multiple paths and assets.
- Exactly \(-100\%\) and below-\(-100\%\) returns fail.
- Positive and zero arithmetic returns behave correctly.
- None currency handler leaves returns and prices unchanged.
- None corporate-action handler emits no events.
- Every live emitted price is finite and positive.
- Incompatible modeled-return metadata is rejected.

### Acceptance criteria

- Pricing contains no sampling logic.
- Default None handlers are explicit and auditable.
- No unsupported log or corporate-action behavior is partially implemented.
- Error messages identify the affected path, date, and asset.

## 18. Step 10 — Implement the simulation engine

**Status:** Complete — approved at Review Gate C on 2026-10-25.

### Objective

Create the single public orchestration entry point.

### Files

- **src/factor_sim/engine.py**
- **src/factor_sim/__init__.py**
- **tests/integration/test_baseline_simulation.py**

### Actions

Implement:

~~~text
simulate(request: SimulationRequest) -> SimulationResult
~~~

Execution order:

1. resolve policies;
2. validate the request;
3. create RNG streams;
4. sample innovations;
5. construct factor and specific returns;
6. construct modeled asset returns;
7. map price-link returns;
8. evolve prices and asset state;
9. compute initial deterministic diagnostics;
10. construct the manifest;
11. transfer engine-owned final-shape arrays into `SimulationResult` without a
    redundant copy, mark them read-only, and return the result.

Additional requirements:

- Failed deterministic validation consumes no random draws.
- The core function performs no file writes.
- The frozen risk state is unchanged after completion.
- The result contains every required array and metadata field.
- The public package exports simulate and the required public dataclasses only.

### Tests

- Canonical three-asset, two-factor integration fixture.
- Exact output shapes and axis order.
- Policy defaults and provenance.
- Repeated fixed-seed runs are identical.
- Different seeds change stochastic outputs.
- Risk-state arrays are unchanged.
- All dates, IDs, currencies, and optional factor metadata survive into the result.
- Default None handlers are recorded.

### Acceptance criteria

- A reader can trace simulate from top to bottom without framework indirection.
- Engine orchestration does not duplicate validation, sampling, or pricing algorithms.
- The result is complete without serialization.

### Manual review checkpoint C1

Review engine.py and pricing.py together. Trace one canonical fixture from request through final prices.

## 19. Review Gate C — Return construction, pricing, and orchestration

**Status:** Passed — user approved Packet C on 2026-10-25. The approved
acceptance-test SHA-256 is
`1EFD00BE8411CFCAAEABB0A5C4707183A9DB292DDD87CFD133F2C4482D213AFA`.
Invasive performance optimization remains deferred until later end-to-end
profiling.

Before Step 11:

- all Step 8–10 focused tests pass;
- all previously approved acceptance tests pass;
- return construction satisfies the frozen-exposure equation;
- incompatible return metadata fails before RNG creation;
- invalid arithmetic returns fail without clipping, flooring, or resampling;
- the public simulation path performs no file writes; and
- engine.py and pricing.py remain directly traceable without framework
  indirection.

The review packet contains:

- the return-construction and pricing control flow;
- the canonical end-to-end fixture;
- manifest and policy provenance examples;
- non-gating canonical and maximum-shape benchmark results; and
- the full changed-file and test summary.

## 20. Step 11 — Implement diagnostics

**Status:** Complete — approved at Review Gate D on 2026-10-25.

### Objective

Create structured deterministic and statistical evidence that outputs satisfy the requested model.

### Files

- **src/factor_sim/diagnostics.py**
- **tests/unit/test_diagnostics.py**
- **tests/statistical/test_covariance_reproduction.py**
- **tests/statistical/test_serial_policy.py**

### Actions

Implement diagnostics for:

- input dimensions and alignment;
- covariance symmetry and minimum eigenvalues;
- factor and specific empirical means;
- factor and specific empirical covariance;
- factor-specific cross-covariance;
- modeled asset covariance;
- marginal volatility;
- cross-asset correlation;
- serial correlation;
- joint-policy metadata;
- equation residuals;
- price validity;
- asset-state consistency; and
- numerical canonicalization.

Every statistical result records:

- statistic name;
- target;
- estimate;
- sample size;
- tolerance or confidence rule;
- pass/fail result; and
- explanatory message.

For the Gaussian default, independence is supported by sampler construction plus zero cross-covariance diagnostics. Do not claim a general empirical proof of independence from covariance alone.

### Tests

- Diagnostics pass on a sufficiently large valid simulation.
- Deliberately corrupted outputs fail the relevant check.
- Singular target covariance does not break correlation reporting; undefined correlations are identified rather than fabricated.
- Tolerances change predictably with sample size.
- Diagnostic calculation does not mutate the result.

### Acceptance criteria

- Diagnostics are structured data, not only log text.
- A failure identifies the violated target.
- Statistical checks distinguish sampling variation from deterministic invariant failure.

## 21. Step 12 — Implement serialization

**Status:** Complete — approved at Review Gate D on 2026-10-25.

### Objective

Persist and reload accepted results without coupling storage to the engine.

### Files

- **src/factor_sim/io.py**
- **tests/unit/test_io.py**
- **tests/integration/test_serialization_round_trip.py**

### Actions

Write:

- numerical arrays to **arrays.npz**;
- manifest and resolved policy to **manifest.json**;
- diagnostics to **diagnostics.json**; and
- asset-state events to **asset_state_events.jsonl** when events exist.

Requirements:

1. include schema versions;
2. preserve array axis order;
3. preserve ordered IDs and dates;
4. serialize enums as stable strings;
5. serialize dates and timestamps in ISO 8601 form;
6. preserve explicit/default policy provenance;
7. record input and output hashes where practical;
8. avoid partial success;
9. support read-back validation; and
10. keep serialization outside simulate.

### Tests

- Full round trip preserves arrays and metadata.
- Empty asset-state events are represented according to the confirmed decision.
- Malformed or incomplete saved results fail clearly.
- Unknown schema versions are rejected.
- Serialization does not modify the in-memory result.

### Acceptance criteria

- A saved run contains enough information to audit and reproduce it.
- The core simulation package works without writing files.
- No pandas or database dependency is introduced.

## 22. Step 13 — Build the complete acceptance suite

**Status:** Complete — approved at Review Gate D on 2026-10-25.

### Objective

Demonstrate end-to-end conformance with v4 and the architecture.

### Files

- **tests/integration/**
- **tests/statistical/**
- **tests/fixtures/**
- an acceptance-report generator under the test or diagnostics boundary

### Actions

#### 13.1 Canonical deterministic fixture

Create a fixture with:

- three assets;
- two factors;
- four future trading dates;
- two paths for deterministic shape tests;
- explicit factor and specific means;
- one zero specific variance;
- a singular but valid factor covariance variant;
- asset currencies; and
- optional factor classes.

Keep the fixture values small enough for manual calculation.

#### 13.2 Statistical fixture

Create a larger path-count fixture for moment tests. Record the fixed seed and justified tolerances.

#### 13.3 Negative fixture collection

Include invalid:

- dimensions;
- identifier orders;
- dates;
- units;
- annualization state;
- nonpositive prices;
- negative variances;
- non-PSD covariance;
- unsupported policies;
- missing currencies;
- invalid return basis;
- nonzero joint cross-covariance; and
- arithmetic returns at or below \(-100\%\).

#### 13.4 Acceptance report

Generate a structured report mapping each v4 Section 10 criterion to:

- test ID;
- test result;
- evidence summary; and
- relevant module.

### Acceptance criteria

- Every v4 acceptance criterion has deterministic or statistical evidence.
- Every architecture component has at least one integration path.
- Statistical tests are repeatable and non-flaky under the supported environment.
- Negative tests prove prohibited assumptions are not silently introduced.

## 23. Step 14 — Readability, performance, and documentation review

**Status:** Complete — approved at Review Gate D on 2026-10-25.

### Objective

Confirm that the accepted behavior is implemented in code the user can reasonably inspect.

### Actions

#### 14.1 Manual readability audit

Review each production module for:

- descriptive names;
- complete public type hints;
- array-shape docstrings;
- short functions with one responsibility;
- direct control flow;
- no unnecessary inheritance;
- no unused extension abstractions;
- no import-time registration;
- no mutable global state;
- equation-oriented comments; and
- consistent terminology.

#### 14.2 Dependency audit

- Confirm NumPy is the only required production dependency.
- Confirm dependency direction remains acyclic.
- Confirm file I/O appears only in io.py.
- Confirm pricing and sampling remain independent.

#### 14.3 Performance check

Measure:

- preflight validation time;
- sampling time;
- return-construction time;
- pricing time;
- diagnostics time; and
- peak estimated and observed memory.

Use representative \(N\), \(K\), \(M\), and \(H\) values confirmed in Step 0.

Do not optimize unless a measured result violates an agreed target.

#### 14.4 Documentation

README should contain:

- environment setup;
- a small direct-Python example;
- array conventions;
- version-one supported policies;
- unsupported policies;
- reproducibility limitations;
- output-file description; and
- how to run unit, integration, statistical, and acceptance tests.

### Acceptance criteria

- No production function requires framework knowledge to follow.
- The canonical example can be read alongside v4 equations.
- Measured scale fits the agreed memory and runtime constraints.
- Deferred features are visibly rejected, not silently approximated.

### Manual review checkpoint D1

Review the complete production code in dependency order:

1. errors.py;
2. config.py;
3. models.py;
4. validation.py;
5. sampling.py;
6. pricing.py;
7. diagnostics.py;
8. engine.py;
9. io.py.

## 24. Step 15 — Final acceptance gate

**Status:** Passed — the user approved Packet D and final version one on
2026-10-25.

### Objective

Decide whether version one is mathematically consistent, architecturally compliant, reproducible, and ready for use.

### Required evidence

- all unit tests pass;
- all integration tests pass;
- all statistical tests pass with recorded tolerances;
- the v4 acceptance report passes;
- fixed-seed reproducibility is demonstrated;
- serialization round trip passes;
- dependency audit passes;
- manual code review is complete;
- performance results are recorded;
- no unsupported policy is silently accepted;
- no unexplained numerical repair occurs; and
- documentation matches actual behavior.

### Final acceptance criteria

Version one is accepted only if:

1. every emitted modeled return satisfies

$$
\mathbf r_h
=
\mathbf X_0\mathbf f_h
+
\boldsymbol\epsilon_h
$$

within the declared equation tolerance;

2. Monte Carlo results reproduce

$$
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\mathbf D_0
$$

within justified statistical tolerances;

3. policy defaults and explicit overrides are distinguishable;
4. the risk state remains frozen;
5. every live price is finite and strictly positive;
6. independent and joint Gaussian policies satisfy their contracts;
7. outputs contain the required provenance and audit information;
8. the same supported environment and fixed seed reproduce the declared result;
9. the user has completed the final manual review; and
10. no unresolved blocking decision remains.

Failure of any condition keeps the implementation unaccepted. A waived condition must be documented as a specification or scope revision, not silently ignored.

## 25. Manual-review strategy

Because the user intends to read the code, implementation should be delivered in four review packets:

| Packet | Steps | Main review question |
|---|---|---|
| A | 1–3 | Are the objects, names, and defaults clear and faithful to v4? |
| B | 4–7 | Are validation, covariance roots, RNG streams, and samplers mathematically transparent? |
| C | 8–10 | Can one trace returns into prices through the public API? |
| D | 11–15 | Do diagnostics, persistence, tests, and documentation prove the implementation is complete? |

Each packet should include:

- a concise change summary;
- exact files added or changed;
- the relevant v4 and architecture sections;
- focused test results;
- known limitations; and
- a suggested file-reading order.

Do not combine all steps into one large implementation change.

## 26. Explicitly deferred work

The following work is outside version one unless Step 0 changes the boundary:

- non-frozen exposure dynamics;
- non-frozen factor-volatility dynamics;
- non-frozen factor-correlation dynamics;
- non-frozen specific-risk dynamics;
- non-Gaussian distributions;
- serially correlated returns;
- nonzero factor-specific cross-covariance;
- nondefault currency conversion;
- corporate actions and delistings;
- new listings;
- log-return price evolution;
- clipping, flooring, truncation, or resampling safeguards;
- path batching;
- parallel execution;
- database storage;
- command-line, web, or graphical interfaces; and
- risk-model estimation from historical data.

Deferred work must not leave partially active code paths in version one.

## 27. Plan completion criteria

This plan is ready for execution when:

1. Step 0 decisions are approved;
2. task order and review gates are accepted;
3. every task has a defined output and test obligation;
4. version-one and deferred behavior are clearly separated;
5. the planned modules match the architecture;
6. v4 requirements are traceable to tests and implementation steps; and
7. the user agrees that implementation will proceed in small review packets rather than one large change.
