# Factor-Risk Equity Simulation — Architectural Design

## 1. Document status and authority

This document defines the software architecture for a Python implementation of the simulation contract in [barra_simulation_specification_manual_v4.md](./barra_simulation_specification_manual_v4.md).

The documents have different responsibilities:

- The v4 simulation specification is authoritative for mathematical behavior, inputs, outputs, policies, invariants, and acceptance requirements.
- This architectural design is authoritative for code organization, component boundaries, data flow, dependency direction, numerical implementation strategy, and extension points.
- A later implementation plan will translate this architecture into ordered coding and testing tasks.

If this document conflicts with the v4 simulation specification on mathematical behavior, v4 takes precedence. The architecture must then be corrected rather than silently changing the simulation contract.

## 2. Architectural goals

The implementation must prioritize:

1. **Mathematical correctness.** Array shapes, covariance targets, return conventions, and timing rules must remain visible in the code.
2. **Manual readability.** A reader familiar with Python and the v4 document should be able to trace a simulation without learning a framework.
3. **Explicit configuration.** Defaults, overrides, units, return bases, and unsupported policies must be resolved and validated before sampling.
4. **Reproducibility.** Random-stream construction and numerical behavior must be recorded sufficiently to reproduce a run.
5. **Auditability.** Every result must retain the resolved policies, aligned identifiers, risk inputs, provenance, and validation results.
6. **Controlled extensibility.** Distribution, currency, corporate-action, and joint-generation extensions must enter through narrow interfaces.
7. **Proportionate performance.** The baseline should use vectorized NumPy operations without introducing distributed systems or complicated optimization.

## 3. Non-goals

The first implementation is not intended to:

- estimate a Barra or custom factor risk model from raw market data;
- update exposures, factor volatility, factor correlation, or specific risk during a run;
- implement a general-purpose workflow engine or plugin framework;
- provide a graphical user interface, web service, or database;
- support distributed computation;
- hide invalid inputs through automatic repair;
- implement every possible non-Gaussian distribution, currency rule, corporate action, or new-listing rule; or
- infer expected returns or a complete price process from a covariance model.

The architecture provides narrow seams for authorized extensions, but it does not pre-build unused abstractions.

## 4. Core design decisions

| Concern | Decision | Reason |
|---|---|---|
| Main language | Python 3.11 or later | Modern type hints and dataclasses without requiring the newest interpreter |
| Numerical arrays | NumPy arrays with float64 values | Direct correspondence to the specification and broadly readable numerical code |
| Domain models | Standard-library dataclasses and enums | Explicit, inspectable objects with little runtime machinery |
| Configuration validation | Ordinary Python validation functions | Avoid hidden coercion and framework-generated behavior |
| Core tabular dependency | None | Pandas is not required inside the simulation engine |
| Test framework | pytest plus NumPy testing helpers | Clear unit, integration, and statistical tests |
| Public API | One primary simulate function | A single traceable entry point |
| Execution model | In-process library | Simplest baseline for manual review and testing |
| State model | Immutable input objects and newly created outputs | Prevent accidental mutation of the frozen risk state |
| Extension registration | Explicit dictionaries of named callables or small protocol implementations | No import-time discovery or dependency-injection framework |
| Default persistence | NumPy NPZ arrays plus JSON metadata and JSON Lines events | Few dependencies and clear separation of arrays from audit records |
| Logging | Boundary-layer logging only | Domain functions return structured diagnostics instead of producing hidden side effects |

### 4.1 Dependency policy

The production core should initially depend on:

- Python standard library; and
- NumPy.

Optional adapters may add:

- PyYAML for YAML configuration files; and
- a future serialization library only if an approved output requirement cannot be met by NPZ and JSON.

SciPy, pandas, Pydantic, a database, and a web framework should not be required by the baseline. A dependency may be added later only when its benefit is concrete and documented.

## 5. Architectural overview

The system is a small library with an imperative orchestration shell around a mostly functional numerical core.

~~~text
SimulationRequest
      |
      v
Resolve policy defaults
      |
      v
Align and validate inputs
      |
      +-----------------------------+
      |                             |
      v                             v
Build RNG streams             Build numerical covariance roots
      |                             |
      +--------------+--------------+
                     v
             Generate innovations
                     |
                     v
       Construct factor and specific returns
                     |
                     v
            Construct modeled returns
                     |
                     v
       Return mapping / currency / events
                     |
                     v
               Evolve prices
                     |
                     v
       Diagnostics and SimulationResult
                     |
                     v
          Optional file serialization
~~~

The orchestration layer controls order. Numerical and policy functions do not call back into the engine.

## 6. Proposed package structure

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
    │   ├── test_config.py
    │   ├── test_validation.py
    │   ├── test_sampling.py
    │   ├── test_pricing.py
    │   └── test_diagnostics.py
    ├── integration/
    │   ├── test_baseline_simulation.py
    │   ├── test_joint_generation.py
    │   └── test_serialization_round_trip.py
    ├── statistical/
    │   ├── test_covariance_reproduction.py
    │   ├── test_means_and_volatility.py
    │   └── test_independence_and_serial_policy.py
    └── fixtures/
~~~

This structure is intentionally shallow. A module should be split only when its responsibilities become difficult to understand in one file.

## 7. Dependency direction

The permitted imports are:

~~~text
config.py ───────────────┐
models.py ───────────────┼──> validation.py
errors.py ───────────────┘

config.py + models.py ──────> sampling.py
config.py + models.py ──────> pricing.py
models.py ──────────────────> diagnostics.py

validation.py
sampling.py
pricing.py
diagnostics.py ─────────────> engine.py

models.py + engine.py ──────> io.py
~~~

Rules:

- Core modules must not import the command line, filesystem adapters, or serialization layer.
- Sampling must not import pricing.
- Pricing must not import sampling.
- Diagnostics may inspect results but must not mutate them.
- The engine may coordinate lower-level modules but must not duplicate their calculations.
- Circular imports are prohibited.

## 8. Domain data model

All public domain objects should be typed dataclasses. Arrays should be documented with their exact axis order.

### 8.1 Enumerations

The configuration module should define string-valued enums for:

- exposure dynamics;
- factor-volatility dynamics;
- factor-correlation dynamics;
- specific-risk dynamics;
- factor-return distribution;
- specific-return distribution;
- serial-correlation policy;
- joint-generation policy;
- price-return convention;
- invalid arithmetic-return policy; and
- asset status.

Only supported values are accepted. Unknown strings are configuration errors, not values to preserve for later interpretation.

The baseline enum values include:

- frozen dynamics;
- Gaussian marginal distributions;
- no serial correlation;
- independent or joint factor-specific generation;
- raw arithmetic price returns; and
- live, terminal, and delisted asset statuses.

Python None represents the default absence of currency, corporate-action, and new-listing handlers.

### 8.2 RiskState

RiskState is immutable and contains:

| Field | Type and shape | Meaning |
|---|---|---|
| exposures | float64, \(N\times K\) | Frozen exposure matrix \(\mathbf X_0\) |
| factor_covariance | float64, \(K\times K\) | Frozen one-day factor covariance \(\mathbf F_0\) |
| specific_variances | float64, \(N\) | Diagonal entries of \(\mathbf D_0\) |
| asset_ids | tuple of length \(N\) | Canonical asset order |
| factor_ids | tuple of length \(K\) | Canonical factor order |
| as_of_date | date or timestamp | Risk-state observation date |
| known_at | timestamp | Time at which the risk state was available |
| return_horizon | explicit value | Must equal one trading day at the engine boundary |
| return_convention | enum or validated string | Arithmetic or log basis used by the risk model |
| return_basis | enum or validated string | Raw price, adjusted price, total, or excess return |
| annualization_state | enum or validated string | Must indicate de-annualized one-day covariance |
| provenance | structured mapping | Producing model, version, run, or source |

The core stores specific variances as a vector rather than a dense diagonal matrix. A helper may expose \(\mathbf D_0\) when required. This makes diagonality explicit and avoids allocating \(N^2\) zero values.

RiskState arrays must be copied or marked read-only when the request is accepted. The engine must never mutate them.

### 8.3 AssetMetadata

AssetMetadata contains one record per asset:

- asset ID;
- asset currency;
- optional display name;
- optional market or exchange identifier; and
- model-independent provenance fields when needed.

AssetMetadata uses the same order as RiskState.asset_ids. Currency factors in the risk model do not replace asset-currency metadata.

### 8.4 FactorMetadata

FactorMetadata is optional and contains one record per factor:

- factor ID;
- optional factor name;
- optional description;
- optional factor class;
- optional exposure class; and
- optional model-specific attributes.

If factor class and exposure class are both present, validation checks their declared mapping. Factor metadata is descriptive and must not alter the numerical exposure matrix.

### 8.5 SimulationPolicy

SimulationPolicy contains the caller-supplied values before default resolution. Each field is optional if v4 defines a default.

A separate ResolvedSimulationPolicy contains:

- one concrete value for every policy;
- handler or sampler names where applicable;
- parameters for nondefault behavior; and
- a provenance mapping from policy name to explicit or default.

Using separate supplied and resolved objects prevents the code from confusing an omitted value with an explicitly supplied default.

### 8.6 SimulationRequest

SimulationRequest contains:

- run ID;
- simulation origin;
- ordered future trading dates of length \(H\);
- number of paths \(M\);
- initial prices \(\mathbf P_0\in\mathbb R^N_{>0}\);
- factor means with shape \(H\times K\);
- specific means with shape \(H\times N\);
- RiskState;
- AssetMetadata;
- optional FactorMetadata;
- supplied SimulationPolicy;
- random seed;
- numerical tolerances; and
- required provenance.

A constant mean vector may be accepted at the adapter boundary, but it must be expanded or exposed through a clear broadcasting helper before the numerical core executes.

### 8.7 InnovationBatch

InnovationBatch contains:

- factor innovations with shape \(M\times H\times K\);
- specific innovations with shape \(M\times H\times N\); and
- sampler metadata, including distribution, joint policy, RNG streams, and numerical factorization diagnostics.

### 8.8 AssetStateEvent

AssetStateEvent contains:

- asset ID;
- path ID;
- effective date;
- status; and
- reason.

Events are emitted only when a supported nondefault handler changes asset liveness or records another required state transition. Terminal state must not be represented solely by a zero, missing, NaN, or infinite price.

### 8.9 SimulationResult

SimulationResult contains:

| Field | Shape or type |
|---|---|
| factor_innovations | \(M\times H\times K\) |
| specific_innovations | \(M\times H\times N\) |
| factor_returns | \(M\times H\times K\) |
| specific_returns | \(M\times H\times N\) |
| modeled_returns | \(M\times H\times N\) |
| price_link_returns | \(M\times H\times N\) |
| prices | \(M\times(H+1)\times N\) |
| asset_state_events | immutable sequence |
| target_asset_covariance | \(N\times N\) |
| resolved_policy | ResolvedSimulationPolicy |
| manifest | structured audit metadata |
| validation_report | structured diagnostics |

The result also retains ordered asset IDs, factor IDs, trading dates, asset currencies, optional factor metadata, and the exact frozen risk state.

## 9. Component responsibilities

### 9.1 Configuration — config.py

Responsibilities:

- define enums and policy dataclasses;
- apply documented defaults;
- normalize None representations at adapter boundaries;
- distinguish explicit from default values;
- reject unsupported policy combinations; and
- resolve named sampler and handler selections.

The module must not inspect numerical matrices or draw random values.

Primary function:

~~~text
resolve_policy(supplied_policy) -> ResolvedSimulationPolicy
~~~

### 9.2 Domain models — models.py

Responsibilities:

- define immutable request, state, metadata, result, event, and diagnostic dataclasses;
- document array shapes;
- provide only small shape-preserving helpers; and
- avoid business logic.

Models must not open files, log, validate covariance eigenvalues, or generate values.

### 9.3 Validation — validation.py

Responsibilities:

- validate dimensions and identifier alignment;
- validate dates and point-in-time availability;
- validate return horizons, bases, and units;
- validate finite values and positive initial prices;
- validate symmetry and positive semidefiniteness;
- validate nonnegative specific variances;
- validate optional factor metadata;
- validate currency requirements;
- validate handler-specific configuration;
- validate supported joint-generation contracts; and
- compute the target covariance

$$
\boldsymbol\Sigma_0
=
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\mathbf D_0.
$$

Validation should collect related preflight problems into one ValidationError containing a list of structured issues. Each issue has a stable code, field path, and readable message.

Validation must not repair the request.

### 9.4 Innovation sampling — sampling.py

Responsibilities:

- construct reproducible random streams;
- construct covariance square roots;
- generate factor and specific innovations;
- support independent and joint factor-specific generation;
- apply the resolved serial policy;
- return sampler diagnostics; and
- expose a narrow callable interface for approved custom samplers.

The initial implementation should contain plainly named functions:

~~~text
sample_independent_gaussian(...)
sample_joint_gaussian(...)
sample_innovations(...)
~~~

The dispatch function sample_innovations selects a concrete function from an explicit mapping. It must not discover implementations by scanning modules or loading arbitrary import paths.

For the independent policy, factor and specific draws use reproducibly separated RNG child streams. For the joint policy, one joint child stream produces a vector in \(\mathbb R^{K+N}\).

The baseline joint covariance is

$$
\begin{bmatrix}
\mathbf F_0 & \mathbf 0_{K\times N}\\
\mathbf 0_{N\times K} & \mathbf D_0
\end{bmatrix}.
$$

A sampler that requests nonzero factor-specific cross-covariance is rejected by the baseline.

### 9.5 Price and event processing — pricing.py

Responsibilities:

- map modeled returns to price-link returns;
- apply the raw arithmetic default;
- invoke an approved currency handler;
- invoke an approved corporate-action handler;
- apply the invalid arithmetic-return policy;
- update prices one trading day at a time;
- update asset state; and
- emit AssetStateEvent records.

The initial functions should be direct and separable:

~~~text
map_modeled_returns(...)
apply_currency_policy(...)
apply_corporate_action_policy(...)
evolve_arithmetic_prices(...)
evolve_log_prices(...)
apply_asset_state_events(...)
~~~

No-op behavior for None policies should be explicit functions, not scattered conditionals throughout the engine.

The price loop proceeds over \(H\) dates because each price depends on the previous price. Operations within a date are vectorized over paths and assets.

### 9.6 Simulation orchestration — engine.py

The engine exposes the primary public entry point:

~~~text
simulate(request: SimulationRequest) -> SimulationResult
~~~

Responsibilities:

1. resolve policy defaults;
2. validate the request;
3. build RNG streams;
4. sample innovations;
5. construct factor and specific returns;
6. construct modeled asset returns;
7. map returns and evolve prices;
8. compute diagnostics;
9. build the manifest; and
10. return a SimulationResult.

The engine coordinates components but contains no covariance-factorization algorithm, file-writing logic, or handler-specific business rules.

### 9.7 Diagnostics — diagnostics.py

Responsibilities:

- verify per-run deterministic invariants;
- calculate empirical means, volatilities, covariance, and correlation;
- check factor-specific cross-covariance;
- check the serial policy;
- evaluate joint-generator diagnostics;
- verify price and asset-state consistency;
- report numerical canonicalization events; and
- create a structured ValidationReport.

Statistical diagnostics return estimates, tolerances, pass/fail results, and sample sizes. They do not hide discrepancies behind a single Boolean.

### 9.8 Serialization — io.py

Responsibilities:

- read optional JSON or YAML adapter inputs;
- write numerical arrays to NPZ;
- write the manifest and diagnostics to JSON;
- write asset-state events to JSON Lines;
- preserve deterministic identifier and field ordering; and
- support a round-trip test.

The core simulate function does not write files. Persistence is an explicit call made after a successful result is returned.

## 10. Execution flow

### 10.1 Policy resolution

The first operation resolves omitted fields to the v4 defaults. Resolution produces a new object and a provenance entry for each policy:

~~~text
factor_exposure_dynamics:
    value: frozen
    value_source: default
~~~

The resolved object is the only policy object used by downstream code.

### 10.2 Preflight validation

All deterministic input checks run before RNG initialization. A failed request consumes no random draws.

Validation order:

1. required scalar fields and dates;
2. unique identifiers;
3. array dimensions;
4. identifier and metadata alignment;
5. finite values and price positivity;
6. units, horizons, return basis, and known-at timing;
7. covariance symmetry and positive semidefiniteness;
8. policy compatibility;
9. handler-specific requirements; and
10. estimated memory requirement.

### 10.3 Innovation generation

The sampling layer produces the complete InnovationBatch.

Under the default policy:

$$
\boldsymbol\eta^f_h\sim\mathcal N(\mathbf 0,\mathbf F_0),
\qquad
\boldsymbol\eta^\epsilon_h\sim\mathcal N(\mathbf 0,\mathbf D_0),
$$

with factor and specific blocks independently sampled and steps serially independent.

For a joint policy, the configured joint sampler must preserve the same marginal covariances and zero cross-covariance.

### 10.4 Return construction

Means are added by broadcasting:

$$
\mathbf f_h^{(m)}
=
\boldsymbol\mu^f_h+\boldsymbol\eta_h^{f,(m)},
$$

$$
\boldsymbol\epsilon_h^{(m)}
=
\boldsymbol\mu^\epsilon_h+\boldsymbol\eta_h^{\epsilon,(m)}.
$$

Modeled returns are computed across all paths and dates with one visible NumPy expression equivalent to

~~~text
modeled_returns = einsum("mhk,nk->mhn", factor_returns, exposures)
modeled_returns = modeled_returns + specific_returns
~~~

The einsum axis notation must be explained in a nearby comment because it is less immediately readable than ordinary matrix multiplication.

### 10.5 Price construction

The pricing layer loops over dates:

1. obtain modeled returns for the date;
2. apply the return mapping;
3. apply the currency policy;
4. apply the corporate-action policy;
5. enforce the invalid-return policy;
6. update prices;
7. update asset state; and
8. append events and diagnostics.

Arrays remain vectorized over paths and assets within each date.

### 10.6 Result construction

The engine builds one immutable SimulationResult. The manifest records:

- run and model identity;
- input hashes or equivalent provenance;
- resolved policies and value sources;
- array shapes and axis order;
- dates, units, currencies, and return basis;
- RNG seed, bit generator, and stream derivation;
- Python and NumPy versions;
- numerical tolerances;
- covariance-root diagnostics;
- handlers and versions;
- safeguard events; and
- validation summary.

## 11. Numerical design

### 11.1 Precision

All numerical calculations use float64 unless a future specification explicitly authorizes another precision.

### 11.2 Symmetry

For each covariance matrix \(\mathbf A\), validation checks

$$
\max_{i,j}|A_{ij}-A_{ji}|
\le
\tau_{\mathrm{sym}}.
$$

If the check fails, the request is rejected. If it passes, the numerical routine may use

$$
\frac{\mathbf A+\mathbf A^{\mathsf T}}{2}
$$

as a canonical symmetric representation. This canonicalization must be recorded; it is not an undisclosed repair.

### 11.3 Positive semidefiniteness and covariance roots

The implementation should use a symmetric eigenvalue decomposition rather than requiring Cholesky factorization, because valid covariance matrices may be singular.

For

$$
\mathbf A=\mathbf Q\boldsymbol\Lambda\mathbf Q^{\mathsf T},
$$

the request is rejected if any eigenvalue is below \(-\tau_{\mathrm{PSD}}\). Eigenvalues in the interval \([-\tau_{\mathrm{PSD}},0)\) may be set to zero when constructing

$$
\mathbf A^{1/2}
=
\mathbf Q\boldsymbol\Lambda_+^{1/2}\mathbf Q^{\mathsf T}.
$$

The number and magnitude of zeroed eigenvalues must be recorded. The original matrix remains in the result manifest.

Specific innovations use the square roots of the supplied specific variances directly.

### 11.4 No silent scaling

The engine accepts one-trading-day covariance inputs. It performs no annualization, de-annualization, square-root-of-time scaling, or half-life conversion.

## 12. Random-number architecture

Use NumPy Generator and SeedSequence.

The derivation is explicit:

1. create one master SeedSequence from the run seed;
2. spawn one child sequence per simulation path;
3. under the independent policy, spawn factor and specific child sequences from each path sequence;
4. under the joint policy, use one joint child sequence per path; and
5. record the derivation policy and bit-generator class.

This design gives each path a stable stream and prevents accidental reuse between factor and specific draws.

Reproducibility is guaranteed only for the recorded Python, NumPy, bit-generator, algorithm version, inputs, and stream policy. Cross-version bitwise identity must not be promised without verification.

Sampling code receives RNG objects as arguments. Global NumPy random state is prohibited.

## 13. Policy extension design

### 13.1 General rule

An extension point is introduced only where v4 defines a policy that can legitimately vary.

Each extension has:

- a stable name;
- an explicit configuration dataclass;
- a small callable interface;
- deterministic validation;
- version metadata;
- tests against the interface contract; and
- an entry in an explicit registry assembled at application startup.

No extension may import or mutate engine internals.

### 13.2 Innovation samplers

An innovation sampler receives:

- resolved distribution and joint policies;
- \(\mathbf F_0\);
- specific variances;
- \(M\) and \(H\);
- RNG streams; and
- sampler-specific parameters.

It returns InnovationBatch.

The initial registry contains independent Gaussian and joint Gaussian samplers. Non-Gaussian samplers are added only with separately specified marginal and dependence tests.

### 13.3 Currency handlers

The None handler returns inputs unchanged and records that prices remain in asset currencies.

A nondefault handler receives asset currencies, target currency, FX inputs, dates, and price-link returns. It returns converted price-link returns plus audit metadata. A handler must specify quote convention and timestamp alignment.

### 13.4 Corporate-action handlers

The None handler emits no events and leaves prices and asset states unchanged.

A nondefault handler receives the date, asset metadata, current state, current prices, and mapped returns. It returns adjusted values and AssetStateEvent records.

### 13.5 Dynamics policies

The initial implementation supports only frozen exposure, factor-volatility, factor-correlation, and specific-risk dynamics.

Do not create four abstract base-class hierarchies for unimplemented dynamics. Unsupported non-frozen values fail configuration validation. Interfaces should be introduced only when a non-frozen method is specified and approved.

### 13.6 New listings

The initial implementation supports None only. The fixed arrays and identifier order make new listings a materially larger extension. It must receive a separate design before implementation.

## 14. Error model

Define a small exception hierarchy:

~~~text
SimulationError
├── ConfigurationError
├── ValidationError
├── NumericalError
└── PolicyExecutionError
~~~

ValidationError contains multiple structured issues when possible. NumericalError includes the matrix name, relevant tolerance, minimum eigenvalue or failed operation, and run context. PolicyExecutionError identifies the handler, version, date, path when applicable, and original exception.

Errors must not include full large arrays. Diagnostics may contain safe summaries, indices, and hashes.

## 15. Validation and testing architecture

### 15.1 Deterministic unit tests

Unit tests cover:

- policy default resolution and provenance;
- invalid enum values and combinations;
- dimensions and identifier alignment;
- optional factor metadata;
- currency metadata;
- symmetry and PSD checks;
- diagonal specific covariance representation;
- covariance-root construction;
- return equations;
- raw and log price links;
- invalid arithmetic returns;
- asset-state events; and
- serialization.

### 15.2 Integration tests

Integration fixtures should be small enough to inspect manually. A recommended canonical fixture has:

- three assets;
- two factors;
- four future dates;
- two paths; and
- simple matrices with analytically checkable results.

Integration tests cover the complete simulate call and verify every output shape, policy value, identifier, date, and manifest field.

### 15.3 Statistical tests

Statistical tests use larger path counts and fixed seeds to verify:

- factor means and covariance;
- specific means and covariance;
- zero factor-specific cross-covariance;
- default block independence using generator construction and suitable diagnostics;
- absence of serial dependence under the default;
- modeled asset covariance;
- marginal asset volatility;
- cross-asset correlation; and
- joint-sampler marginal and dependence contracts.

Tolerances must be justified from sample size and the tested statistic. Tests must not require exact equality of empirical moments.

### 15.4 Acceptance test

One end-to-end acceptance test must map directly to every criterion in v4 Section 10. Its output is a structured report, not merely a pytest success code.

## 16. Output and serialization design

The initial file output is a directory containing:

~~~text
simulation-output/
├── arrays.npz
├── manifest.json
├── diagnostics.json
└── asset_state_events.jsonl
~~~

asset_state_events.jsonl may be omitted when no events exist, provided the manifest records that the applicable handlers were None.

Serialization requirements:

- write schema versions;
- preserve axis order;
- preserve ordered identifiers and dates;
- never serialize enums using language-specific object representations;
- use ISO 8601 dates and timestamps;
- use deterministic key ordering where practical;
- fail rather than partially report success when required files cannot be written; and
- support a read-back validation test.

## 17. Readability rules for the implementation

The codebase should follow these rules:

1. Prefer named functions over clever expressions.
2. Use domain names from v4 consistently.
3. Keep mathematical transformations close to their equations in docstrings or comments.
4. Give every public function complete type hints.
5. Document array shapes in docstrings.
6. Avoid inheritance unless a callable protocol cannot express the interface clearly.
7. Avoid decorators that change control flow.
8. Avoid implicit global registries and import side effects.
9. Avoid mutable module-level state.
10. Do not combine validation, sampling, pricing, and serialization in one function.
11. Comments should explain financial or mathematical intent, not restate Python syntax.
12. A manually constructed small example should be runnable without a configuration framework.
13. Prefer explicit loops when they materially improve clarity and the scale is small; use vectorization where it directly represents the matrix calculation.
14. Keep optimization changes behind tests proving identical required behavior.

Code review should reject abstractions that have only one use and do not clarify a contract.

## 18. Performance and memory strategy

The first implementation operates in memory and uses NumPy vectorization.

Approximate storage for the principal \(M\times H\times N\) float64 arrays is

$$
8MHN\ \text{bytes per array}.
$$

Because several arrays coexist, validation must estimate total peak memory before execution and reject or warn on a configured threshold.

Initial optimization order:

1. vectorize across assets and paths;
2. avoid dense \(\mathbf D_0\);
3. reuse temporary arrays only when readability is preserved;
4. add path batching only after profiling; and
5. consider parallelism only after deterministic single-process behavior is stable.

Path batching must not change RNG stream assignment or results for a fixed seed.

## 19. Traceability to v4

| V4 area | Primary module | Primary evidence |
|---|---|---|
| Purpose and compatibility | models.py, validation.py | request validation tests |
| Notation and dimensions | models.py | type and shape tests |
| Input contract | config.py, models.py, validation.py | schema and preflight tests |
| Policy defaults | config.py | default provenance tests |
| Innovation generation | sampling.py | unit and statistical tests |
| Return construction | engine.py | algebraic invariant tests |
| Price evolution | pricing.py | price-link and positivity tests |
| Outputs | models.py, io.py | shape and round-trip tests |
| Invariants and validation | validation.py, diagnostics.py | deterministic and statistical reports |
| Prohibited assumptions | config.py, validation.py | negative tests |
| Pseudocode flow | engine.py | integration tests |
| Acceptance criteria | all modules | end-to-end acceptance report |

Every implementation-plan task should cite at least one v4 requirement and one architectural component.

## 20. Version-one implementation boundary

The first complete implementation should support:

- frozen exposure, factor-volatility, factor-correlation, and specific-risk policies;
- Gaussian factor and specific innovations;
- independent factor-specific generation;
- joint Gaussian generation with zero cross-covariance;
- no serial correlation;
- explicit factor and specific means;
- raw arithmetic price returns;
- currency handling set to None;
- corporate-action adjustment set to None;
- new listings set to None;
- optional factor metadata;
- deterministic validation;
- statistical covariance validation;
- reproducible RNG streams;
- in-memory SimulationResult; and
- optional NPZ and JSON serialization.

The architecture permits later currency and corporate-action handlers, but those handlers should not be included in version one unless their behavior is specified before the implementation plan is finalized.

## 21. Decisions required before the implementation plan

The following decisions should be made or explicitly deferred:

1. Maximum expected values of \(N\), \(K\), \(M\), and \(H\).
2. Whether JSON, YAML, or Python construction is the primary user-facing input.
3. Which invalid arithmetic-return policies version one will implement.
4. Whether log-return price mapping belongs in version one or a later increment.
5. Whether nondefault currency handling is required in version one.
6. Whether nondefault corporate-action or delisting handling is required in version one.
7. Required operating systems and Python versions.
8. Whether saved results need compression, and the acceptable size and speed tradeoff.

These are product-scope decisions. They should not be left for individual coding tasks to infer.

## 22. Architectural acceptance criteria

The architecture is ready to translate into an implementation plan when:

1. every v4 input, output, invariant, policy, and acceptance criterion maps to an identified module;
2. the public entry point and domain objects are agreed;
3. dependency direction is acyclic and understandable;
4. the version-one policy boundary is explicit;
5. RNG and covariance-root behavior are specified;
6. currency, corporate-action, joint-generation, and asset-state boundaries are explicit;
7. error and validation behavior is explicit;
8. output and audit artifacts are identified;
9. deterministic, statistical, and acceptance testing layers are defined;
10. unresolved product decisions are either answered or scheduled as explicit implementation-plan prerequisites; and
11. a reviewer can trace the full baseline execution without following framework-generated control flow.
