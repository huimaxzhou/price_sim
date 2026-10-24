# Packet B Acceptance Contract

**Status:** Approved and active
**Initial draft:** 2026-10-22
**Revised for approved Packet A:** 2026-10-23
**Approval date:** 2026-10-24
**Scope:** Implementation-plan Steps 4–7
**Test module:** `test_packet_b_contracts.py`

## Review-state boundary

Review Gate A passed on 2026-10-23. Packet B production work was separately
authorized on 2026-10-23, and the user approved Packet B without issue on
2026-10-24. The acceptance module now follows pytest's `test_*.py` discovery
pattern and is part of the normal executable suite. Under `AGENTS.md`, its
behavior, thresholds, and tolerances are protected from material change without
explicit user approval.

## Purpose

Packet B covers:

- deterministic request validation before random-number initialization;
- one-time resolution of typed schedules against the supplied deterministic
  trading calendar;
- construction of the frozen target covariance
  \(\boldsymbol\Sigma_0=\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}+\mathbf D_0\);
- transparent symmetric canonicalization and PSD square-root construction;
- reproducible, separated NumPy random streams;
- independent Gaussian factor and specific innovations; and
- block-diagonal joint Gaussian innovations with zero factor-specific
  cross-covariance.

It does not cover return construction, price evolution, engine orchestration,
diagnostics outside sampler metadata, persistence, or final end-to-end
acceptance.

## Approved callable surface

The draft uses the approved Packet A dataclasses and a small function-oriented
Packet B surface:

- `validation.validate_request(request, resolved_policy)`, reading the memory
  limit, RNG identity, stream policy, schedule, and calendar from the approved
  `SimulationRequest` fields;
- `sampling.covariance_root(matrix, *, matrix_name, symmetry_tolerance,
  positive_semidefinite_tolerance)` returning `(root, diagnostics)`;
- `sampling.build_random_streams(*, seed, path_count, joint_policy,
  bit_generator_name, stream_policy)`;
- `sampling.sample_independent_gaussian(...)`;
- `sampling.sample_joint_gaussian(...)`; and
- `sampling.sample_innovations(...)` for dispatch from the approved
  `ResolvedSimulationPolicy`, without a second caller-supplied sampler string.

Both concrete samplers accept the frozen factor covariance, specific-variance
vector, step count, explicit RNG-stream bundle, and numerical tolerances. They
return the already-approved `InnovationBatch`. This keeps RNG creation separate
from drawing, as required by the architecture, without prescribing the private
stream-container type.

The draft uses the approved grouped `JointGeneratorConfig`; it no longer assumes
the superseded five parallel joint-generator fields. Sampler outputs must be
engine-owned `float64` arrays and are transferred to `InnovationBatch` without
a second full-array copy.

## Approved Packet B decisions

- Version one registers NumPy `PCG64` as its RNG bit generator.
- Version one registers `per_path` as its stream-allocation policy, with the
  exact `SeedSequence -> path -> factor,specific` and
  `SeedSequence -> path -> joint` derivations tested below.
- `sample_innovations` dispatches from the typed resolved policy. It does not
  accept a second free-form sampler name that could disagree with that policy.
- `validate_request` returns one immutable reusable context rather than a loose
  tuple. The exact public class name remains open below.

## Requirement traceability

| Test ID | Governing requirement | Evidence type |
|---|---|---|
| `B-VAL-001` | Valid inputs produce resolved dates, calendar provenance, aligned dimensions, and the exact frozen target covariance | Deterministic fixture |
| `B-CAL-001` | Count and end-date schedules resolve to the supplied calendar sessions | Deterministic schedule fixtures |
| `B-CAL-002` | A non-session end date resolves through the last preceding session without invented weekday rules | Deterministic boundary fixture |
| `B-VAL-002` | Invalid schedule selections, calendars, scalar fields, identifiers, shapes, metadata, finiteness, prices, variances, timing, units, basis, RNG controls, tolerances, and covariance conditions produce field-addressed issues | Parameterized negative tests |
| `B-VAL-003` | Related preflight problems are collected and validation consumes no random state | Negative test with RNG tripwire |
| `B-VAL-004` | Zero specific variance and singular PSD covariance are valid | Deterministic boundary fixture |
| `B-VAL-005` | Missing optional factor metadata is valid; conflicting supplied classifications are not | Positive and negative tests |
| `B-VAL-006` | Currency-factor exposure does not replace asset-currency metadata | Negative test |
| `B-VAL-007` | A grouped joint-generator configuration requesting nonzero factor-specific cross-covariance is outside the baseline contract | Negative test |
| `B-VAL-008` | Estimated peak memory above `SimulationRequest.memory_limit_bytes` fails before RNG initialization | Negative test with RNG tripwire |
| `B-MEM-001` | The recorded estimate grows with path count and covers at least the final innovation arrays | Deterministic lower-bound test |
| `B-NUM-001` | PSD roots reconstruct positive-definite and singular matrices | Deterministic numerical test |
| `B-NUM-002` | Small negative eigenvalues inside tolerance are zeroed and disclosed | Deterministic boundary test |
| `B-NUM-003` | Material indefiniteness and material asymmetry fail without mutating input | Negative test |
| `B-NUM-004` | Within-tolerance asymmetry is canonically symmetrized and disclosed | Deterministic boundary test |
| `B-RNG-001` | Fixed inputs and seed reproduce identical innovations; path and block streams are distinct | Deterministic repeatability test |
| `B-RNG-002` | Sampling uses the recorded NumPy Generator/SeedSequence policy, not legacy global state | RNG tripwire and metadata inspection |
| `B-SAMPLE-001` | Independent Gaussian output shapes, ownership, dtype, immutability, zero-variance behavior, and input preservation are correct | Deterministic test |
| `B-SAMPLE-002` | Independent Gaussian moments, zero cross-covariance, and no serial dependence meet sample-size-aware tolerances | Fixed-seed statistical test |
| `B-JOINT-001` | Joint Gaussian output and metadata use one block-diagonal joint stream per path | Deterministic test |
| `B-JOINT-002` | Joint Gaussian marginals and zero cross-covariance meet sample-size-aware tolerances | Fixed-seed statistical test |
| `B-DISPATCH-001` | Dispatch maps the typed resolved independent/joint policies to the corresponding explicitly registered Packet B samplers | Deterministic dispatch test |

## Statistical design

The draft uses 40,000 one-step paths for small two-factor/three-asset fixtures.
Mean and covariance assertions use five estimated standard errors plus a small
floating-point allowance. Lag-one correlation and a cross-block squared-return
correlation diagnostic use 40,000 observations and a `5/sqrt(n) + 0.01` bound.
The squared-return diagnostic supplements—not replaces—the separate-stream
construction evidence for independence. These are fixed-seed regression checks,
not claims that empirical moments equal their population targets.

The statistical sample size and formulas remain protected contract details once
approved. Any proposed change after approval requires explicit user approval.

## Performance evidence for Review Gate B

Packet B implementation includes a non-collecting benchmark harness at
`tests/performance/benchmark_packet_b.py`. It reports separate timings for:

- schedule and deterministic request validation;
- factor covariance-root construction;
- RNG-stream construction;
- independent Gaussian sampling; and
- block-diagonal joint Gaussian sampling.

The harness will run both the small canonical fixture and the approved design
maximum of (N=250), (K=100), (M=100), and (H=252), while reporting the
validated peak-memory estimate and output bytes. It is review evidence, not an
acceptance test, and normal pytest discovery must not run it.

No hardware-dependent runtime threshold is proposed before a baseline is
measured on the intended machine. Packet B code must nevertheless preserve the
performance-shaping contracts already approved at Review Gate A: validation and
schedule resolution occur once, covariance roots are reused, outputs are
preallocated, specific variance remains a vector outside the required joint
block, large engine-owned arrays are not recopied, and no per-asset or per-date
Python loop is introduced in the sampling hot path. Path batching, parallelism,
and compiled extensions remain deferred.

## Approved implementation latitude

- The acceptance contract asserts the behavior and fields of the frozen
  validated context but does not require the name
  `ValidatedContext` or require that it live in `models.py`.
- Tested issue codes and field paths are protected observable behavior.
- Tested sampler-metadata keys are protected observable behavior; untested
  internal layout remains implementation detail.
- `PCG64` is the registered version-one bit generator for this packet.
- The exact peak-memory estimator formula remains private. The contract checks rejection and
  reporting behavior, not a private allocation-count formula.

## Packet A reconciliation performed

This revision removes every pre-amendment assumption found in the initial
draft:

- `future_dates` is replaced by `SimulationSchedule` plus `TradingCalendar`;
- risk-model identity, RNG identity, and stream policy use their approved
  `SimulationRequest` fields rather than provenance conventions;
- the memory limit is read from `SimulationRequest.memory_limit_bytes`;
- joint-generator fields use one approved `JointGeneratorConfig`;
- core policy and joint-policy values use typed enums rather than strings; and
- sampler-output checks include the approved no-redundant-copy ownership rule.

## How to run

Run the approved Packet B acceptance contract with:

```powershell
python -m pytest tests/acceptance/test_packet_b_contracts.py -q
```

The normal `python -m pytest` command also collects this active module.

## Contract hash

The approved SHA-256 of `test_packet_b_contracts.py` is
`462E143E18AFFF5CFB72AD21918E7C880696480861281DAAB6510FC0AA1759A0`.
