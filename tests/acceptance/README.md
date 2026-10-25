# Acceptance Contracts

Packets A through D are approved executable behavioral contracts. Their test
modules are collected by the normal pytest suite. Under `AGENTS.md`, approved
tests, thresholds, and tolerances cannot be materially changed without explicit
user approval.

## Packet A

**Status:** Approved
**Approval date:** 2026-10-22
**Scope:** Implementation-plan Steps 1–3
**Test module:** `test_packet_a_contracts.py`

## Purpose

These tests define observable contracts for the package bootstrap, error and
domain vocabulary, and policy-default resolution. They do not authorize or test
simulation, validation, sampling, pricing, diagnostics, or persistence behavior.

This approved revision supersedes the initial Packet A contract. Production
code must conform to these tests. Any later material change to these tests
requires explicit user approval under `AGENTS.md`.

## Approved public names

Packet A uses the following direct, readable names:

- errors: `SimulationError`, `ConfigurationError`, `ValidationError`,
  `NumericalError`, `PolicyExecutionError`, and `ValidationIssue`;
- policy enums: `ExposureDynamics`, `FactorVolatilityDynamics`,
  `FactorCorrelationDynamics`, `SpecificRiskDynamics`,
  `FactorReturnDistribution`, `SpecificReturnDistribution`,
  `SerialCorrelationPolicy`, `JointDistributionPolicy`,
  `PriceReturnConvention`, and `InvalidReturnPolicy`;
- asset enum: `AssetStatus`;
- models: `RiskState`, `AssetMetadata`, `FactorMetadata`,
  `SimulationSchedule`, `TradingCalendar`, `JointGeneratorConfig`,
  `SimulationPolicy`, `ResolvedSimulationPolicy`, `NumericalTolerances`,
  `SimulationRequest`, `InnovationBatch`, `AssetStateEvent`,
  `ValidationReport`, and `SimulationResult`; and
- policy function: `resolve_policy`.

The direct Python API accepts enum members. Serialized string parsing is outside
Packet A and remains deferred to a future adapter.

## Requirement traceability

| Test ID | Governing requirement | Evidence type |
|---|---|---|
| `A-BOOT-001` | Python 3.11+, source layout, NumPy-only production dependency, pytest and linter development tools | Deterministic metadata inspection |
| `A-BOOT-002` | Planned modules import without circular-import failure | Deterministic import test |
| `A-ERR-001` | Required exception hierarchy | Deterministic type inspection |
| `A-ERR-002` | Validation errors preserve multiple structured issues | Deterministic construction |
| `A-ENUM-001` | Policy and asset-status enums are stable string-valued enums | Deterministic value inspection |
| `A-MODEL-001` | Required domain models are frozen dataclasses | Deterministic type inspection |
| `A-MODEL-002` | Risk-state identifiers preserve order and numerical inputs cannot be changed through caller aliases | Deterministic construction and mutation attempt |
| `A-MODEL-003` | Factor metadata is optional | Deterministic construction |
| `A-MODEL-004` | Asset-state events contain every required field | Deterministic construction |
| `A-MODEL-005` | All three schedule alternatives are representable | Deterministic construction |
| `A-MODEL-006` | Requests copy and protect external numerical inputs | Deterministic construction and mutation attempt |
| `A-MODEL-007` | Engine-owned output arrays are made read-only without being recopied | Deterministic identity and rejection tests |
| `A-POLICY-001` | The minimally valid policy defaults every optional policy and records provenance | Deterministic resolution |
| `A-POLICY-002` | Explicit default values remain distinguishable from omitted values | Deterministic provenance inspection |
| `A-POLICY-003` | Missing mandatory invalid-return policy is rejected | Negative test |
| `A-POLICY-004` | The typed core boundary rejects a wrong runtime type | Negative test |
| `A-POLICY-005` | Joint generation requires a grouped joint-generator configuration | Negative test |
| `A-POLICY-006` | A registered grouped joint configuration resolves | Deterministic resolution |
| `A-POLICY-007` | Joint-generator version must match the registry | Negative test |
| `A-POLICY-008` | Independent mode rejects joint-generator configuration | Negative test |
| `A-POLICY-009` | Non-`None` optional handlers are unsupported | Negative test |
| `A-POLICY-010` | Resolution is pure, deterministic, and does not mutate supplied policy | Deterministic repeatability test |

## Deliberately unresolved implementation details

The tests require observable behavior but do not prescribe:

- the private sentinel used to distinguish omission from explicit `None`;
- the internal joint-generator registry representation;
- the internal storage type for validation-issue collections;
- the implementation of frozen-array copying; or
- concrete fields of result objects belonging to later packets beyond their
  existence as frozen dataclasses.

## How to run

```powershell
python -m pytest tests/acceptance/test_packet_a_contracts.py -q
```

The module must collect without skips and pass with the approved Packet A
implementation. The approved SHA-256 recorded below is the protected baseline.

## Contract hash history

| Revision | SHA-256 of `test_packet_a_contracts.py` | Status |
|---|---|---|
| Initial approved Packet A | `55F7EADEA235ED0C1009F5CC7681F8547257FE4639F6BA80E823EDA373045EA6` | Superseded by the approved Review Gate A amendment |
| Review Gate A amendment | `9CC9EAB2F29DD8297036488E83C99913C42F153E30251D6757919DCCB3A556C7` | Approved 2026-10-23 |

## Packet B

**Status:** Approved
**Approval date:** 2026-10-24
**Scope:** Implementation-plan Steps 4–7
**Test module:** `test_packet_b_contracts.py`

Packet B protects deterministic validation, covariance-root construction,
reproducible per-path RNG streams, and independent and joint Gaussian sampling.
Its approved SHA-256 is
`462E143E18AFFF5CFB72AD21918E7C880696480861281DAAB6510FC0AA1759A0`.

## Packet C

**Status:** Approved
**Approval date:** 2026-10-25
**Scope:** Implementation-plan Steps 8–10
**Test module:** `test_packet_c_contracts.py`

Packet C protects return construction, raw arithmetic pricing, explicit None
handlers, and complete in-memory orchestration. Packet D's approved successor
amendment replaced the preflight-only diagnostic assertion while preserving all
preflight summaries. Its current approved SHA-256 is
`E4CE706DB87B9D87F8D5D0F11F8E15F806951579C7DFEEA3F50DB5ABEA1D4B0B`.

## Packet D

**Status:** Approved contract; implementation and Review Gate D pending
**Approval date:** 2026-10-25
**Scope:** Implementation-plan Steps 11–15
**Test module:** `test_packet_d_contracts.py`

Packet D protects complete diagnostics, exact auditable persistence, final
acceptance reporting, documentation, and non-gating performance evidence. Its
approved SHA-256 is
`C1B99F0E64579EABEEEBCE6E8AE7CF17423958F1FEBAFE60EE6C59B3CE48B4B6`.
