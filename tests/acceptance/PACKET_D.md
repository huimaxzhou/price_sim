# Packet D Acceptance Contract

**Status:** Active; implementation authorized, Review Gate D pending
**Prepared:** 2026-10-25
**Scope:** Implementation-plan Steps 11–15
**Acceptance module:** `test_packet_d_contracts.py`

## Review-state boundary

Review Gate C passed on 2026-10-25. The user approved the Packet D decisions and
authorized implementation on 2026-10-25. Activating this contract authorizes
Steps 11–15 implementation but does not itself establish Review Gate D or final
version-one acceptance. Those remain pending the required evidence and manual
review.
The approved acceptance-test SHA-256 is
`C1B99F0E64579EABEEEBCE6E8AE7CF17423958F1FEBAFE60EE6C59B3CE48B4B6`.

## Purpose

Packet D supplies the evidence layer around the accepted simulator:

1. deterministic and Monte Carlo diagnostics;
2. external, atomic serialization and validated read-back;
3. complete v4 acceptance traceability;
4. final readability, dependency, and performance evidence; and
5. the final version-one acceptance decision.

Packet D does not add a new stochastic model, policy, dependency, database,
parallel execution, path batching, or invasive optimization.

## Proposed diagnostic interface

`diagnostics.py` exposes:

```python
evaluate_result(result: SimulationResult) -> ValidationReport
```

`simulate()` calls this function before returning, so the final
`SimulationResult.validation_report` has `diagnostic_scope="complete"`.
The same function remains directly callable for review and corruption tests.

The existing `ValidationReport` is retained; no new large diagnostic object is
introduced. Its `summaries` mapping contains:

- `diagnostic_scope="complete"`;
- `overall_status`, one of `pass` or `fail`;
- `checks`, an ordered tuple of JSON-ready mappings; and
- the deterministic preflight and numerical-normalization summaries already
  produced by Packets B and C.

Each check contains exactly:

- `name` — stable identifier;
- `category` — `deterministic` or `statistical`;
- `target` — scalar or nested JSON-ready values;
- `estimate` — scalar, nested JSON-ready values, or `None`;
- `sample_size` — integer or `None`;
- `rule` — a JSON-ready mapping naming the equality, tolerance, or confidence
  method and any numerical threshold;
- `status` — `pass`, `fail`, or `not_applicable`; and
- `message` — concise interpretation.

Nested diagnostic mappings are not required to be recursively immutable.

### Deterministic checks

The report verifies:

- every array shape and axis order;
- ordered IDs, dates, and currencies;
- the exact return equation within the request equation tolerance;
- finite factor, specific, modeled, and price-link returns;
- finite and strictly positive live prices;
- initial-price preservation;
- fixed risk-state identity and target covariance;
- price-link and modeled-return identity under the raw mapping;
- empty asset-state events under the three `None` handlers;
- resolved-policy and sampler metadata consistency; and
- disclosed covariance-root canonicalization.

A deterministic diagnostic failure is recorded as `fail`. `evaluate_result`
does not repair arrays or mutate the result.

### Statistical checks and sample rules

The routine acceptance fixture uses `M=400` paths, `H=2` dates, `N=3` assets,
and `K=2` factors with a fixed seed. Pooling by date supplies 800 observations
for the conditionally identical moment checks. This is a fast statistical smoke
test intended to detect material errors; deterministic equation and corruption
tests remain the primary correctness evidence. No 40,000-path run is required
for routine completion. A larger, separately invoked run may be used later for
non-gating precision evidence.

For Gaussian samples, the draft uses five-standard-error bounds:

- mean: `5 * sqrt(target_variance / sample_size) + 1e-12`;
- covariance `(i,j)`:
  `5 * sqrt((Sigma_ij**2 + Sigma_ii*Sigma_jj)/(sample_size-1)) + 1e-12`;
- zero factor-specific cross-covariance:
  `5 * sqrt(F_ii*D_jj/(sample_size-1)) + 1e-12`; and
- lag-one correlation: `5 / sqrt(M) + 0.01` using one adjacent-date pair per
  path in the acceptance fixture.

Factor, specific, and modeled covariance checks pool the `M*H` conditionally
identical innovations after their date-specific means are removed. Statistical
failure is evidence, not a runtime exception: a simulation result is still
returned with `overall_status="fail"` for the caller to inspect. Structural
inability to calculate a required deterministic diagnostic raises
`NumericalError`. A statistically undefined quantity in an otherwise valid run
is reported as `not_applicable`, not raised.

Zero-variance correlation entries use `estimate=None`,
`status="not_applicable"`, and a message naming the zero-variance component.
They are not fabricated as zero and do not by themselves make the overall
report fail.

The independent Gaussian claim is supported by both recorded generator
construction and distribution-sensitive evidence. Zero empirical
cross-covariance alone is never labeled proof of independence.

## Proposed serialization interface

`io.py` exposes:

```python
save_result(result: SimulationResult, output_dir: Path) -> Path
load_result(output_dir: Path) -> SimulationResult
```

### Directory and atomicity policy

- `output_dir` must not already exist; overwrite and merge are refused.
- The writer creates a unique sibling temporary directory on the same volume.
- All files are written, flushed, closed, hashed, and validated in the temporary
  directory.
- Only then is the temporary directory renamed to `output_dir`.
- Failure removes only that writer-owned temporary directory and leaves no
  final target.
- `simulate()` remains file-free.

This policy prevents a normal process failure from publishing a visibly partial
bundle. It does not claim durability against storage-device failure or sudden
power loss.

### Saved layout

```text
output_dir/
├── arrays.npz
├── manifest.json
├── diagnostics.json
└── asset_state_events.jsonl   # only when events exist
```

- `arrays.npz` is uncompressed and contains both the eight documented generated
  result arrays and the complete numerical input state needed to audit and
  reproduce the run: exposures, factor covariance, specific variances, initial
  prices, factor means, and specific means. Exact member names are fixed by the
  acceptance test.
- JSON is UTF-8, `allow_nan=False`, uses stable key ordering, and represents
  enums as their values and dates/timestamps as ISO 8601 strings.
- `manifest.json` contains serialization schema version `1`, file inventory,
  byte counts, SHA-256 hashes for every other emitted file, event-file presence,
  and the complete run metadata needed to reconstruct the result.
- `diagnostics.json` contains the complete `ValidationReport`.
- Empty events omit `asset_state_events.jsonl`; the manifest records
  `present=false`, `count=0`, and that the applicable handlers were `None`.
- Version-one `save_result` rejects a result containing events while all three
  supported handlers are `None`; such a result is internally inconsistent.
- The conditional nonempty JSON Lines encoding is implemented and unit-tested
  at the I/O boundary, but it is not activated as a successful version-one
  bundle until a non-`None` handler receives its own approved contract.

These events are audit records for a future corporate-action, delisting, or
new-listing handler that changes an asset's path-specific state. They do not
implement or imply active corporate-action support.

SHA-256 values cover each complete emitted file other than the manifest that
stores those hashes. They detect truncation, accidental edits, corruption, and
mixing files from different runs during read-back. They are not digital
signatures and do not establish authorship against an intentional editor who
can also rewrite the manifest. Caller-supplied source hashes are preserved as
input provenance; Packet D does not hash an arbitrary in-memory Python request
representation. Packet D also does not promise that two NPZ archives written by
different supported platforms are byte-for-byte identical; fixed-seed
reproducibility is tested on reconstructed array values and metadata.

### Read-back policy

`load_result` fails closed on:

- missing required files or array members;
- unknown schema versions;
- hash or byte-count mismatch;
- malformed JSON or JSON Lines;
- nonfinite numerical data;
- inconsistent dimensions, IDs, dates, currencies, policies, or diagnostics;
  and
- unexpected files not listed in the manifest.

Successful read-back reconstructs the typed result with exact array values,
ordered identifiers, dates, metadata, policy values and value sources,
diagnostics, risk state, and optional factor metadata. Loaded numerical arrays
are owned `float64` and read-only.

## Proposed acceptance evidence

Packet D introduces one canonical deterministic fixture shared by integration,
serialization, and reporting tests:

- three assets;
- two factors;
- four future dates;
- two deterministic paths where appropriate;
- explicit factor and specific means;
- one zero specific variance;
- a singular-PSD factor-covariance variant;
- asset currencies and factor classes; and
- values small enough for manual reconstruction.

The acceptance-report generator lives under `tests/acceptance/`, not production
code. It emits a JSON-ready report mapping all ten manual-v4 Section 10 criteria
to stable test IDs, evidence summaries, relevant modules, and actual statuses.
Missing, unknown, skipped, or failed required evidence makes the report fail.
The report does not execute pytest or infer success from test names alone; it
consumes explicit test-result records produced by the acceptance run.

## Performance and optimization posture

The non-gating benchmark reports validation, sampling, return construction,
pricing, diagnostics, serialization, read-back, end-to-end time, and estimated
and observed result bytes for the approved canonical and maximum shapes.

No hardware-dependent pass threshold is proposed. Packet D records results and
identifies the largest measured stage. Path batching, parallelism, compiled
extensions, storage changes, and other invasive optimization remain deferred
unless a measured result violates a separately agreed target.

## Proposed public surface

The package-root API remains unchanged. `evaluate_result`, `save_result`, and
`load_result` are available from `factor_sim.diagnostics` and `factor_sim.io`.
No file-I/O function is re-exported from the package root.

To make exact input persistence possible, `SimulationResult` gains read-only
`initial_prices`, `factor_means`, and `specific_means` arrays. Its existing
`risk_state` already retains exposures, factor covariance, and specific
variances. This is retained simulation evidence, not a new computational API.

## Requirement traceability

| Test ID | Governing requirement | Evidence |
|---|---|---|
| `D-DIAG-001` | Final result carries complete structured diagnostics | Report-schema inspection |
| `D-DIAG-002` | Return equation and raw identity mapping are checked | Exact and corrupted fixtures |
| `D-DIAG-003` | Prices, initial slice, and asset state are checked | Exact and corrupted fixtures |
| `D-DIAG-004` | Result alignment, policy, provenance, and frozen state are checked | Metadata corruption fixtures |
| `D-DIAG-005` | Diagnostics do not mutate the result | Before/after snapshots |
| `D-STAT-001` | Factor means and covariance reproduce targets | 400-path Gaussian smoke fixture |
| `D-STAT-002` | Specific means, diagonal covariance, and zero off-diagonals reproduce targets | 400-path Gaussian smoke fixture |
| `D-STAT-003` | Factor-specific cross-covariance reproduces zero | Five-standard-error rule |
| `D-STAT-004` | Modeled asset covariance reproduces `X F X.T + D` | Five-standard-error rule |
| `D-STAT-005` | Cross-asset correlations are checked without fabricating zero-variance entries | Regular and singular fixtures |
| `D-STAT-006` | No-serial policy is checked | Fixed-seed lag-one fixture |
| `D-STAT-007` | Tolerance scales predictably with sample size | Analytic rule comparison |
| `D-STAT-008` | Independence claims include construction and distribution-sensitive evidence | Diagnostic-content inspection |
| `D-IO-001` | Saved layout and uncompressed NPZ members include generated arrays and complete numerical inputs | Temporary-directory inspection |
| `D-IO-002` | JSON values are portable and finite | Parsed JSON inspection |
| `D-IO-003` | Empty event files are omitted and disclosed | Manifest inspection |
| `D-IO-004` | Events cannot be published under the three `None` handlers | Inconsistent-result rejection |
| `D-IO-005` | Full round trip preserves arrays and metadata | Save/load comparison |
| `D-IO-006` | File hashes and byte counts are verified | Tamper tests |
| `D-IO-007` | Missing, malformed, unknown-version, and extra files fail closed | Parameterized negative tests |
| `D-IO-008` | Save is atomic and refuses overwrite | Failure injection and existing-target tests |
| `D-IO-009` | Serialization does not mutate the result or write from `simulate` | Snapshot and I/O tripwire tests |
| `D-ACC-001` | Canonical and singular deterministic fixtures remain manually traceable | End-to-end exact tests |
| `D-ACC-002` | Every v4 Section 10 criterion maps to explicit passing evidence | Acceptance-report inspection |
| `D-ACC-003` | Missing, unknown, skipped, or failed evidence prevents acceptance | Negative report tests |
| `D-ARCH-001` | NumPy remains the only production dependency and module boundaries remain intact | Dependency and import inspection |
| `D-PERF-001` | Required stage timings and memory observations are recorded without a hidden threshold | Benchmark-schema inspection |
| `D-DOC-001` | README describes actual setup, API, policies, outputs, limitations, and test commands | Documentation inspection |
| `D-FINAL-001` | Final acceptance requires all Step 15 evidence and completed manual review | Final-gate report inspection |

## Decisions requiring user approval

1. **Diagnostic execution:** automatically compute complete diagnostics during
   `simulate`, while statistical failures return evidence instead of raising.
2. **Diagnostic representation:** retain `ValidationReport` and store ordered,
   JSON-ready check mappings rather than introducing another large dataclass.
3. **Statistical contract:** use a 400-path, two-date routine smoke fixture and
   the five-standard-error formulas above. A larger precision run is optional,
   separately invoked, and non-gating.
4. **Undefined correlation:** record `not_applicable` with `estimate=None`; do
   not fabricate zero or fail the whole report solely for zero variance.
5. **Serialization target:** refuse any existing output directory and publish by
   same-volume temporary-directory rename.
6. **Empty events:** omit the JSON Lines file and disclose the omission in the
   manifest. Reject nonempty events under the only supported (`None`) handlers;
   keep the dormant nonempty JSON Lines codec under unit test rather than
   treating an impossible version-one bundle as valid. These records are only
   for future corporate-action, delisting, or new-listing state changes.
7. **Hash meaning:** verify file integrity, but do not promise cross-platform
   byte-identical NPZ archives or cryptographic authorship. Preserve supplied
   source hashes as provenance rather than hashing an arbitrary Python object.
8. **Public API:** keep diagnostics and I/O functions in their modules rather
   than expanding the package root.
9. **Performance:** record both scales without a hardware-dependent gate and
   continue deferring invasive optimization.

## Focused check

```powershell
python -m pytest tests/acceptance/test_packet_d_contracts.py -q
```
