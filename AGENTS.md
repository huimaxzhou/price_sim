# Repository Instructions

## Project purpose

This repository implements a readable Python simulator for equity returns and
price paths under a supplied factor-risk model. Preserve the distinction between
a covariance risk model and a complete stochastic return or price model.

## Governing documents

Use these documents when their subject is relevant:

1. `references/barra_simulation_specification_manual_v4.md` governs mathematical
   behavior, inputs, outputs, policies, invariants, and acceptance criteria.
2. `references/barra_simulation_architectural_design.md` governs module boundaries,
   dependency direction, domain objects, numerical strategy, and readability.
3. `references/barra_simulation_implementation_plan.md` governs implementation
   order, intermediate deliverables, tests, and review gates.

They have authority in that order. If they conflict, stop and report the
conflict. Do not resolve it silently in code or tests.

## Implementation scope

- Implement only the user-approved step or review packet from the implementation
  plan. Do not begin later steps automatically.
- Keep the Python implementation direct, typed, and small enough for manual
  review.
- Do not add abstractions solely for deferred features.
- Preserve unrelated user changes.
- Do not commit, push, delete files, or change dependencies without explicit
  authorization.

## Acceptance-test rule

Files under `tests/acceptance/` become the executable behavioral contract after
the user approves them.

After approval, do not weaken, remove, skip, rename, or materially reinterpret an
acceptance test without explicit user approval. If production code and an
approved acceptance test disagree, treat the production code as incorrect unless
the governing documents establish that the test is wrong.

The implementation agent may add unit, integration, statistical, and negative
tests outside the approved acceptance-test baseline.

## Protected contract changes

Obtain explicit user approval before changing:

- a governing document listed above;
- an approved file under `tests/acceptance/`;
- an acceptance threshold or numerical tolerance;
- a documented equation, dimension, axis order, unit, or default policy;
- the definition of completion;
- the supported version-one scope; or
- a deferred feature into an active feature.

Harmless formatting and spelling corrections are permitted only when they do not
change meaning.

## Verification and completion

Do not declare a step or review packet complete merely because selected tests
pass. Completion requires:

1. all focused tests for the step pass;
2. all previously approved acceptance tests pass;
3. required negative tests pass;
4. statistical tests use approved sample sizes and tolerances;
5. the implementation satisfies the step's acceptance criteria;
6. no unsupported policy is silently accepted;
7. changed files and actual test results are reported; and
8. the applicable manual-review checkpoint is ready for user review.

Plans, intended behavior, and unexecuted tests are not evidence of completion.

## If an acceptance test appears incorrect

1. Do not modify production code merely to exploit the test.
2. Identify the governing requirement.
3. Explain why the test and requirement differ.
4. Propose the exact test change.
5. Wait for explicit user approval before changing the approved test.
