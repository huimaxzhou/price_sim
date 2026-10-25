"""Build the explicit v4 acceptance report from recorded test evidence."""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_CRITERIA = {str(index) for index in range(1, 11)}
ACCEPTED_EVIDENCE_STATUS = "pass"
ACCEPTED_MANUAL_STATUS = "approved"


def build_acceptance_report(
    evidence: Mapping[str, Mapping[str, object]],
    *,
    manual_review_status: str,
) -> dict[str, object]:
    """Return a fail-closed report for the ten v4 acceptance criteria."""

    supplied = set(evidence)
    exact_criteria = supplied == REQUIRED_CRITERIA
    complete_records = all(
        bool(record.get("test_ids"))
        and bool(record.get("evidence_summary"))
        and bool(record.get("modules"))
        and record.get("status") == ACCEPTED_EVIDENCE_STATUS
        for record in evidence.values()
    )
    passed = (
        exact_criteria
        and complete_records
        and manual_review_status == ACCEPTED_MANUAL_STATUS
    )
    return {
        "schema_version": "1",
        "criteria": dict(evidence),
        "manual_review_status": manual_review_status,
        "overall_status": "pass" if passed else "fail",
    }
