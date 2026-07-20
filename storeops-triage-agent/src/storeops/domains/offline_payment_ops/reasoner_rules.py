"""Offline payment reasoner rules."""

from __future__ import annotations

from storeops.domains.offline_payment_ops.safety_rules import (
    OFFLINE_PAYMENT_FORBIDDEN_ACTIONS,
)
from storeops.core.contracts import Assessment, CauseAssessment, EvidenceRecord


class OfflinePaymentReasoner:
    """Ranks offline payment causes from normalized evidence."""

    def reason(
        self,
        *,
        evidence: list[EvidenceRecord],
        parsed_case,
    ) -> CauseAssessment:
        supporting_by_cause: dict[str, list[str]] = {}
        contradicting_by_cause: dict[str, list[str]] = {}
        for record in evidence:
            for cause in record.supports:
                supporting_by_cause.setdefault(cause, []).append(record.evidence_id)
            for cause in record.contradicts:
                contradicting_by_cause.setdefault(cause, []).append(record.evidence_id)

        if "temporary_duplicate_tid" in supporting_by_cause and "temporary_duplicate_tid" in contradicting_by_cause:
            return CauseAssessment(
                primary_cause=None,
                assessment=Assessment.NEEDS_CONFIRMATION,
                supporting_evidence_ids=supporting_by_cause["temporary_duplicate_tid"],
                contradicting_evidence_ids=contradicting_by_cause["temporary_duplicate_tid"],
                alternative_causes=[
                    "temporary_duplicate_tid",
                    "post_incident_configuration_change",
                    "incomplete_activation_history",
                ],
                next_checks=[
                    "inspect_incident_time_tid_history",
                    "confirm_activation_sequence",
                    "compare_current_and_incident_time_configuration",
                ],
                forbidden_actions=OFFLINE_PAYMENT_FORBIDDEN_ACTIONS,
            )

        priority = [
            "duplicate_tid",
            "terminal_identifier_mismatch",
            "van_merchant_registration_missing",
            "pos_front_connection_issue",
        ]
        for cause in priority:
            if cause in supporting_by_cause:
                alternatives = [
                    candidate
                    for candidate in priority
                    if candidate != cause and candidate in supporting_by_cause
                ]
                return CauseAssessment(
                    primary_cause=cause,
                    assessment=Assessment.LIKELY,
                    supporting_evidence_ids=supporting_by_cause[cause],
                    alternative_causes=alternatives,
                    next_checks=self._next_checks_for(cause),
                    forbidden_actions=OFFLINE_PAYMENT_FORBIDDEN_ACTIONS,
                )

        return CauseAssessment(
            primary_cause=None,
            assessment=Assessment.UNAVAILABLE,
            missing_evidence=parsed_case.missing_fields or ["supporting_system_evidence"],
            next_checks=["request_missing_merchant_context", "manual_review_available_system_logs"],
            forbidden_actions=OFFLINE_PAYMENT_FORBIDDEN_ACTIONS,
        )

    @staticmethod
    def _next_checks_for(cause: str) -> list[str]:
        return {
            "duplicate_tid": ["confirm_tid_mapping_with_installation_owner"],
            "terminal_identifier_mismatch": ["compare_physical_device_number_and_registered_identity"],
            "van_merchant_registration_missing": ["confirm_van_registration_status_with_responsible_team"],
            "pos_front_connection_issue": ["verify_pos_front_pairing_and_request_delivery_logs"],
        }[cause]


__all__ = ["OFFLINE_PAYMENT_FORBIDDEN_ACTIONS", "OfflinePaymentReasoner"]
