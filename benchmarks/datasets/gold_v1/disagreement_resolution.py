"""Deterministic resolution for Reviewer A/B disagreements.

Rule:
- RETRIEVAL_FAILURE: the required fact exists in the authoritative
  source knowledge, but is missing from retrieved contexts.
- KNOWLEDGE_BASE_FAILURE: the required fact does not exist in the
  authoritative source knowledge.
"""

RESOLVER_ID = "deterministic_source_knowledge_rule_v1"

DISAGREEMENT_RESOLUTIONS = {
    "candidate_004": {
        "label": "KNOWLEDGE_BASE_FAILURE",
        "reason": "Return-shipping cost policy is not present in the source knowledge."
    },
    "candidate_005": {
        "label": "KNOWLEDGE_BASE_FAILURE",
        "reason": "Required browser for the VPN portal is not present in the source knowledge."
    },
    "candidate_006": {
        "label": "KNOWLEDGE_BASE_FAILURE",
        "reason": "Required browser for the VPN portal is not present in the source knowledge."
    },
    "candidate_018": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Standard support hours exist in source knowledge but are missing from retrieved contexts."
    },
    "candidate_024": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Damaged-item reporting period exists in source knowledge but is missing from retrieved contexts."
    },
    "candidate_025": {
        "label": "KNOWLEDGE_BASE_FAILURE",
        "reason": "Exchange-instead-of-return policy is not present in the source knowledge."
    },
    "candidate_031": {
        "label": "KNOWLEDGE_BASE_FAILURE",
        "reason": "Weekend IT support availability is not present in the source knowledge."
    },
    "candidate_033": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Riyadh hotel reimbursement cap exists in source knowledge but is missing from retrieved contexts."
    },
    "candidate_035": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Password reset link validity exists in source knowledge but is missing from retrieved contexts."
    },
    "candidate_042": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Critical-incident escalation timing exists in source knowledge but is missing from retrieved contexts."
    },
    "candidate_047": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "VPN MFA requirement exists in source knowledge but is missing from retrieved contexts."
    },
    "candidate_049": {
        "label": "RETRIEVAL_FAILURE",
        "reason": "Approved refund timing exists in source knowledge but is missing from retrieved contexts."
    },
}
