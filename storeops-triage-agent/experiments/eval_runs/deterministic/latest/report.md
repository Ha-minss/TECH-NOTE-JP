# Evaluation Report

- total_cases: 50
- passed_cases: 38
- state_accuracy: 0.90
- cause_accuracy: 0.98
- abstention_safety_accuracy: 1.00
- unsupported_claim_count: 0

## Failing Cases

- GOLD-SYN-032: missing required tools
- GOLD-SYN-033: missing required tools
- GOLD-SYN-034: expected_state=NEEDS_CLARIFICATION actual_state=DEGRADED_REVIEW; missing required tools
- GOLD-SYN-035: expected_state=NEEDS_CLARIFICATION actual_state=DEGRADED_REVIEW; missing required tools
- GOLD-SYN-036: missing required tools
- GOLD-SYN-037: missing required tools
- GOLD-SYN-038: missing required tools
- GOLD-SYN-042: expected_state=DEGRADED_REVIEW actual_state=NEEDS_CLARIFICATION; missing required tools
- GOLD-SYN-043: missing required tools
- GOLD-SYN-044: expected_state=READY_FOR_REVIEW actual_state=DEGRADED_REVIEW; expected_cause=duplicate_tid actual_cause=None; displayed cause without evidence citations; missing required tools
- GOLD-SYN-045: missing required tools
- GOLD-SYN-049: expected_state=CONFLICT_REVIEW actual_state=DEGRADED_REVIEW; missing required tools