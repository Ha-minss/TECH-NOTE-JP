from __future__ import annotations

OFFLINE_PAYMENT_TOOL_PLANNING_POLICY = """
Offline payment operations tool-selection policy.

Data need definitions:
- payment_identifier_config:
  Use this to inspect TID/config mapping evidence.
  This maps to get_tid_config.

- payment_identifier_history:
  Use this to inspect current vs incident-time TID records.
  This maps to get_tid_history.

- approval_failure_history:
  Use this to inspect recent card approval failure logs.
  This maps to get_recent_approval_errors.

- van_registration_status:
  Use this to inspect VAN merchant registration evidence.
  This maps to get_van_registration.

- terminal_identity:
  Use this to inspect terminal serial/model identity evidence.
  This maps to get_terminal_identity.

- pos_front_connection_logs:
  Use this to inspect POS-to-front-terminal request delivery or timeout logs.
  This maps to get_pos_front_connection_logs.

Planning rules:
- If a case mentions new terminal installation plus approval failures,
  include payment_identifier_config as required.

- If a case mentions existing terminal approval failures after installation,
  include payment_identifier_config and approval_failure_history as required.

- If a case mentions payment identifier, TID, identifier mismatch, serial mismatch,
  or current-vs-incident TID disagreement,
  include payment_identifier_config as required.

- If a case mentions missing or incomplete VAN merchant registration with approval failures,
  include van_registration_status, payment_identifier_config, and approval_failure_history as required.

- If a case mentions POS cannot deliver payment requests to the front terminal,
  include pos_front_connection_logs as required.

Schema rules:
- issue_family must be exactly one of:
  payment_approval_failure, pos_front_connection_issue.

- selected_data_needs[].priority must be exactly one of:
  required, supporting, optional.

- confidence must be a number between 0 and 1.
  Do not use strings such as low, medium, or high.
"""
