"""Offline payment safety-specific rules."""

OFFLINE_PAYMENT_FORBIDDEN_ACTIONS = [
    "change_tid_without_confirmation",
    "execute_payment",
    "payment_cancellation",
    "refund",
    "config_mutation",
    "external_handoff_without_approval",
]


def offline_payment_forbidden_actions() -> list[str]:
    return list(OFFLINE_PAYMENT_FORBIDDEN_ACTIONS)


__all__ = ["OFFLINE_PAYMENT_FORBIDDEN_ACTIONS", "offline_payment_forbidden_actions"]
