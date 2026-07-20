"""Placeholder handler for H02 Fee Overcharge.

Do not implement product logic here until real fee-waiver policy documents,
ledger schemas, and approved SQL are provided. Keeping this file as a placeholder
shows the extension point without inventing financial rules.
"""

from __future__ import annotations

from typing import Any


class FeeOverchargePlaceholderHandler:
    """Non-executable placeholder for the future H02 template."""

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "H02 fee_overcharge is a placeholder. Add real Product Config, "
            "Data Contract, approved SQL, and tests before enabling execution."
        )
