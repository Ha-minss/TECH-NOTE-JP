"""Base interface for LLM providers.

Providers call the model and return raw structured output.
They do not update RecoveryCase.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Minimal provider contract used by extraction.py and answers.py.

    Implementations may return either:
    - a Python dict already parsed from JSON, or
    - a JSON string / text containing a JSON object.
    """

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        """Return structured output from the model."""
        ...
