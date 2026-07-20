"""LLM-assisted layers for the synthetic workflow portfolio.

These modules are intentionally scoped as proposal layers:

- parse richer natural language into the existing deterministic contract;
- propose clarification questions limited to merchant-observable fields;
- select allowed data-needs that map back into the read-only tool catalog;
- draft merchant-facing copy without changing safety or execution logic.
"""

from .case_parser import LLMCaseParser
from .clarification import ClarificationQuestion, ClarificationQuestionGenerator
from .client import ScriptedLLMClient
from .drafting import MerchantResponseDrafter
from .planner import LLMPlanner
from .runtime import LLMCallTrace, LLMRuntime

__all__ = [
    "ClarificationQuestion",
    "ClarificationQuestionGenerator",
    "LLMCallTrace",
    "LLMCaseParser",
    "LLMPlanner",
    "LLMRuntime",
    "MerchantResponseDrafter",
    "ScriptedLLMClient",
]
