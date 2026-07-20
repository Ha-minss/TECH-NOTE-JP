"""Gemma + validator + template fallback generator."""

from __future__ import annotations

import time
from typing import Any

from evaluation.dataset_loader import GoldCase
from evaluation.generators.gemma_generator import GemmaGenerator
from evaluation.generators.template_generator import TemplateGenerator
from evaluation.validators.fact_validator import validate_generated_record
from recover24.providers.base import LLMProvider


class GemmaValidatorGenerator:
    name = "gemma_validator"

    def __init__(self, provider: LLMProvider | None = None, claim_provider: LLMProvider | None = None) -> None:
        self.gemma = GemmaGenerator(provider)
        self.claim_provider = claim_provider

    def generate(self, case: GoldCase) -> dict[str, Any]:
        record = self.gemma.generate(case)
        validation = validate_generated_record(case, record, claim_provider=self.claim_provider)
        record["method"] = self.name
        record["meta"]["blocked_by_validator"] = not validation["safe_to_use"]
        record["meta"]["validation"] = validation
        return record


class HybridGenerator:
    name = "gemma_validator_fallback"

    def __init__(
        self,
        provider: LLMProvider | None = None,
        claim_provider: LLMProvider | None = None,
        max_retries: int = 1,
    ) -> None:
        self.provider = provider
        self.claim_provider = claim_provider
        self.max_retries = max_retries
        self.template = TemplateGenerator()

    def generate(self, case: GoldCase) -> dict[str, Any]:
        start = time.perf_counter()
        llm_calls = 0
        last_record: dict[str, Any] | None = None
        last_validation: dict[str, Any] | None = None
        gemma = GemmaGenerator(self.provider)

        for _ in range(self.max_retries + 1):
            record = gemma.generate(case)
            llm_calls += record["meta"].get("llm_calls", 0)
            validation = validate_generated_record(case, record, claim_provider=self.claim_provider)
            last_record, last_validation = record, validation
            if validation["safe_to_use"]:
                record["method"] = self.name
                record["meta"].update({
                    "fallback_used": False,
                    "blocked_by_validator": False,
                    "validation": validation,
                    "llm_calls": llm_calls,
                    "latency_sec": round(time.perf_counter() - start, 4),
                })
                return record

        fallback = self.template.generate(case)
        fallback_validation = validate_generated_record(case, fallback, claim_provider=self.claim_provider)
        fallback["method"] = self.name
        fallback["meta"].update({
            "fallback_used": True,
            "blocked_by_validator": True,
            "validation": fallback_validation,
            "previous_validation": last_validation,
            "previous_outputs": last_record.get("outputs") if last_record else None,
            "llm_calls": llm_calls,
            "latency_sec": round(time.perf_counter() - start, 4),
        })
        return fallback
