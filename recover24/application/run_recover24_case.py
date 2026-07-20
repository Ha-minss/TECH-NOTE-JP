"""Application workflow for a single Recover24 V3 case.

This module wires the core components together without owning their internal
logic. It is intentionally thin:

- extraction.py reads the first statement and returns Patch[].
- answers.py reads a follow-up answer and returns Patch[].
- patching.py is the only code that updates RecoveryCase.
- questions.py decides what to ask next.
- readiness.py decides what the case can do next.
- document_view.py builds HTML-ready values.
- html_renderer.py renders the final official HTML.

Use this file from CLI, tests, or Streamlit. Keep UI-specific code out of here.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from recover24.answers import extract_answer_patches
from recover24.document_view import OfficialDocumentView, build_document_view
from recover24.field_policy import should_mark_unknown_on_llm_failure, text_fallback_path
from recover24.extraction import extract_initial_statement
from recover24.html_renderer import render_official_html
from recover24.models import FieldStatus, Patch, Question, RecoveryCase
from recover24.patching import apply_patches
from recover24.providers.base import LLMProvider
from recover24.questions import build_next_questions
from recover24.readiness import ReadinessReport, evaluate_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "templates"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True, slots=True)
class Recover24CaseResult:
    """One complete snapshot after a workflow step.

    The result deliberately carries both machine state and UI-friendly outputs so
    callers do not need to know which module to call next.
    """

    case: RecoveryCase
    patches: list[Patch]
    questions: list[Question]
    readiness: ReadinessReport
    document_view: OfficialDocumentView

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "patches": [_patch_to_dict(patch) for patch in self.patches],
            "questions": [_question_to_dict(question) for question in self.questions],
            "readiness": self.readiness.to_dict(),
            "document_view": self.document_view.to_dict(),
        }


class DemoKeywordProvider:
    """Tiny deterministic provider for local demos when no real LLM is wired.

    This is not the production extraction strategy. It only helps the CLI and
    Streamlit app show the full Recover24 loop without a Colab/Gemma endpoint.
    Real deployments should pass a provider that calls the approved LLM backend.
    """

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if "User answer:" in prompt:
            return {"patches": self._answer_patches(prompt)}
        if "User statement:" in prompt:
            return {"patches": self._initial_patches(prompt)}
        return {"patches": []}

    def _initial_patches(self, prompt: str) -> list[dict[str, Any]]:
        text = prompt.split("User statement:", 1)[-1]
        patches: list[dict[str, Any]] = []

        if any(token in text for token in ("검찰", "경찰", "수사관", "금감원")):
            patches.append(_json_patch("incident.fraud_type", "authority_impersonation", "수사기관 사칭", 0.85))
        elif any(token in text for token in ("자녀", "아들", "딸", "엄마", "아빠", "카카오톡", "카톡")):
            patches.append(_json_patch("incident.fraud_type", "family_impersonation", "가족/지인 사칭", 0.85))
        elif any(token in text for token in ("대출", "저금리", "대환")):
            patches.append(_json_patch("incident.fraud_type", "loan_scam", "대출빙자", 0.80))
        elif any(token in text for token in ("악성앱", "원격", "스미싱", "링크")):
            patches.append(_json_patch("incident.fraud_type", "smishing_malware", "스미싱/악성앱", 0.80))

        amount = _extract_amount_krw(text)
        if amount is not None:
            patches.append(_json_patch("transactions.0.amount_krw", amount, str(amount), 0.80))

        if "모바일" in text or "앱" in text or "뱅킹" in text:
            patches.append(_json_patch("transactions.0.transaction_type", "mobile_banking_transfer", "모바일/앱/뱅킹", 0.75))

        for path, label, keywords in (
            ("transactions.0.source_bank", "출금은행", ("출금은행", "보낸 은행")),
            ("transactions.0.destination_bank", "입금은행", ("입금은행", "받는 은행", "상대 은행")),
            ("transactions.0.destination_account_holder", "수취인", ("수취인", "예금주")),
        ):
            value = _extract_labeled_text(text, keywords)
            if value:
                patches.append(_json_patch(path, value, f"{label}: {value}", 0.65))

        if "경찰" in text and any(token in text for token in ("신고", "접수")):
            patches.append(_json_patch("investigation.status", "reported", "경찰 신고", 0.75))
        elif "신고" in text and any(token in text for token in ("못", "아직", "안")):
            patches.append(_json_patch("investigation.status", "not_reported", "아직 신고 안 함", 0.75))

        if text.strip():
            patches.append(_json_patch("incident.overview", _shorten(text.strip(), 120), text.strip()[:80], 0.60))

        return patches

    def _answer_patches(self, prompt: str) -> list[dict[str, Any]]:
        answer = prompt.split("User answer:", 1)[-1].strip()
        allowed = set(_extract_allowed_paths_from_prompt(prompt))
        patches: list[dict[str, Any]] = []

        if not answer:
            return patches

        compact = re.sub(r"\s+", "", answer)
        unknown = any(token in answer for token in ("모르", "몰라", "모름", "기억 안", "기억안", "확실하지"))
        not_applicable = any(token in answer for token in ("해당 없", "해당없", "상관 없", "상관없")) and not unknown
        all_no = any(
            token in compact
            for token in (
                "전부다아님",
                "전부아님",
                "모두아님",
                "다아님",
                "전부아니",
                "모두아니",
                "전부아닙니다",
                "모두아닙니다",
            )
        )
        all_yes = any(token in compact for token in ("모두동의", "전부동의", "다동의", "전체동의"))

        for path in sorted(allowed):
            # If the user explicitly says they do not know, mark only the current
            # target paths as UNKNOWN so questions.py does not repeat forever.
            if unknown:
                patches.append(_json_patch(path, None, answer, 0.80, status="unknown"))
                continue

            # Exclusion checklist: "전부 다 아님" means every exclusion item is false.
            # This must become ANSWERED False, not a free-text string.
            if all_no and _looks_boolean_path(path):
                patches.append(_json_patch(path, False, answer, 0.90))
                continue

            if all_yes and path.startswith("consent."):
                patches.append(_json_patch(path, True, answer, 0.95))
                continue

            if not_applicable and path.startswith(("optional_reports.", "delegation.", "survey.")):
                patches.append(_json_patch(path, None, answer, 0.80, status="not_applicable"))
                continue

            value = _guess_value_for_path(path, answer)
            if value is not None:
                patches.append(_json_patch(path, value, answer, 0.70))

        return patches


def start_case(
    initial_statement: str,
    provider: LLMProvider,
    *,
    case_id: str | None = None,
    question_limit: int = 3,
    strict_document_completion: bool = True,
    today: date | None = None,
) -> Recover24CaseResult:
    """Create a case from the first user statement and return the next snapshot."""

    case = RecoveryCase.new(case_id or _new_case_id())
    patches = extract_initial_statement(initial_statement, provider)
    updated_case = apply_patches(case, patches)
    return build_case_result(
        updated_case,
        patches=patches,
        question_limit=question_limit,
        strict_document_completion=strict_document_completion,
        today=today,
    )


def answer_question(
    case: RecoveryCase,
    question: Question,
    answer_text: str,
    provider: LLMProvider,
    *,
    question_limit: int = 3,
    strict_document_completion: bool = True,
    today: date | None = None,
) -> Recover24CaseResult:
    """Apply a free-form follow-up answer through answers.py/LLM."""

    patches = extract_answer_patches(question, answer_text, provider)
    if not patches and answer_text.strip():
        patches = _fallback_patches_for_unstructured_answer(question, answer_text)
    return apply_form_patches(
        case,
        patches,
        question_limit=question_limit,
        strict_document_completion=strict_document_completion,
        today=today,
    )


def _fallback_patches_for_unstructured_answer(question: Question, answer_text: str) -> list[Patch]:
    """Keep the workflow moving when the LLM returns no usable JSON patches.

    This is intentionally conservative: raw text is stored only in text-like
    fields, while other scoped target paths are marked UNKNOWN so questions.py
    will not repeat the exact same blocked LLM question forever. Staff/readiness
    can still review the unknown values later.
    """

    patches: list[Patch] = []
    raw_text = answer_text.strip()
    target_paths = list(question.target_paths)

    fallback_path = text_fallback_path(target_paths)
    if fallback_path is not None:
        patches.append(
            Patch(
                path=fallback_path,
                value=_shorten(raw_text, 600),
                status=FieldStatus.ANSWERED,
                source_text="llm_fallback_raw_text",
                confidence=0.30,
            )
        )

    for path in target_paths:
        if path == fallback_path:
            continue
        if should_mark_unknown_on_llm_failure(path):
            patches.append(
                Patch(
                    path=path,
                    value=None,
                    status=FieldStatus.UNKNOWN,
                    source_text="llm_fallback_unknown",
                    confidence=0.20,
                )
            )

    return patches


def apply_form_patches(
    case: RecoveryCase,
    patches: list[Patch],
    *,
    question_limit: int = 3,
    strict_document_completion: bool = True,
    today: date | None = None,
) -> Recover24CaseResult:
    """Apply already-structured Patch[] from forms/radios/selectboxes.

    This is the key bridge for the mixed-input UX: form patches and LLM patches
    both enter the same patching.py -> RecoveryCase -> document pipeline.
    """

    updated_case = apply_patches(case, patches)
    return build_case_result(
        updated_case,
        patches=patches,
        question_limit=question_limit,
        strict_document_completion=strict_document_completion,
        today=today,
    )


def build_case_result(
    case: RecoveryCase,
    *,
    patches: list[Patch] | None = None,
    question_limit: int = 3,
    strict_document_completion: bool = True,
    today: date | None = None,
) -> Recover24CaseResult:
    """Build the standard workflow snapshot for UI/CLI callers."""

    return Recover24CaseResult(
        case=case,
        patches=list(patches or []),
        questions=build_next_questions(case, limit=question_limit),
        readiness=evaluate_readiness(case, strict_document_completion=strict_document_completion),
        document_view=build_document_view(case, today=today),
    )


def render_case_html(
    case: RecoveryCase,
    *,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    today: date | None = None,
) -> str:
    """Render a case to official HTML through document_view.py + html_renderer.py."""

    return render_official_html(build_document_view(case, today=today), template_dir=template_dir)


def write_case_artifacts(
    result: Recover24CaseResult,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> dict[str, Path]:
    """Write JSON snapshot, readiness JSON, and official HTML to disk."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = result.case.case_id

    case_json = out / f"{base}_case.json"
    readiness_json = out / f"{base}_readiness.json"
    html_path = out / f"{base}_official_report.html"

    case_json.write_text(_json_dumps(result.to_dict()), encoding="utf-8")
    readiness_json.write_text(_json_dumps(result.readiness.to_dict()), encoding="utf-8")
    html_path.write_text(render_case_html(result.case, template_dir=template_dir), encoding="utf-8")

    return {"case_json": case_json, "readiness_json": readiness_json, "html": html_path}


def _new_case_id() -> str:
    return f"CASE-{uuid4().hex[:8].upper()}"


def _patch_to_dict(patch: Patch) -> dict[str, Any]:
    return {
        "path": patch.path,
        "value": patch.value.value if hasattr(patch.value, "value") else patch.value,
        "status": patch.status.value,
        "source_text": patch.source_text,
        "confidence": patch.confidence,
    }


def _question_to_dict(question: Question) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "category": question.category.value,
        "prompt": question.prompt,
        "target_paths": list(question.target_paths),
        "required": question.required,
        "input_type": question.input_type.value,
    }


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _json_patch(path: str, value: Any, source_text: str, confidence: float, *, status: str = "answered") -> dict[str, Any]:
    return {"path": path, "value": value, "status": status, "source_text": source_text, "confidence": confidence}


def _extract_amount_krw(text: str) -> int | None:
    match = re.search(r"(\d[\d,]*)\s*만\s*원", text)
    if match:
        return int(match.group(1).replace(",", "")) * 10_000
    match = re.search(r"(\d[\d,]*)\s*원", text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _extract_labeled_text(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*(?:은|는|:)?\s*([^,\.\n]+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_allowed_paths_from_prompt(prompt: str) -> list[str]:
    paths: list[str] = []
    capture = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Allowed patch paths for this answer"):
            capture = True
            continue
        if capture and stripped.startswith("Output format"):
            break
        if capture and stripped.startswith("- "):
            path = stripped[2:].split(":", 1)[0].strip()
            if path:
                paths.append(path)
    return paths


def _guess_value_for_path(path: str, answer: str) -> Any:
    # Applicant basics: parse comma-separated answers instead of writing the
    # whole sentence into every field.
    if path == "applicant.name":
        return _extract_person_name(answer)
    if path == "applicant.birth_date":
        return _extract_birth_date(answer)
    if path == "applicant.mobile_number":
        return _extract_mobile_number(answer)
    if path == "applicant.phone_number":
        return _extract_phone_number(answer)
    if path == "applicant.email":
        return _extract_email(answer)
    if path == "applicant.address":
        return _extract_address(answer)
    if path in {"applicant.memo", "narrative.incident_circumstances", "narrative.post_action", "incident.overview"}:
        return _shorten(answer, 300)

    if path.endswith("amount_krw"):
        return _extract_amount_krw(answer)
    if path.endswith("transaction_type"):
        if any(token in answer for token in ("모바일", "앱", "스마트폰")):
            return "mobile_banking_transfer"
        if "인터넷" in answer:
            return "internet_banking_transfer"
        if "ATM" in answer.upper():
            return "atm_transfer"
        return None
    if path.endswith("fraud_type"):
        if "자녀" in answer or "카톡" in answer or "카카오톡" in answer:
            return "family_impersonation"
        if "검찰" in answer or "경찰" in answer or "수사" in answer:
            return "authority_impersonation"
        if "대출" in answer:
            return "loan_scam"
        if "악성앱" in answer or "원격" in answer or "스미싱" in answer or "링크" in answer:
            return "smishing_malware"
        if "보이스피싱" in answer or "사기" in answer:
            return "other"
        return None
    if path.endswith("status") and path.startswith("evidence."):
        if any(token in answer for token in ("있", "보유", "첨부", "업로드")):
            return "available"
        if any(token in answer for token in ("추후", "나중", "예정")):
            return "planned"
        if any(token in answer for token in ("없", "삭제")):
            return "missing"
        return None
    if path.endswith(".note") and path.startswith("evidence."):
        return _shorten(answer, 300)
    if _looks_boolean_path(path):
        compact = re.sub(r"\s+", "", answer)
        if any(token in answer for token in ("네", "예", "맞", "동의", "했", "있")):
            return True
        if any(token in answer for token in ("아니", "아님", "안", "없", "미동의")) or any(
            token in compact for token in ("전부다아님", "전부아님", "모두아님", "다아님")
        ):
            return False
        return None

    # For unknown generic string fields, do not guess. Returning the full answer
    # here caused the same sentence to be copied into many unrelated fields.
    return None




def _split_answer_parts(answer: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,/|\n]+", answer) if part.strip()]


def _extract_person_name(answer: str) -> str | None:
    parts = _split_answer_parts(answer)
    if parts:
        first = parts[0]
        if re.fullmatch(r"[가-힣]{2,5}", first):
            return first
    match = re.search(r"(?:성명|이름)\s*(?:은|는|:)?\s*([가-힣]{2,5})", answer)
    if match:
        return match.group(1)
    return None


def _extract_birth_date(answer: str) -> str | None:
    match = re.search(r"(\d{4})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})", answer)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None


def _extract_mobile_number(answer: str) -> str | None:
    match = re.search(r"01\d[-\s]?\d{3,4}[-\s]?\d{4}", answer)
    if match:
        raw = re.sub(r"\D", "", match.group(0))
        return f"{raw[:3]}-{raw[3:-4]}-{raw[-4:]}"
    return None


def _extract_phone_number(answer: str) -> str | None:
    match = re.search(r"0(?:2|[3-6]\d|70)[-\s]?\d{3,4}[-\s]?\d{4}", answer)
    if match:
        raw = re.sub(r"\D", "", match.group(0))
        if raw.startswith("02"):
            return f"02-{raw[2:-4]}-{raw[-4:]}"
        return f"{raw[:3]}-{raw[3:-4]}-{raw[-4:]}"
    return None


def _extract_email(answer: str) -> str | None:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", answer)
    return match.group(0) if match else None


def _extract_address(answer: str) -> str | None:
    parts = _split_answer_parts(answer)
    address_markers = ("시", "군", "구", "동", "읍", "면", "로", "길", "번지", "아파트", "빌라")
    for part in reversed(parts):
        if any(marker in part for marker in address_markers) and not re.fullmatch(r"[가-힣]{2,5}", part):
            return part
    match = re.search(r"(?:주소)\s*(?:은|는|:)?\s*(.+)$", answer)
    if match:
        return match.group(1).strip()
    return None
def _looks_boolean_path(path: str) -> bool:
    return any(
        token in path
        for token in (
            "consent.",
            "sms_consent",
            "reported",
            "clicked",
            "installed",
            "provided_",
            "used",
            "id_lent",
            "phone_lent",
            "enabled",
            "stored",
            "proxy_used",
            "exclude_",
            "final_has_exclusion",
        )
    )


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Recover24 V3 case workflow from the command line.")
    parser.add_argument("--statement", default="검찰 사칭 전화를 받고 모바일뱅킹으로 100만원을 보냈어요. 아직 경찰 신고는 못 했어요.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--question-limit", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--no-write", action="store_true", help="Do not write output artifacts to disk.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    provider = DemoKeywordProvider()
    result = start_case(
        args.statement,
        provider,
        case_id=args.case_id,
        question_limit=args.question_limit,
    )

    print(f"case_id: {result.case.case_id}")
    print(f"readiness: {result.readiness.status.value}")
    print(f"completion: {result.readiness.document_completion_rate:.1%}")
    print("next_questions:")
    for index, question in enumerate(result.questions, start=1):
        print(f"  {index}. [{question.category.value}] {question.prompt}")

    if not args.no_write:
        paths = write_case_artifacts(result, output_dir=args.output_dir, template_dir=args.template_dir)
        print("written:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
