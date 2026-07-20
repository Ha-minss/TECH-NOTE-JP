"""Tests for OfficialDocumentView -> HTML."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from recover24.document_view import build_document_view
from recover24.html_renderer import render_official_case_html, render_official_html
from recover24.models import FieldValue, RecoveryCase


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def test_render_official_html_uses_view_values_only():
    case = RecoveryCase.new("CASE-HTML-001")
    case.applicant.name = FieldValue.answered("김영희")
    view = build_document_view(case, today=date(2026, 6, 22))

    html = render_official_html(view, template_dir=TEMPLATE_DIR)

    assert "김영희" in html
    assert "전자금융거래 사고 피해 신고서" in html
    assert "{{" not in html
    assert "{%" not in html


def test_render_official_case_html_is_convenience_wrapper_only():
    case = RecoveryCase.new("CASE-HTML-002")
    case.applicant.name = FieldValue.answered("박동하")

    html = render_official_case_html(case, template_dir=TEMPLATE_DIR, today=date(2026, 6, 22))

    assert "박동하" in html
    assert "전자금융거래 사고 피해 신고서" in html
    assert "{{" not in html
