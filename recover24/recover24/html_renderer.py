"""Render official Recover24 HTML from an OfficialDocumentView.

Rule: renderer must not infer facts, calculate values, mutate cases, or call an LLM.
It only injects the already-built document view into the Jinja template.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .document_view import OfficialDocumentView, build_document_view
from .models import RecoveryCase

DEFAULT_TEMPLATE_NAME = "recover24_official_report_v1.html"


class HtmlRenderError(RuntimeError):
    """Raised when an official report HTML cannot be rendered safely."""


def render_official_html(
    view: OfficialDocumentView | dict[str, Any],
    template_dir: str | Path = "templates",
    template_name: str = DEFAULT_TEMPLATE_NAME,
) -> str:
    """Render an already-built document view into official HTML.

    This function deliberately accepts view data, not raw RecoveryCase logic.
    The only convenience conversion is OfficialDocumentView.to_dict(). All labels,
    checkboxes, money strings, and status text must already be prepared by
    document_view.py.
    """

    context = view.to_dict() if isinstance(view, OfficialDocumentView) else view
    env = _build_environment(template_dir)
    template = env.get_template(template_name)
    html = template.render(**context)
    _assert_no_unresolved_jinja_markers(html)
    return html


def render_official_case_html(
    case: RecoveryCase,
    template_dir: str | Path = "templates",
    template_name: str = DEFAULT_TEMPLATE_NAME,
    today: date | None = None,
) -> str:
    """Convenience wrapper: RecoveryCase -> document_view.py -> HTML.

    The conversion remains delegated to document_view.py, so renderer policy stays
    simple and deterministic.
    """

    return render_official_html(build_document_view(case, today=today), template_dir=template_dir, template_name=template_name)


def _build_environment(template_dir: str | Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(enabled_extensions=("html", "xml"), default_for_string=False),
        trim_blocks=False,
        lstrip_blocks=False,
    )


def _assert_no_unresolved_jinja_markers(html: str) -> None:
    # StrictUndefined catches missing variables, and this catches accidental raw
    # Jinja placeholders left in output when templates are edited incorrectly.
    if "{{" in html or "}}" in html or "{%" in html or "%}" in html:
        raise HtmlRenderError("Rendered HTML still contains unresolved Jinja markers")
