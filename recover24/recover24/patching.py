"""Apply Patch objects to a RecoveryCase.

patching.py is the only runtime module allowed to change RecoveryCase state.

Rules:
- Do not parse Korean text here.
- Do not call an LLM here.
- Do not infer missing facts here.
- Do not generate questions or HTML here.
- Only follow Patch.path and write FieldValue objects to the case.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import is_dataclass
from typing import Any

from .models import EvidenceItem, EvidenceStatus, FieldStatus, FieldValue, Patch, RecoveryCase, Transaction


class PatchPathError(ValueError):
    """Raised when a Patch path cannot be applied to RecoveryCase."""


class PatchTargetError(TypeError):
    """Raised when a Patch points to something that is not a FieldValue."""


def apply_patches(case: RecoveryCase, patches: list[Patch] | tuple[Patch, ...]) -> RecoveryCase:
    """Return a new RecoveryCase with all patches applied.

    The input case is not mutated. This makes debugging/evaluation easier:
    old_case -> apply_patches(...) -> new_case.
    """

    next_case = deepcopy(case)
    for patch in patches:
        apply_patch_in_place(next_case, patch)
    return next_case


def apply_patch(case: RecoveryCase, patch: Patch) -> RecoveryCase:
    """Return a new RecoveryCase with one patch applied."""

    return apply_patches(case, [patch])


def apply_patch_in_place(case: RecoveryCase, patch: Patch) -> None:
    """Apply one patch to an existing RecoveryCase object.

    Most application code should prefer apply_patches() because it returns a copy.
    This lower-level function exists for controlled internal use and tests.
    """

    parts = _split_path(patch.path)

    if _is_evidence_patch(parts):
        _apply_evidence_patch(case, parts, patch)
        return

    _ensure_supported_containers(case, parts)

    parent, field_name = _resolve_parent(case, parts)
    current = _read_child(parent, field_name)

    if not isinstance(current, FieldValue):
        raise PatchTargetError(
            f"Patch path '{patch.path}' resolved to {type(current).__name__}, not FieldValue"
        )

    _write_child(parent, field_name, _field_value_from_patch(patch))


def _is_evidence_patch(parts: list[str]) -> bool:
    return len(parts) == 3 and parts[0] == "evidence" and parts[2] in {"status", "note"}


def _apply_evidence_patch(case: RecoveryCase, parts: list[str], patch: Patch) -> None:
    """Apply patches such as evidence.id_card_copy.status.

    Evidence is modeled as a list[EvidenceItem], not FieldValue fields, because
    Page 7 is an attachment checklist. We still let extraction.py/answers.py
    express evidence updates as Patch objects so state changes keep one door.
    """

    kind = parts[1]
    field_name = parts[2]
    item = _get_or_create_evidence_item(case, kind)

    if field_name == "status":
        if isinstance(patch.value, EvidenceStatus):
            item.status = patch.value
        elif isinstance(patch.value, str):
            try:
                item.status = EvidenceStatus(patch.value)
            except ValueError as exc:
                raise PatchTargetError(f"Unsupported evidence status: {patch.value!r}") from exc
        else:
            raise PatchTargetError(f"Evidence status patch needs EvidenceStatus or str, got {type(patch.value).__name__}")
    elif field_name == "note":
        item.note = None if patch.value is None else str(patch.value)

    if patch.source_text:
        item.source_text = patch.source_text


def _get_or_create_evidence_item(case: RecoveryCase, kind: str) -> EvidenceItem:
    for item in case.evidence:
        if item.kind == kind:
            return item

    item = EvidenceItem(kind=kind)
    case.evidence.append(item)
    return item


def _split_path(path: str) -> list[str]:
    if not isinstance(path, str) or not path.strip():
        raise PatchPathError("Patch path must be a non-empty string")

    parts = [part.strip() for part in path.split(".")]
    if any(not part for part in parts):
        raise PatchPathError(f"Invalid patch path: {path!r}")
    return parts


def _ensure_supported_containers(case: RecoveryCase, parts: list[str]) -> None:
    """Create supported list containers before resolving a path.

    V3 currently supports dynamic transaction rows such as:
    - transactions.0.amount_krw
    - transactions.1.amount_krw

    Other dynamic containers can be added here later, deliberately.
    """

    if len(parts) >= 3 and parts[0] == "transactions" and parts[1].isdigit():
        index = int(parts[1])
        while len(case.transactions) <= index:
            case.transactions.append(Transaction())


def _resolve_parent(root: Any, parts: list[str]) -> tuple[Any, str]:
    """Resolve all path parts except the last one."""

    current = root
    for part in parts[:-1]:
        current = _read_child(current, part)
    return current, parts[-1]


def _read_child(parent: Any, key: str) -> Any:
    if isinstance(parent, list):
        if not key.isdigit():
            raise PatchPathError(f"List path segment must be numeric, got {key!r}")
        index = int(key)
        try:
            return parent[index]
        except IndexError as exc:
            raise PatchPathError(f"List index out of range: {index}") from exc

    if isinstance(parent, dict):
        if key not in parent:
            raise PatchPathError(f"Unknown dict key in patch path: {key!r}")
        return parent[key]

    if is_dataclass(parent) and hasattr(parent, key):
        return getattr(parent, key)

    raise PatchPathError(f"Cannot resolve path segment {key!r} on {type(parent).__name__}")


def _write_child(parent: Any, key: str, value: Any) -> None:
    if isinstance(parent, list):
        if not key.isdigit():
            raise PatchPathError(f"List path segment must be numeric, got {key!r}")
        parent[int(key)] = value
        return

    if isinstance(parent, dict):
        if key not in parent:
            raise PatchPathError(f"Unknown dict key in patch path: {key!r}")
        parent[key] = value
        return

    if is_dataclass(parent) and hasattr(parent, key):
        setattr(parent, key, value)
        return

    raise PatchPathError(f"Cannot write path segment {key!r} on {type(parent).__name__}")


def _field_value_from_patch(patch: Patch) -> FieldValue[Any]:
    """Convert a Patch into a FieldValue to store in RecoveryCase."""

    if patch.status == FieldStatus.ANSWERED:
        return FieldValue(value=patch.value, status=FieldStatus.ANSWERED, source_text=patch.source_text)

    if patch.status == FieldStatus.UNKNOWN:
        return FieldValue(value=None, status=FieldStatus.UNKNOWN, source_text=patch.source_text)

    if patch.status == FieldStatus.NOT_APPLICABLE:
        return FieldValue(value=None, status=FieldStatus.NOT_APPLICABLE, source_text=patch.source_text)

    if patch.status == FieldStatus.NOT_ASKED:
        return FieldValue(value=None, status=FieldStatus.NOT_ASKED, source_text=patch.source_text)

    raise PatchTargetError(f"Unsupported patch status: {patch.status!r}")
