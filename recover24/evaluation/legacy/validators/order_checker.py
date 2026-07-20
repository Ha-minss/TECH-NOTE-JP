"""Event-order scoring utilities."""

from __future__ import annotations


def order_score(expected: list[str], predicted: list[str]) -> dict[str, float | int | None]:
    """Return a conservative order score.

    The score is based on pairwise ordering among expected events that also
    appear in predicted. If fewer than two expected events are extracted, the
    score is None because there is not enough evidence to judge order.
    """
    if not expected or not predicted:
        return {"score": None, "matched_events": 0, "ordered_pairs": 0, "correct_pairs": 0}
    pos = {event: i for i, event in enumerate(predicted)}
    matched = [event for event in expected if event in pos]
    if len(matched) < 2:
        return {"score": None, "matched_events": len(matched), "ordered_pairs": 0, "correct_pairs": 0}
    total = 0
    correct = 0
    for i, left in enumerate(matched):
        for right in matched[i + 1:]:
            total += 1
            if pos[left] < pos[right]:
                correct += 1
    return {
        "score": correct / total if total else None,
        "matched_events": len(matched),
        "ordered_pairs": total,
        "correct_pairs": correct,
    }
