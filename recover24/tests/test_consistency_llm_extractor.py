from evaluation.consistency.conflict_checker import check_consistency
from evaluation.consistency.llm_extractor import extract_statement_facts_llm


class FakeProvider:
    def generate_json(self, prompt: str) -> str:
        return '{"statement_facts":{"damage_amount_krw":30000000,"police_status":"reported"}}'


def test_llm_extractor_feeds_deterministic_conflict_checker():
    facts = extract_statement_facts_llm(
        "총 3000만원을 송금했고 경찰에 신고했습니다.",
        FakeProvider(),
    )

    assert facts == {
        "damage_amount_krw": 30000000,
        "police_status": "reported",
    }

    result = check_consistency(
        form_facts={
            "damage_amount_krw": 17500000,
            "police_status": "not_reported",
        },
        statement_facts=facts,
    )

    assert result["can_generate_document"] is False
    assert {item["field"] for item in result["conflicts"]} == {
        "damage_amount_krw",
        "police_status",
    }
