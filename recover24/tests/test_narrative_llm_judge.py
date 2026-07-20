from evaluation.narrative.judge import evaluate_narrative_with_llm


class FakeJudgeProvider:
    def generate_json(self, prompt: str) -> str:
        return """
{
  "elements": [
    {"id": "fraud_type", "label": "present", "evidence": "대출사기"},
    {"id": "amount", "label": "present", "evidence": "17,500,000원"},
    {"id": "police_status", "label": "missing", "evidence": ""}
  ],
  "unsupported_claims": [],
  "contradictions": []
}
"""


def test_llm_judge_scores_required_elements_by_id():
    result = evaluate_narrative_with_llm(
        canonical_case={
            "damage_amount_krw": 17500000,
            "fraud_type": "loan_scam",
        },
        generated_text="대출사기 피해로 17,500,000원 손해가 발생했습니다.",
        required_elements=[
            {"id": "fraud_type", "description": "대출사기 유형을 언급해야 함"},
            {"id": "amount", "description": "피해금액을 보존해야 함"},
            {"id": "police_status", "description": "경찰 신고 여부를 언급해야 함"},
        ],
        provider=FakeJudgeProvider(),
    )

    assert result["included_elements"] == ["fraud_type", "amount"]
    assert result["missing_elements"] == ["police_status"]
    assert result["judge_method"] == "llm"
    assert result["passed"] is False
