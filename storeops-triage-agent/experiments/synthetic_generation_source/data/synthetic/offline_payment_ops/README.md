# Offline Payment Ops Synthetic 50 Dataset

이 데이터셋은 StoreOps 결제 장애 triage agent의 데이터 기반 판단을 평가하기 위한 50개 합성 운영 사건입니다.

## 구성

- `synthetic_case_plan_50.json`: 사람이 검토하는 case 설계표
- `raw/*.csv`: 에이전트 tool이 조회할 운영 raw fact 테이블
- `../../fixtures/offline_payment_ops_synthetic_50.sqlite3`: raw CSV를 적재한 SQLite fixture
- `../../golden/offline_payment_ops_cases_50.json`: 평가기가 보는 정답 label
- `../../evaluation/retrieval_cases_50.json`: RAG retrieval 평가 케이스
- `../../evaluation/planner_cases_50.json`: planner/tool 선택 평가 케이스

## 분포

- S1 duplicate_tid: 10
- S2 terminal_identifier_mismatch: 7
- S3 van_merchant_registration_missing: 7
- S4 pos_front_connection_issue: 7
- S5 clarification_required: 7
- S6A required_tool_failure: 4
- S6B optional_tool_failure: 3
- S7 temporal_conflict: 5

## 중요한 원칙

운영 raw CSV/SQLite에는 `expected_primary_cause` 같은 정답 원인을 넣지 않습니다. 정답은 `golden` JSON과 검증 리포트에만 있습니다.
