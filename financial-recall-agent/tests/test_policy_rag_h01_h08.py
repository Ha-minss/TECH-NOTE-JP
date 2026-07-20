from src.recall_agent.policy_rag.policy_basis_retriever import (
    basis_ids_for_harm_type,
    retrieve_policy_basis,
    search_policy_basis,
    validate_h01_h08_coverage,
)


def test_h01_h08_map_has_all_harm_types():
    coverage = validate_h01_h08_coverage()

    assert set(coverage["covered_harm_types"]) == {
        "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08"
    }
    assert coverage["is_complete_for_id_lookup"] is True


def test_h07_basis_is_connected_for_demo():
    basis_ids = basis_ids_for_harm_type("H07")
    rows = retrieve_policy_basis(basis_ids)

    assert len(rows) == 4
    assert all(row["retrieval_status"] == "FOUND" for row in rows)
    assert all(row["usable_as_demo_basis"] is True for row in rows)
    assert any(row["basis_id"] == "H07-BASIS-RATE" for row in rows)
    assert any("익월 27일" in (row.get("source_text") or "") for row in rows)


def test_h08_basis_is_demo_safe_not_real_policy_final():
    basis_ids = basis_ids_for_harm_type("H08")
    rows = retrieve_policy_basis(basis_ids)

    assert len(rows) == 3
    assert all(row["retrieval_status"] == "FOUND" for row in rows)
    assert any(row["source_type"] == "FICTIONAL_DEMO_INTERNAL_RULE" for row in rows)
    assert all(row["usable_as_real_policy_evidence"] is False for row in rows)


def test_h01_placeholder_requires_human_review():
    rows = retrieve_policy_basis(basis_ids_for_harm_type("H01"))

    assert len(rows) == 3
    assert all(row["requires_human_review"] is True for row in rows)
    assert all(row["source_status"] == "SOURCE_REQUIRED_NOT_CONNECTED" for row in rows)


def test_search_policy_basis_h07():
    rows = search_policy_basis("캐시백 익월 27일 지급", harm_type="H07", top_k=3)

    assert rows
    assert any(row["basis_id"] == "H07-BASIS-PAYMENT-DATE" for row in rows)
    assert all(row.get("retrieval_backend") for row in rows)
