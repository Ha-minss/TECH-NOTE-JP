from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from src.recall_agent.application.complaint_investigation import (
    ComplaintInvestigationResult,
    ProductVerificationResult,
    RuleExecutionResult,
    investigate_complaint,
)
from src.recall_agent.core.bundle_loader import DEFAULT_BUNDLE_DIR
from src.recall_agent.core.registry_loader import DEFAULT_RULE_REGISTRY_PATH
from src.recall_agent.interfaces.cli.h07_reward_missing_demo import H07_BUNDLE_PATH


def test_demo_paths_live_under_data_demo():
    assert H07_BUNDLE_PATH == "data/demo/rules/bundles/h07_reward_missing_mvp.json"
    assert DEFAULT_BUNDLE_DIR == "data/demo/rules/bundles"
    assert DEFAULT_RULE_REGISTRY_PATH == "data/demo/rules/rule_registry.json"


def test_investigation_executes_h07_through_direct_service(monkeypatch, tmp_path):
    complaints_path = tmp_path / "complaints.csv"
    complaints_path.write_text(
        "\n".join(
            [
                "complaint_id,customer_id,channel,product_id_claimed,narrative_ko",
                "CP0001,C0001,APP,JB_SMART_CASHBACK_CHECK,missing reward",
            ]
        ),
        encoding="utf-8",
    )
    contracts_path = tmp_path / "card_contracts.csv"
    contracts_path.write_text(
        "\n".join(
            [
                "customer_id,product_id,status",
                "C0001,JB_SMART_CASHBACK_CHECK,ACTIVE",
            ]
        ),
        encoding="utf-8",
    )

    router = SimpleNamespace(
        route=lambda complaint: SimpleNamespace(
            schema_valid=True,
            result=SimpleNamespace(
                route=SimpleNamespace(name="H07_CANDIDATE"),
                model_dump=lambda mode="json": {"route": "H07_CANDIDATE"},
            ),
        )
    )

    monkeypatch.setattr(
        "src.recall_agent.application.complaint_investigation.should_enter_product_verification",
        lambda route: True,
    )
    monkeypatch.setattr(
        "src.recall_agent.application.complaint_investigation.should_run_h07_rule",
        lambda **kwargs: True,
    )

    captured: dict[str, object] = {}

    def fake_run_h07_demo(
        complaint_id: str,
        *,
        dataset_dir: str | None = None,
        max_customer_rows: int = 50,
        audit_log_path: str | None = None,
        dev_mode: bool | None = None,
    ) -> dict[str, object]:
        captured["complaint_id"] = complaint_id
        captured["dataset_dir"] = dataset_dir
        captured["max_customer_rows"] = max_customer_rows
        captured["audit_log_path"] = audit_log_path
        captured["dev_mode"] = dev_mode
        return {"status": "ok"}

    monkeypatch.setattr(
        "src.recall_agent.application.complaint_investigation.run_h07_demo",
        fake_run_h07_demo,
    )

    result = investigate_complaint(
        complaint_id="CP0001",
        router=router,
        complaints_path=complaints_path,
        card_contracts_path=contracts_path,
        dataset_dir=tmp_path,
    )

    assert captured == {
        "complaint_id": "CP0001",
        "dataset_dir": str(tmp_path),
        "max_customer_rows": 50,
        "audit_log_path": None,
        "dev_mode": True,
    }
    assert result.should_run_h07_rule is True
    assert result.rule_execution["service"] == "run_h07_demo"
