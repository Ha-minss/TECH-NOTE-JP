"""Demo entrypoint for the executable H07 Reward Missing MVP.

H07/Smart Cashback defaults live here, not in core.rule_runner. This keeps the
core runner product-agnostic while preserving a one-command demo path.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.recall_agent.application import run_rule
from src.recall_agent.core.demo_paths import DEMO_BUNDLE_PATH
from src.recall_agent.core.reporting import print_report_summary

H07_RULE_ID = "H07-REWARD-MISSING-TEMPLATE"
H07_PRODUCT_CONFIG_ID = "JB_SMART_CASHBACK_CHECK__2022-07__v2"
H07_BUNDLE_PATH = DEMO_BUNDLE_PATH


def run_h07_demo(
    complaint_id: str,
    *,
    dataset_dir: str | None = None,
    max_customer_rows: int = 50,
    audit_log_path: str | None = None,
    dev_mode: bool | None = None,
) -> dict[str, Any]:
    """Run the approved H07 demo bundle for one complaint."""
    return run_rule(
        rule_id=H07_RULE_ID,
        complaint_id=complaint_id,
        product_config_id=H07_PRODUCT_CONFIG_ID,
        bundle_path=H07_BUNDLE_PATH,
        dataset_base_path=dataset_dir,
        max_customer_rows=max_customer_rows,
        audit_log_path=audit_log_path,
        dev_mode=dev_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H07 Reward Missing demo.")
    parser.add_argument("complaint_id", help="Complaint ID, for example CP0001.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--max-customer-rows", type=int, default=50)
    parser.add_argument("--audit-log-path", default=None)
    parser.add_argument("--dev-mode", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_h07_demo(
        args.complaint_id,
        dataset_dir=args.dataset_dir,
        max_customer_rows=args.max_customer_rows,
        audit_log_path=args.audit_log_path,
        dev_mode=args.dev_mode,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report_summary(report)


if __name__ == "__main__":
    main()


