# Financial Recall Rule Assets

## Asset boundaries

| Asset | Responsibility |
|---|---|
| `product_policy_configs/` | Versioned contractual product conditions |
| `rule_registry.json` | Controls routing, execution, data access, output, audit, and failure behavior for approved rules |
| `data_contracts/` | Declares required source tables, columns, keys, and joins. Active H07 contract: `h07_synthetic_v3.json` |
| `governance/` | Defines LLM limits, human approvals, audit requirements, and failure policy |
| `rule_assets_validation_report.json` | Latest automated integrity-check result |

## Active H07 product policy

The active MVP configuration is:

`product_policy_configs/JB_SMART_CASHBACK_CHECK__2022-07__v2.json`

It independently recalculates the expected cashback from the product policy,
then reconciles that result with the bank-system expected reward and actual
reward ledger. It is approved for the MVP demo only and is not approved for
production use.

The active registry uses the `RECALL_RULE_REGISTRY` V2 contract. It permits the
LLM to recommend a registered `rule_id`, while deterministic execution is
restricted to active rules, approved repositories, and approved product
configs.

## Validation

```powershell
python -m src.recall_agent.rule_assets.validate_rule_assets
python -m pytest tests/test_rule_assets.py -v -p no:cacheprovider
```

The runtime investigation engine must not read `private_ground_truth`.
