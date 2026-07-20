# Reward Missing Template

This template implements the active MVP path for H07 reward/cashback/point/mileage missing cases.

It is intentionally product-agnostic:

- the template owns the reconciliation pattern,
- Product Config owns product-specific policy,
- the approved bundle owns allowed product configs and SQL hashes.

Current executable product config: `JB_SMART_CASHBACK_CHECK__2022-07__v2`.
