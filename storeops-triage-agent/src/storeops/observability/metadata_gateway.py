"""Tool gateway that enriches responses with provenance and freshness."""

from __future__ import annotations

from datetime import datetime

from storeops.domains.offline_payment_ops.scenario_runtime import (
    OfflinePaymentScenarioGateway,
    seed_offline_payment_scenarios,
)


class MetadataScenarioGateway(OfflinePaymentScenarioGateway):
    """Scenario gateway with record provenance and freshness classification."""

    def _query(self, **kwargs):
        response = super()._query(**kwargs)
        if not response.data:
            return response

        response.provenance = [
            f"{response.tool_name}:{next(iter(row.values()))}"
            for row in response.data
        ]

        delays = []
        for row in response.data:
            observed_at = row.get("observed_at")
            available_at = row.get("available_at")
            if not observed_at or not available_at:
                continue
            observed = datetime.fromisoformat(str(observed_at))
            available = datetime.fromisoformat(str(available_at))
            delays.append(available - observed)

        if any(delay.total_seconds() >= 24 * 60 * 60 for delay in delays):
            response.freshness = "delayed"
            response.warnings.append(
                "Source data became available at least 24 hours after observation."
            )
        elif delays:
            response.freshness = "current"
        else:
            response.freshness = "unknown"
            response.warnings.append(
                "Source rows do not expose freshness timestamps."
            )
        return response


__all__ = ["MetadataScenarioGateway", "seed_offline_payment_scenarios"]
