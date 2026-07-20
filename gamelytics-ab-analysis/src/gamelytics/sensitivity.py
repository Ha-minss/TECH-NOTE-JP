from __future__ import annotations

import pandas as pd

from gamelytics.inference import bootstrap_mean_difference


def common_winsorize(
    a: pd.Series, b: pd.Series, upper_quantile: float
) -> tuple[pd.Series, pd.Series, float]:
    combined = pd.concat([a, b], ignore_index=True)
    cutoff = float(combined.quantile(upper_quantile, interpolation="lower"))
    return a.clip(upper=cutoff), b.clip(upper=cutoff), cutoff



def sensitivity_table(
    df: pd.DataFrame,
    iterations: int = 5000,
    seed: int = 42,
    control: str = "a",
    treatment: str = "b",
) -> pd.DataFrame:
    rows = []
    scenarios = [("raw_all_users", df.copy(), None)]
    a = df.loc[df["testgroup"] == control, "revenue"]
    b = df.loc[df["testgroup"] == treatment, "revenue"]
    for q in (0.999, 0.995):
        wa, wb, cutoff = common_winsorize(a, b, q)
        scenario = df.copy()
        scenario.loc[scenario["testgroup"] == control, "revenue"] = wa.values
        scenario.loc[scenario["testgroup"] == treatment, "revenue"] = wb.values
        scenarios.append((f"common_top_{(1-q)*100:.1f}pct_winsorized", scenario, cutoff))

    for name, scenario, cutoff in scenarios:
        a_values = scenario.loc[scenario["testgroup"] == control, "revenue"].to_numpy()
        b_values = scenario.loc[scenario["testgroup"] == treatment, "revenue"].to_numpy()
        result = bootstrap_mean_difference(a_values, b_values, iterations=iterations, seed=seed)
        arpu_a = float(a_values.mean())
        arpu_b = float(b_values.mean())
        rows.append(
            {
                "scenario": name,
                "cutoff": cutoff,
                "arpu_a": arpu_a,
                "arpu_b": arpu_b,
                "b_minus_a": result.observed_diff,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "direction": "B observed ARPU higher" if result.observed_diff > 0 else ("A observed ARPU higher" if result.observed_diff < 0 else "No observed difference"),
            }
        )
    return pd.DataFrame(rows)
