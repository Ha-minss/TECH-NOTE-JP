from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gamelytics.inference import bootstrap_mean_difference, permutation_test_mean_difference
from gamelytics.io import read_semicolon_csv
from gamelytics.metrics import group_summary
from gamelytics.retention import exact_day_retention
from gamelytics.sensitivity import common_winsorize
from gamelytics.validation import (
    assert_single_group_per_user,
    sample_ratio_mismatch_test,
)


def test_reads_semicolon_csv(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("user_id;revenue;testgroup\n1;0;a\n2;10;b\n", encoding="utf-8")

    df = read_semicolon_csv(path)

    assert list(df.columns) == ["user_id", "revenue", "testgroup"]
    assert df.loc[1, "revenue"] == 10


def test_rejects_user_in_multiple_testgroups():
    df = pd.DataFrame(
        {"user_id": [1, 1, 2], "testgroup": ["a", "b", "a"], "revenue": [0, 1, 0]}
    )

    with pytest.raises(ValueError, match="multiple test groups"):
        assert_single_group_per_user(df)


def test_group_summary_uses_assigned_users_and_arpu_identity():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "revenue": [0, 100, 0, 300],
            "testgroup": ["a", "a", "b", "b"],
        }
    )

    summary = group_summary(df)

    assert summary.loc["a", "arpu"] == 50
    assert summary.loc["b", "conversion_rate"] == 0.5
    assert summary.loc["b", "arppu"] == 300
    assert summary.loc["b", "arpu"] == pytest.approx(
        summary.loc["b", "conversion_rate"] * summary.loc["b", "arppu"]
    )


def test_bootstrap_mean_difference_is_reproducible():
    a = np.array([0, 0, 10, 10])
    b = np.array([0, 10, 10, 20])

    result1 = bootstrap_mean_difference(a, b, iterations=200, seed=7)
    result2 = bootstrap_mean_difference(a, b, iterations=200, seed=7)

    assert result1.observed_diff == pytest.approx(5.0)
    assert result1.ci_low <= result1.observed_diff <= result1.ci_high
    assert np.array_equal(result1.samples, result2.samples)


def test_permutation_test_does_not_over_reject_same_distribution():
    values = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    a = values[:4]
    b = values[4:]

    result = permutation_test_mean_difference(a, b, iterations=500, seed=42)

    assert 0 <= result.p_value <= 1
    assert result.p_value > 0.01


def test_common_winsorize_uses_shared_cutoff():
    a = pd.Series([1, 2, 100])
    b = pd.Series([1, 2, 3])

    wa, wb, cutoff = common_winsorize(a, b, upper_quantile=0.8)

    assert cutoff == pytest.approx(3.0)
    assert wa.max() == pytest.approx(3.0)
    assert wb.max() == pytest.approx(3.0)


def test_srm_chi_square_for_balanced_groups():
    result = sample_ratio_mismatch_test({"a": 50, "b": 50})

    assert result["chi_square"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_latest_cohorts_excluded_from_mature_retention_denominator():
    reg = pd.DataFrame(
        {
            "uid": [1, 2],
            "reg_ts": [1577836800, 1578614400],
        }
    )
    auth = pd.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "auth_ts": [1577836800, 1578441600, 1578614400, 1578700800],
        }
    )

    retention = exact_day_retention(reg, auth, days=[1, 7], observation_end="2020-01-12")

    assert retention.loc[1, "eligible_users"] == 2
    assert retention.loc[7, "eligible_users"] == 1
    assert retention.loc[7, "retained_users"] == 1
