from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from gamelytics.io import load_retention_data
from gamelytics.report import write_retention_report
from gamelytics.retention import exact_day_retention, monthly_retention_heatmap_data
from gamelytics.visualization import plot_retention_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    tables = reports / "tables"
    figures = reports / "figures"
    reports.mkdir(exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    reg, auth = load_retention_data(args.data_dir)
    summary = exact_day_retention(reg, auth, days=[1, 2, 3, 7, 14, 30])
    heatmap_data = monthly_retention_heatmap_data(reg, auth, days=[1, 7, 14, 30])
    summary.to_csv(tables / "retention_summary.csv")
    heatmap_data.to_csv(tables / "retention_monthly_heatmap_data.csv", index=False)
    plot_retention_heatmap(heatmap_data, figures / "monthly_retention_heatmap")
    write_retention_report(reports / "retention_analysis.md", summary, heatmap_data)


if __name__ == "__main__":
    main()
