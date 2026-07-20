from __future__ import annotations

import glob
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]  # repo root (Refugees_Crime_Panel)


def run(cmd: str) -> None:
    print("\n" + "=" * 100)
    print("[RUN]", cmd)
    subprocess.check_call(cmd, cwd=str(ROOT), shell=True)


def show_all_tables() -> None:
    paths = sorted(glob.glob(str(ROOT / "outputs" / "tables" / "*.csv")))
    print("\n" + "=" * 100)
    print(f"[INFO] Found {len(paths)} table(s) in outputs/tables/")
    for p in paths:
        df = pd.read_csv(p)
        rel = str(Path(p).relative_to(ROOT))
        print("\n" + "-" * 100)
        print(f"[TABLE] {rel}   shape={df.shape}")
        print(df.to_string(index=False))


def main() -> None:
    # 1) reproduce all outputs
    run("python scripts/00_prepare_panel.py")
    run("python scripts/01_main_twfe_refugees_only.py")
    run("python scripts/06_make_coefplot.py")
    run("python scripts/02_dynamic_lags_leads_placebo.py")
    run("python scripts/03_robustness_country_trends.py")
    run("python scripts/04_robustness_drop_years.py")
    run("python scripts/05_first_difference.py")

    # 2) print all saved tables
    show_all_tables()

    print("\n" + "=" * 100)
    print("[DONE] Reproduced + printed all results.")


if __name__ == "__main__":
    main()
