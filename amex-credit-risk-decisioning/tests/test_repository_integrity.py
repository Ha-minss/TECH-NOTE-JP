from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_internal_links_and_paths_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    internal_links = [link for link in links if not link.startswith(("http://", "https://"))]
    missing = [link for link in internal_links if not (ROOT / link).exists()]
    assert not missing


def test_no_large_or_private_files_are_committed() -> None:
    forbidden_suffixes = {".parquet", ".pkl", ".joblib", ".cbm", ".pt", ".duckdb", ".zip"}
    offenders = []
    large_files = []
    for path in ROOT.rglob("*"):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in forbidden_suffixes:
                offenders.append(rel)
            if path.stat().st_size > 1_000_000:
                large_files.append(rel)

    assert offenders == []
    assert large_files == []


def test_no_private_paths_kaggle_keys_or_raw_customer_ids_in_public_files() -> None:
    patterns = [
        "C:/Us" + "ers/",
        "C:\\Us" + "ers\\",
        "kaggle" + ".json",
        "KAGGLE" + "_KEY",
        "0f75afc82a612c9b35f3d54b03c2ecd5d24533ccefc62c" + "251fb29bc40de3ae5e",
    ]
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".csv", ".yaml", ".yml", ".json", ".ipynb", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                if pattern in text:
                    offenders.append(f"{path.relative_to(ROOT).as_posix()} contains {pattern}")

    assert offenders == []


def test_sample_file_is_synthetic_not_public_top1000() -> None:
    assert not (ROOT / "data" / "sample" / "public_sample_scores.csv").exists()
    sample = ROOT / "data" / "sample" / "synthetic_scores.csv"
    assert sample.exists()
    text = sample.read_text(encoding="utf-8")
    assert "synthetic_test_fixture" in text



