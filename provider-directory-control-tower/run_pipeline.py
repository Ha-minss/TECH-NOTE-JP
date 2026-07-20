from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import ProviderDirectoryPipeline
from src.repository import JsonlDirectoryRepository
from src.report import (
    export_audit,
    export_evidence,
    export_executive_summary,
    export_recommendations,
    export_connector_diagnostics,
)
from src.utils import read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Provider Directory Control Tower MVP")
    parser.add_argument("--input", default="data/input/provider_records.jsonl", help="Input provider records JSONL")
    parser.add_argument("--config", default="configs/pipeline_config.json", help="Pipeline config JSON")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--use-real-npi", action="store_true", help="Also query public NPI Registry API v2.1")
    parser.add_argument("--use-cms", action="store_true", help="Also query CMS Provider Data whitelist by NPI")
    parser.add_argument("--cms-source", choices=["minimal", "ffs", "revoked", "facility", "all"], default=None, help="CMS source mode: minimal=FFS+Revoked, facility=facility datasets+Revoked, all=all whitelisted datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.config)
    if args.cms_source:
        config["cms_source_mode"] = args.cms_source
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = JsonlDirectoryRepository(args.input, output_dir / "recommendations.jsonl")
    records = repo.load_records()

    pipeline = ProviderDirectoryPipeline(
        config=config,
        use_real_npi=args.use_real_npi,
        use_cms=args.use_cms,
    )
    recommendations, evidence, audit = pipeline.process_batch(records)

    export_recommendations(output_dir, recommendations)
    export_evidence(output_dir, evidence)
    export_audit(output_dir, audit)
    export_connector_diagnostics(output_dir, audit)
    export_executive_summary(output_dir, recommendations)

    print(f"Processed {len(records)} provider record(s).")
    print(f"Wrote outputs to: {output_dir.resolve()}")
    for rec in recommendations:
        print(rec.model_dump_json(indent=2))


if __name__ == "__main__":
    main()