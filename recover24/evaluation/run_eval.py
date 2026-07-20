"""Compatibility entry point for Recover24 evaluation modes."""

from evaluation.cli import generate_record, main


def _generate(method, case, provider, claim_provider, max_retries):
    """Backward-compatible helper used by tests and external scripts."""
    return generate_record(method, case, provider, claim_provider, max_retries)


if __name__ == "__main__":
    main()