"""Mode-specific policy for Recover24 evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPolicy:
    mode: str
    dataset: Path
    output_dir: Path
    final: bool = False

    def validate(
        self,
        *,
        provider_kind: str | None,
        claim_provider_kind: str | None,
        force_final: bool = False,
    ) -> None:
        if not self.final:
            return
        if provider_kind in (None, "", "none"):
            raise ValueError("Final mode requires a real generation provider.")
        if claim_provider_kind in (None, "", "none"):
            raise ValueError("Final mode requires a real claim provider.")
        lock_path = self.output_dir / "FINAL_EVAL_LOCK.json"
        if lock_path.exists() and not force_final:
            raise FileExistsError(
                f"Final evaluation is already locked at {lock_path}. "
                "Use --force-final only for an intentional replacement."
            )


def resolve_run_policy(
    mode: str,
    *,
    project_root: str | Path,
    dataset: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> RunPolicy:
    root = Path(project_root)
    defaults = {
        "dev": (
            root / "evaluation/dataset/dev.jsonl",
            root / "evaluation/runs/dev",
            False,
        ),
        "challenge": (
            root / "evaluation/dataset/challenge.jsonl",
            root / "evaluation/runs/challenge",
            False,
        ),
        "final": (
            root / "evaluation/dataset/test.jsonl",
            root / "evaluation/final_submission",
            True,
        ),
    }
    if mode not in defaults:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    default_dataset, default_output, final = defaults[mode]
    return RunPolicy(
        mode=mode,
        dataset=Path(dataset) if dataset else default_dataset,
        output_dir=Path(output_dir) if output_dir else default_output,
        final=final,
    )
