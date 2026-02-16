from __future__ import annotations

import json
from datetime import UTC, datetime

from kagglebot.kernel_runner import run_kernel_local
from kagglebot.runners.base import RunContext, RunResult


class LocalKernelRunner:
    name = "local_kernel"

    def run(self, context: RunContext) -> RunResult:
        run_dir = context.paths.run_dir(context.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"

        result = run_kernel_local(
            slug=context.slug,
            run_id=context.run_id,
            iteration=0,
            base_dir=context.paths.base_dir.parent,
            accelerator=context.accelerator,
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=max(2, int(context.cv_folds)),
            seed=42,
            dry_run=context.dry_run,
            timeout_minutes=context.time_budget_minutes,
            strict_accelerator=context.strict_accelerator,
        )

        summary = {
            "schema_version": 1,
            "runner": self.name,
            "slug": context.slug,
            "run_id": context.run_id,
            "kernel_id": result.kernel_id,
            "output_dir": str(result.output_dir),
            "submission_path": str(result.submission_path) if result.submission_path else None,
            "metrics_path": str(result.metrics_path) if result.metrics_path else None,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return RunResult(
            run_id=context.run_id,
            runner=self.name,
            submission_path=result.submission_path,
            summary_path=summary_path,
            analysis_path=None,
            kernel_slug=result.kernel_id,
        )
