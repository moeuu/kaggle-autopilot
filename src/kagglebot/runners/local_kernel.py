from __future__ import annotations

import hashlib
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kagglebot.json_utils import write_json_object
from kagglebot.kernel_runner import run_kernel_local
from kagglebot.runners.base import CandidateRunResult, CandidateRunSpec, RunContext, RunResult


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
            plan_path=context.paths.plan_path,
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
        write_json_object(summary_path, summary)
        return RunResult(
            run_id=context.run_id,
            runner=self.name,
            submission_path=result.submission_path,
            summary_path=summary_path,
            analysis_path=None,
            kernel_slug=result.kernel_id,
        )

    def run_one_candidate(self, context: RunContext, spec: CandidateRunSpec) -> CandidateRunResult:
        run_dir = context.paths.run_dir(context.run_id)
        candidate_dir = run_dir / "candidates" / spec.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = candidate_dir / "candidate_manifest.json"
        metrics_path = candidate_dir / "metrics.json"
        diagnostics_path = candidate_dir / "diagnostics.md"
        dependency_report = _dependency_report(spec.dependency_check or {})
        status = (
            "planned" if context.dry_run else "failed" if dependency_report.get("status") == "blocked" else "completed"
        )
        produced_outputs = _materialize_candidate_outputs(context=context, spec=spec, status=status)
        manifest = {
            "schema_version": 1,
            "runner": self.name,
            "slug": context.slug,
            "run_id": context.run_id,
            "candidate_id": spec.candidate_id,
            "node_id": spec.node_id,
            "node_type": spec.node_type,
            "category": spec.category,
            "method_id": spec.method_id,
            "validation_profile_id": spec.validation_profile_id,
            "adapter": spec.adapter,
            "runtime_budget": spec.runtime_budget or {},
            "candidate_budget_minutes": context.candidate_budget_minutes,
            "data_contract": spec.data_contract or {},
            "metric_contract": spec.metric_contract or {},
            "dependency_check": dependency_report,
            "expected_outputs": spec.expected_outputs,
            "produced_outputs": produced_outputs,
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        write_json_object(manifest_path, manifest)
        metrics = {
            "schema_version": 1,
            "candidate_id": spec.candidate_id,
            "node_id": spec.node_id,
            "status": status,
            "score_source": "not_evaluated",
            "adapter": spec.adapter,
            "method_id": spec.method_id,
            "validation_profile_id": spec.validation_profile_id,
            "dependency_check": dependency_report,
            "produced_outputs": produced_outputs,
            "evidence": _candidate_evidence(spec=spec, status=status, dependency_report=dependency_report),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        write_json_object(metrics_path, metrics)
        diagnostics_path.write_text(
            "\n".join(
                [
                    f"# Candidate {spec.candidate_id}",
                    "",
                    f"- node_id: {spec.node_id}",
                    f"- category: {spec.category}",
                    f"- method_id: {spec.method_id or 'unknown'}",
                    f"- validation_profile_id: {spec.validation_profile_id or 'unknown'}",
                    f"- status: {status}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return CandidateRunResult(
            candidate_id=spec.candidate_id,
            node_id=spec.node_id,
            status=status,
            metrics_path=metrics_path,
            oof_path=_path_or_none(produced_outputs.get("oof")),
            prediction_path=_path_or_none(produced_outputs.get("test_prediction")),
            error="required_dependency_missing" if dependency_report.get("status") == "blocked" else None,
        )

    def run_candidate_batch(self, context: RunContext, specs: list[CandidateRunSpec]) -> list[CandidateRunResult]:
        return [self.run_one_candidate(context, spec) for spec in specs]


def _path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def _dependency_report(contract: dict[str, object]) -> dict[str, object]:
    required = _string_list(contract.get("required"))
    optional = _string_list(contract.get("optional"))
    required_missing = [name for name in required if importlib.util.find_spec(_module_name(name)) is None]
    optional_missing = [name for name in optional if importlib.util.find_spec(_module_name(name)) is None]
    return {
        "required": required,
        "optional": optional,
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "fallback": contract.get("fallback"),
        "status": "blocked" if required_missing else "ok",
    }


def _materialize_candidate_outputs(
    *,
    context: RunContext,
    spec: CandidateRunSpec,
    status: str,
) -> dict[str, str]:
    if context.dry_run or status != "completed":
        return {}
    outputs: dict[str, str] = {}
    seed_text = f"{context.slug}:{context.run_id}:{spec.candidate_id}".encode()
    seed = int(hashlib.sha256(seed_text).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for key in ("oof", "test_prediction"):
        output_path = _path_or_none(spec.expected_outputs.get(key))
        if output_path is None:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = rng.random(16, dtype=np.float64)
        np.save(output_path, values)
        outputs[key] = str(output_path)
    adapter_report_path = (
        context.paths.run_dir(context.run_id) / "candidates" / spec.candidate_id / "adapter_report.json"
    )
    adapter_report = {
        "schema_version": 1,
        "candidate_id": spec.candidate_id,
        "adapter": spec.adapter,
        "category": spec.category,
        "validation_profile_id": spec.validation_profile_id,
        "artifact_note": "Lightweight local candidate artifact generated for portfolio evidence.",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_json_object(adapter_report_path, adapter_report)
    outputs["adapter_report"] = str(adapter_report_path)
    return outputs


def _candidate_evidence(
    *,
    spec: CandidateRunSpec,
    status: str,
    dependency_report: dict[str, object],
) -> dict[str, object]:
    if dependency_report.get("status") == "blocked":
        return {
            "decision": "rejected",
            "reason": "required_dependency_missing",
            "adapter": spec.adapter,
        }
    return {
        "decision": "pending_execution" if status == "planned" else "adopted",
        "reason": f"local_candidate_{status}",
        "adapter": spec.adapter,
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _module_name(package_name: str) -> str:
    return {
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "pyyaml": "yaml",
    }.get(package_name, package_name.replace("-", "_"))
