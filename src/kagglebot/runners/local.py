from __future__ import annotations

from rich import print

from kagglebot.analyzer import analyze_competition
from kagglebot.compute import Compute, detect_local_gpu
from kagglebot.kaggle_cli import download_competition
from kagglebot.runners.base import RunContext, RunResult
from kagglebot.training import predict_tabular, train_tabular, train_torch_tabular


class LocalRunner:
    name = "local"

    def run(self, context: RunContext) -> RunResult:
        paths = context.paths
        analysis_path = None
        compute = Compute(context.compute)
        accelerator = context.accelerator

        if not context.dry_run:
            if not context.force:
                raise RuntimeError("Refusing to download competition data without --force.")
            download_competition(context.slug, paths.data_raw, overwrite=False)

        analysis = analyze_competition(
            slug=context.slug,
            paths=paths,
            time_budget_minutes=context.time_budget_minutes,
            cv_folds=context.cv_folds,
            models=context.model_names,
            use_stacking=context.use_stacking,
        )
        analysis_path = analysis.analysis_path
        print(f"[green]analysis saved[/green]: {analysis_path}")

        model_names = context.model_names
        if compute == Compute.local_gpu:
            availability = detect_local_gpu()
            if availability.cuda:
                print("[cyan]local GPU detected[/cyan]: CUDA available")
                if model_names is None:
                    baseline = "logreg" if analysis.metadata.task == "classification" else "ridge"
                    model_names = ["catboost_gpu", baseline]
                elif "catboost_gpu" not in model_names:
                    print("[yellow]GPU available but no GPU model requested; using provided models[/yellow]")
            elif availability.mps:
                print("[cyan]local GPU detected[/cyan]: MPS available")
                try:
                    train_torch_tabular(
                        analysis.metadata,
                        paths=paths,
                        time_budget_minutes=context.time_budget_minutes,
                        device="mps",
                    )
                    submission_path = predict_tabular(analysis.metadata, paths=paths)
                    return RunResult(
                        run_id=context.run_id,
                        runner=self.name,
                        submission_path=submission_path,
                        summary_path=None,
                        analysis_path=analysis_path,
                        kernel_slug=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    if context.strict_accelerator:
                        raise RuntimeError("MPS training failed.") from exc
                    print(f"[yellow]MPS training failed, falling back to CPU[/yellow]: {exc}")
                    accelerator = "none"
                    model_names = _strip_gpu_models(model_names)
            else:
                message = "No local GPU detected. Falling back to CPU training."
                if context.strict_accelerator:
                    raise RuntimeError(
                        "No local GPU detected for --compute local_gpu. "
                        "Disable --strict-accelerator to fall back to CPU."
                    )
                print(f"[yellow]{message}[/yellow]")
                accelerator = "none"
                model_names = _strip_gpu_models(model_names)

        train_tabular(
            analysis.metadata,
            paths=paths,
            time_budget_minutes=context.time_budget_minutes,
            model_names=model_names,
            cv_folds=context.cv_folds,
            strict_accelerator=context.strict_accelerator,
            accelerator=accelerator,
        )
        submission_path = predict_tabular(analysis.metadata, paths=paths)

        return RunResult(
            run_id=context.run_id,
            runner=self.name,
            submission_path=submission_path,
            summary_path=None,
            analysis_path=analysis_path,
            kernel_slug=None,
        )


def _strip_gpu_models(model_names: list[str] | None) -> list[str] | None:
    if model_names is None:
        return None
    filtered = [name for name in model_names if name.lower() != "catboost_gpu"]
    return filtered or None
