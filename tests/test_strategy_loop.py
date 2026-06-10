"""Tests for the GPT-5.5 planning pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.orchestrator.agent_pipeline import (
    _STRATEGY_PROMPT_MAX_CHARS,
    AgentPipelineConfig,
    _extract_plan_json,
    run_agent_pipeline,
)
from kagglebot.paths import CompetitionPaths

pytestmark = pytest.mark.slow


class DummyCodexResult:
    def __init__(self, output_dir: Path, message: str = "brief summary") -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""
        self.last_message_path = output_dir / "codex_last_message.txt"
        self.last_message_path.write_text(message + "\n", encoding="utf-8")


class DummyStrategyResult:
    def __init__(self, output: str, *, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = output
        self.stderr = stderr


def _long_strategy_text() -> str:
    base = "\n".join(
        [
            "## Problem",
            "This section frames the problem and target metric.",
            "## Data",
            "We summarize data structure and submission format.",
            "## Candidate Approaches",
            "Candidate models and alternatives with pros/cons.",
            "## Final Approach",
            "Final model choice and rationale.",
            "## Training & Evaluation",
            "Train/validation plan and evaluation strategy.",
            "## Compute Plan",
            "GPU usage plan and time budget.",
            "## Error Analysis & Ablation",
            "Ablation plan and error analysis steps.",
            "## Risks",
            "Risks and rule constraints.",
            "## Search Notes",
            "Search queries and key findings.",
            "## Sources",
            "- Source A (example.com)",
            "- Source B (example.org)",
            "- Source C (example.net)",
        ]
    )
    filler = "Details and rationale. " * 80
    return base + "\n" + filler


def _long_instructions_text() -> str:
    steps = [
        "1) Update kernel.py with the new model pipeline.",
        "2) Add helper modules under kernel/ for parsing and preprocessing.",
        "3) Ensure training/evaluation flow uses the chosen metric.",
        "4) Write outputs to submission.csv with correct format.",
        "5) Validate against sample submission and log metrics.",
    ]
    return "\n".join(steps) + "\n" + ("More detail. " * 40)


def _long_instructions_without_kernel_text() -> str:
    steps = [
        "1) Update the primary kernel entrypoint with the new model pipeline.",
        "2) Add helper modules under kernel/ for parsing and preprocessing.",
        "3) Ensure training/evaluation flow uses the chosen metric.",
        "4) Write outputs to submission.csv with correct format.",
        "5) Validate against sample submission and log metrics.",
    ]
    return "\n".join(steps) + "\n" + ("More detail. " * 40)


def _plan_json_text() -> str:
    payload = {
        "target_metric": "auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "catboost_freq",
                "features": ["numeric_impute", "frequency_encoding"],
                "models": ["catboost_classifier"],
                "key_hyperparameters": {"iterations": 2000},
                "runtime_memory": "medium",
                "failure_modes": ["slow runtime"],
                "fallbacks": ["reduce iterations"],
            },
            {
                "name": "lgbm_te",
                "features": ["numeric_impute", "target_encoding_cv"],
                "models": ["lightgbm_classifier"],
                "key_hyperparameters": {"n_estimators": 2500},
                "runtime_memory": "low-medium",
                "failure_modes": ["encoding leakage if incorrect"],
                "fallbacks": ["disable target encoding"],
            },
        ],
        "toggles": {
            "USE_CATBOOST": True,
            "USE_XGB": True,
            "USE_LGBM": True,
            "USE_STACKING": True,
            "FAST_DEV": False,
        },
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "AUC",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _research_sources_jsonl_text() -> str:
    rows = [
        {
            "url": "https://www.kaggle.com/competitions/example/discussion/1",
            "title": "Kaggle discussion",
            "date": "2025-01-01",
            "why_relevant": "competition-specific pattern",
            "extracted_technique": "catboost + cv target encoding",
            "query": "example kaggle binary classification discussion",
            "top_urls": [
                "https://www.kaggle.com/competitions/example/discussion/1",
                "https://www.kaggle.com/code/example/notebook",
            ],
            "publish_dates": ["2025-01-01", "2025-01-02"],
            "takeaway": "Fold-safe target encoding with boosting is strong.",
        },
        {
            "url": "https://github.com/example/tabular-ensemble",
            "title": "GitHub tabular ensemble",
            "date": "2024-11-10",
            "why_relevant": "reusable OOF stacking recipe",
            "extracted_technique": "OOF stack + rank blend",
            "query": "tabular stacking oof github",
            "top_urls": [
                "https://github.com/example/tabular-ensemble",
                "https://github.com/example/catboost-lgbm-stack",
            ],
            "publish_dates": ["2024-11-10", "2024-06-01"],
            "takeaway": "Stacking with logistic regression is practical and robust.",
        },
        {
            "url": "https://arxiv.org/abs/2106.11959",
            "title": "Tabular DL survey",
            "date": "2021-06-22",
            "why_relevant": "broad guidance for tabular model families",
            "extracted_technique": "strong GBDT baselines + careful CV",
            "query": "arxiv tabular deep learning survey",
            "top_urls": [
                "https://arxiv.org/abs/2106.11959",
                "https://arxiv.org/abs/1908.07442",
            ],
            "publish_dates": ["2021-06-22", "2019-08-20"],
            "takeaway": "GBDT remains a strong default for structured data.",
        },
    ]
    return "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)


def _research_summary_text() -> str:
    return "\n".join(
        [
            "# Ranked Pipelines",
            "",
            "1. CatBoost + frequency encoding",
            "- Pros: strong categorical handling.",
            "- Cons: can be slower than LGBM.",
            "- Runtime risk: medium.",
            "- Leakage risk: low with fold-safe stats.",
            "",
            "2. LightGBM + CV target encoding",
            "- Pros: fast, scalable, high AUC.",
            "- Cons: encoding mistakes cause leakage.",
            "- Runtime risk: low-medium.",
            "- Leakage risk: medium if not fold-safe.",
            "",
            "3. XGBoost + rank-mean blend",
            "- Pros: robust complement model.",
            "- Cons: added compute.",
            "- Runtime risk: medium.",
            "- Leakage risk: low.",
        ]
    )


def _write_context(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://example.com/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    paths.overview_md_path.write_text("overview text\n", encoding="utf-8")
    paths.data_md_path.write_text("data text\n", encoding="utf-8")
    paths.rules_md_path.write_text("rules text\n", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")


def test_agent_pipeline_runs_all_stages(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 2:
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===RESEARCH_SOURCES_JSONL===",
                _research_sources_jsonl_text(),
                "===RESEARCH_SUMMARY_MD===",
                _research_summary_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)
    monkeypatch.setattr(
        "kagglebot.orchestrator.agent_pipeline._load_problem_type_knowledge_text",
        lambda *args, **kwargs: "Problem-type knowledge (test fixture)",
    )

    config = AgentPipelineConfig(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )
    run_agent_pipeline(paths=paths, config=config)

    agent_dir = paths.context_agent_dir
    assert (agent_dir / "brief_for_strategy.md").exists()
    assert (agent_dir / "strategy_plan.md").exists()
    assert (agent_dir / "codex_instructions.md").exists()
    assert (agent_dir / "strategy_transcript.txt").exists()
    assert (paths.context_dir / "research_sources.jsonl").exists()
    assert (paths.context_dir / "research_summary.md").exists()
    assert (tmp_path / "knowledge" / "research" / "unknown" / "demo" / "research_sources.jsonl").exists()
    assert (tmp_path / "knowledge" / "research" / "unknown" / "demo" / "research_summary.md").exists()
    assert len(codex_calls) == 2
    brief_prompt = (agent_dir / "brief" / "prompt.md").read_text(encoding="utf-8")
    assert str(paths.overview_md_path) in brief_prompt
    assert str(paths.data_md_path) in brief_prompt
    assert str(paths.rules_md_path) in brief_prompt
    assert "Problem-type knowledge (test fixture)" in brief_prompt


def test_agent_pipeline_does_not_raise_on_successful_strategy_result(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        return DummyStrategyResult(
            "\n".join(
                [
                    "===STRATEGY===",
                    _long_strategy_text(),
                    "===RESEARCH_SOURCES_JSONL===",
                    _research_sources_jsonl_text(),
                    "===RESEARCH_SUMMARY_MD===",
                    _research_summary_text(),
                    "===PLAN_JSON===",
                    _plan_json_text(),
                    "===CODEX_INSTRUCTIONS===",
                    _long_instructions_text(),
                ]
            )
        )

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)

    assert (paths.context_agent_dir / "strategy_plan.md").exists()
    assert (paths.context_agent_dir / "codex_instructions.md").exists()


def test_agent_pipeline_falls_back_when_strategy_times_out(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        message = "Strategy runner timed out after 600s (elapsed=600s)."
        return DummyStrategyResult("", returncode=124, stderr=message)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)

    agent_dir = paths.context_agent_dir
    assert len(codex_calls) == 2
    assert (agent_dir / "strategy_plan.md").exists()
    assert (agent_dir / "codex_instructions.md").exists()
    assert (agent_dir / "strategy_transcript.txt").read_text(encoding="utf-8").find("timed out") != -1
    assert (paths.context_dir / "research_sources.jsonl").exists()
    assert (paths.context_dir / "research_summary.md").exists()


def test_agent_pipeline_caps_compact_strategy_prompt(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        assert len(prompt_text) <= _STRATEGY_PROMPT_MAX_CHARS
        return DummyStrategyResult(
            "\n".join(
                [
                    "===STRATEGY===",
                    _long_strategy_text(),
                    "===RESEARCH_SOURCES_JSONL===",
                    _research_sources_jsonl_text(),
                    "===RESEARCH_SUMMARY_MD===",
                    _research_summary_text(),
                    "===PLAN_JSON===",
                    _plan_json_text(),
                    "===CODEX_INSTRUCTIONS===",
                    _long_instructions_text(),
                ]
            )
        )

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)
    monkeypatch.setattr(
        "kagglebot.orchestrator.agent_pipeline._load_problem_type_knowledge_text",
        lambda *args, **kwargs: "Prior knowledge.\n" * (_STRATEGY_PROMPT_MAX_CHARS // 8),
    )

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)

    assert len(codex_calls) == 2
    assert (paths.context_agent_dir / "strategy" / "prompt.md").exists()


def test_agent_pipeline_write_guard_blocks_data_dir(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            train_path = paths.data_dir / "train.csv"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_text("id,target\n1,1\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===RESEARCH_SOURCES_JSONL===",
                _research_sources_jsonl_text(),
                "===RESEARCH_SUMMARY_MD===",
                _research_summary_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert not (paths.data_dir / "train.csv").exists()


def test_agent_pipeline_allows_kernel_write(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('ok')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===RESEARCH_SOURCES_JSONL===",
                _research_sources_jsonl_text(),
                "===RESEARCH_SUMMARY_MD===",
                _research_summary_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert (paths.kernel_source_dir / "kernel.py").exists()


def test_agent_pipeline_allows_src_write(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            support_path = tmp_path / "src" / "support_fix.py"
            support_path.parent.mkdir(parents=True, exist_ok=True)
            support_path.write_text("READY = True\n", encoding="utf-8")
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('ok')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===RESEARCH_SOURCES_JSONL===",
                _research_sources_jsonl_text(),
                "===RESEARCH_SUMMARY_MD===",
                _research_summary_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert (tmp_path / "src" / "support_fix.py").read_text(encoding="utf-8") == "READY = True\n"


def test_agent_pipeline_injects_kernel_reference_when_missing(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 2:
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===RESEARCH_SOURCES_JSONL===",
                _research_sources_jsonl_text(),
                "===RESEARCH_SUMMARY_MD===",
                _research_summary_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_without_kernel_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    instructions = (paths.context_agent_dir / "codex_instructions.md").read_text(encoding="utf-8")
    assert "kernel.py" in instructions


def test_agent_pipeline_uses_full_transcript_when_last_message_is_truncated(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 2:
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    class DummyStrategyWithTranscript:
        def __init__(self, output_dir: Path) -> None:
            self.returncode = 0
            self.stdout = "short last message"
            self.stderr = ""
            self.transcript_path = output_dir / "strategy_exec.txt"
            self.transcript_path.write_text(
                "\n".join(
                    [
                        "===STRATEGY===",
                        _long_strategy_text(),
                        "===RESEARCH_SOURCES_JSONL===",
                        _research_sources_jsonl_text(),
                        "===RESEARCH_SUMMARY_MD===",
                        _research_summary_text(),
                        "===PLAN_JSON===",
                        _plan_json_text(),
                        "===CODEX_INSTRUCTIONS===",
                        _long_instructions_text(),
                    ]
                ),
                encoding="utf-8",
            )

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        return DummyStrategyWithTranscript(output_dir)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert len(codex_calls) == 2
    instructions = (paths.context_agent_dir / "codex_instructions.md").read_text(encoding="utf-8")
    assert "kernel.py" in instructions


def test_extract_plan_json_normalizes_repeated_error_fingerprint_abort_alias() -> None:
    payload = {
        "target_metric": "auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "catboost_freq",
                "features": ["basic"],
                "models": ["catboost"],
                "key_hyperparameters": {"iterations": 1000},
                "runtime_memory": "medium",
                "failure_modes": ["slow runtime"],
                "fallbacks": ["reduce iterations"],
            },
            {
                "name": "xgb",
                "features": ["basic"],
                "models": ["xgboost"],
                "key_hyperparameters": {"n_estimators": 1000},
                "runtime_memory": "medium",
                "failure_modes": ["overfit"],
                "fallbacks": ["regularize"],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "AUC",
        },
        "stop_policy": {
            "max_iterations": 3,
            "repeated_error_fingerprint_abort": {"max_repeats": 2, "window_runs": 5},
        },
    }
    response_text = "\n".join(
        [
            "===PLAN_JSON===",
            json.dumps(payload),
            "===CODEX_INSTRUCTIONS===",
            "kernel.py",
        ]
    )
    plan_payload, issue = _extract_plan_json(response_text)
    assert issue is None
    assert isinstance(plan_payload, dict)
    stop_policy = plan_payload.get("stop_policy")
    assert isinstance(stop_policy, dict)
    assert "error_fingerprint_abort" in stop_policy
    assert "repeated_error_fingerprint_abort" in stop_policy


def test_extract_plan_json_backfills_stop_policy_defaults() -> None:
    payload = {
        "target_metric": "auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 7,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "name": "catboost_freq",
                "features": ["basic"],
                "models": ["catboost"],
                "key_hyperparameters": {"iterations": 1000},
                "runtime_memory": "medium",
                "failure_modes": ["slow runtime"],
                "fallbacks": ["reduce iterations"],
            },
            {
                "name": "xgb",
                "features": ["basic"],
                "models": ["xgboost"],
                "key_hyperparameters": {"n_estimators": 1000},
                "runtime_memory": "medium",
                "failure_modes": ["overfit"],
                "fallbacks": ["regularize"],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "AUC",
        },
        "stop_policy": {},
    }
    response_text = "\n".join(
        [
            "===PLAN_JSON===",
            json.dumps(payload),
            "===CODEX_INSTRUCTIONS===",
            "kernel.py",
        ]
    )
    plan_payload, issue = _extract_plan_json(response_text)
    assert issue is None
    assert isinstance(plan_payload, dict)
    stop_policy = plan_payload.get("stop_policy")
    assert isinstance(stop_policy, dict)
    assert stop_policy.get("max_iterations") == 7
    assert stop_policy.get("error_fingerprint_abort") is True
