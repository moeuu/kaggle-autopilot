from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.orchestrator.agent_pipeline import (
    _ensure_context_materials,
    _read_sample_submission_head,
    _resolve_blocked_modules_for_runtime,
)
from kagglebot.paths import CompetitionPaths
from kagglebot.solver.io import read_table, write_table


def test_ensure_context_materials_refreshes_header_only_sample_submission(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    (paths.context_dir / "submission_format.md").write_text(
        "## Submission File\n\n```csv\nfilename,right_place,prediction_string\n0.jpg,0,-\n```\n",
        encoding="utf-8",
    )

    images_test_dir = paths.data_dir / "images" / "test"
    images_test_dir.mkdir(parents=True, exist_ok=True)
    (images_test_dir / "0.jpg").write_bytes(b"")
    (images_test_dir / "1.jpg").write_bytes(b"")

    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")

    _ensure_context_materials(paths)

    sample = pd.read_csv(paths.sample_submission_path)
    assert sample.columns.tolist() == ["filename", "right_place", "prediction_string"]
    assert len(sample) == 2


def test_ensure_context_materials_creates_community_placeholders(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    _ensure_context_materials(paths)

    assert "/code" in paths.code_md_path.read_text(encoding="utf-8")
    assert "/models" in paths.models_md_path.read_text(encoding="utf-8")
    assert "/discussions" in paths.discussion_md_path.read_text(encoding="utf-8")
    assert paths.code_notebooks_index_path.exists()
    assert paths.discussion_threads_index_path.exists()
    assert paths.code_notebooks_dir.exists()
    assert paths.discussion_threads_dir.exists()


def test_ensure_context_materials_uses_prediction_header_for_placeholder_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "id,prediction"


def test_ensure_context_materials_uses_submission_format_suffix_for_placeholder_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nThe submission file must be a TSV file.\n\n```tsv\nimage_id\tscore\n```\n",
        encoding="utf-8",
    )

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.tsv"
    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "image_id\tscore"
    assert paths.sample_submission_head_path.name == "sample_submission_head.tsv"
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").strip() == "image_id\tscore"


def test_ensure_context_materials_placeholder_write_fallback_preserves_text_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nThe submission file must be a TSV file.\n\n```tsv\nimage_id\tscore\n```\n",
        encoding="utf-8",
    )

    def fail_write_table(*_args, **_kwargs):
        raise RuntimeError("writer unavailable")

    monkeypatch.setattr("kagglebot.solver.io.write_table", fail_write_table)

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.tsv"
    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "image_id\tscore"
    assert not (paths.context_dir / "sample_submission.csv").exists()


def test_ensure_context_materials_placeholder_write_fallback_preserves_compressed_text_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nUpload a gzip-compressed TSV file with columns image_id,score.\n",
        encoding="utf-8",
    )

    def fail_write_table(*_args, **_kwargs):
        raise RuntimeError("writer unavailable")

    monkeypatch.setattr("kagglebot.solver.io.write_table", fail_write_table)

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.tsv.gz"
    with gzip.open(paths.sample_submission_path, "rt", encoding="utf-8") as handle:
        assert handle.read().strip() == "image_id\tscore"
    assert not (paths.context_dir / "sample_submission.csv").exists()


def test_ensure_context_materials_uses_binary_submission_format_suffix_for_placeholder_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nUpload `submission.feather` with columns id,target.\n",
        encoding="utf-8",
    )

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.feather"
    assert read_table(paths.sample_submission_path).columns.tolist() == ["id", "target"]
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").strip() == "id,target"


def test_ensure_context_materials_uses_html_submission_format_suffix_for_placeholder_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nUpload `submission.html` with columns id,target.\n",
        encoding="utf-8",
    )

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.html"
    assert read_table(paths.sample_submission_path).columns.tolist() == ["id", "target"]
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").strip() == "id,target"


def test_ensure_context_materials_uses_csv_placeholder_for_rowless_hdf_format(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nUpload `submission.hdf5` with columns id,target.\n",
        encoding="utf-8",
    )

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.csv"
    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "id,target"


def test_ensure_context_materials_uses_csv_placeholder_for_rowless_jsonl_format(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.submission_format_md_path.write_text(
        "## Submission File\n\nSubmit `submission.jsonl` with columns id,target.\n",
        encoding="utf-8",
    )

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.csv"
    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "id,target"


def test_ensure_context_materials_refreshes_placeholder_with_jsonl_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")
    data_sample = paths.data_dir / "sample_submission.jsonl"
    data_sample.write_text('{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n', encoding="utf-8")

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.jsonl"
    assert paths.sample_submission_path.read_text(encoding="utf-8") == data_sample.read_text(encoding="utf-8")
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").startswith("id,target\n")


def test_ensure_context_materials_preserves_compressed_sample_submission_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")
    data_sample = paths.data_dir / "sample_submission.csv.gz"
    with gzip.open(data_sample, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n2,0.2\n")

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.csv.gz"
    assert paths.context_sample_submission_path_for_suffix(".csv.gz").is_file()
    assert paths.sample_submission_head_path.name == "sample_submission_head.csv"
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").startswith("id,target\n")


def test_read_sample_submission_head_preserves_compressed_text_delimiter(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = paths.context_sample_submission_path_for_suffix(".tsv.gz")
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n001\t0.0\n002\t0.0\n")

    assert _read_sample_submission_head(paths) == "id\ttarget\n001\t0.0\n002\t0.0"


def test_competition_paths_uses_uncompressed_text_head_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)

    assert paths.context_sample_submission_head_path_for_suffix(".tsv").name == "sample_submission_head.tsv"
    assert paths.context_sample_submission_head_path_for_suffix(".jsonl.zst").name == "sample_submission_head.csv"
    assert paths.context_sample_submission_head_path_for_suffix(".parquet").name == "sample_submission_head.csv"


def test_read_sample_submission_head_limits_non_text_table_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = paths.context_sample_submission_path_for_suffix(".parquet")
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_parquet(sample_path, index=False)
    calls: list[int | None] = []

    import kagglebot.solver.io as solver_io

    real_read_table = solver_io.read_table

    def spy_read_table(path: Path, *, nrows=None):
        calls.append(nrows)
        return real_read_table(path, nrows=nrows)

    monkeypatch.setattr(solver_io, "read_table", spy_read_table)

    assert _read_sample_submission_head(paths, max_lines=2) == "id,target\n1,0.1\n2,0.2"
    assert calls == [2]


def test_ensure_context_materials_preserves_excel_sample_submission_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")
    data_sample = paths.data_dir / "sample_submission.xlsx"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_excel(data_sample, index=False)

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == "sample_submission.xlsx"
    assert paths.context_sample_submission_path_for_suffix(".xlsx").is_file()
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").startswith("id,target\n")


@pytest.mark.parametrize("suffix", [".parquet", ".orc", ".hdf", ".hdf5", ".html", ".html.zst", ".pkl.zst", ".json.zst"])
def test_ensure_context_materials_preserves_generalized_sample_submission_suffix(tmp_path: Path, suffix: str) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")
    data_sample = paths.data_dir / f"sample_submission{suffix}"
    write_table(pd.DataFrame({"id": [1, 2], "target": [0, 1]}), data_sample)

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.name == f"sample_submission{suffix}"
    assert paths.context_sample_submission_path_for_suffix(suffix).is_file()
    assert read_table(paths.sample_submission_path).to_dict("list") == {"id": [1, 2], "target": [0, 1]}
    assert paths.sample_submission_head_path.read_text(encoding="utf-8").startswith("id,target\n")


def test_competition_paths_preserves_html_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    sample = paths.context_dir / "sample_submission.html"
    pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_html(sample, index=False)

    assert paths.sample_submission_path.name == "sample_submission.html"
    assert paths.context_sample_submission_path_for_suffix(".html").name == "sample_submission.html"


def test_competition_paths_preserves_compressed_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "sample_submission.csv.gz").write_bytes(b"compressed bytes")

    assert paths.sample_submission_path.name == "sample_submission.csv.gz"
    assert paths.context_sample_submission_path_for_suffix(".csv.gz").name == "sample_submission.csv.gz"


def test_competition_paths_preserves_compressed_jsonl_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "sample_submission.jsonl.zst").write_bytes(b"jsonl zstd placeholder")

    assert paths.sample_submission_path.name == "sample_submission.jsonl.zst"
    assert paths.context_sample_submission_path_for_suffix(".jsonl.zst").name == "sample_submission.jsonl.zst"
    assert paths.context_sample_submission_path_for_suffix("jsonl.zst").name == "sample_submission.jsonl.zst"


def test_competition_paths_preserves_excel_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    sample = paths.context_dir / "sample_submission.xlsx"
    pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_excel(sample, index=False)

    assert paths.sample_submission_path.name == "sample_submission.xlsx"
    assert paths.context_sample_submission_path_for_suffix(".xlsx").name == "sample_submission.xlsx"


def test_competition_paths_preserves_sqlite_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    sample = paths.context_dir / "sample_submission.sqlite"
    sample.write_bytes(b"sqlite placeholder")

    assert paths.sample_submission_path.name == "sample_submission.sqlite"
    assert paths.context_sample_submission_path_for_suffix(".sqlite").name == "sample_submission.sqlite"


def test_competition_paths_preserves_stata_context_sample_suffix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "sample_submission.dta").write_bytes(b"stata placeholder")

    assert paths.sample_submission_path.name == "sample_submission.dta"
    assert paths.context_sample_submission_path_for_suffix(".dta").name == "sample_submission.dta"


def test_competition_paths_uses_context_sample_alias_when_canonical_missing(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "AnswerTemplate.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    assert paths.sample_submission_path.name == "AnswerTemplate.csv"


def test_competition_paths_does_not_treat_head_or_output_as_sample_alias(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "sample_submission_head.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (paths.context_dir / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    assert paths.sample_submission_path.name == "sample_submission.csv"


def test_resolve_blocked_modules_for_runtime_adds_missing_xgboost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "blocked_modules.json").write_text('["custom_pkg"]\n', encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.orchestrator.agent_pipeline.importlib.util.find_spec",
        lambda name: None if name == "xgboost" else object(),
    )
    resolved = _resolve_blocked_modules_for_runtime(context_dir, compute="local_gpu")
    assert resolved == ["custom_pkg", "xgboost"]
