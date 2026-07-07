from __future__ import annotations

import gzip
import io
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.local_sample_submission import (
    copy_or_convert_sample_submission,
    ensure_local_sample_submission_file,
    expand_placeholder_sample_submission,
)
from kagglebot.solver.io import read_table


def test_ensure_local_sample_submission_file_expands_placeholder_template(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text(
        "id,feature\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n7,70\n8,80\n9,90\n10,100\n11,110\n12,120\n13,130\n14,140\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = (data_dir / "sample_submission.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[0] == "id,target"
    assert lines[1].startswith("1,")
    assert lines[14].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_non_csv_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.tsv").write_text("id\ttarget\n1\t0\n2\t1\n3\t1\n", encoding="utf-8")
    (data_dir / "test.jsonl").write_text(
        "\n".join(f'{{"id": {idx}, "feature": {idx * 10}}}' for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[0] == "id,target"
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_does_not_expand_idless_prediction_template(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": [0, 1, 0], "score": [0.1, 0.2, 0.3]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "target": [0] * 14,
            "score": [0.0] * 14,
            "feature": list(range(14)),
        }
    ).to_csv(data_dir / "test.csv", index=False)

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    frame = pd.read_csv(resolved)
    assert frame.to_dict("list") == {"target": [0, 1, 0], "score": [0.1, 0.2, 0.3]}


def test_ensure_local_sample_submission_file_expands_placeholder_from_public_test_name(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
        }
    ).to_csv(data_dir / "TrainingSet.csv", index=False)
    pd.DataFrame({"id": list(range(1, 15)), "feature": list(range(14))}).to_csv(
        data_dir / "PublicTest.csv",
        index=False,
    )
    pd.DataFrame({"id": [100], "feature": [9], "target": [1]}).to_csv(
        data_dir / "contest.csv",
        index=False,
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_eval_features_name(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    (data_dir / "eval_features.csv").write_text(
        "id,feature\n" + "\n".join(f"{idx},{idx * 10}" for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_validation_features_name(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    (data_dir / "validation_features.csv").write_text(
        "id,feature\n" + "\n".join(f"{idx},{idx * 10}" for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_scoring_parquet(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": list(range(1, 15)), "feature": list(range(14))}).to_parquet(
        data_dir / "scoring.parquet",
        index=False,
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_holdout_parquet(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": list(range(1, 15)), "feature": list(range(14))}).to_parquet(
        data_dir / "holdout_features.parquet",
        index=False,
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_leaderboard_features(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    (data_dir / "leaderboard_features.csv").write_text(
        "id,feature\n" + "\n".join(f"{idx},{idx * 10}" for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_expands_placeholder_from_nested_compressed_test(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    nested_dir = data_dir / "split"
    nested_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(nested_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,feature\n")
        for idx in range(1, 15):
            handle.write(f"{idx},{idx * 10}\n")
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = resolved.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert lines[-1].startswith("14,")


def test_ensure_local_sample_submission_file_copies_context_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == tmp_path / "demo" / "data" / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0\n"


def test_ensure_local_sample_submission_file_converts_tsv_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample_submission.tsv").write_text("id\tlabel\n1\ta\n2\tb\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,label\n1,a\n2,b\n"


def test_ensure_local_sample_submission_file_preserves_context_tsv_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.tsv").write_text("id\tlabel\n1\ta\n2\tb\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,label\n1,a\n2,b\n"
    assert (data_dir / "sample_submission.tsv").read_text(encoding="utf-8") == "id\tlabel\n1\ta\n2\tb\n"


def test_copy_or_convert_sample_submission_does_not_mislabel_unreadable_non_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample_submission.xlsx"
    destination = tmp_path / "sample_submission.csv"
    source.write_bytes(b"not-readable-as-table")

    def fail_read_table(_path: Path):
        raise ValueError("cannot read table")

    monkeypatch.setattr("kagglebot.solver.io.read_table", fail_read_table)

    assert copy_or_convert_sample_submission(source=source, destination=destination) is False
    assert not destination.exists()


def test_copy_or_convert_sample_submission_uses_shared_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample_submission.jsonl"
    source.write_text('{"id":1,"target":0.1}\n', encoding="utf-8")
    destination = tmp_path / "sample_submission.csv"
    calls: list[tuple[list[str], Path]] = []

    def spy_write_table(frame, path: Path):
        calls.append((list(frame.columns), path))
        raise ValueError("writer failed")

    monkeypatch.setattr("kagglebot.solver.io.write_table", spy_write_table)

    assert copy_or_convert_sample_submission(source=source, destination=destination) is False
    assert calls == [(["id", "target"], destination)]
    assert not destination.exists()


def test_ensure_local_sample_submission_file_expands_preserved_tsv_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.tsv").write_text("id\ttarget\n1\t0\n2\t0\n3\t0\n", encoding="utf-8")
    (data_dir / "test.tsv").write_text(
        "id\tfeature\n" + "\n".join(f"{idx}\t{idx * 10}" for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    csv_sample = read_table(resolved)
    tsv_sample = read_table(data_dir / "sample_submission.tsv")
    assert csv_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]
    assert tsv_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]
    assert (data_dir / "sample_submission.tsv").read_text(encoding="utf-8").splitlines()[0] == "id\ttarget"


def test_expand_placeholder_sample_submission_fallback_preserves_tsv_delimiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_path = data_dir / "sample_submission.tsv"
    sample_path.write_text("id\ttarget\n1\t0\n2\t0\n3\t0\n", encoding="utf-8")
    (data_dir / "test.tsv").write_text(
        "id\tfeature\n" + "\n".join(f"{idx}\t{idx * 10}" for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )

    def fail_write_table(*args: object, **kwargs: object) -> Path:  # noqa: ARG001
        raise ValueError("write failed")

    monkeypatch.setattr("kagglebot.solver.io.write_table", fail_write_table)

    expand_placeholder_sample_submission(canonical_path=sample_path, data_dir=data_dir)

    lines = sample_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id\ttarget"
    assert lines[1].startswith("1\t")
    assert lines[-1].startswith("14\t")


def test_ensure_local_sample_submission_file_preserves_context_jsonl_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source_payload = '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n'
    (context_dir / "sample_submission.jsonl").write_text(source_payload, encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.jsonl").read_text(encoding="utf-8") == source_payload


def test_ensure_local_sample_submission_file_preserves_context_compressed_jsonl_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source_payload = b'{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n'
    compressed = zstd.ZstdCompressor().compress(source_payload)
    (context_dir / "sample_submission.jsonl.zst").write_bytes(compressed)

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.jsonl.zst").read_bytes() == compressed


def test_ensure_local_sample_submission_file_preserves_context_wrapped_json_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source_payload = '{"records":[{"id":1,"target":0.1},{"id":2,"target":0.2}]}'
    (context_dir / "sample_submission.json").write_text(source_payload, encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.json").read_text(encoding="utf-8") == source_payload


def test_ensure_local_sample_submission_file_preserves_context_html_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source = context_dir / "sample_submission.html"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_html(source, index=False)

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    mirrored = data_dir / "sample_submission.html"
    assert mirrored.is_file()
    mirrored_frame = read_table(mirrored)
    assert list(mirrored_frame.columns) == ["id", "target"]
    assert mirrored_frame["id"].tolist() == [1, 2]


def test_ensure_local_sample_submission_file_preserves_context_compressed_wrapped_json_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source_payload = b'{"data":[{"id":1,"target":0.1},{"id":2,"target":0.2}]}'
    compressed = zstd.ZstdCompressor().compress(source_payload)
    (context_dir / "sample_submission.json.zst").write_bytes(compressed)

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.json.zst").read_bytes() == compressed


def test_ensure_local_sample_submission_file_synthesizes_from_format_and_assets(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    image_dir = data_dir / "images" / "test"
    context_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a CSV with this header:\n\n```csv\nimage_id,label\n```\n",
        encoding="utf-8",
    )
    (image_dir / "img_002.jpg").write_bytes(b"image-2")
    (image_dir / "img_001.jpg").write_bytes(b"image-1")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    sample = read_table(resolved)
    assert list(sample.columns) == ["image_id", "label"]
    assert sample["image_id"].tolist() == ["img_001.jpg", "img_002.jpg"]
    assert sample["label"].tolist() == [0, 0]


def test_ensure_local_sample_submission_file_expands_preserved_jsonl_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.jsonl").write_text(
        "\n".join(f'{{"id": {idx}, "target": 0}}' for idx in range(1, 4)) + "\n",
        encoding="utf-8",
    )
    (data_dir / "eval_features.jsonl").write_text(
        "\n".join(f'{{"id": {idx}, "feature": {idx * 10}}}' for idx in range(1, 15)) + "\n",
        encoding="utf-8",
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    csv_sample = read_table(resolved)
    jsonl_sample = read_table(data_dir / "sample_submission.jsonl")
    assert csv_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]
    assert jsonl_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]


def test_ensure_local_sample_submission_file_expands_preserved_html_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]}).to_html(
        context_dir / "sample_submission.html",
        index=False,
    )
    pd.DataFrame({"id": list(range(1, 15)), "feature": list(range(14))}).to_html(
        data_dir / "leaderboard_features.html",
        index=False,
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    csv_sample = read_table(resolved)
    html_sample = read_table(data_dir / "sample_submission.html")
    assert csv_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]
    assert html_sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]


def test_ensure_local_sample_submission_file_preserves_context_compressed_csv_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(context_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n2,0.2\n")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.csv.gz").is_file()


def test_ensure_local_sample_submission_file_preserves_context_excel_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_excel(
        context_dir / "sample_submission.xlsx",
        index=False,
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.xlsx").is_file()


def test_ensure_local_sample_submission_file_preserves_context_orc_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source = context_dir / "sample_submission.orc"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_orc(source, index=False)

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    mirrored = data_dir / "sample_submission.orc"
    assert mirrored.is_file()
    mirrored_frame = read_table(mirrored)
    assert list(mirrored_frame.columns) == ["id", "target"]
    assert mirrored_frame["id"].tolist() == [1, 2]


def test_ensure_local_sample_submission_file_preserves_context_zip_wrapped_parquet_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source = context_dir / "sample_submission.parquet.zip"
    payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_parquet(payload, index=False)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("nested/sample_submission.parquet", payload.getvalue())

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    mirrored = data_dir / "sample_submission.parquet.zip"
    assert mirrored.is_file()
    assert mirrored.read_bytes() == source.read_bytes()
    mirrored_frame = read_table(mirrored)
    assert mirrored_frame.to_dict("list") == {"id": [1, 2], "target": [0.1, 0.2]}


def test_ensure_local_sample_submission_file_expands_zip_wrapped_parquet_sample_to_canonical_csv(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": list(range(1, 15)), "feature": list(range(14))}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    source = context_dir / "sample_submission.parquet.zip"
    payload = io.BytesIO()
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]}).to_parquet(payload, index=False)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sample_submission.parquet", payload.getvalue())

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    sample = read_table(resolved)
    assert sample["id"].astype(str).tolist() == [str(idx) for idx in range(1, 15)]
    assert (data_dir / "sample_submission.parquet.zip").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("suffix", [".hdf", ".hdf5"])
def test_ensure_local_sample_submission_file_preserves_context_hdf_sample(tmp_path: Path, suffix: str) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    source = context_dir / f"sample_submission{suffix}"
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_hdf(
        source,
        key="sample_submission",
        mode="w",
        format="table",
        index=False,
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    mirrored = data_dir / f"sample_submission{suffix}"
    assert mirrored.is_file()
    mirrored_frame = read_table(mirrored)
    assert list(mirrored_frame.columns) == ["id", "target"]
    assert mirrored_frame["id"].tolist() == [1, 2]


def test_ensure_local_sample_submission_file_preserves_context_sqlite_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    sqlite_sample = context_dir / "sample_submission.sqlite"
    with sqlite3.connect(sqlite_sample) as conn:
        conn.execute("CREATE TABLE sample_submission (id INTEGER, target REAL)")
        conn.executemany("INSERT INTO sample_submission VALUES (?, ?)", [(1, 0.1), (2, 0.2)])

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    mirrored = data_dir / "sample_submission.sqlite"
    assert mirrored.is_file()
    with sqlite3.connect(mirrored) as conn:
        rows = conn.execute("SELECT id, target FROM sample_submission ORDER BY id").fetchall()
    assert rows == [(1, 0.1), (2, 0.2)]


def test_ensure_local_sample_submission_file_prefers_real_data_sample_over_header_only_context(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.csv").write_text("id,target\n", encoding="utf-8")
    (data_dir / "SampleSubmission.jsonl").write_text(
        '{"id": 1, "target": 0.1}\n{"id": 2, "target": 0.2}\n',
        encoding="utf-8",
    )

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"


def test_ensure_local_sample_submission_file_replaces_header_only_canonical_with_real_sample(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample_submission.csv").write_text("id,target\n", encoding="utf-8")
    source_payload = '{"id": 1, "target": 0.1}\n{"id": 2, "target": 0.2}\n'
    (data_dir / "SampleSubmission.jsonl").write_text(source_payload, encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.jsonl").read_text(encoding="utf-8") == source_payload


def test_ensure_local_sample_submission_file_mirrors_context_suffix_when_canonical_exists(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    data_dir = tmp_path / "demo" / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    source_payload = '{"id": 1, "target": 0.1}\n{"id": 2, "target": 0.2}\n'
    (context_dir / "sample_submission.jsonl").write_text(source_payload, encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0.1\n2,0.2\n"
    assert (data_dir / "sample_submission.jsonl").read_text(encoding="utf-8") == source_payload
