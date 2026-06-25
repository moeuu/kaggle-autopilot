from __future__ import annotations

from pathlib import Path

from kagglebot.local_sample_submission import ensure_local_sample_submission_file


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


def test_ensure_local_sample_submission_file_copies_context_sample(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    resolved = ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == tmp_path / "demo" / "data" / "sample_submission.csv"
    assert resolved.read_text(encoding="utf-8") == "id,target\n1,0\n"
