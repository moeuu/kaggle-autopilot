from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.code_reference import (
    CodeReferenceNotebook,
    code_reference_marker,
    extract_code_reference_score,
    load_ensemble_reference_notebook,
    load_required_reference_notebook,
    reference_requires_tabicl,
    validate_code_reference_implementation,
)
from kagglebot.paths import CompetitionPaths


def test_extract_code_reference_score_prefers_required_index_entry(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [
                    {"kernel_id": "alice/other-kernel", "score": 0.701},
                    {"kernel_id": "alice/ref-kernel", "score": "0.741"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.code_md_path.write_text("notebook_score: 0.612\n", encoding="utf-8")

    score, source = extract_code_reference_score(paths)

    assert score == pytest.approx(0.741)
    assert source == "code_index:alice/ref-kernel"


def test_extract_code_reference_score_falls_back_to_required_markdown_section(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_md_path.write_text(
        "\n".join(
            [
                "Top public notebooks",
                "unrelated score 0.601",
                "Required reference notebook",
                "The required implementation has score 0.833 and should be used.",
            ]
        ),
        encoding="utf-8",
    )

    score, source = extract_code_reference_score(paths)

    assert score == pytest.approx(0.833)
    assert source == "code_md:required_reference_section"


def test_load_reference_notebooks_and_validate_marker_requirements(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/gold-main",
                "ensemble_reference_kernel_id": "bob/blend",
                "notebooks": [
                    {
                        "kernel_id": "alice/gold-main",
                        "title": "TabICL gold solution",
                        "source_file": "kernel.py",
                        "summary": "Uses TabICL over tabular prompts.",
                    },
                    {"kernel_id": "bob/blend", "title": "OOF blend"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    required = load_required_reference_notebook(paths)
    ensemble = load_ensemble_reference_notebook(paths)

    assert required == CodeReferenceNotebook(
        kernel_id="alice/gold-main",
        title="TabICL gold solution",
        source_file="kernel.py",
        summary="Uses TabICL over tabular prompts.",
    )
    assert ensemble is not None
    assert ensemble.kernel_id == "bob/blend"
    assert reference_requires_tabicl(required)

    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(f"{code_reference_marker(required)}\nprint('baseline')\n", encoding="utf-8")

    assert validate_code_reference_implementation(kernel_path=kernel_path, reference=required) == [
        "missing_tabicl_implementation_path"
    ]

    kernel_path.write_text(
        f"{code_reference_marker(required)}\n# real tabicl implementation path\nTABICL_ENABLED = True\n",
        encoding="utf-8",
    )

    assert validate_code_reference_implementation(kernel_path=kernel_path, reference=required) == []
