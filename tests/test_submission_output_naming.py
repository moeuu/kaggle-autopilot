from __future__ import annotations

import pytest

from kagglebot.submission_output_naming import (
    all_submission_output_suffixes,
    all_submission_output_suffixes_ordered,
    configured_submission_filename_is_template,
    expected_output_filename_from_text,
    first_allowed_expected_output_suffix,
    non_tabular_submission_output_suffixes,
    non_tabular_submission_output_suffixes_ordered,
    output_filename_from_format_text,
    tabular_submission_output_suffixes,
    tabular_submission_output_suffixes_ordered,
)


def test_all_submission_output_suffixes_covers_tabular_assets_models_and_archives() -> None:
    suffixes = all_submission_output_suffixes()

    for suffix in (
        ".csv",
        ".html.zst",
        ".jsonl",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".jxl",
        ".exr",
        ".svg.gz",
        ".svgz",
        ".epub",
        ".pptx",
        ".vcf.gz",
        ".bam",
        ".fasta.gz",
        ".smi",
        ".smiles.gz",
        ".graphml.bz2",
        ".mid",
        ".edf",
        ".hea.gz",
        ".mrc",
        ".ccp4",
        ".hgt",
        ".ecw",
        ".vhdr",
        ".ttl",
        ".jsonld",
        ".py",
        ".ipynb",
        ".nii.gz",
        ".npz",
        ".mpg",
        ".ply.zst",
        ".n5",
        ".ome.zarr",
        ".zarr",
        ".onnx",
        ".xgb",
        ".cbm",
        ".engine",
        ".rknn",
        ".hef",
        ".dlc",
        ".savedmodel",
        ".hfmodel",
        ".mlflowmodel",
        ".mlpackage",
        ".mlmodelc",
        ".tfcheckpoint",
        ".safetensors.index.json",
        ".tar.zst",
        ".7z",
        ".rar",
    ):
        assert suffix in suffixes


def test_all_submission_output_suffixes_ordered_prefers_longest_suffixes() -> None:
    suffixes = all_submission_output_suffixes_ordered()

    assert suffixes == tuple(sorted(all_submission_output_suffixes(), key=len, reverse=True))
    assert suffixes.index(".safetensors.index.json") < suffixes.index(".json")
    assert suffixes.index(".tar.zst") < suffixes.index(".tar")


def test_tabular_submission_output_suffixes_excludes_non_tabular_artifacts() -> None:
    suffixes = tabular_submission_output_suffixes()

    assert ".csv" in suffixes
    assert ".orc" in suffixes
    assert ".html" in suffixes
    assert ".html.zst" in suffixes
    assert ".nii.gz" not in suffixes
    assert ".tar.zst" not in suffixes


def test_tabular_submission_output_suffixes_ordered_prefers_longest_suffixes() -> None:
    suffixes = tabular_submission_output_suffixes_ordered()

    assert suffixes == tuple(sorted(tabular_submission_output_suffixes(), key=len, reverse=True))
    assert suffixes.index(".jsonl.zst") < suffixes.index(".jsonl")
    assert suffixes.index(".csv.gz") < suffixes.index(".csv")


def test_non_tabular_submission_output_suffixes_cover_assets_and_archives_only() -> None:
    suffixes = non_tabular_submission_output_suffixes()

    assert suffixes == all_submission_output_suffixes() - tabular_submission_output_suffixes()
    assert ".nii.gz" in suffixes
    assert ".safetensors.index.json" in suffixes
    assert ".tar.zst" in suffixes
    assert ".sqlite" in suffixes
    assert ".sqlite3" in suffixes
    assert ".db" in suffixes
    assert ".csv" not in suffixes
    assert ".jsonl.zst" not in suffixes


def test_non_tabular_submission_output_suffixes_ordered_prefers_longest_suffixes() -> None:
    suffixes = non_tabular_submission_output_suffixes_ordered()

    assert suffixes == tuple(sorted(non_tabular_submission_output_suffixes(), key=len, reverse=True))
    assert suffixes.index(".safetensors.index.json") < suffixes.index(".safetensors")
    assert suffixes.index(".tar.zst") < suffixes.index(".tar")


def test_configured_submission_filename_is_template_detects_sample_and_template_names() -> None:
    for name in (
        "sample_submission.csv",
        "sample-submission.csv.gz",
        "sample_solution.csv",
        "solutions_template.jsonl",
        "submission_template.parquet",
    ):
        assert configured_submission_filename_is_template(name)

    for name in ("answers.csv", "predictions.zarr", "submission.tar.zst"):
        assert not configured_submission_filename_is_template(name)


def test_first_allowed_expected_output_suffix_normalizes_missing_dot_and_order() -> None:
    assert (
        first_allowed_expected_output_suffix(
            ["tar.zst", ".csv"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == ".tar.zst"
    )


def test_expected_output_filename_from_text_handles_prose_filename_variants() -> None:
    assert (
        expected_output_filename_from_text(
            "Submission File Format\nFilename: answers.nii.gz\n",
            expected_suffixes=[".nii.gz"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == "answers.nii.gz"
    )
    assert (
        expected_output_filename_from_text(
            "Save your predictions as predictions.zarr before submitting.",
            expected_suffixes=[".zarr"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == "predictions.zarr"
    )
    assert (
        expected_output_filename_from_text(
            "The output should be named as model_bundle.tar.zst.",
            expected_suffixes=[".tar.zst"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == "model_bundle.tar.zst"
    )
    assert (
        expected_output_filename_from_text(
            "Upload `model.safetensors.index.json` as the final output.",
            expected_suffixes=[".safetensors.index.json"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == "model.safetensors.index.json"
    )


def test_expected_output_filename_from_text_still_ignores_sample_template_names() -> None:
    assert (
        expected_output_filename_from_text(
            "Use sample_submission.csv as the template, then save predictions as answers.csv.",
            expected_suffixes=[".csv"],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == "answers.csv"
    )


def test_output_filename_from_format_text_prefers_explicit_name_then_suffix_fallback() -> None:
    suffixes = all_submission_output_suffixes()

    assert (
        output_filename_from_format_text(
            "Upload `model.safetensors.index.json` as the final output.",
            expected_suffixes=[".safetensors.index.json"],
            allowed_suffixes=suffixes,
        )
        == "model.safetensors.index.json"
    )
    assert (
        output_filename_from_format_text(
            "Upload a safetensors index JSON file for scoring.",
            expected_suffixes=["safetensors.index.json"],
            allowed_suffixes=suffixes,
        )
        == "submission.safetensors.index.json"
    )
    assert (
        output_filename_from_format_text(
            "Upload a Hugging Face model directory for scoring.",
            expected_suffixes=["hfmodel"],
            allowed_suffixes=suffixes,
        )
        == "submission.hfmodel"
    )
    assert (
        output_filename_from_format_text(
            "Upload a SQLite database for scoring.",
            expected_suffixes=["sqlite"],
            allowed_suffixes=suffixes,
        )
        == "submission.sqlite"
    )
    assert (
        output_filename_from_format_text(
            "Upload `predictions.sqlite3` as the final output.",
            expected_suffixes=[".sqlite3"],
            allowed_suffixes=suffixes,
        )
        == "predictions.sqlite3"
    )


@pytest.mark.parametrize(
    "suffix",
    [
        ".edf",
        ".fasta.gz",
        ".graphml.bz2",
        ".hea.gz",
        ".ply.zst",
        ".pptx",
        ".smi",
        ".smiles.gz",
        ".svg.gz",
        ".vcf.gz",
        ".mrc",
        ".hgt",
        ".vhdr",
        ".jsonld",
        ".sqlite",
        ".xgb",
        ".cbm",
        ".engine",
        ".rknn",
        ".dlc",
        ".npz",
        ".py",
        ".tar.zst",
    ],
)
def test_output_filename_from_format_text_falls_back_for_inferred_artifact_suffixes(suffix: str) -> None:
    assert (
        output_filename_from_format_text(
            "Upload a file for scoring.",
            expected_suffixes=[suffix],
            allowed_suffixes=all_submission_output_suffixes(),
        )
        == f"submission{suffix}"
    )
