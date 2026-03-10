from __future__ import annotations

from kagglebot.bootstrap import _extract_usable_submission_section
from kagglebot.submission_format import extract_submission_section, parse_submission_format


def test_extract_submission_section_skips_submission_code_requirements_heading() -> None:
    markdown = (
        "## Foundational Rules\n\n"
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n"
    )
    assert extract_submission_section(markdown) is None


def test_extract_submission_section_prefers_real_submission_block() -> None:
    markdown = (
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n\n"
        "## Submission\n\n"
        "```csv\n"
        "id,prediction\n"
        "```\n"
    )
    section = extract_submission_section(markdown)
    assert section is not None
    hint = parse_submission_format(section)
    assert hint.columns == ["id", "prediction"]
    assert hint.expected_suffixes and hint.expected_suffixes[0] == ".csv"


def test_extract_usable_submission_section_rejects_rules_text() -> None:
    markdown = (
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, not a CSV header.\n"
    )
    assert _extract_usable_submission_section(markdown) is None


def test_parse_submission_format_prefers_zip_when_rules_require_zip_file() -> None:
    markdown = (
        "## Submission Format\n\n"
        "You must submit a ZIP file containing per-record predictions.\n"
        "```csv\nid,prediction\n```\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes is not None
    assert hint.expected_suffixes[0] == ".zip"
    assert ".csv" in hint.expected_suffixes
    assert hint.artifact_class == "multi_file_zip"


def test_parse_submission_format_reads_bullet_column_definitions() -> None:
    markdown = (
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename\n"
        "* `Category`: The predicted class\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.columns == ["Id", "Category"]
    assert hint.artifact_class == "tabular"


def test_parse_submission_format_treats_weights_and_inference_script_as_bundle() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Submit a ZIP archive containing model weights (.pt or .pth) and the inference script.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes == [".zip"]
    assert hint.artifact_class == "bundle"


def test_parse_submission_format_ignores_topology_json_noise() -> None:
    markdown = (
        "## Submission Format\n\n"
        "Include the topology JSON description in your documentation.\n"
        "Submit predictions through the official submission channel.\n"
    )
    hint = parse_submission_format(markdown)
    assert hint.expected_suffixes is None
    assert hint.artifact_class == "unknown"
