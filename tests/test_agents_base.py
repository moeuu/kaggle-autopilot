"""
Tests for agent base utilities (delimiter parsing, template rendering).

Uses tmp_path fixtures to avoid touching real artifacts.
"""

from pathlib import Path

import pytest

from kagglebot.agents import (
    DelimiterParseError,
    parse_claude_strategy_output,
    render_prompt_template,
    verify_outputs_exist,
)


def test_parse_claude_output_valid():
    """Test parsing valid Claude strategy output."""
    output = """
Some preamble text that should be ignored.

===CLAUDE_STRATEGY===
This is the strategy section.
It can have multiple lines.
===CODEX_IMPLEMENTATION_INSTRUCTIONS===
These are the implementation instructions.
Step 1: Do this
Step 2: Do that
===REFERENCES===
- Reference 1
- Reference 2
"""

    parsed = parse_claude_strategy_output(output)

    assert "strategy section" in parsed.strategy
    assert "multiple lines" in parsed.strategy
    assert "implementation instructions" in parsed.codex_instructions
    assert "Step 1" in parsed.codex_instructions
    assert "Reference 1" in parsed.references
    assert parsed.raw_output == output


def test_parse_claude_output_no_references():
    """Test parsing output without optional REFERENCES section."""
    output = """
===CLAUDE_STRATEGY===
Strategy content here.
===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Implementation instructions here.
"""

    parsed = parse_claude_strategy_output(output)

    assert "Strategy content" in parsed.strategy
    assert "Implementation instructions" in parsed.codex_instructions
    assert parsed.references == ""


def test_parse_claude_output_missing_strategy():
    """Test that missing CLAUDE_STRATEGY section raises error."""
    output = """
===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Instructions only.
"""

    with pytest.raises(DelimiterParseError, match="CLAUDE_STRATEGY"):
        parse_claude_strategy_output(output)


def test_parse_claude_output_missing_codex_instructions():
    """Test that missing CODEX_IMPLEMENTATION_INSTRUCTIONS section raises error."""
    output = """
===CLAUDE_STRATEGY===
Strategy only.
"""

    with pytest.raises(DelimiterParseError, match="CODEX_IMPLEMENTATION_INSTRUCTIONS"):
        parse_claude_strategy_output(output)


def test_parse_claude_output_empty_sections():
    """Test parsing with empty but present sections."""
    output = """
===CLAUDE_STRATEGY===

===CODEX_IMPLEMENTATION_INSTRUCTIONS===

===REFERENCES===

"""

    parsed = parse_claude_strategy_output(output)

    assert parsed.strategy == ""
    assert parsed.codex_instructions == ""
    assert parsed.references == ""


def test_render_prompt_template_basic(tmp_path: Path):
    """Test basic template rendering."""
    template_path = tmp_path / "template.md"
    template_path.write_text("Hello {{name}}, your age is {{age}}.")

    rendered = render_prompt_template(template_path, {"name": "Alice", "age": "30"})

    assert rendered == "Hello Alice, your age is 30."


def test_render_prompt_template_multiline(tmp_path: Path):
    """Test rendering multi-line template."""
    template_path = tmp_path / "template.md"
    template_path.write_text(
        """
# Competition: {{slug}}

Metric: {{metric}}
Target: {{target}}
"""
    )

    rendered = render_prompt_template(
        template_path,
        {
            "slug": "titanic",
            "metric": "accuracy",
            "target": "0.80",
        },
    )

    assert "Competition: titanic" in rendered
    assert "Metric: accuracy" in rendered
    assert "Target: 0.80" in rendered


def test_render_prompt_template_missing_variable(tmp_path: Path):
    """Test that undefined variables raise KeyError."""
    template_path = tmp_path / "template.md"
    template_path.write_text("Hello {{name}}, you are {{age}} years old.")

    with pytest.raises(KeyError, match="age"):
        render_prompt_template(template_path, {"name": "Alice"})


def test_render_prompt_template_extra_variables(tmp_path: Path):
    """Test that extra variables don't cause errors."""
    template_path = tmp_path / "template.md"
    template_path.write_text("Hello {{name}}!")

    rendered = render_prompt_template(
        template_path,
        {"name": "Alice", "age": "30", "city": "NYC"},
    )

    assert rendered == "Hello Alice!"


def test_render_prompt_template_no_variables(tmp_path: Path):
    """Test rendering template with no variables."""
    template_path = tmp_path / "template.md"
    template_path.write_text("Static template content.")

    rendered = render_prompt_template(template_path, {})

    assert rendered == "Static template content."


def test_verify_outputs_exist_all_present(tmp_path: Path):
    """Test verification when all files exist."""
    (tmp_path / "file1.txt").write_text("content")
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.txt").write_text("content")

    # Should not raise
    verify_outputs_exist(tmp_path, ["file1.txt", "dir1/file2.txt"])


def test_verify_outputs_exist_missing_file(tmp_path: Path):
    """Test verification raises error for missing files."""
    (tmp_path / "file1.txt").write_text("content")

    with pytest.raises(FileNotFoundError, match="file2.txt"):
        verify_outputs_exist(tmp_path, ["file1.txt", "file2.txt"])


def test_verify_outputs_exist_multiple_missing(tmp_path: Path):
    """Test verification reports all missing files."""
    with pytest.raises(FileNotFoundError, match="file1.txt.*file2.txt"):
        verify_outputs_exist(tmp_path, ["file1.txt", "file2.txt"])


def test_verify_outputs_exist_empty_list(tmp_path: Path):
    """Test verification with empty required files list."""
    # Should not raise
    verify_outputs_exist(tmp_path, [])
