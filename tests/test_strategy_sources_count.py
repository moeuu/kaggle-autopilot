from __future__ import annotations

import kagglebot.orchestrator.agent_pipeline as agent_pipeline


def test_count_source_items_accepts_numbered_sources_heading() -> None:
    text = "\n".join(
        [
            "## 12) Sources (>=3)",
            "- A (example.com)",
            "- B (example.org)",
            "- C (example.net)",
        ]
    )
    assert agent_pipeline._count_source_items(text) == 3


def test_count_source_items_tolerates_blank_lines_in_sources_list() -> None:
    text = "\n".join(
        [
            "## Sources",
            "- A (example.com)",
            "",
            "- B (example.org)",
            "",
            "- C (example.net)",
            "",
        ]
    )
    assert agent_pipeline._count_source_items(text) == 3
