from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_pipeline_cfg import inject_pipeline_cfg_fallback


def test_inject_pipeline_cfg_fallback_replaces_keyerror(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "def get_pipeline_cfg(plan, name):",
                "    for p in plan.get('pipelines', []):",
                "        if p.get('name') == name:",
                "            return p",
                '    raise KeyError(f"Pipeline not found in plan: {name}")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_pipeline_cfg_fallback(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:pipeline_cfg_fallback" in text
    assert "raise KeyError" not in text
    assert "missing_pipeline_in_plan" in text


def test_inject_pipeline_cfg_fallback_is_idempotent(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "def get_pipeline_cfg(plan, name):",
                '    raise KeyError(f"Pipeline not found in plan: {name}")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_pipeline_cfg_fallback(kernel_dir)
    first = kernel_path.read_text(encoding="utf-8")
    inject_pipeline_cfg_fallback(kernel_dir)
    second = kernel_path.read_text(encoding="utf-8")

    assert second == first
    assert second.count("kagglebot:pipeline_cfg_fallback") == 1
