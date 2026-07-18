from __future__ import annotations

from pathlib import Path

from kagglebot.local_kernel_pipeline_cfg import (
    inject_pipeline_cfg_fallback,
    inject_staged_plan_path_fallback,
    inject_staged_plan_payload_fallback,
)


def test_inject_staged_plan_payload_fallback_embeds_plan_for_kaggle_working(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text('{"pipelines":[{"name":"strong"}]}\n', encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "from __future__ import annotations\nprint('run')\n",
        encoding="utf-8",
    )

    inject_staged_plan_payload_fallback(kernel_dir)
    first = kernel_path.read_text(encoding="utf-8")
    inject_staged_plan_payload_fallback(kernel_dir)

    assert kernel_path.read_text(encoding="utf-8") == first
    assert first.count("kagglebot:staged_plan_payload_fallback") == 1
    assert '"/kaggle/working/plan.json"' in first
    assert '"pipelines":[{"name":"strong"}]' in first


def test_inject_staged_plan_path_fallback_prefers_packaged_plan(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text("{}\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "KERNEL_DIR = Path(__file__).resolve().parent",
                "ARTIFACT_ROOT = KERNEL_DIR.parent",
                'PLAN_PATH = ARTIFACT_ROOT / "plan.json"',
                "if not PLAN_PATH.exists():",
                '    PLAN_PATH = Path("/kaggle/working/plan.json")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_staged_plan_path_fallback(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:staged_plan_path_fallback" in text
    assert 'PLAN_PATH = KERNEL_DIR / "plan.json"' in text
    assert 'PLAN_PATH = KERNEL_DIR / "/kaggle/working/plan.json"' in text
    assert text.index("staged_plan_path_fallback") < text.index('Path("/kaggle/working/plan.json")')


def test_inject_staged_plan_path_fallback_is_idempotent(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text("{}\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        'KERNEL_DIR = Path(__file__).parent\nPLAN_PATH = KERNEL_DIR.parent / "plan.json"\n',
        encoding="utf-8",
    )

    inject_staged_plan_path_fallback(kernel_dir)
    first = kernel_path.read_text(encoding="utf-8")
    inject_staged_plan_path_fallback(kernel_dir)

    assert kernel_path.read_text(encoding="utf-8") == first
    assert first.count("kagglebot:staged_plan_path_fallback") == 1


def test_inject_staged_plan_path_fallback_supports_frozen_plan_path(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text("{}\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "KERNEL_DIR = Path(__file__).resolve().parent",
                'FROZEN_PLAN_PATH = KERNEL_DIR / "plan.json"',
                'UPSTREAM_PLAN_PATH = KERNEL_DIR.parent / "plan.json"',
                "def load_frozen_plan():",
                "    return FROZEN_PLAN_PATH.read_text()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_staged_plan_path_fallback(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:staged_plan_path_fallback" in text
    assert 'FROZEN_PLAN_PATH = KERNEL_DIR / "plan.json"' in text
    assert 'FROZEN_PLAN_PATH = KERNEL_DIR / "/kaggle/working/plan.json"' in text


def test_inject_staged_plan_path_fallback_supports_plan_loader_candidates(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text('{"pipelines": []}\n', encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "KERNEL_DIR = Path(__file__).resolve().parent",
                "ARTIFACT_ROOT = KERNEL_DIR.parent",
                "def _load_frozen_plan():",
                '    for path in (ARTIFACT_ROOT / "plan.json", KERNEL_DIR / "plan.json"):',
                "        if path.exists():",
                "            return path.read_text()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_staged_plan_path_fallback(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:staged_plan_path_fallback" in text
    assert 'for path in (Path("/kaggle/working/plan.json"), ARTIFACT_ROOT' in text


def test_inject_staged_plan_path_fallback_supports_multiline_plan_loader(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "plan.json").write_text("{}\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "KERNEL_DIR = Path(__file__).resolve().parent",
                "ARTIFACT_ROOT = KERNEL_DIR.parent",
                "def _load_frozen_plan():",
                "    for path in (",
                '        ARTIFACT_ROOT / "plan.json",',
                '        KERNEL_DIR / "plan.json",',
                "    ):",
                "        if path.exists():",
                "            return path.read_text()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_staged_plan_path_fallback(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:staged_plan_path_fallback" in text
    assert '        Path("/kaggle/working/plan.json"),' in text


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
