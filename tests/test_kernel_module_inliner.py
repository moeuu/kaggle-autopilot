from __future__ import annotations

from pathlib import Path

from kagglebot.kernel_module_inliner import (
    discover_inline_modules,
    inline_kernel_modules,
    modules_with_alias_imports,
    strip_module_headers,
    strip_module_import,
)


def test_inline_kernel_modules_inserts_module_before_main_guard(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "helper.py").write_text(
        "\n".join(
            [
                "#!/usr/bin/env python",
                "# -*- coding: utf-8 -*-",
                "from __future__ import annotations",
                "",
                "VALUE = 42",
                "",
                "def answer():",
                "    return VALUE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from helper import answer",
                "",
                "print(answer())",
                "",
                "if __name__ == '__main__':",
                "    print('main')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inline_kernel_modules(kernel_dir, modules=("helper",))

    text = kernel_path.read_text(encoding="utf-8")
    assert "from helper import answer" not in text
    assert "# --- Begin inlined module: helper.py ---" in text
    assert "VALUE = 42" in text
    assert text.index("VALUE = 42") < text.index("if __name__ == '__main__':")
    assert text.endswith("\n")


def test_inline_kernel_modules_skips_alias_imports(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text("import helper as hp\nprint(hp.VALUE)\n", encoding="utf-8")

    inline_kernel_modules(kernel_dir, modules=("helper",))

    assert kernel_path.read_text(encoding="utf-8") == "import helper as hp\nprint(hp.VALUE)\n"


def test_discover_inline_modules_only_returns_imported_local_modules(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "used.py").write_text("VALUE = 1\n", encoding="utf-8")
    (kernel_dir / "unused.py").write_text("VALUE = 2\n", encoding="utf-8")
    (kernel_dir / "not-a-module.py").write_text("VALUE = 3\n", encoding="utf-8")
    lines = ["from used import VALUE", "print(VALUE)"]

    assert discover_inline_modules(kernel_dir, lines) == ("used",)


def test_modules_with_alias_imports_falls_back_on_syntax_error() -> None:
    lines = ["import helper as hp", "def broken(: pass"]

    assert modules_with_alias_imports(lines, ("helper",)) == {"helper"}


def test_strip_module_import_handles_multiline_from_import() -> None:
    lines = [
        "from helper import (",
        "    answer,",
        "    value,",
        ")",
        "print('ok')",
    ]

    assert strip_module_import(lines, "helper") == ["print('ok')"]


def test_strip_module_headers_removes_executable_headers_and_future_imports() -> None:
    lines = [
        "#!/usr/bin/env python",
        "# -*- coding: utf-8 -*-",
        "from __future__ import annotations",
        "",
        "VALUE = 1",
    ]

    assert strip_module_headers(lines) == ["VALUE = 1"]
