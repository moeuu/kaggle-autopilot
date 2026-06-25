from __future__ import annotations

import ast
import re
from pathlib import Path


def inline_kernel_modules(kernel_dir: Path, modules: tuple[str, ...] | None = None) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if modules is None:
        modules = discover_inline_modules(kernel_dir, lines)
    if not modules or not kernel_imports_local_modules(lines, modules):
        return
    alias_modules = modules_with_alias_imports(lines, modules)
    if alias_modules:
        modules = tuple(module for module in modules if module not in alias_modules)
        if not modules:
            return

    stripped = lines
    for module in modules:
        stripped = strip_module_import(stripped, module)

    module_blocks: list[str] = []
    for module in modules:
        module_path = kernel_dir / f"{module}.py"
        if not module_path.exists():
            continue
        module_lines = module_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        cleaned = strip_module_headers(module_lines)
        cleaned = strip_local_module_imports(cleaned, modules)
        if not cleaned:
            continue
        module_blocks.append(f"# --- Begin inlined module: {module}.py ---")
        module_blocks.extend(cleaned)
        module_blocks.append(f"# --- End inlined module: {module}.py ---")

    if not module_blocks:
        return

    insert_at = find_main_guard_index(stripped)
    new_lines = stripped[:insert_at] + [""] + module_blocks + [""] + stripped[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def kernel_imports_local_modules(lines: list[str], modules: tuple[str, ...]) -> bool:
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                return True
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                return True
    return False


def modules_with_alias_imports(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    if not modules:
        return set()
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return modules_with_alias_imports_fallback(lines, modules)

    alias_modules: set[str] = set()
    module_set = set(modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".", 1)[0]
                if base in module_set and alias.asname:
                    alias_modules.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module.split(".", 1)[0]
            if base not in module_set:
                continue
            for alias in node.names:
                if alias.asname:
                    alias_modules.add(base)
                    break
    return alias_modules


def modules_with_alias_imports_fallback(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    alias_modules: set[str] = set()
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*import\s+{re.escape(module)}\s+as\s+\w+", line):
                alias_modules.add(module)
                continue
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                if " as " in line:
                    alias_modules.add(module)
    return alias_modules


def strip_module_import(lines: list[str], module: str) -> list[str]:
    output: list[str] = []
    skipping = False
    paren_depth = 0
    for line in lines:
        if not skipping:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                skipping = True
                paren_depth = line.count("(") - line.count(")")
                if paren_depth <= 0 and not line.rstrip().endswith("\\"):
                    skipping = False
                continue
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                continue
            output.append(line)
            continue
        paren_depth += line.count("(") - line.count(")")
        if paren_depth <= 0 and not line.rstrip().endswith("\\"):
            skipping = False
        continue
    return output


def discover_inline_modules(kernel_dir: Path, lines: list[str]) -> tuple[str, ...]:
    module_names: list[str] = []
    for path in kernel_dir.glob("*.py"):
        if path.name == "kernel.py":
            continue
        name = path.stem
        if name.isidentifier():
            module_names.append(name)
    if not module_names:
        return ()
    used: list[str] = []
    for name in module_names:
        if kernel_imports_local_modules(lines, (name,)):
            used.append(name)
    return tuple(used)


def strip_module_headers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if not cleaned and line.startswith("#!"):
            continue
        if not cleaned and re.match(r"^#.*coding[:=]\s*[-\w.]+", line):
            continue
        if re.match(r"^\s*from\s+__future__\s+import\s+", line):
            continue
        cleaned.append(line)
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)
    return cleaned


def strip_local_module_imports(lines: list[str], modules: tuple[str, ...]) -> list[str]:
    cleaned = lines
    for module in modules:
        cleaned = strip_module_import(cleaned, module)
    return cleaned


def find_main_guard_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            return idx
    return len(lines)
