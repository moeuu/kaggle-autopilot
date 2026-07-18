from __future__ import annotations

import ast
import re
from pathlib import Path

from kagglebot.exceptions import KernelFailedError


def sanitize_submit_inference_output_roots(kernel_dir: Path) -> None:
    """Rewrite staged submit-kernel output roots from the source tree into `/kaggle/working`."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    working_root = "Path('/kaggle/working')"
    updated = text
    patterns = (
        re.compile(r"\b(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*/\s*(['\"])(?:output|outputs)\1"),
        re.compile(r"\b(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\.joinpath\(\s*(['\"])(?:output|outputs)\1\s*\)"),
        re.compile(
            r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)"
            r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*/\s*(['\"])(?:output|outputs)\2",
            re.MULTILINE,
        ),
        re.compile(
            r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)"
            r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\.joinpath\(\s*(['\"])(?:output|outputs)\2\s*\)",
            re.MULTILINE,
        ),
    )
    for pattern in patterns:
        if pattern.groups >= 2:
            updated = pattern.sub(rf"\1{working_root}", updated)
        else:
            updated = pattern.sub(working_root, updated)
    if updated != text:
        kernel_path.write_text(updated, encoding="utf-8")


def validate_inference_submit_kernel(kernel_dir: Path) -> None:
    """Reject notebook submit kernels that still look like local wrapper artifacts or write to read-only paths."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        raise KernelFailedError("Notebook submit kernel is missing kernel.py.")
    if (kernel_dir / "output").exists():
        raise KernelFailedError(
            "Invalid notebook submit artifact for code competition inference mode: "
            "found staged output directory in notebook package."
        )
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    _validate_inference_server_lifecycle(text)
    if (kernel_dir / "plan.json").exists() and "plan_path" in lowered:
        if "# kagglebot:staged_plan_payload_fallback" not in text:
            raise KernelFailedError(
                "Invalid notebook submit artifact for code competition inference mode: "
                "staged plan is not embedded for the relocated Kaggle runtime."
            )
    suspicious_fragments = (
        ("submit_only metrics payload", '"kind": "submit_only"'),
        ("embedded submission wrapper payload", "submission_gzip_b64"),
        ("read-only kaggle source output path", "/kaggle/src/output"),
        ("read-only kaggle source outputs path", "/kaggle/src/outputs"),
    )
    for label, fragment in suspicious_fragments:
        if fragment in lowered:
            raise KernelFailedError(
                f"Invalid notebook submit artifact for code competition inference mode: found {label} in staged kernel."
            )
    direct_readonly_patterns = (
        (
            "read-only kaggle source joinpath output path",
            re.compile(r"/kaggle/src['\"]?\s*\)?\s*\.joinpath\(\s*[\"']outputs?[\"']\s*\)", re.IGNORECASE),
        ),
        (
            "staged output root assignment under kaggle source",
            re.compile(
                r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
                r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*(?:/\s*[\"']outputs?[\"']|\.joinpath\(\s*[\"']outputs?[\"']\s*\))",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
    )
    for label, pattern in direct_readonly_patterns:
        if pattern.search(text):
            raise KernelFailedError(
                f"Invalid notebook submit artifact for code competition inference mode: found {label} in staged kernel."
            )
    persisted_cache_patterns = (
        re.compile(r"/kaggle/working/[^\n'\"]*(?:cache|site-packages|wheelhouse)", re.IGNORECASE),
        re.compile(
            r"\b(?:KAGGLE_WORKING_DIR|WORKING_DIR)\s*/\s*['\"][^'\"]*(?:cache|site-packages|wheelhouse)[^'\"]*['\"]",
            re.IGNORECASE,
        ),
    )
    if any(pattern.search(text) for pattern in persisted_cache_patterns):
        raise KernelFailedError(
            "Invalid notebook submit artifact for code competition inference mode: "
            "dependency caches must use /tmp, not /kaggle/working, because persisted cache trees can break rerun "
            "submission."
        )
    readonly_root_patterns = {
        var_name: re.compile(rf"\b{var_name}\s*=.*?/kaggle/src\b", re.IGNORECASE)
        for var_name in ("kernel_dir", "artifact_dir", "artifact_root")
    }
    output_usage_templates = (
        ("output mirror path", r"\b{var}\s*/\s*[\"']outputs?[\"']"),
        ("output mirror joinpath", r"\b{var}\.joinpath\(\s*[\"']outputs?[\"']\s*\)"),
    )
    for var_name, root_pattern in readonly_root_patterns.items():
        if not root_pattern.search(text):
            continue
        for label_suffix, usage_template in output_usage_templates:
            if re.search(usage_template.format(var=var_name), text, re.IGNORECASE):
                raise KernelFailedError(
                    "Invalid notebook submit artifact for code competition inference mode: "
                    f"found staged {var_name} {label_suffix} in staged kernel."
                )
    if "/kaggle/working" not in lowered and "kaggle_working" not in lowered:
        raise KernelFailedError(
            "Invalid notebook submit artifact for code competition inference mode: "
            "kernel does not appear to write outputs under /kaggle/working."
        )


def _validate_inference_server_lifecycle(text: str) -> None:
    """Require the hosted server to be smoke-started in the visible notebook run."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    serve_calls: list[ast.Call] = []
    fallback_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "run_local_gateway":
            raise KernelFailedError(
                "Invalid notebook submit artifact for code competition inference mode: "
                "run_local_gateway is a local-only smoke test; the submitted kernel must call serve()."
            )
        if isinstance(function, ast.Attribute) and function.attr == "serve":
            serve_calls.append(node)
        if isinstance(function, ast.Name) and function.id == "write_fallback_submission":
            fallback_calls.append(node)

    for call in serve_calls:
        ancestor = parents.get(call)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                condition = ast.unparse(ancestor.test)
                if "KAGGLE_IS_COMPETITION_RERUN" in condition:
                    raise KernelFailedError(
                        "Invalid notebook submit artifact for code competition inference mode: "
                        "serve() is hidden behind KAGGLE_IS_COMPETITION_RERUN, so the visible run cannot smoke-test "
                        "the hosted inference server. Call serve() unconditionally."
                    )
            ancestor = parents.get(ancestor)

    if serve_calls and fallback_calls:
        first_serve_line = min(call.lineno for call in serve_calls)
        first_fallback_line = min(call.lineno for call in fallback_calls)
        if first_fallback_line < first_serve_line:
            raise KernelFailedError(
                "Invalid notebook submit artifact for code competition inference mode: "
                "the visible fallback runs before serve(), so server imports and construction are not validated. "
                "Call serve() before writing the visible-run fallback."
            )
