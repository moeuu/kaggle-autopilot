from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.competition_policy import load_competition_policy
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.kaggle_api import download_competition, download_dataset, kernels_pull
from kagglebot.paths import CompetitionPaths
from kagglebot.validators import safe_extract_zip

_NOTEBOOK_DATASET_REF_RE = re.compile(
    r"(?:kagglehub\.dataset_download|/datasets/)(?:\(|)(?:['\"])?"
    r"(?P<ref>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
_NOTEBOOK_COMPETITION_REF_RE = re.compile(r"/competitions/(?P<slug>[A-Za-z0-9_.-]+)")
_NOTEBOOK_KERNEL_REF_RE = re.compile(r"/code/(?P<ref>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def stage_reference_notebook_inputs(
    *,
    paths: CompetitionPaths,
    slug: str,
    download: bool,
    quiet: bool,
    dry_run: bool,
    download_competition_fn=download_competition,
    download_dataset_fn=download_dataset,
    download_kernel_fn=kernels_pull,
) -> None:
    index_path = paths.code_notebooks_index_path
    manifest_payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "required_reference_kernel_id": "",
        "ensemble_reference_kernel_id": "",
        "required_datasets": [],
        "required_capabilities": [],
        "missing_required_sources": [],
        "policy_tags": [],
        "reference_notebooks": [],
    }
    if not index_path.exists():
        write_json_object(paths.reference_inputs_manifest_path, manifest_payload)
        return
    index_payload = load_json_object(index_path, errors="ignore")
    if index_payload is None:
        write_json_object(paths.reference_inputs_manifest_path, manifest_payload)
        return

    notebooks = index_payload.get("notebooks")
    if not isinstance(notebooks, list):
        notebooks = []
    competition_policy = load_competition_policy(paths)
    required_kernel_id = str(index_payload.get("required_reference_kernel_id") or "").strip()
    ensemble_kernel_id = str(index_payload.get("ensemble_reference_kernel_id") or "").strip()
    manifest_payload["required_reference_kernel_id"] = required_kernel_id
    manifest_payload["ensemble_reference_kernel_id"] = ensemble_kernel_id
    manifest_payload["required_datasets"] = list(competition_policy.reference_inputs.required_datasets)
    manifest_payload["required_capabilities"] = list(competition_policy.required_capabilities)
    manifest_payload["policy_tags"] = list(competition_policy.archetype_tags)
    effective_download = bool(download or competition_policy.reference_inputs.proactive)
    target_ids = [kernel_id for kernel_id in (required_kernel_id, ensemble_kernel_id) if kernel_id]
    target_entries = [
        item for item in notebooks if isinstance(item, dict) and str(item.get("kernel_id") or "").strip() in target_ids
    ]
    if not target_entries:
        target_entries = [item for item in notebooks if isinstance(item, dict)][:1]

    manifest_entries: list[dict[str, object]] = []
    discovered_required_sources: set[tuple[str, str]] = set()
    for entry in target_entries:
        notebook_dir = Path(str(entry.get("local_dir") or "")).expanduser() if entry.get("local_dir") else None
        source_file = (
            Path(str(entry.get("source_file") or "")).expanduser()
            if entry.get("source_file")
            else (_choose_notebook_source_file(notebook_dir) if notebook_dir else None)
        )
        metadata = _load_notebook_metadata(notebook_dir) if notebook_dir else {}
        metadata_sources = _collect_metadata_input_sources(metadata)
        notebook_sources = _collect_notebook_text_input_sources(source_file)
        summary_sources = _collect_free_text_input_sources(str(entry.get("summary") or ""), source="summary_text")
        input_sources = _merge_reference_input_sources(
            metadata_sources,
            notebook_sources,
            summary_sources,
            _policy_reference_sources(paths=paths),
        )
        staged_sources: list[dict[str, object]] = []
        if input_sources and notebook_dir is not None:
            staged_sources = _stage_reference_sources(
                paths=paths,
                current_slug=slug,
                kernel_id=str(entry.get("kernel_id") or ""),
                input_sources=input_sources,
                download=effective_download,
                quiet=quiet,
                dry_run=dry_run,
                download_competition_fn=download_competition_fn,
                download_dataset_fn=download_dataset_fn,
                download_kernel_fn=download_kernel_fn,
            )
        for source in _merge_reference_input_sources(metadata_sources, notebook_sources, summary_sources):
            kind = str(source.get("kind") or "").strip().lower()
            ref = str(source.get("ref") or "").strip()
            if kind and ref:
                discovered_required_sources.add((kind, ref))
        manifest_entries.append(
            {
                "kernel_id": str(entry.get("kernel_id") or ""),
                "title": str(entry.get("title") or ""),
                "local_dir": str(notebook_dir) if notebook_dir else None,
                "source_file": str(source_file) if source_file else None,
                "metadata_path": str(notebook_dir / "kernel-metadata.json") if notebook_dir else None,
                "input_sources": input_sources,
                "staged_sources": staged_sources,
            }
        )
    manifest_payload["reference_notebooks"] = manifest_entries
    manifest_payload["missing_required_sources"] = [
        ref
        for ref in competition_policy.reference_inputs.required_datasets
        if ("dataset", ref) not in discovered_required_sources
    ]
    write_json_object(paths.reference_inputs_manifest_path, manifest_payload)


def _safe_kernel_dir_name(kernel_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", kernel_id.replace("/", "__")).strip("_")


def _choose_notebook_source_file(directory: Path) -> Path | None:
    for suffix in (".ipynb", ".py"):
        files = sorted(directory.glob(f"*{suffix}"))
        if files:
            return files[0]
    return None


def _normalize_cell_source(source: object) -> str:
    if isinstance(source, list):
        return "".join(str(chunk) for chunk in source)
    if isinstance(source, str):
        return source
    return ""


def _load_notebook_metadata(directory: Path) -> dict[str, object]:
    candidates = [directory / "kernel-metadata.json", *sorted(directory.glob("*metadata*.json"))]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        payload = load_json_object(path, errors="ignore")
        if payload is not None:
            return payload
    return {}


def _collect_metadata_input_sources(metadata: dict[str, object]) -> list[dict[str, str]]:
    if not metadata:
        return []

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_source(kind: str, ref: object, *, source: str) -> None:
        value = str(ref or "").strip().strip("/")
        if not value:
            return
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        sources.append({"kind": kind, "ref": value, "source": source})

    def handle_item(kind: str, item: object, *, source: str) -> None:
        if isinstance(item, str):
            add_source(kind, item, source=source)
            return
        if not isinstance(item, dict):
            return
        if kind == "dataset":
            for key in ("dataset", "source", "ref", "slug", "dataset_ref", "datasetSlug"):
                if key in item:
                    add_source(kind, item.get(key), source=source)
                    return
            owner = str(item.get("ownerSlug") or item.get("owner_slug") or item.get("owner") or "").strip()
            slug = str(item.get("datasetSlug") or item.get("dataset_slug") or item.get("slug") or "").strip()
            if owner and slug:
                add_source(kind, f"{owner}/{slug}", source=source)
            return
        if kind == "competition":
            for key in ("competition", "source", "ref", "slug", "competitionSlug"):
                if key in item:
                    add_source(kind, item.get(key), source=source)
                    return
        if kind == "kernel":
            for key in ("kernel", "source", "ref", "slug", "kernelSlug"):
                if key in item:
                    add_source(kind, item.get(key), source=source)
                    return

    for key, kind in (
        ("dataset_sources", "dataset"),
        ("competition_sources", "competition"),
        ("kernel_sources", "kernel"),
    ):
        raw = metadata.get(key)
        if isinstance(raw, list):
            for item in raw:
                handle_item(kind, item, source=f"metadata:{key}")

    raw_sources = metadata.get("dataSources")
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            source_type = str(item.get("sourceType") or item.get("type") or "").strip().lower()
            if "dataset" in source_type:
                handle_item("dataset", item, source="metadata:dataSources")
            elif "competition" in source_type:
                handle_item("competition", item, source="metadata:dataSources")
            elif "kernel" in source_type or "notebook" in source_type:
                handle_item("kernel", item, source="metadata:dataSources")
    return sources


def _collect_notebook_text_input_sources(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    text = content
    if path.suffix.lower() == ".ipynb":
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {}
        cells = payload.get("cells", []) if isinstance(payload, dict) else []
        if isinstance(cells, list):
            text = "\n".join(_normalize_cell_source(cell.get("source")) for cell in cells if isinstance(cell, dict))

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_source(kind: str, ref: str, source: str) -> None:
        value = ref.strip().strip("/")
        if not value:
            return
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        sources.append({"kind": kind, "ref": value, "source": source})

    for match in _NOTEBOOK_DATASET_REF_RE.finditer(text):
        add_source("dataset", match.group("ref"), "notebook_text")
    for match in _NOTEBOOK_COMPETITION_REF_RE.finditer(text):
        add_source("competition", match.group("slug"), "notebook_text")
    for match in _NOTEBOOK_KERNEL_REF_RE.finditer(text):
        add_source("kernel", match.group("ref"), "notebook_text")
    return sources


def _collect_free_text_input_sources(text: str, *, source: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_source(kind: str, ref: str) -> None:
        value = ref.strip().strip("/")
        if not value:
            return
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        sources.append({"kind": kind, "ref": value, "source": source})

    for match in _NOTEBOOK_DATASET_REF_RE.finditer(text):
        add_source("dataset", match.group("ref"))
    for match in _NOTEBOOK_COMPETITION_REF_RE.finditer(text):
        add_source("competition", match.group("slug"))
    for match in _NOTEBOOK_KERNEL_REF_RE.finditer(text):
        add_source("kernel", match.group("ref"))
    return sources


def _policy_reference_sources(*, paths: CompetitionPaths) -> list[dict[str, str]]:
    policy = load_competition_policy(paths)
    if not policy.active:
        return []
    sources: list[dict[str, str]] = []
    for ref in policy.reference_inputs.required_datasets:
        sources.append({"kind": "dataset", "ref": ref, "source": "competition_policy.required_datasets"})
    for ref in policy.reference_inputs.extra_dataset_refs:
        sources.append({"kind": "dataset", "ref": ref, "source": "competition_policy.extra_dataset_refs"})
    for ref in policy.reference_inputs.extra_kernel_refs:
        sources.append({"kind": "kernel", "ref": ref, "source": "competition_policy.extra_kernel_refs"})
    for ref in policy.reference_inputs.extra_competition_refs:
        sources.append({"kind": "competition", "ref": ref, "source": "competition_policy.extra_competition_refs"})
    return sources


def _merge_reference_input_sources(*sources_lists: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sources in sources_lists:
        for item in sources:
            kind = str(item.get("kind") or "").strip().lower()
            ref = str(item.get("ref") or "").strip().strip("/")
            if not kind or not ref:
                continue
            key = (kind, ref)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"kind": kind, "ref": ref, "source": str(item.get("source") or "").strip()})
    return merged


def _stage_reference_sources(
    *,
    paths: CompetitionPaths,
    current_slug: str,
    kernel_id: str,
    input_sources: list[dict[str, str]],
    download: bool,
    quiet: bool,
    dry_run: bool,
    download_competition_fn,
    download_dataset_fn,
    download_kernel_fn,
) -> list[dict[str, object]]:
    staged: list[dict[str, object]] = []
    for source in input_sources:
        kind = str(source.get("kind") or "").strip().lower()
        ref = str(source.get("ref") or "").strip().strip("/")
        if not kind or not ref:
            continue
        stage_name = _safe_kernel_dir_name(f"{kind}__{ref}")
        stage_dir = paths.reference_inputs_dir / stage_name
        status = "discovered"
        error = ""
        if kind == "competition" and ref == current_slug:
            status = "already_present_current_competition"
        elif _stage_dir_has_content(stage_dir):
            status = f"already_staged_{kind}"
        elif not download:
            status = "discovered_not_downloaded"
        elif dry_run:
            status = "dry_run"
        else:
            stage_dir.mkdir(parents=True, exist_ok=True)
            try:
                if kind == "dataset":
                    download_dataset_fn(ref, stage_dir, slug=current_slug, dry_run=False, force=True, quiet=quiet)
                    _unzip_downloads(stage_dir)
                    status = "staged_dataset"
                elif kind == "competition":
                    download_competition_fn(ref, stage_dir, force=True, quiet=quiet)
                    _unzip_downloads(stage_dir)
                    status = "staged_competition"
                elif kind == "kernel":
                    download_kernel_fn(ref, stage_dir, slug=current_slug, dry_run=False, metadata=True)
                    status = "staged_kernel"
                else:
                    status = "unsupported_source_type"
            except Exception as exc:  # noqa: BLE001
                status = "stage_error"
                error = str(exc)
        staged.append(
            {
                "kernel_id": kernel_id,
                "kind": kind,
                "ref": ref,
                "source": str(source.get("source") or ""),
                "stage_dir": str(stage_dir),
                "status": status,
                "error": error or None,
            }
        )
    return staged


def _stage_dir_has_content(stage_dir: Path) -> bool:
    if not stage_dir.exists() or not stage_dir.is_dir():
        return False
    try:
        return any(stage_dir.iterdir())
    except OSError:
        return False


def _unzip_downloads(data_dir: Path) -> None:
    for zip_path in data_dir.glob("*.zip"):
        safe_extract_zip(zip_path, data_dir)
