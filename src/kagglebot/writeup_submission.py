from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot.agents import strategy_runner
from kagglebot.exec_utils import run_command
from kagglebot.json_utils import append_jsonl_record, load_jsonl_records
from kagglebot.kaggle_api import EnteredCompetition, check_rules_accepted, list_entered_competitions


class WriteupBrowserAdapter(Protocol):
    def submit(
        self,
        *,
        slug: str,
        title: str,
        body: str,
        artifact_paths: list[Path],
        track_label: str | None,
        card_image_path: Path | None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class WriteupSubmissionRequest:
    slug: str
    metadata: dict[str, object]
    attempts_path: Path
    force: bool
    dry_run: bool


def submit_validated_writeup(
    request: WriteupSubmissionRequest,
    *,
    adapter: WriteupBrowserAdapter | None = None,
    entered_loader: Callable[..., list[EnteredCompetition]] = list_entered_competitions,
    rules_checker: Callable[..., bool] = check_rules_accepted,
) -> dict[str, object]:
    report_path = Path(str(request.metadata.get("report_path") or ""))
    validation = request.metadata.get("validation")
    if request.metadata.get("status") == "ready_for_notebook_publish":
        return {"status": "blocked_notebook_required", "report_path": str(report_path)}
    if request.metadata.get("status") != "ready_for_submit" or not isinstance(validation, dict):
        return {"status": "blocked_invalid_writeup", "report_path": str(report_path)}
    if validation.get("valid") is not True or not report_path.is_file():
        return {"status": "blocked_invalid_writeup", "report_path": str(report_path)}
    body = report_path.read_text(encoding="utf-8")
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if content_sha256 != request.metadata.get("content_sha256"):
        return {"status": "blocked_content_changed", "content_sha256": content_sha256}
    artifact_error = _required_artifact_error(request.metadata)
    if artifact_error is not None:
        return {
            "status": "blocked_required_artifact",
            "content_sha256": content_sha256,
            "reason": artifact_error,
        }
    artifact_paths = _required_artifact_paths(request.metadata)
    card_image_error = _card_image_error(request.metadata)
    if card_image_error is not None:
        return {
            "status": "blocked_card_image",
            "content_sha256": content_sha256,
            "reason": card_image_error,
        }
    card_image_path = _card_image_path(request.metadata)
    track_label = _resolved_track_label(request.metadata.get("track"))
    notebook = request.metadata.get("notebook")
    if isinstance(notebook, dict) and notebook.get("required") is True:
        if notebook.get("status") != "ready" or not str(notebook.get("kernel_id") or "").strip():
            return {"status": "blocked_notebook_required", "content_sha256": content_sha256}
    submission_sha256 = _writeup_submission_sha256(request.metadata, content_sha256=content_sha256)
    if request.dry_run:
        return {
            "status": "dry_run",
            "content_sha256": content_sha256,
            "submission_sha256": submission_sha256,
            "report_path": str(report_path),
            "artifact_paths": [str(path) for path in artifact_paths],
            "track_label": track_label,
            "card_image_path": str(card_image_path) if card_image_path is not None else None,
        }
    if not request.force:
        return {"status": "blocked_force_required", "content_sha256": content_sha256}

    prior = _matching_attempt(
        request.attempts_path,
        slug=request.slug,
        content_sha256=content_sha256,
        submission_sha256=submission_sha256,
    )
    if prior is not None and prior.get("status") == "submitted":
        return {
            "status": "blocked_duplicate",
            "content_sha256": content_sha256,
            "previous_status": prior.get("status"),
        }
    resolved_adapter = adapter or KaggleWriteupCdpAdapter()
    if (artifact_paths or track_label or card_image_path) and not _adapter_supports_submission_contract(
        resolved_adapter
    ):
        return {
            "status": "blocked_adapter_contract_unsupported",
            "content_sha256": content_sha256,
            "required_artifacts": [path.name for path in artifact_paths],
            "track_label": track_label,
            "card_image_path": str(card_image_path) if card_image_path is not None else None,
        }
    existing = find_submitted_writeup(
        slug=request.slug,
        title=_report_title(body, fallback=request.slug),
        adapter=resolved_adapter,
    )
    if existing.get("status") == "submitted":
        result = {
            "status": "submitted",
            "slug": request.slug,
            "content_sha256": content_sha256,
            "submission_sha256": submission_sha256,
            "report_path": str(report_path),
            "browser": existing,
            "reconciled": True,
        }
        append_jsonl_record(request.attempts_path, result, sort_keys=True)
        return result
    reconciled_draft = (
        existing.get("status") == "not_submitted" and existing.get("reason") == "kaggle-draft-in-progress"
    )
    if prior is not None and not (reconciled_draft and str(prior.get("status") or "") in {"started", "ambiguous"}):
        return {
            "status": "blocked_duplicate",
            "content_sha256": content_sha256,
            "previous_status": prior.get("status"),
        }
    entered = entered_loader(page_limit=20, dry_run=False)
    competition = next((item for item in entered if item.slug == request.slug), None)
    if competition is None:
        return {"status": "blocked_not_entered", "content_sha256": content_sha256}
    if competition.submissions_disabled:
        return {"status": "blocked_submissions_disabled", "content_sha256": content_sha256}
    if not rules_checker(request.slug, dry_run=False):
        return {"status": "blocked_rules_not_accepted", "content_sha256": content_sha256}

    title = _report_title(body, fallback=request.slug)
    append_jsonl_record(
        request.attempts_path,
        {
            "slug": request.slug,
            "content_sha256": content_sha256,
            "submission_sha256": submission_sha256,
            "status": "started",
            "report_path": str(report_path),
            "title": title,
        },
        sort_keys=True,
    )
    try:
        submit_kwargs: dict[str, object] = {"slug": request.slug, "title": title, "body": body}
        if _adapter_supports_submission_contract(resolved_adapter):
            submit_kwargs.update(
                {
                    "artifact_paths": artifact_paths,
                    "track_label": track_label,
                    "card_image_path": card_image_path,
                }
            )
        browser_result = resolved_adapter.submit(**submit_kwargs)
    except Exception as exc:  # noqa: BLE001
        browser_result = {
            "status": "ambiguous",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    status = str(browser_result.get("status") or "ambiguous")
    if status not in {"submitted", "failed", "ambiguous"}:
        status = "ambiguous"
    result = {
        "status": status,
        "slug": request.slug,
        "content_sha256": content_sha256,
        "submission_sha256": submission_sha256,
        "report_path": str(report_path),
        "browser": browser_result,
    }
    append_jsonl_record(request.attempts_path, result, sort_keys=True)
    return result


class KaggleWriteupCdpAdapter:
    """Submit through the authenticated Kaggle UI and fail closed on unknown DOM states."""

    def find_submitted(self, *, slug: str, title: str) -> dict[str, object]:
        return self._run_cdp_script(
            script=_KAGGLE_WRITEUP_STATUS_CDP_SCRIPT,
            args=[slug, title],
            timeout=90.0,
        )

    def submit(
        self,
        *,
        slug: str,
        title: str,
        body: str,
        artifact_paths: list[Path],
        track_label: str | None,
        card_image_path: Path | None,
    ) -> dict[str, object]:
        return self._run_cdp_script(
            script=_KAGGLE_WRITEUP_CDP_SCRIPT,
            args=[
                slug,
                title,
                body,
                json.dumps([str(path.resolve()) for path in artifact_paths]),
                track_label or "",
                str(card_image_path.resolve()) if card_image_path is not None else "",
            ],
            timeout=300.0,
        )

    def _run_cdp_script(
        self,
        *,
        script: str,
        args: list[str],
        timeout: float,
    ) -> dict[str, object]:
        bootstrap = strategy_runner._maybe_start_oracle_browser(["--engine", "browser"])  # noqa: SLF001
        try:
            remote = strategy_runner._oracle_remote_chrome_endpoint(bootstrap.args)  # noqa: SLF001
            cdp_module = strategy_runner._oracle_cdp_module_path()  # noqa: SLF001
            node = strategy_runner._oracle_node_command()  # noqa: SLF001
            if remote is None or cdp_module is None or node is None:
                return {"status": "failed", "reason": "authenticated-browser-unavailable"}
            host, port = remote
            result = run_command(
                [
                    node,
                    "-e",
                    script,
                    str(cdp_module),
                    host,
                    str(port),
                    *args,
                ],
                timeout=timeout,
            )
            try:
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                payload = {}
            if result.returncode != 0 or not isinstance(payload, dict):
                return {
                    "status": "ambiguous",
                    "reason": (result.stderr or result.stdout or "cdp-writeup-submit-failed")[-1000:],
                }
            return payload
        finally:
            bootstrap.close()


def find_submitted_writeup(
    *,
    slug: str,
    title: str,
    adapter: WriteupBrowserAdapter | None = None,
) -> dict[str, object]:
    """Read the authenticated Kaggle project state without submitting anything."""
    resolved_adapter = adapter or KaggleWriteupCdpAdapter()
    finder = getattr(resolved_adapter, "find_submitted", None)
    if not callable(finder):
        return {"status": "not_checked", "reason": "adapter-does-not-support-reconciliation"}
    try:
        result = finder(slug=slug, title=title)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ambiguous",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(result, dict):
        return {"status": "ambiguous", "reason": "invalid-reconciliation-result"}
    return result


def _matching_attempt(
    path: Path,
    *,
    slug: str,
    content_sha256: str,
    submission_sha256: str,
) -> dict[str, object] | None:
    for row in reversed(load_jsonl_records(path)):
        if row.get("slug") != slug:
            continue
        previous_submission_sha256 = str(row.get("submission_sha256") or "")
        if previous_submission_sha256:
            if previous_submission_sha256 != submission_sha256:
                continue
        elif row.get("content_sha256") != content_sha256:
            continue
        status = str(row.get("status") or "")
        if status == "failed":
            # Adapter failures are emitted only before the Submit control is clicked.
            # A read-only Kaggle reconciliation still runs before a retry.
            return None
        if status in {"started", "submitted", "ambiguous"}:
            return row
    return None


def _required_artifact_error(metadata: dict[str, object]) -> str | None:
    notebook = metadata.get("notebook")
    notebook_required = isinstance(notebook, dict) and notebook.get("required") is True
    key = "published_required_artifacts" if notebook_required else "required_artifacts"
    records = metadata.get(key)
    contract = metadata.get("artifact_contract")
    required_names = contract.get("required_output_names") if isinstance(contract, dict) else []
    if required_names and not isinstance(records, list):
        kind = "published notebook" if notebook_required else "writeup attachment"
        return f"{kind} required-artifact evidence is missing"
    if records is None:
        records = metadata.get("required_artifacts")
    if not isinstance(records, list):
        return None
    recorded_names = {str(record.get("name") or "") for record in records if isinstance(record, dict)}
    missing_names = [str(name) for name in required_names or [] if str(name) not in recorded_names]
    if missing_names:
        return f"required artifact records are missing: {', '.join(missing_names)}"
    for record in records:
        if not isinstance(record, dict):
            return f"invalid {key} record"
        name = str(record.get("name") or "").strip()
        path = Path(str(record.get("path") or ""))
        expected_hash = str(record.get("sha256") or "")
        if not name or not path.is_file() or not expected_hash:
            return f"required artifact is missing or incomplete: {name or path}"
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            return f"required artifact changed after validation: {name}"
    return None


def _required_artifact_paths(metadata: dict[str, object]) -> list[Path]:
    notebook = metadata.get("notebook")
    notebook_required = isinstance(notebook, dict) and notebook.get("required") is True
    records = metadata.get("published_required_artifacts") if notebook_required else metadata.get("required_artifacts")
    if not isinstance(records, list):
        return []
    return [Path(str(record.get("path"))) for record in records if isinstance(record, dict)]


def _card_image_error(metadata: dict[str, object]) -> str | None:
    required = metadata.get("card_image_required") is True
    record = metadata.get("card_image")
    if record is None and not required:
        return None
    if not isinstance(record, dict):
        return "required 560x280 writeup card image evidence is missing"
    path = Path(str(record.get("path") or ""))
    expected_hash = str(record.get("sha256") or "")
    if not path.is_file() or not expected_hash:
        return "required writeup card image is missing or incomplete"
    if _sha256_file(path) != expected_hash:
        return "writeup card image changed after validation"
    if record.get("width") != 560 or record.get("height") != 280:
        return "writeup card image must be recorded as 560x280"
    return None


def _card_image_path(metadata: dict[str, object]) -> Path | None:
    record = metadata.get("card_image")
    return Path(str(record.get("path"))) if isinstance(record, dict) else None


def _resolved_track_label(value: object) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"static", "static_skill", "static_skills", "track_1"}:
        return "Static Skills"
    if normalized in {"meta", "meta_skill", "meta_skills", "track_2"}:
        return "Meta Skills"
    return str(value).strip() if str(value or "").strip() else None


def _adapter_supports_submission_contract(adapter: WriteupBrowserAdapter) -> bool:
    try:
        parameters = inspect.signature(adapter.submit).parameters
    except (TypeError, ValueError):
        return False
    return all(name in parameters for name in ("artifact_paths", "track_label", "card_image_path"))


def _writeup_submission_sha256(metadata: dict[str, object], *, content_sha256: str) -> str:
    notebook = metadata.get("notebook") if isinstance(metadata.get("notebook"), dict) else {}
    artifacts = metadata.get("published_required_artifacts")
    if not isinstance(artifacts, list):
        artifacts = metadata.get("required_artifacts")
    artifact_hashes = sorted(
        (
            str(record.get("name") or ""),
            str(record.get("sha256") or ""),
        )
        for record in artifacts or []
        if isinstance(record, dict)
    )
    payload = {
        "content_sha256": content_sha256,
        "notebook_id": str(notebook.get("kernel_id") or ""),
        "required_artifacts": artifact_hashes,
        "track": _resolved_track_label(metadata.get("track")) or "",
        "card_image_sha256": str(
            metadata.get("card_image", {}).get("sha256") if isinstance(metadata.get("card_image"), dict) else ""
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_title(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()[:120]
    return fallback[:120]


_KAGGLE_WRITEUP_CDP_SCRIPT = r"""
const CDP = require(process.argv[1]);
const host = process.argv[2];
const port = Number(process.argv[3]);
const slug = process.argv[4];
const title = process.argv[5];
const body = process.argv[6];
const artifactPaths = JSON.parse(process.argv[7] || '[]');
const trackLabel = process.argv[8] || '';
const cardImagePath = process.argv[9] || '';
const normalizeText = (value) => String(value || '')
  .toLowerCase()
  .normalize('NFKC')
  .replace(/[“”]/g, '"')
  .replace(/[‘’]/g, "'")
  .replace(/[\\u2010-\\u2015]/g, '-')
  .replace(/[^a-z0-9\\s]/g, ' ')
  .replace(/\\s+/g, ' ')
  .trim();
const hasWriteupSubmissionSignal = (text) => {
  const normalized = String(text || '').toLowerCase();
  return /submitted!|successfully submitted|submission complete|writeup submitted/.test(
    normalized,
  );
};
const projectsUrl = `https://www.kaggle.com/competitions/${slug}/projects`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
(async () => {
  let client;
  let target;
  try {
    target = await CDP.New({host, port, url: projectsUrl});
    client = await CDP({host, port, target});
    await client.Page.enable();
    await client.Page.navigate({url: projectsUrl});
    await client.Page.loadEventFired();
    await sleep(2500);
    const readPage = async () => {
      const evaluated = await client.Runtime.evaluate({
        expression: `JSON.stringify({
          title: document.title,
          text: (document.body && document.body.innerText) || '',
          url: location.href,
          links: [...document.querySelectorAll('a[href]')].map((node) => ({
            text: (node.textContent || '').replace(/\\s+/g, ' ').trim(),
            href: node.href
          }))
        })`,
        returnByValue: true,
      });
      const value = evaluated.result && evaluated.result.value;
      return value ? JSON.parse(value) : {};
    };
    const page = await readPage();
    if (/sign in|log in/i.test(page.text || '') && /kaggle/i.test(page.title || '')) {
      console.log(JSON.stringify({status: 'failed', reason: 'kaggle-authentication-required'}));
      return;
    }
    if (/join competition|i understand and accept/i.test(page.text || '')) {
      console.log(JSON.stringify({status: 'failed', reason: 'competition-entry-or-rules-action-required'}));
      return;
    }
    const prefix = `/competitions/${slug}/writeups/`;
    const draftLink = (page.links || []).find((item) => {
      let path;
      try {
        path = new URL(item.href, page.url).pathname;
      } catch {
        return false;
      }
      return path.startsWith(prefix) && /new writeup|draft/i.test(item.text || '');
    });
    if (draftLink && draftLink.href) {
      await client.close();
      client = undefined;
      await CDP.Close({host, port, id: target.id}).catch(() => {});
      target = await CDP.New({host, port, url: draftLink.href});
      client = await CDP({host, port, target});
      await client.Page.enable();
      await sleep(2500);
    } else {
      const opened = await client.Runtime.evaluate({
        expression: `(() => {
          const nodes = [...document.querySelectorAll('button,a,[role="button"]')];
          const match = nodes.find(
            (node) => /new writeup/i.test((node.textContent || '').trim()) && !node.disabled);
          if (!match) return false;
          match.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!opened.result || opened.result.value !== true) {
        console.log(JSON.stringify({
          status: 'failed',
          reason: 'new-writeup-control-not-found',
          url: page.url,
        }));
        return;
      }
      await sleep(2500);
    }
    const editorReady = async () => {
      const result = await client.Runtime.evaluate({
        expression: `!!document.querySelector('input[name="title"], textarea[aria-label="Project Description"]')`,
        returnByValue: true,
      });
      return !!(result.result && result.result.value);
    };
    if (!(await editorReady())) {
      const editOpened = await client.Runtime.evaluate({
        expression: `(() => {
          const edit = [...document.querySelectorAll('button,a,[role="button"]')].find(
            (node) => /^edit$/i.test((node.textContent || '').trim()) && !node.disabled);
          if (!edit) return false;
          edit.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!editOpened.result || editOpened.result.value !== true) {
        console.log(JSON.stringify({status: 'failed', reason: 'writeup-edit-control-not-found'}));
        return;
      }
      await sleep(1800);
    }
    const fillExpression = `(() => {
      const title = ${JSON.stringify(title)};
      const body = ${JSON.stringify(body)};
      const pageText = (document.body && document.body.innerText) || '';
      if (/join competition|i understand and accept/i.test(pageText)) {
        return {ok: false, reason: 'competition-entry-or-rules-action-required', url: location.href};
      }
      const descriptor = (node) => (
        (node.name || '') + ' ' + (node.placeholder || '') + ' ' +
        (node.getAttribute('aria-label') || '')
      );
      const titleInput = [...document.querySelectorAll('input')].find((node) =>
        /(?:^|\\s)title(?:\\s|$)/i.test(descriptor(node)) && !/subtitle/i.test(descriptor(node)));
      const subtitleInput = [...document.querySelectorAll('input')].find(
        (node) => /subtitle/i.test(descriptor(node)));
      const textArea = [...document.querySelectorAll('textarea')].find(
        (node) => /body|content|description|writeup|markdown/i.test(descriptor(node)));
      const editor = textArea || document.querySelector('[contenteditable="true"], .ProseMirror');
      if (!titleInput || !editor) {
        return {ok: false, reason: 'unrecognized-writeup-editor', url: location.href};
      }
      const setNativeValue = (element, value) => {
        const prototype = element instanceof HTMLTextAreaElement
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
        if (setter) setter.call(element, value); else element.value = value;
        element.dispatchEvent(new Event('input', {bubbles: true}));
        element.dispatchEvent(new Event('change', {bubbles: true}));
      };
      setNativeValue(titleInput, title);
      if (subtitleInput) {
        const subtitle = body.split('\\n')
          .map((line) => line.trim())
          .find((line) => line && !line.startsWith('#') && line.length >= 20) || title;
        setNativeValue(subtitleInput, subtitle.slice(0, 140));
      }
      if (editor instanceof HTMLTextAreaElement || editor instanceof HTMLInputElement) {
        setNativeValue(editor, body);
      } else {
        editor.focus();
        editor.textContent = body;
        editor.dispatchEvent(
          new InputEvent('input', {bubbles: true, inputType: 'insertText', data: null}));
      }
      return {ok: true, url: location.href};
    })()`;
    const filled = await client.Runtime.evaluate({expression: fillExpression, returnByValue: true});
    const fillResult = filled.result && filled.result.value;
    if (!fillResult || fillResult.ok !== true) {
      console.log(JSON.stringify({
        status: 'failed',
        reason: (fillResult && fillResult.reason) || 'writeup-fill-failed',
        url: fillResult && fillResult.url,
      }));
      return;
    }
    let cardImageConfirmed = !cardImagePath;
    if (cardImagePath) {
      await client.DOM.enable();
      const imageEditorOpened = await client.Runtime.evaluate({
        expression: `(() => {
          const control = [...document.querySelectorAll('button,a,[role="button"]')].find(
            (node) => /edit image/i.test((node.textContent || '').trim()) &&
              !node.disabled && node.getAttribute('aria-disabled') !== 'true');
          if (!control) return false;
          control.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!imageEditorOpened.result || imageEditorOpened.result.value !== true) {
        console.log(JSON.stringify({status: 'failed', reason: 'card-image-control-not-found'}));
        return;
      }
      await sleep(300);
      try {
        const documentNode = await client.DOM.getDocument({depth: -1, pierce: true});
        const imageInput = await client.DOM.querySelector({
          nodeId: documentNode.root.nodeId,
          selector: 'input[type="file"][accept*=".png"]',
        });
        if (!imageInput || !imageInput.nodeId) {
          throw new Error('card-image-file-input-not-resolved');
        }
        await client.DOM.setFileInputFiles({files: [cardImagePath], nodeId: imageInput.nodeId});
      } catch (error) {
        console.log(JSON.stringify({status: 'failed', reason: String(error)}));
        return;
      }
      for (let attempt = 0; attempt < 20; attempt += 1) {
        await sleep(300);
        const preview = await client.Runtime.evaluate({
          expression: `!!document.querySelector('img[alt="Uploaded Image"], img[alt="cropped cover"]')`,
          returnByValue: true,
        });
        if (preview.result && preview.result.value === true) {
          cardImageConfirmed = true;
          break;
        }
      }
      if (!cardImageConfirmed) {
        console.log(JSON.stringify({status: 'failed', reason: 'card-image-preview-not-observed'}));
        return;
      }
      const cropSaved = await client.Runtime.evaluate({
        expression: `(() => {
          const control = [...document.querySelectorAll('button,[role="button"]')].find(
            (node) => /^save$/i.test((node.textContent || '').trim()) &&
              !node.disabled && node.getAttribute('aria-disabled') !== 'true');
          if (!control) return false;
          control.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!cropSaved.result || cropSaved.result.value !== true) {
        console.log(JSON.stringify({status: 'failed', reason: 'card-image-crop-save-not-found'}));
        return;
      }
      cardImageConfirmed = false;
      for (let attempt = 0; attempt < 60; attempt += 1) {
        await sleep(500);
        const cropCommitted = await client.Runtime.evaluate({
          expression: `(() => {
            const text = (document.body && document.body.innerText) || '';
            const preview = document.querySelector('img[alt="cropped cover"]');
            return !text.includes('Upload Card and Thumbnail Image') && !!preview;
          })()`,
          returnByValue: true,
        });
        if (cropCommitted.result && cropCommitted.result.value === true) {
          cardImageConfirmed = true;
          break;
        }
      }
      if (!cardImageConfirmed) {
        console.log(JSON.stringify({status: 'failed', reason: 'card-image-crop-not-committed'}));
        return;
      }
    }
    if (trackLabel) {
      const selectTrack = await client.Runtime.evaluate({
        expression: `(() => {
          const desired = ${JSON.stringify(trackLabel)};
          const normalizedDesired = desired.toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
          const controls = [...document.querySelectorAll('button,a,[role="button"],[role="checkbox"]')];
          const selected = controls.find((node) => {
            const text = (node.textContent || '').toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
            const ariaLabel = (node.getAttribute('aria-label') || '').toLowerCase()
              .replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
            return (text === normalizedDesired && node.getAttribute('aria-checked') === 'true') ||
              ariaLabel === normalizedDesired;
          });
          if (selected) return {ok: true, alreadySelected: true};
          const picker = controls.find((node) =>
            /select track/i.test(((node.textContent || '') + ' ' + (node.getAttribute('aria-label') || '')).trim()) &&
            !node.disabled);
          if (!picker) return {ok: false, reason: 'track-picker-not-found'};
          picker.click();
          return {ok: true, alreadySelected: false};
        })()`,
        returnByValue: true,
      });
      const selectTrackResult = selectTrack.result && selectTrack.result.value;
      if (!selectTrackResult || selectTrackResult.ok !== true) {
        console.log(JSON.stringify({
          status: 'failed',
          reason: (selectTrackResult && selectTrackResult.reason) || 'track-picker-failed',
        }));
        return;
      }
      if (!selectTrackResult.alreadySelected) {
        await sleep(700);
        const optionClicked = await client.Runtime.evaluate({
          expression: `(() => {
            const desired = ${JSON.stringify(trackLabel)};
            const normalize = (value) => String(value || '').toLowerCase()
              .replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
            const candidates = [...document.querySelectorAll('label,[role="option"],[role="radio"]')];
            const option = candidates.find((node) => normalize(node.textContent) === normalize(desired));
            if (!option) return false;
            const radio = option.matches('input[type="radio"]')
              ? option
              : option.querySelector('input[type="radio"]');
            (radio || option).click();
            return true;
          })()`,
          returnByValue: true,
        });
        if (!optionClicked.result || optionClicked.result.value !== true) {
          console.log(JSON.stringify({status: 'failed', reason: 'requested-track-not-found'}));
          return;
        }
        await sleep(700);
      }
      const trackConfirmed = await client.Runtime.evaluate({
        expression: `(() => {
          const desired = ${JSON.stringify(trackLabel)};
          const normalize = (value) => String(value || '').toLowerCase()
            .replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
          const candidates = [...document.querySelectorAll(
            'button,[role="button"],[role="checkbox"],input[type="radio"]',
          )];
          return candidates.some((node) => {
            const label = node.labels && node.labels.length
              ? node.labels[0].textContent
              : (node.getAttribute('aria-label') || node.textContent);
            const checked = node.checked === true || node.getAttribute('aria-checked') === 'true';
            const selectedChip = node.getAttribute('role') === 'button' &&
              normalize(node.getAttribute('aria-label')) === normalize(desired);
            return (checked || selectedChip) && normalize(label) === normalize(desired);
          });
        })()`,
        returnByValue: true,
      });
      if (!trackConfirmed.result || trackConfirmed.result.value !== true) {
        console.log(JSON.stringify({status: 'failed', reason: 'requested-track-not-confirmed'}));
        return;
      }
    }
    if (artifactPaths.length) {
      await client.DOM.enable();
      const uploadClicked = await client.Runtime.evaluate({
        expression: `(() => {
          const upload = [...document.querySelectorAll('button,a,[role="button"]')].find(
            (node) => /^upload files$/i.test((node.textContent || '').trim()) && !node.disabled);
          if (!upload) return false;
          upload.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!uploadClicked.result || uploadClicked.result.value !== true) {
        console.log(JSON.stringify({status: 'failed', reason: 'artifact-upload-control-not-found'}));
        return;
      }
      await sleep(300);
      try {
        const documentNode = await client.DOM.getDocument({depth: -1, pierce: true});
        const fileInput = await client.DOM.querySelector({
          nodeId: documentNode.root.nodeId,
          selector: 'input[type="file"][multiple]',
        });
        if (!fileInput || !fileInput.nodeId) {
          throw new Error('artifact-file-input-not-resolved');
        }
        await client.DOM.setFileInputFiles({files: artifactPaths, nodeId: fileInput.nodeId});
      } catch (error) {
        console.log(JSON.stringify({status: 'failed', reason: String(error)}));
        return;
      }
      const artifactNames = artifactPaths.map((path) => path.split(/[\\/]/).pop());
      let artifactsObserved = false;
      for (let attempt = 0; attempt < 60; attempt += 1) {
        await sleep(1000);
        const observed = await client.Runtime.evaluate({
          expression: `(() => {
            const names = ${JSON.stringify(artifactNames)};
            const text = (document.body && document.body.innerText) || '';
            return names.every((name) => text.includes(name));
          })()`,
          returnByValue: true,
        });
        if (observed.result && observed.result.value === true) {
          artifactsObserved = true;
          break;
        }
      }
      if (!artifactsObserved) {
        console.log(JSON.stringify({
          status: 'failed',
          reason: 'artifact-attachment-not-observed',
          artifact_names: artifactNames,
        }));
        return;
      }
    }
    const saveClicked = await client.Runtime.evaluate({
      expression: `(() => {
        const save = [...document.querySelectorAll('button,a,[role="button"]')].find(
          (node) => /^save draft$/i.test((node.textContent || '').trim()) &&
            !node.disabled && node.getAttribute('aria-disabled') !== 'true');
        if (!save) return false;
        save.click();
        return true;
      })()`,
      returnByValue: true,
    });
    if (!saveClicked.result || saveClicked.result.value !== true) {
      console.log(JSON.stringify({status: 'failed', reason: 'save-control-not-found'}));
      return;
    }
    await sleep(3000);
    const savedPage = await readPage();
    const artifactNames = artifactPaths.map((path) => path.split(/[\\/]/).pop());
    const normalizedSavedText = normalizeText(savedPage.text || '');
    const artifactsConfirmed = artifactNames.every((name) =>
      normalizedSavedText.includes(normalizeText(name)));
    const trackConfirmed = !trackLabel || normalizedSavedText.includes(normalizeText(trackLabel));
    if (!artifactsConfirmed || !trackConfirmed || !cardImageConfirmed) {
      console.log(JSON.stringify({
        status: 'failed',
        reason: 'saved-writeup-contract-not-observed',
        artifact_names: artifactNames,
        artifacts_confirmed: artifactsConfirmed,
        track: trackLabel || null,
        track_confirmed: trackConfirmed,
        card_image_confirmed: cardImageConfirmed,
      }));
      return;
    }
    const submitClicked = await client.Runtime.evaluate({
      expression: `(() => {
        const submit = [...document.querySelectorAll('button,a,[role="button"]')].find(
          (node) => /^submit$/i.test((node.textContent || '').trim()) &&
            !node.disabled && node.getAttribute('aria-disabled') !== 'true');
        if (!submit) return false;
        submit.click();
        return true;
      })()`,
      returnByValue: true,
    });
    if (!submitClicked.result || submitClicked.result.value !== true) {
      const current = await readPage();
      console.log(JSON.stringify({
        status: 'failed',
        reason: 'submit-control-disabled-or-not-found-after-save',
        url: current.url,
      }));
      return;
    }
    await sleep(1200);
    const immediatePage = await readPage();
    let immediateConfirmed = hasWriteupSubmissionSignal(immediatePage.text || '');
    let confirmationClicked = false;
    if (!immediateConfirmed) {
      const confirmation = await client.Runtime.evaluate({
      expression: `(() => {
        const dialog = document.querySelector('[role="dialog"]');
        if (!dialog) return false;
        const confirm = [...dialog.querySelectorAll('button,[role="button"]')].find(
          (node) => /^(?:submit|confirm submission)$/i.test((node.textContent || '').trim()) &&
            !node.disabled && node.getAttribute('aria-disabled') !== 'true');
        if (!confirm) return false;
        confirm.click();
        return true;
      })()`,
      returnByValue: true,
      });
      confirmationClicked = !!(confirmation.result && confirmation.result.value);
    }
    await sleep(3500);
    await client.Page.navigate({url: projectsUrl});
    await client.Page.loadEventFired().catch(() => {});
    await sleep(2500);
    const finalPage = await readPage();
    const expected = normalizeText(title);
    const matchingProjects = (finalPage.links || []).filter((item) =>
      normalizeText(item.text).includes(expected));
    const matchingProject = matchingProjects.find((item) => {
      const text = String(item.text || '').toLowerCase();
      return text.includes('submitted') || text.includes('draft') || text.includes('in progress');
    }) || matchingProjects[0];
    const projectText = String((matchingProject && matchingProject.text) || '').toLowerCase();
    const draftObserved = projectText.includes('draft') || projectText.includes('in progress');
    const submittedObserved = projectText.includes('submitted') && !draftObserved;
    const confirmed = submittedObserved && artifactsConfirmed && trackConfirmed && cardImageConfirmed;
    console.log(JSON.stringify({
      status: confirmed ? 'submitted' : 'ambiguous',
      reason: confirmed
        ? 'kaggle-submission-contract-confirmed'
        : draftObserved
          ? 'kaggle-draft-remains-after-submit-action'
          : 'kaggle-project-submission-state-not-confirmed',
      url: (matchingProject && matchingProject.href) || finalPage.url,
      artifact_names: artifactNames,
      artifacts_confirmed: artifactsConfirmed,
      track: trackLabel || null,
      track_confirmed: trackConfirmed,
      card_image_confirmed: cardImageConfirmed,
      confirmation_clicked: confirmationClicked,
      immediate_confirmation_observed: immediateConfirmed,
    }));
  } catch (error) {
    console.log(JSON.stringify({status: 'ambiguous', reason: String(error)}));
    process.exitCode = 1;
  } finally {
    if (client) await client.close().catch(() => {});
    if (target) await CDP.Close({host, port, id: target.id}).catch(() => {});
  }
})();
"""


_KAGGLE_WRITEUP_STATUS_CDP_SCRIPT = r"""
const CDP = require(process.argv[1]);
const host = process.argv[2];
const port = Number(process.argv[3]);
const slug = process.argv[4];
const title = process.argv[5];
const normalizeText = (value) => String(value || '')
  .toLowerCase()
  .normalize('NFKC')
  .replace(/[“”]/g, '"')
  .replace(/[‘’]/g, "'")
  .replace(/[\\u2010-\\u2015]/g, '-')
  .replace(/[^a-z0-9\\s]/g, ' ')
  .replace(/\\s+/g, ' ')
  .trim();
const projectsUrl = `https://www.kaggle.com/competitions/${slug}/projects`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
(async () => {
  let client;
  let target;
  try {
    target = await CDP.New({host, port, url: projectsUrl});
    client = await CDP({host, port, target});
    await client.Page.enable();
    await client.Page.navigate({url: projectsUrl});
    await client.Page.loadEventFired();
    await sleep(2500);
    const expected = normalizeText(title);
    const prefix = `/competitions/${slug}/writeups/`;
    const listing = await client.Runtime.evaluate({
      expression: `JSON.stringify({
        title: document.title,
        text: (document.body && document.body.innerText) || '',
        url: location.href,
        links: [...document.querySelectorAll('a[href]')].map((node) => ({
          text: (node.textContent || '').replace(/\\s+/g, ' ').trim(),
          href: node.href
        }))
      })`,
      returnByValue: true,
    });
    const listingValue = listing.result && listing.result.value;
    const page = listingValue ? JSON.parse(listingValue) : {};
    if (/sign in|log in/i.test(page.text || '') && /kaggle/i.test(page.title || '')) {
      console.log(JSON.stringify({status: 'failed', reason: 'kaggle-authentication-required'}));
      return;
    }
    const links = (page.links || []).filter((item) => {
      try {
        return new URL(item.href, page.url).pathname.startsWith(prefix);
      } catch {
        return false;
      }
    });
    const matches = links.filter((item) => {
      const linkText = normalizeText(item.text);
      return (
        linkText === expected ||
        (expected.length >= 6 && (linkText.includes(expected) || expected.includes(linkText)))
      );
    });
    const match = matches.find((item) => {
      const text = String(item.text || '').toLowerCase();
      return text.includes('submitted') || text.includes('draft') || text.includes('in progress');
    }) || matches[0];
    if (!match || !match.href) {
      console.log(JSON.stringify({
        status: 'not_found',
        reason: 'matching-writeup-not-listed',
        writeup_count: links.length,
        url: page.url,
      }));
      return;
    }
    const projectText = String(match.text || '').toLowerCase();
    const draftObserved = projectText.includes('draft') || projectText.includes('in progress');
    const submittedObserved = projectText.includes('submitted') && !draftObserved;
    console.log(JSON.stringify({
      status: submittedObserved ? 'submitted' : draftObserved ? 'not_submitted' : 'ambiguous',
      reason: submittedObserved
        ? 'kaggle-project-listed-submitted'
        : draftObserved
          ? 'kaggle-draft-in-progress'
          : 'matching-project-state-unrecognized',
      title,
      url: match.href,
    }));
  } catch (error) {
    console.log(JSON.stringify({status: 'ambiguous', reason: String(error)}));
    process.exitCode = 1;
  } finally {
    if (client) await client.close().catch(() => {});
    if (target) await CDP.Close({host, port, id: target.id}).catch(() => {});
  }
})();
"""
