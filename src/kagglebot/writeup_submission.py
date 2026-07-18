from __future__ import annotations

import hashlib
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
    def submit(self, *, slug: str, title: str, body: str) -> dict[str, object]: ...


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
        }
    if not request.force:
        return {"status": "blocked_force_required", "content_sha256": content_sha256}

    prior = _matching_attempt(
        request.attempts_path,
        slug=request.slug,
        content_sha256=content_sha256,
        submission_sha256=submission_sha256,
    )
    if prior is not None:
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
    resolved_adapter = adapter or KaggleWriteupCdpAdapter()
    try:
        browser_result = resolved_adapter.submit(slug=request.slug, title=title, body=body)
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

    def submit(self, *, slug: str, title: str, body: str) -> dict[str, object]:
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
                    _KAGGLE_WRITEUP_CDP_SCRIPT,
                    str(cdp_module),
                    host,
                    str(port),
                    slug,
                    title,
                    body,
                ],
                timeout=180.0,
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
        if row.get("content_sha256") == content_sha256 and row.get("status") in {
            "started",
            "submitted",
            "ambiguous",
        }:
            return row
        previous_submission_sha256 = str(row.get("submission_sha256") or "")
        if previous_submission_sha256:
            if previous_submission_sha256 != submission_sha256:
                continue
        elif row.get("content_sha256") != content_sha256:
            continue
        if row.get("status") in {"started", "submitted", "ambiguous"}:
            return row
    return None


def _required_artifact_error(metadata: dict[str, object]) -> str | None:
    notebook = metadata.get("notebook")
    notebook_required = isinstance(notebook, dict) and notebook.get("required") is True
    key = "published_required_artifacts" if notebook_required else "required_artifacts"
    records = metadata.get(key)
    contract = metadata.get("artifact_contract")
    required_names = contract.get("required_output_names") if isinstance(contract, dict) else []
    if notebook_required and required_names and not isinstance(records, list):
        return "published notebook required-artifact evidence is missing"
    if records is None:
        records = metadata.get("required_artifacts")
    if not isinstance(records, list):
        return None
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
    const pageFunction = async (slug, title, body) => {
      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const visibleText = () => (document.body && document.body.innerText) || '';
      const clickByText = (pattern) => {
        const candidates = [...document.querySelectorAll('button,a,[role="button"]')];
        const match = candidates.find((node) => pattern.test((node.textContent || '').trim()) && !node.disabled);
        if (!match) return false;
        match.click();
        return true;
      };
      const setNativeValue = (element, value) => {
        const prototype = element instanceof HTMLTextAreaElement
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
        if (setter) setter.call(element, value); else element.value = value;
        element.dispatchEvent(new Event('input', {bubbles: true}));
        element.dispatchEvent(new Event('change', {bubbles: true}));
      };
      if (/sign in|log in/i.test(visibleText()) && /kaggle/i.test(document.title)) {
        return {status: 'failed', reason: 'kaggle-authentication-required'};
      }
      if (/join competition|i understand and accept/i.test(visibleText())) {
        return {status: 'failed', reason: 'competition-entry-or-rules-action-required'};
      }
      const writeupLink = [...document.querySelectorAll('a')].find(
        (node) => /new writeup/i.test(node.textContent || ''));
      if (writeupLink && writeupLink.href) location.href = writeupLink.href;
      else if (!clickByText(/new writeup/i)) {
        location.href = `https://www.kaggle.com/competitions/${slug}/projects/new`;
      }
      await sleep(3500);
      const competitionPrefix = `/competitions/${slug}/`;
      if (!location.pathname.includes(competitionPrefix) ||
          !/(?:project|writeup)/i.test(location.pathname)) {
        return {status: 'failed', reason: 'unexpected-writeup-route', url: location.href};
      }
      if (/join competition|i understand and accept/i.test(visibleText())) {
        return {status: 'failed', reason: 'competition-entry-or-rules-action-required'};
      }
      const titleInput = [...document.querySelectorAll('input')].find((node) =>
        /title/i.test((node.name || '') + ' ' + (node.placeholder || '') + ' ' +
          (node.getAttribute('aria-label') || '')));
      const textArea = [...document.querySelectorAll('textarea')].find((node) =>
        /body|content|description|writeup|markdown/i.test(
          (node.name || '') + ' ' + (node.placeholder || '') + ' ' +
          (node.getAttribute('aria-label') || '')));
      const editor = textArea || document.querySelector('[contenteditable="true"], .ProseMirror');
      if (!titleInput || !editor) {
        return {status: 'failed', reason: 'unrecognized-writeup-editor', url: location.href};
      }
      setNativeValue(titleInput, title);
      if (editor instanceof HTMLTextAreaElement || editor instanceof HTMLInputElement) {
        setNativeValue(editor, body);
      } else {
        editor.focus();
        editor.textContent = body;
        editor.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: null}));
      }
      await sleep(500);
      if (!clickByText(/^save(?: writeup)?$/i)) {
        return {status: 'failed', reason: 'save-control-not-found', url: location.href};
      }
      await sleep(3000);
      if (!clickByText(/^submit$/i)) {
        return {status: 'failed', reason: 'submit-control-not-found-after-save', url: location.href};
      }
      await sleep(1000);
      clickByText(/^(?:submit|confirm submission)$/i);
      await sleep(3500);
      const finalText = visibleText();
      const confirmed = /successfully submitted|submission complete|writeup submitted/i.test(finalText);
      return {
        status: confirmed ? 'submitted' : 'ambiguous',
        reason: confirmed ? 'kaggle-confirmed' : 'submission-confirmation-not-observed',
        url: location.href,
      };
    };
    const expression = '(' + pageFunction.toString() + ')(' +
      JSON.stringify(slug) + ',' + JSON.stringify(title) + ',' + JSON.stringify(body) + ')';
    const evaluated = await client.Runtime.evaluate({expression, awaitPromise: true, returnByValue: true});
    const value = evaluated.result && evaluated.result.value;
    console.log(JSON.stringify(value || {status: 'ambiguous', reason: 'empty-cdp-result'}));
  } catch (error) {
    console.log(JSON.stringify({status: 'ambiguous', reason: String(error)}));
    process.exitCode = 1;
  } finally {
    if (client) await client.close().catch(() => {});
    if (target) await CDP.Close({host, port, id: target.id}).catch(() => {});
  }
})();
"""
