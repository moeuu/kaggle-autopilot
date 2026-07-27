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
    if prior is not None and prior.get("status") == "submitted":
        return {
            "status": "blocked_duplicate",
            "content_sha256": content_sha256,
            "previous_status": prior.get("status"),
        }
    resolved_adapter = adapter or KaggleWriteupCdpAdapter()
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

    def find_submitted(self, *, slug: str, title: str) -> dict[str, object]:
        return self._run_cdp_script(
            script=_KAGGLE_WRITEUP_STATUS_CDP_SCRIPT,
            args=[slug, title],
            timeout=90.0,
        )

    def submit(self, *, slug: str, title: str, body: str) -> dict[str, object]:
        return self._run_cdp_script(
            script=_KAGGLE_WRITEUP_CDP_SCRIPT,
            args=[slug, title, body],
            timeout=180.0,
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
        setNativeValue(subtitleInput, subtitle.slice(0, 160));
      }
      if (editor instanceof HTMLTextAreaElement || editor instanceof HTMLInputElement) {
        setNativeValue(editor, body);
      } else {
        editor.focus();
        editor.textContent = body;
        editor.dispatchEvent(
          new InputEvent('input', {bubbles: true, inputType: 'insertText', data: null}));
      }
      const save = [...document.querySelectorAll('button,a,[role="button"]')].find(
        (node) => /^save(?: draft| writeup)?$/i.test((node.textContent || '').trim()) && !node.disabled);
      if (!save) return {ok: false, reason: 'save-control-not-found', url: location.href};
      save.click();
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
    await sleep(3000);
    const submitClicked = await client.Runtime.evaluate({
      expression: `(() => {
        const submit = [...document.querySelectorAll('button,a,[role="button"]')].find(
          (node) => /^submit$/i.test((node.textContent || '').trim()) && !node.disabled);
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
        reason: 'submit-control-not-found-after-save',
        url: current.url,
      }));
      return;
    }
    await sleep(1200);
    await client.Runtime.evaluate({
      expression: `(() => {
        const confirm = [...document.querySelectorAll('button,[role="button"]')].find(
          (node) => /^(?:submit|confirm submission)$/i.test((node.textContent || '').trim()) &&
            !node.disabled);
        if (!confirm) return false;
        confirm.click();
        return true;
      })()`,
      returnByValue: true,
    });
    await sleep(3500);
    const finalPage = await readPage();
    const confirmed = /submitted!|successfully submitted|submission complete|writeup submitted/i.test(
      finalPage.text || '');
    console.log(JSON.stringify({
      status: confirmed ? 'submitted' : 'ambiguous',
      reason: confirmed ? 'kaggle-confirmed' : 'submission-confirmation-not-observed',
      url: finalPage.url,
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
    const expected = String(title || '').replace(/\s+/g, ' ').trim().toLowerCase();
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
    const match = links.find((item) => {
      const linkText = String(item.text || '').replace(/\s+/g, ' ').trim().toLowerCase();
      return linkText === expected || (expected.length >= 6 && linkText.includes(expected));
    });
    if (!match || !match.href) {
      console.log(JSON.stringify({
        status: 'not_found',
        reason: 'matching-writeup-not-listed',
        writeup_count: links.length,
        url: page.url,
      }));
      return;
    }
    await client.close();
    client = undefined;
    await CDP.Close({host, port, id: target.id}).catch(() => {});
    target = await CDP.New({host, port, url: match.href});
    client = await CDP({host, port, target});
    await client.Page.enable();
    await sleep(3000);
    const finalState = await client.Runtime.evaluate({
      expression: `JSON.stringify({
        title: document.title,
        text: (document.body && document.body.innerText) || '',
        url: location.href
      })`,
      returnByValue: true,
    });
    const finalValue = finalState.result && finalState.result.value;
    const finalPage = finalValue ? JSON.parse(finalValue) : {};
    const currentTitle = String(finalPage.title || '')
      .replace(/\s*\|\s*Kaggle\s*$/i, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
    const confirmed = /submitted!/i.test(finalPage.text || '');
    const titleMatches = currentTitle === expected;
    console.log(JSON.stringify({
      status: confirmed && titleMatches ? 'submitted' : 'not_submitted',
      reason: confirmed
        ? (titleMatches ? 'kaggle-submitted-confirmation-observed' : 'writeup-title-mismatch')
        : 'submitted-confirmation-not-observed',
      title: finalPage.title,
      url: finalPage.url,
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
