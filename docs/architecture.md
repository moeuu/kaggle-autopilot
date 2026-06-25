# Architecture

This document describes the current autopilot execution architecture.

## Execution Overview

`kagglebot autopilot` executes in three major phases:

1. Bootstrap context
2. Agent pipeline (`gpt -> gpt -> gpt`)
3. Iterative train/evaluate/improve/submit loop

## 1) Bootstrap Context

Bootstrap prepares `artifacts/<slug>/context/`:
- rules URL and rules text
- overview/data/submission-format summaries
- dataset profile
- sample submission snapshot
- top1 public score snapshot

## 2) Agent Pipeline (`gpt -> gpt -> gpt`)

Implemented in `src/kagglebot/orchestrator/agent_pipeline.py`.

### Stage A: Codex Brief
- Reads local context files
- Produces `context/agent/brief_for_strategy.md`

### Stage B: GPT Strategy
- Builds strategy from brief + context + knowledge hints
- Uses live web search when available in CLI runtime
- Must emit structured sections:
  - `===STRATEGY===`
  - `===RESEARCH_SOURCES_JSONL===`
  - `===RESEARCH_SUMMARY_MD===`
  - `===PLAN_JSON===`
  - `===CODEX_INSTRUCTIONS===`
- Persists:
  - `context/research_sources.jsonl` (working copy)
  - `context/research_summary.md` (working copy)
  - `context/research_storage.json` (links to persistent storage)
  - `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent copy)
  - `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent copy)
  - `context/agent/strategy_plan.md`
  - `context/agent/codex_instructions.md`
  - `context/agent/strategy_transcript.txt`

### Stage C: Codex Implementation
- Reads frozen strategy artifacts (`plan.json`, research summary, codex instructions)
- Updates only `artifacts/<slug>/kernel/`
- Intended to execute Prompt 3/4/5 style implementation flow

## 3) Iteration Loop

Main loop is in `src/kagglebot/autopilot.py`.
Supporting state/resume helpers now live in `src/kagglebot/autopilot_state.py`, and score/policy helpers live in
`src/kagglebot/autopilot_helpers.py`, so the main file stays focused on orchestration.
Competition rule parsing now lives in `src/kagglebot/competition_rules.py`; the loop calls that public module directly
instead of carrying private rule-parsing aliases in `autopilot.py`.
Offline score-source normalization and trust checks live in `src/kagglebot/score_sources.py` for the same reason:
the loop should consume normalized policy answers rather than own every parsing rule inline.
Split-strategy policy and competition-specific evaluation overrides live in `src/kagglebot/plan_policy.py`. This keeps
plan resolution moving toward a set of small policy functions while the larger `_resolve_plan` orchestrator is still
being retired incrementally.
Submit-gate normalization, target/top1 checks, quality reason soft overrides, daily-limit row counting, and slot spacing live in
`src/kagglebot/submission_policy.py`, leaving Kaggle API quota lookup and ledger side effects in the orchestration layer.
Explicit submit decision objects such as `QualitySubmitOverrideDecision`, `InitialSubmitProbeDecision`, and
`LimitedSubmissionHoldbackDecision` now carry soft override/probe/holdback results back to the loop instead of spreading
that state across several booleans.
Submit failure classification, notebook-submit fallback detection, and repair-target selection live in
`src/kagglebot/submit_failure_policy.py`. The loop still owns persistence, ledgers, and knowledge recording, but it no
longer owns the pure decision table for whether a failed submit should repair the artifact, submit mode/kernel, platform
polling path, or wait for manual intervention.
Submit failure context persistence, reference parsing, prompt formatting, stale repaired-artifact decisions, and submit
autofix artifact resolution live in `src/kagglebot/submit_failure_context.py`, keeping submit recovery state handling out
of the main loop. Submit failure-context payload creation also lives there; the loop supplies repair decisions and
runtime state snapshots.
Deterministic submit file repair preparation lives in `src/kagglebot/submit_autofix.py`; the loop supplies persistence
and validation callbacks while the module owns the repair-required check and result summary.
Submit code fingerprinting, same-error-fingerprint retry allowance, duplicate-submission skip decisions, and
same-submission-path retry/skip decisions live in `src/kagglebot/submit_retry_policy.py`; the loop supplies paths,
hashing, and state persistence callbacks.
Submit attempt payloads, submit run-state updates, submit knowledge-record message/fix summaries, submit result payloads,
and submit success outcome display/ledger-recording decisions live in `src/kagglebot/submit_attempts.py`. The same module
now owns `submit_attempts.jsonl` append, duplicate SHA lookup, and tolerant row readers used by resume state and
self-improvement reporting. This keeps the submit attempt record shape and JSONL parsing rules centralized instead of
duplicated across the loop, state helpers, and improvement analysis.
Historical Kaggle submission row normalization, best/latest public-score summary construction, online-regression
detection against historical submissions, and prompt formatting for that history live in
`src/kagglebot/submission_history.py`; the loop only fetches/cache-persist rows and consumes the resulting summary.
Notebook submit artifact-mode normalization, tiny public sample hidden-test guards, submit-kernel run kwargs construction,
kernel output artifact/reference handling, output file selection, Kaggle submit-kernel kwargs construction, ambiguous
submit retry execution, push-error text detection, and CPU fallback execution live in `src/kagglebot/submit_notebook.py`.
Shared JSON object loading lives in `src/kagglebot/json_utils.py` so policy and state modules do not reimplement
permissive artifact reads.
Initial submit-stage mode decisions, file/notebook submit attempt dispatch, successful submit result normalization,
submit-error classification normalization, submit-error retry/abort decisions, submission outcome abort/classification
decisions, rank payload/guard/display normalization, iteration submit-status formatting, and campaign-aware submission
message/score tracking resolution live in `src/kagglebot/submit_stage.py`, starting the split of `_attempt_submit` into typed
file-submit/notebook-submit stage services.

For each iteration:
1. Train (`local_gpu` or Kaggle kernel mode)
2. Evaluate (holdout/cv/test/consensus based on resolved plan)
3. Write `metrics.json` and `diagnostics.md`
4. Decide:
   - submit current iteration output
   - wait for submission result
   - if submission score is top1-tier: stop loop
   - if max iteration reached: stop loop
   - else: run improvement and continue

Loop decision source:
- primary: readiness score (SRS) from offline evaluation + uncertainty
- secondary: submission score/rank guardrails when submission outcomes are available

## Submission Safety

Before any submit call:
- rules accepted check
- strict submission format validation
- duplicate hash check
- rate-limit and retry policy
- repeated error fingerprint abort

## Artifact Map

```text
artifacts/<slug>/
  plan.json
  context/
    research_sources.jsonl
    research_summary.md
    research_storage.json
    agent/
      brief_for_strategy.md
      strategy_plan.md
      codex_instructions.md
      strategy_transcript.txt
  kernel/
    kernel.py
  runs/<run-id>/iter-<k>/
    metrics.json
    diagnostics.md
    submission.csv

knowledge/
  kb.sqlite
  taxonomy.yml
  research/<problem_type>/<slug>/
    research_sources.jsonl
    research_summary.md
```

## Notes

- Git branch/stash automation is not part of current implementation.
- Autofix can patch `src/` when runtime errors require framework-level fixes.

## Architecture Improvement Direction

The main architectural risk is still the size of `src/kagglebot/autopilot.py`. Future updates should keep extracting
stable, testable orchestration policies out of that file rather than adding new loop branches inline.

### Target Boundaries

The long-term shape should keep four responsibilities separate:

1. **Adapters**: Kaggle CLI/API, filesystem, process execution, and kernel runtimes. These modules translate external
   side effects into typed results and should not own loop policy.
2. **Policies**: pure decision modules for plan normalization, score trust, submission gating, retry/abort decisions,
   rank interpretation, and campaign selection. These modules should be unit-testable without Kaggle credentials or
   artifact directories.
3. **State and artifacts**: JSON/JSONL readers, writers, manifests, fingerprints, and durable run/iteration/submission
   records. These modules should own schema tolerance and migration behavior.
4. **Orchestration**: `autopilot.py` and `supervisor.py` should compose adapters, policies, and state services. They
   should avoid direct parsing of CLI text, JSON payload internals, and scalar coercion unless doing one-off wiring.

Dependency direction should be:

```text
cli/supervisor/autopilot
  -> services/stages
    -> policies + state/artifact helpers
      -> scalar/json/path/hash utilities
  -> adapters
```

Policies and state helpers should not import `autopilot.py`, `supervisor.py`, or CLI modules. Adapter modules may import
shared utilities and exception types, but should not import campaign or submission policy modules.

### Current Modernization Themes

Recent cleanup has started consolidating scalar and environment parsing into `scalar_utils.py` and `env_utils.py`. Keep
following that pattern: when two modules parse the same external shape, put the tolerant conversion in a shared utility
and leave call-site wrappers only where they document a local policy choice, such as comma handling or accepting
integral float strings.

The next high-value modernization work is:

1. **Typed submit service**: move the remaining `_attempt_submit` side-effect choreography into a `SubmitService` or
   `SubmitStageRunner` that composes `submit_stage`, `submit_attempts`, `submit_notebook`, `submit_failure_context`,
   and `submission_service`. The loop should receive one typed result with outcome, artifact reference, retry summary,
   and persistence payloads.
2. **Plan resolution service**: finish moving `_resolve_plan` into a typed plan-resolution module. The output should be
   one immutable resolved-plan object consumed by training, evaluation, submission, and prompts instead of a mutable
   dictionary with repeated ad hoc coercion.
3. **Kernel/run adapter split**: keep runner implementations behind `runners/` and prevent `kernel_runner.py` from
   accumulating unrelated hardware, parsing, and local-runtime policy. Shared runtime parsing belongs in small helpers;
   competition-specific generated code belongs in `kernel_runtime/`.
4. **Artifact schema registry**: centralize durable artifact shapes for `metrics.json`, `diagnostics.md`,
   `submit_attempts.jsonl`, candidate manifests, and self-improvement outputs. New artifact readers should use schema
   helpers rather than open-coding tolerant dictionary access in orchestration modules.
5. **Compatibility wrapper retirement**: after call sites move to extracted modules, remove private wrappers in
   `autopilot.py` and sibling orchestrators instead of preserving multiple names for the same policy.

Each modernization step should come with focused tests for the extracted module plus the standard full gate. Prefer
small extractions that make import direction clearer over broad refactors that only move code.

Recommended extraction order:

1. Plan resolution: continue moving `_resolve_plan` into `plan_policy.py`; split strategy normalization, score-source
   normalization, and competition-specific overrides are already out of the main loop.
2. Submission decision policy: keep moving candidate quality holdback, forced-submit reasons, and submit deferral into
   `submission_policy.py` until the loop consumes one explicit end-to-end submit decision object.
   Score progress helpers for official metric overrides, top1 gap classification, code-reference score normalization,
   severe-regression checks, conservative-collapse detection, and best-candidate priority comparison are now in
   `score_progress.py`.
   Top1 campaign metadata extraction from kernel metrics and validation-redesign preference checks are now in
   `campaign_metrics.py`.
   Kernel metrics payload parsing, CV fallback extraction, baseline score extraction, and kernel-log metric scraping are
   now in `kernel_metrics.py`.
   Kernel quality guard constants and payload-level detectors for subgroup collapse, external test-label transfer,
   candidate-selection mismatch, and prediction-distribution collapse are now in `kernel_quality.py`.
   Iteration metrics payload assembly, evaluation-result serialization, evaluation data-cache/fingerprint helpers, and
   iteration submit eligibility, plus run evaluation-report resume/persistence, are now in `iteration_metrics.py`.
   Diagnostics rendering and stable pipeline config hashing are now in `diagnostics.py`.
3. Kernel repair/autofix policy: continue isolating submit-error recovery from the loop. Submit failure classification,
   context formatting, artifact resolution, deterministic file repair preparation, same-fingerprint retry allowance, and
   submit retry decisions are now extracted.
   Lightweight runtime-fix artifact writers for missing columns, column aliases, object dtype coercion, device coercion,
   and blocked modules are now in `runtime_fixes.py`.
   Code-reference score extraction, reference notebook lookup, implementation marker construction, and reference
   implementation validation are now in `code_reference.py`; keep additional code-reference policy there unless it needs
   direct agent prompt rendering.
   Best-kernel snapshot capture/restore helpers are now in `kernel_snapshot.py`.
4. Submit state persistence: submit attempt JSONL writing/reading, duplicate SHA lookup, submit attempt/run-state payloads,
   submit failure-context payloads, submit knowledge-record payloads, and submit result payload construction are now
   centralized. Duplicate-submit skip decisions are extracted.
   Submit success outcome/ledger recording decisions, notebook submit kernel reference handling, ambiguous notebook submit
   retry decisions, CPU fallback decisions, push-error text detection, tiny public sample guards, notebook submit
   kernel-run kwargs construction, notebook submit result artifact/reference handling, and notebook submit exception/retry
   orchestration are extracted. Initial submit-stage mode decisions, file/notebook submit attempt dispatch, successful
   submit result normalization, submit-error classification normalization, submit-error retry/abort decisions, submission
   outcome abort/classification decisions, rank payload/guard/display normalization, iteration submit-status formatting,
   campaign-aware submission message resolution, submission iteration inference, tracking score selection, and
   file-submit-to-notebook fallback decisions are now in `submit_stage.py`. Next, move the remaining `_attempt_submit`
   side-effect orchestration into a typed service that coordinates the existing `submit_attempts`, `submit_stage`,
   `submit_notebook`, and `submit_failure_context` modules rather than adding more private wrappers in `autopilot.py`.
5. Runtime adapters: keep Kaggle CLI subprocess execution in adapter modules, and keep loop code dependent on typed result
   objects rather than raw CLI stdout/stderr parsing.
6. Artifact serialization: keep JSON object reads/writes for durable artifacts behind `json_utils` helpers. New modules
   should avoid ad hoc `json.dumps(...).write_text(...)` for artifact files unless they need non-object JSON, JSONL, or
   generated kernel code that runs outside the package.

Each extraction should preserve private compatibility names only where downstream tests/extensions still import them.
New code should call the smaller public modules directly, and obsolete private wrappers in `autopilot.py` should be
removed once repo-wide references are gone.
