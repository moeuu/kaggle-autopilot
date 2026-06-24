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
Competition rule parsing now lives in `src/kagglebot/competition_rules.py`; `autopilot.py` only keeps compatibility
aliases for older tests/extensions that imported private rule helpers from the main module.
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
Submit attempt payload and submit run-state update creation live in `src/kagglebot/submit_attempts.py`, keeping the JSONL
record shape, state update fields, submit knowledge-record message/fix summaries, and submit result payloads centralized
while the loop remains responsible for persistence. Submit success outcome display and ledger-recording decisions also
live there.
Notebook submit artifact-mode normalization, kernel reference construction, output file selection, submit-kernel kwargs
construction, ambiguous submit retry decisions, and CPU fallback decisions live in `src/kagglebot/submit_notebook.py`.
Shared JSON object loading lives in `src/kagglebot/json_utils.py` so policy and state modules do not reimplement
permissive artifact reads.

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

Recommended extraction order:

1. Plan resolution: continue moving `_resolve_plan` into `plan_policy.py`; split strategy normalization, score-source
   normalization, and competition-specific overrides are already out of the main loop.
2. Submission decision policy: keep moving candidate quality holdback, forced-submit reasons, and submit deferral into
   `submission_policy.py` until the loop consumes one explicit end-to-end submit decision object.
3. Kernel repair/autofix policy: continue isolating submit-error recovery from the loop. Submit failure classification,
   context formatting, artifact resolution, deterministic file repair preparation, same-fingerprint retry allowance, and
   submit retry decisions are now extracted.
4. Submit state persistence: submit attempt, submit run-state, submit failure-context, and submit knowledge-record payload
   creation are now centralized; duplicate-submit skip decisions and submit result payload construction are extracted.
   Submit success outcome/ledger recording decisions, notebook submit kernel reference handling, and ambiguous notebook
   submit retry decisions are extracted; CPU fallback decisions are extracted. Next, move submit kernel push-error text
   detection behind a small adapter.
5. Runtime adapters: keep Kaggle CLI subprocess execution in adapter modules, and keep loop code dependent on typed result
   objects rather than raw CLI stdout/stderr parsing.

Each extraction should preserve the existing private compatibility names in `autopilot.py` until downstream tests and
extensions have moved to the new public module.
