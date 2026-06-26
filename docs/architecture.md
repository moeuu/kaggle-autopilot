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
Supporting state/resume helpers now live in `src/kagglebot/autopilot_state.py`; score, leaderboard/online-best-score,
and iteration-signal policy helpers live in `src/kagglebot/score_utils.py`, `src/kagglebot/leaderboard_policy.py`, and
`src/kagglebot/iteration_signals.py`, so the main file stays focused on orchestration.
Competition rule parsing now lives in `src/kagglebot/competition_rules.py`; the loop calls that public module directly
instead of carrying private rule-parsing aliases in `autopilot.py`.
Offline score-source normalization, user-selectable score-source validation, and trust checks live in
`src/kagglebot/score_sources.py` for the same reason: the loop should consume normalized policy answers rather than own
every parsing rule inline.
Campaign mode normalization lives in `src/kagglebot/campaign.py`; CLI commands and the loop share the same aliases and
validation instead of keeping separate option parsers.
Method-scout, research-scout, portfolio-execution, validation-lab, top1-submit policy, and watch submit-policy
normalization similarly stay in their owning modules; CLI commands translate their policy errors into parameter errors.
Verify command execution support, external artifact mirroring for pytest verification, pytest environment isolation, and
competition-specific verify compatibility shims live in `src/kagglebot/verify_artifacts.py`; `autopilot.py` only invokes
that module from the verify step.
Kernel error formatting, normalized same-error fingerprints, and pushed-kernel registration failure detection live in
`src/kagglebot/kernel_errors.py`; the loop only records the resulting text/fingerprint and enforces retry limits.
Kaggle CLI error-shape helpers such as missing-credentials detection live in `src/kagglebot/kaggle_cli_errors.py`; the
loop maps those typed predicates to submit abort reasons instead of parsing credential hints inline.
Shared runtime policy such as `local_gpu` compute detection, heavy deep-learning modality classification, and local GPU
time-budget environment parsing live in `src/kagglebot/runtime_policy.py`; plan guardrails and autopilot execution use
the same definitions.
Compute-to-runner and compute-compatible accelerator resolution live in `src/kagglebot/compute.py`; CLI commands only
translate policy errors into CLI parameter errors.
Loop-control decisions such as explicit `max_total_min` wall-clock stops, no-improvement patience, repeated same-config
counter updates/stops, no-improve major-overhaul escalation, first-place stops, and max-iteration completion live in
`src/kagglebot/loop_control.py`, keeping budget stops and runaway-loop guards testable outside the main orchestration.
Agent prompt/response file I/O, capacity-error detection, failure detail formatting, and retry-feedback prompt appending
live in `src/kagglebot/agent_io.py`; the loop supplies agent identity and paths but does not own transcript formatting.
Context artifact reads such as dataset profile loading, evaluation-spec validation/override application, and capped CSV
data-row counting live in `src/kagglebot/context_artifacts.py`; orchestration code consumes normalized context payloads.
Metric aliases, metric direction inference, and canonical direction normalization live in
`src/kagglebot/solver/metrics.py`; callers should not hand-roll `minimize|maximize` parsing.
Split-strategy policy, planning necessity/resume-skip checks, evaluation seed/repeat normalization, rank-force
thresholds, improvement-mode upgrades, and competition-specific evaluation overrides live in
`src/kagglebot/plan_policy.py`. This keeps plan resolution moving toward a set of small policy functions while the
larger `_resolve_plan` orchestrator is still being retired incrementally.
Submit-gate normalization, target/top1 checks, quality reason soft overrides, daily-limit row counting, daily quota
fallback policy, and slot spacing live in `src/kagglebot/submission_policy.py`; the loop supplies the Kaggle fetch
adapter and ledger fallback count.
Explicit submit decision objects such as `QualitySubmitOverrideDecision`, `InitialSubmitProbeDecision`, and
`LimitedSubmissionHoldbackDecision` now carry soft override/probe/holdback results back to the loop instead of spreading
that state across several booleans.
Submit failure classification, notebook-submit fallback detection, and repair-target selection live in
`src/kagglebot/submit_failure_policy.py`. The loop still owns persistence, ledgers, and knowledge recording, but it no
longer owns the pure decision table for whether a failed submit should repair the artifact, submit mode/kernel, platform
polling path, or wait for manual intervention.
Submit failure context persistence, reference parsing, prompt formatting, stale repaired-artifact decisions/application,
run-level submit-attempt autofix context/input resolution, and submit autofix artifact resolution live in
`src/kagglebot/submit_failure_context.py`, keeping submit recovery state handling out of the main loop. Submit
failure-context payload creation plus submitted/duplicate-skip resolution marking also live there; the loop supplies
repair decisions and runtime state snapshots.
Deterministic submit file repair preparation lives in `src/kagglebot/submit_autofix.py`; the loop supplies persistence
and validation callbacks while the module owns the repair-required check and result summary.
Submit code fingerprinting, same-error-fingerprint reuse/allowance decisions, duplicate-submission source collection
and skip decisions, and same-submission-path retry/skip decisions live in `src/kagglebot/submit_retry_policy.py`; the
loop supplies paths, hashing, and state persistence callbacks.
Submit attempt payloads, submit run-state updates, duplicate-skip record payloads, submit knowledge-record message/fix
summaries, retry attempt/knowledge recording, retry knowledge detail formatting, submit result payload construction, and
submit success outcome display/ledger-recording decisions live in
`src/kagglebot/submit_attempts.py`. The same module now owns successful/duplicate-skip result timestamp/iteration
wiring, outcome ledger recording callback dispatch, `submit_attempts.jsonl` append, duplicate SHA lookup,
seen-fingerprint set assembly, and tolerant row readers
used by resume state and self-improvement reporting. This keeps the submit attempt record shape and JSONL parsing rules
centralized instead of duplicated across the loop, state helpers, and improvement analysis.
Historical Kaggle submission row normalization, best/latest public-score summary construction, online-regression
detection against historical submissions, history fetch/cache fallback, and prompt formatting for that history live in
`src/kagglebot/submission_history.py`; the loop supplies the Kaggle fetch adapter and consumes the resulting summary.
Notebook submit artifact-mode normalization, initial artifact-mode resolution, path-based artifact-mode decisions, tiny
public sample hidden-test guards, submit-kernel run kwargs construction, kernel output artifact/reference handling,
kernel push version-label inference, output file selection, Kaggle submit-kernel kwargs construction, ambiguous submit
retry execution, push-error text detection, and CPU fallback execution live in `src/kagglebot/submit_notebook.py`.
Shared JSON object loading, including permissive empty-dict fallback, lives in `src/kagglebot/json_utils.py` so policy,
knowledge, state, notification, and restart modules do not reimplement artifact reads.
Initial submit-stage mode decisions, file/notebook submit attempt dispatch, successful submit result normalization,
submit-error classification normalization, submit-error retry/abort decisions, manual submit-blocker abort specs,
submission outcome abort/classification decisions, poll-result outcome normalization/detail construction, rank
payload/guard/display normalization, iteration submit-status formatting, and campaign-aware submission message/score
tracking resolution live in `src/kagglebot/submit_stage.py`, starting the split of `_attempt_submit` into typed
file-submit/notebook-submit stage services.

For each iteration:
1. Train (`local_gpu` or Kaggle kernel mode)
2. Evaluate (holdout/cv based on resolved plan)
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
and leave call-site wrappers only where they document a local policy choice, such as comma handling, accepting integral
float strings, or choosing an environment-flag default.

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
   local training-progress parsing is now in `kernel_progress.py`, and kernel output/submission discovery is now in
   `kernel_outputs.py`; local artifact resolution/copying is now also in `kernel_outputs.py`; kernel log
   tailing/JSON-log formatting is now in `kernel_logs.py`; local duration-history estimation is now in
   `local_kernel_duration.py`; local memory/stall guard policy is now in `local_kernel_limits.py`; kernel
   slug/metadata construction is now in `kernel_metadata.py`; local staged-plan runtime-parameter validation is now
   in `kernel_plan_validation.py`; local sample-submission mirroring/placeholder expansion is now in
   `local_sample_submission.py`; local data/context profile staging is now in `local_kernel_context.py`;
   local generated-kernel data-dir resolver injection is now in `local_kernel_data_resolver.py`; local generated-kernel
   pipeline-config fallback injection is now in `local_kernel_pipeline_cfg.py`; local progress tracking, stall
   detection, and heartbeat formatting is now in `local_kernel_progress.py`;
   local runtime env defaults, optional-backend downgrades, and CUDA-OOM fallback env policy are now in
   `local_kernel_runtime_env.py`; local metrics normalization for competition-specific full-data guards is now in
   `local_kernel_metrics_normalization.py`; local model-cache discovery and staging is now in `local_kernel_models.py`;
   local text-runtime aux input staging is now in `local_kernel_aux_inputs.py`; kernel package source/runtime/external-file
   staging is now in `kernel_package_files.py`; local-module inlining for packaged kernels is now in
   `kernel_module_inliner.py`;
   zero-overlap drift guard generation is now in `local_kernel_drift_guard.py`; kernel bootstrap/env injection is now
   in `kernel_bootstrap.py`; local sitecustomize shim injection groups are now in `local_kernel_shims.py`; runner code
   should call those groups instead of enumerating individual shim functions; static submit
   wrapper rendering and code-competition tiny-submission rejection are now in `kernel_submit_wrapper.py`; Kaggle push
   source validation is now in `kernel_push_validation.py`; notebook submit-inference validation and output-root
   sanitization are now in `kernel_submit_inference.py`; competition-specific local-kernel contract checks are now in
   `kernel_contracts.py`;
   competition-specific generated code belongs in `kernel_runtime/`.
4. **Artifact schema registry**: centralize durable artifact shapes for `metrics.json`, `diagnostics.md`,
   `submit_attempts.jsonl`, candidate manifests, and self-improvement outputs. New artifact readers should use schema
   helpers rather than open-coding tolerant dictionary access in orchestration modules.
5. **Compatibility wrapper retirement**: after call sites move to extracted modules, remove private wrappers in
   `autopilot.py` and sibling orchestrators instead of preserving multiple names for the same policy. Obsolete
   autopilot wrappers around kernel capacity/data-tier inference have been removed; new call sites should use
   `kernel_quality.py` directly. Autopilot medal/rank normalization wrappers have also been removed; call sites now use
   `medals.py` directly. Online-best, rank-overhaul, and medal-target policy calls now use `leaderboard_policy.py`
   directly. Dataset-profile/evaluation-spec wrapper reads in `autopilot.py` have also been retired in
   favor of direct `context_artifacts.py` calls. Method-registry prompt formatting and effective method-scout mode
   resolution now read `method_scout.py` directly. Submit retry backoff and force-resubmit checks now call
   `submit_retry_policy.py` and `submit_failure_context.py` directly, same-fingerprint retry allowance now calls
   `submit_retry_policy.py` directly, and submit-abort deferral now calls `submit_failure_context.py` directly.
   Planning necessity and resume-skip checks now call `plan_policy.py` directly.
   Iteration resume, submit-retry artifact resume, best-submission resume paths, run payload/summary construction,
   run-state reads, iteration-state marker writes, and iteration submission-artifact resolution now call
   `autopilot_state.py` directly. Submit-attempt status/fingerprint lookups now call `submit_attempts.py` directly
   instead of passing through state wrappers. Submit autofix context formatting, stale repaired-artifact decisions and
   application,
   autofix artifact
   resolution, submit-failure improvement context, and submit-file repair contract checks now call
   `submit_failure_context.py` directly, and submit code fingerprinting now calls `submit_retry_policy.py` directly.
   Submit-kernel CPU fallback
   decisions, initial/path-based artifact-mode resolution, and kernel push version-label inference now call
   `submit_notebook.py` directly. Previous-submission history loading, regression checks, and prompt formatting now
   call `submission_history.py` directly.
   Metric-recheck OOF column
   selection and fold-score list parsing now call `kernel_metrics.py` directly.
   Daily quota count/fallback decisions now call `submission_policy.py` directly.
   Loop stagnation track selection, same-config counter updates, stop-reason construction, explicit wall-clock budget
   stops, terminal iteration stops, and no-improve major-overhaul escalation now call `loop_control.py` directly.
   Autopilot, iteration metrics, kernel quality, autopilot state, campaign metrics,
   submission history, iteration signals, score progress, kernel metrics, submission outcome, and code-reference scalar
   parsing wrappers have also been removed in favor of public helpers in `scalar_utils.py`.
   Kernel source preflight error construction now calls `validators.py` directly; `autopilot.py` keeps only the fix loop
   policy. Top1 public-score display formatting now calls `top1_exhaustive.py` directly.
   Agent/autopilot plan payload normalization, validation, high-accuracy suite repair, guardrail application, plan
   persistence, and resolved-plan-to-`PlanConfig` conversion now call `plan_policy.py` directly instead of living inside
   orchestration modules; obsolete autopilot plan load/write/resolved conversion wrappers have been retired.
   Submit-file autofix source resolution and deterministic repair preparation now call `submit_autofix.py` directly,
   leaving `autopilot.py` to provide run-specific services and state callbacks.
   Submit abort autofixability decisions now call `submit_failure_context.py` directly, removing the obsolete
   `autopilot.py` private wrapper around that policy.
   Target request selection, base evaluation request selection, runtime request selection, loop-control submit request
   selection, readiness, drift, no-improvement stop-policy, and rank-force threshold resolution now also live in
   `plan_policy.py`. Resolved-plan payload schema assembly now also lives there through `ResolvedPlan`, leaving the
   loop to orchestrate policy calls instead of owning the final artifact shape.
   Autopilot resolved-plan orchestration now lives in `plan_resolution.py`, so `autopilot.py` only adapts
   `AutopilotConfig` and default constants while existing plan policies remain reusable.
   Agent write-guard policy, repair edit-scope construction, snapshots, repairs, and secret prompt checks now live in
   `write_guard.py`; agent and autopilot orchestration import that shared module instead of sharing guard internals
   through `agent_pipeline.py`.
   Watch-state phase updates now live in `watch_state.py`; autopilot orchestration reports phase transitions without
   owning the state-file environment lookup or JSON update details.
   Initial `run.json` payload construction, run status/stop-reason application, final status resolution, and final run
   summary payload construction now live in `autopilot_state.py`, keeping run-state schema assembly with the rest of the
   state/artifact helpers.
   Tiny public `sample_submission.csv` expansion to authoritative test ids now lives in `submission_templates.py`, so
   solver and kernel-runtime submission writers share one row-count contract. Direction-aware score gaps, best-score
   selection, and candidate-vs-baseline comparisons now live in `score_utils.py`, leaving campaign modules as
   compatibility call sites and self-improvement run summaries on the shared score policy. Timestamp parsing for ISO
   values and Kaggle CLI date formats now normalizes values to UTC in `datetime_utils.py`, so bootstrap, Kaggle API
   adapters, notification, supervisor, submission ledger, submission quota policy, and submission-outcome code share one
   timezone policy.

Each modernization step should come with focused tests for the extracted module plus the standard full gate. Prefer
small extractions that make import direction clearer over broad refactors that only move code.

Recommended extraction order:

1. Plan resolution: continue moving `_resolve_plan` into `plan_policy.py`; split strategy normalization/override,
   target request selection, base evaluation request selection, runtime request selection, loop-control submit request
   selection, metric/direction override policy, plan score-source normalization, evaluation-spec value extraction,
   local-GPU evaluation budget/max-iteration policy, submit/runtime constraint application, planning
   necessity/resume-skip checks, readiness/stop-policy resolution, rank-force threshold resolution, and
   competition-specific overrides are already out of the main loop. Leaderboard medal/rank objective resolution, plan
   file I/O, and resolved-plan config conversion are now also in `plan_policy.py`.
2. Submission decision policy: keep moving candidate quality holdback, forced-submit reasons, and submit deferral into
   `submission_policy.py` until the loop consumes one explicit end-to-end submit decision object. Plan-level
   submit-policy and submission-gate resolution, latest-iteration/high-potential fallback-submit blockers, and initial
   major-overhaul reason aggregation now also live there.
   Score progress helpers for official metric overrides, top1 gap classification, code-reference score normalization,
   severe-regression checks, conservative-collapse detection, and best-candidate priority comparison are now in
   `score_progress.py`. Iteration-level score delta/update policy now also lives there instead of in `autopilot.py`.
   Top1 campaign metadata extraction from kernel metrics, post-submit state decisions, and validation-redesign
   preference checks are now in `campaign_metrics.py`.
   Kernel metrics payload parsing, evaluation-result normalization, CV fallback extraction, metric-recheck OOF column
   helpers, OOF metric recomputation, and payload persistence, fold-score list parsing, baseline score extraction, and
   kernel-log metric scraping are now in `kernel_metrics.py`.
   Kernel quality guard constants, payload-level detectors for subgroup collapse, external test-label transfer,
   candidate-selection mismatch, prediction-distribution collapse, and the aggregate submit quality-guard payload builder
   are now in `kernel_quality.py`.
   Iteration metrics payload/final guard-section composition, evaluation-result serialization, evaluation
   data-cache/fingerprint helpers, iteration submit eligibility, submit-phase completion decisions, iteration record
   kwargs construction, and run evaluation-report resume/persistence are now in `iteration_metrics.py`.
   Iteration repair-signal collection/extraction and next-iteration policy/knowledge payload assembly/dispatch are now
   in `iteration_signals.py`.
   Diagnostics rendering and stable pipeline config hashing are now in `diagnostics.py`.
   Loop score-update/readiness-noise/streak decisions and terminal/stagnation stop policy are now in `loop_control.py`.
3. Kernel repair/autofix policy: continue isolating submit-error recovery from the loop. Submit failure classification,
   context formatting, submit-file repair contract prompts/retry feedback, artifact resolution, deterministic file
   repair preparation, same-fingerprint retry allowance, and submit retry decisions are now extracted.
   Lightweight runtime-fix artifact writers for missing columns, column aliases, object dtype coercion, device coercion,
   blocked modules, deterministic strategy-skip decisions, and kernel-first non-autofixable runtime checks are now in
   `runtime_fixes.py`.
   Code-reference score extraction, reference notebook lookup, implementation marker construction, and reference
   implementation validation are now in `code_reference.py`; keep additional code-reference policy there unless it needs
   direct agent prompt rendering.
   Best-kernel snapshot capture/restore helpers are now in `kernel_snapshot.py`.
4. Submit state persistence: submit attempt JSONL writing/reading, run-bound submit attempt recorder/aborter callback binding,
   duplicate SHA lookup, submit attempt/run-state payloads, submit-abort artifact path resolution, repair-classified submit failure-context payloads, submit knowledge-record
   payloads/orchestration, same-submission-path skip payloads, submit-abort attempt/context persistence, submit result
   payload construction, resume-time submit attempt completion/iteration inference, run-level submit autofix context
   resolution, and run-level submit-abort autofixability resolution are now centralized.
   Seen-fingerprint set assembly/run loading, duplicate-submit source collection, and skip decisions are extracted.
   Submit success outcome/ledger recording decisions, notebook submit kernel reference handling, ambiguous notebook submit
   retry decisions, CPU fallback decisions, push-error text detection, initial artifact-mode resolution, tiny public
   sample guards, notebook submit kernel-run kwargs construction, notebook submit result artifact/reference handling,
   submit-kernel error wrapping, and notebook submit exception/retry orchestration are extracted. Initial submit-stage mode
   decisions, file/notebook submit attempt dispatch, successful
   submit result normalization, submit-stage runtime state transitions, initial submit runtime-state resolution,
   initial artifact-mode decision application,
   submit-error classification normalization, submit CLI error resolution/run binding, classification-driven retry/abort decisions,
   submit-error action abort specs,
   submit-retry attempt/knowledge recording orchestration, abort-spec kwargs mapping,
   duplicate-submit resolution/decision application, local validation/prepared-path resolution,
   rules-acceptance blocker resolution,
   local/Kaggle submit blocker abort-spec resolution,
   same-submission-path resolution/decision application, notebook fallback runtime-state resolution and decision
   application/state/message assembly,
   manual local validation/submit-blocker abort specs,
   submission polling/outcome abort specs, submission outcome
   classification and poll-result post-processing decisions, rank
   payload/guard/display/state normalization, iteration submit-status formatting,
   campaign-aware submission message resolution, submission iteration inference, iteration/fallback submit improvement
   gate decisions, tracking score selection/update, submission knowledge orchestration/context/default-insight
   preparation/record dispatch resolution, successful submit attempt/outcome/failure-context recording orchestration,
   file-submit-to-notebook fallback decisions, submission
   outcome polling orchestration, post-poll abort-spec resolution, and submit-abort persistence/knowledge recording are
   now in `submit_stage.py`; submission knowledge
   recording now calls the submit-stage helper directly from the loop. Notebook kernel submit execution, run-specific
   iteration/log path resolution, output-reference construction, kernel-output submit retry orchestration, and run-bound
   notebook submit callback wiring now live together in `submit_notebook.py`.
   Run-bound duplicate-submission and same-path skip handling now use
   `submit_stage.resolve_duplicate_submission_for_run` and `submit_stage.resolve_same_submission_path_for_run`.
   Submit retry attempt/knowledge recording is bound by `submit_stage.SubmitRunRetryRecorder`.
   Submit-abort recorder construction and exception raising now also run through `submit_stage.SubmitRunAborter`;
   successful submit ledger/outcome/failure-context finalization uses `submit_stage.record_successful_submit_for_run`.
   This removes the old private submit-abort wrapper from `autopilot.py` and keeps run-specific submit side effects
   behind typed submit-stage helpers.
   Next, move the remaining `_attempt_submit` side-effect orchestration into a typed service that coordinates the
   existing `submit_attempts`, `submit_stage`, `submit_notebook`, and `submit_failure_context` modules rather than
   adding more private wrappers in `autopilot.py`.
5. Runtime adapters: keep Kaggle CLI subprocess execution in adapter modules, and keep loop code dependent on typed result
   objects and shared `kaggle_cli_errors.py`, `kernel_status.py`, and `remote_kernel_state.py` helpers rather than raw
   CLI stdout/stderr parsing or ad hoc pending-run files.
6. Runtime policy: keep shared compute/modality/time-budget policy in `runtime_policy.py` and compute/accelerator
   compatibility in `compute.py` so agent plan guardrails, CLI commands, and autopilot execution cannot drift.
7. Agent I/O helpers: keep prompt/response transcript display, response-file reads, Codex sandbox-fallback logging,
   capacity-error detection, retry feedback prompt construction, and live stdout rendering in `agent_io.py`;
   orchestration code should pass identity/path context only.
   Strategy, improvement, autofix, error-repair, and mandatory code-reference repair prompt rendering now lives in
   `agent_prompts.py`, keeping long prompt templates out of `autopilot.py`.
   Strategy prompt file persistence, runner invocation, response loading, and failed/empty fallback handling now lives in
   `agent_strategy.py`; orchestration wrappers should only supply stage-specific labels and fallback messages.
   Problem-type knowledge context rendering and knowledge-hints file generation are shared through
   `knowledge_context.py`, so planning and improvement prompts do not duplicate insight/research lookup policy.
   Dataset-profile-to-problem-type resolution also lives there, keeping orchestration code from directly owning profile
   parsing for knowledge lookup. Improvement orchestration calls these public helpers directly instead of preserving
   private problem-type knowledge wrappers in `autopilot.py`. Dataset-profile cardinality helpers now live in
   `knowledge/profile_utils.py`. The legacy `knowledge_init.py` module is only a compatibility shim that re-exports
   `kagglebot.knowledge`; do not add behavior back to that old mirror.
8. Context artifacts: keep dataset-profile loading, evaluation-spec validation/override normalization, and capped CSV
   row-count helpers in `context_artifacts.py`; loop code should not open these context files directly.
9. Verify execution/staging: keep verify command execution policy, local/external artifact mirroring, pytest environment
   isolation, and competition-specific compatibility shims in `verify_artifacts.py`; avoid adding generated shim strings
   or pytest-specific execution rules back into `autopilot.py` or CLI commands.
10. Kernel error policy: keep exception formatting, same-error fingerprinting, pushed-kernel registration failure
   classification, kernel-error artifact writing, and repeated-error abort policy in `kernel_errors.py`; `autopilot.py`
   should only decide when to invoke that policy.
11. Artifact serialization: keep JSON object, array, and JSONL reads/writes for durable artifacts behind `json_utils`
   helpers. Forward and reverse JSONL history reads now share the same tolerant loader, so duration histories and
   crawled submission-format records do not open-code line parsing. JSON object/array parsing from already-read text or
   response bytes also lives in `json_utils.py`, so metadata readers, notebook input discovery, taxonomy loading,
   research artifact metadata, notifier responses, Codex event lines, advisor responses, dataset-profile summaries, and
   kernel-log parsing share the same typed parse contracts. New modules should avoid ad hoc JSON parsing/serialization
   for artifact files unless they need generated kernel code that runs outside the package or intentionally strict parser
   behavior.

Each extraction should preserve private compatibility names only where downstream tests/extensions still import them.
New code should call the smaller public modules directly, and obsolete private wrappers in `autopilot.py` should be
removed once repo-wide references are gone.
