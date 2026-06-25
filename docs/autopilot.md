# Autopilot

Autopilot is a non-interactive Kaggle loop with readiness-score iteration control.
It always follows this high-level path:

1. Bootstrap competition context
2. Plan and implement initial kernel via `gpt -> gpt -> gpt`
3. Train/evaluate per iteration
4. Improve if needed
5. Submit and use submission outcomes as secondary guardrails

## Quick Start

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/<slug> \
  --compute local_gpu
```

For long-running operation across competitions already entered by the Kaggle account:

```bash
uv run kagglebot --force watch --compute local_gpu
```

`watch` selects from `group=entered`, runs one competition at a time, records its loop state under
`artifacts/_watch/`, and then selects again. It does not accept rules, join competitions, or consider
unentered competitions.

Supported compute values:
- `local_gpu`
- `kaggle_gpu`
- `kaggle_tpu`

## Planning Flow (gpt -> gpt -> gpt)

Autopilot planning is fixed to:

1. GPT (`gpt-5.5`, xhigh): reads local context and writes a brief.
2. GPT (`gpt-5.5`, xhigh): performs strategy planning with live web search when available.
3. GPT (`gpt-5.5`, xhigh): implements kernel code from frozen instructions.

The GPT stage now produces and the pipeline persists:
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/method_scout_queries.json`
- `artifacts/<slug>/context/source_registry.json`
- `artifacts/<slug>/context/method_registry.json`
- `artifacts/<slug>/context/validation_registry.json`
- `artifacts/<slug>/context/validation_lab_report.json`
- `artifacts/<slug>/context/win_contract.json`
- `artifacts/<slug>/context/private_robustness_report.json`
- `artifacts/<slug>/context/top1_exhaustion_report.json`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent copy)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent copy)
- `artifacts/<slug>/plan.json`

`research_sources.jsonl` stores per-source research metadata (query, top URLs, publish dates, takeaway, and extracted technique).

## Iteration and Submission Policy

Per iteration, autopilot does:
1. Train and evaluate
2. Write metrics and diagnostics
3. Check top1-tier condition (direction-aware)
4. If not top1-tier and iterations remain, run improvement

Submission behavior:
- Default: submit every iteration
- Leaderboard CLI runs default to `--campaign-mode top1`; use `--campaign-mode baseline` for the older lightweight loop
- Top1 campaign mode writes `context/campaign_state.json`, `context/candidate_registry.json`, `context/reference_reproduction_report.json`, and `context/experiment_graph.json`, tracks historical best public score, top1 gap, validation trust, candidate metadata, and remaining daily slots
- `--portfolio-execution off|serial|parallel|budgeted` controls the top1 candidate DAG. The default is `serial`; `budgeted` runs the highest evidence-value ready node first and records candidate budget hints.
- Portfolio execution writes a graph execution report and per-candidate manifests/metrics so independent candidates can fail or complete without collapsing the whole iteration.
- `--method-scout auto|off|refresh` controls competition-specific method discovery. In top1 mode it defaults to `auto`, generates modality/metric/domain-specific research queries, ranks methods from research sources, and blocks unsafe or infeasible methods before prompt injection
- `--research-scout auto|off|refresh` writes attributed source evidence and planned retrieval queries to `source_registry.json`; `--validation-lab auto|off|force` calibrates split profiles and records adoption evidence in `validation_lab_report.json`
- `--top1-exhaustive` applies safe exhaustive defaults (`method/research refresh`, `validation-lab force`, `portfolio budgeted`) and records win contract, private robustness, portfolio optimizer, and exhaustion reports.
- `--top1-submit-policy value_only|calibration|final_lock` controls portfolio-level submit ranking without bypassing campaign baseline, duplicate, rate-limit, or rules guardrails.
- In top1 campaign mode, candidates below the historical/champion baseline are treated as regressions and are not submitted unless explicitly selected as calibration candidates or `--force-submit` is used
- `submission_gate` is activated only when rules indicate submission-count limits
- The first leaderboard iteration is submitted to establish an online checkpoint when submit is enabled, even with `submit_policy=improved`
- When daily/rolling submission slots remain for all remaining iterations, spare-slot policy can submit non-improving or soft quality-guarded candidates while still respecting hard safety guards
- Duplicate submission SHA is skipped before file or notebook submit unless explicitly forced
- Markdown rule text such as `five (5) Submissions per day` and rolling 24h limits such as `2 submissions within 24 hours` are parsed as daily limits
- Loop decision uses readiness score (SRS); submission score/rank are secondary guardrails
- Repeated submit-error fingerprints are aborted safely
- `deliverable_mode` is canonicalized to `leaderboard|writeup`; legacy `csv` values are accepted for backward compatibility
- `submit_mode` is resolved separately as `file|notebook`, with notebook-only rules able to force notebook submit without changing `deliverable_mode`
- notebook submissions with tiny public `test.csv`/`sample_submission.csv` fixtures are treated as hidden/full-test code competitions and use inference-mode notebook submit instead of embedding a local public-test CSV in a wrapper kernel
- static wrapper submit kernels fail fast for detected code competitions when the embedded CSV has only tiny public-test rows, preventing accidental 3-row notebook submissions
- submission writers and fold-intermediate artifact writers expand tiny public `sample_submission.csv` templates to the actual `test.csv` ids during notebook reruns, so completed folds can produce row-count-valid `submission_<candidate>_fold<N>.csv` artifacts
- heuristic `writeup` inference is conservative and ignores negative mentions such as `not a judged/writeup competition`
- leaderboard runs default to `target_medal=winner` and `target_rank_percentile=0.001`; until that near-first-place band is reached, autopilot will not collapse into `minor_tuning`
- for large tabular binary datasets with meaningful categoricals, planning quality gates require multi-family search plus at least one OOF blend candidate
- required reference notebooks emit `context/reference_inputs_manifest.json`; with `--download`, referenced datasets/competitions are staged under `context/reference_inputs/`
- if pseudo-labeling fully fails or an external/original-data feature path collapses to constants, the next iteration gets explicit repair targets instead of silently accepting the degraded path
- if CV improves but public LB regresses, autopilot treats that as a validation mismatch first and forces validation redesign with group/time/leak/proxy split candidates before model-only changes
- when kernels report multiple candidate pipelines, autopilot blocks CV-only winners whose holdout/validation score is materially worse than another candidate, especially if their test/submission prediction distribution collapses to sparse or constant-like outputs
- long-running multi-fold candidates are instructed to persist fold-level OOF/test predictions, metadata, and a valid `submission_<candidate>_fold<N>.csv` after each completed fold so an interrupted run can still submit the completed-fold candidate

## Important Defaults

- `--max-iterations`: default runtime behavior is 12 unless overridden by CLI
- `--internet`: default `on` for autopilot, but forced to `off` when captured competition rules ban notebook internet access
- Submission in autopilot is enabled by default
- Data bootstrap checks existing competition files and skips re-download when local file count/size already matches
- `--agent` and `--submit` are not part of autopilot CLI
- RNA sequence/structure datasets with residue-level coordinate submissions are profiled as `rna_structure` instead of generic tabular when the schema matches that family.
- Local kernel runs default to conservative worker/runtime guards: `KAGGLEBOT_NUM_WORKERS=0`, torch shared-memory fallback `file_system`, a best-effort higher `RLIMIT_NOFILE`, and a local stall watchdog so `watch` can fail and resume cleanly instead of showing a stale `local kernel running` state forever.
- Submission schema handling is flexible:
  - supports ID-based alignment when an ID column exists
  - falls back to row-order alignment when no reliable ID column exists
  - supports multi-target submission columns at I/O/validation layer
- CV strategy auto-selects from:
  - `TimeSeriesSplit` when reliable time columns exist
  - `GroupKFold` when group-like columns are detected
  - `StratifiedKFold` for classification
  - `KFold` otherwise
- If `plan.json` includes `evaluation_protocol.cv_type` (or `toggles.CV_TYPE`), autopilot now treats it as a strong hint and auto-upgrades a default `kfold` to the matching split (`group_kfold` / `timeseries_split` / `stratified_kfold`) to reduce local-public mismatch.
- Model-family selection can be customized by `KAGGLEBOT_MODEL_CANDIDATES`

## Main Flags

```text
--compute local_gpu|kaggle_gpu|kaggle_tpu   (required)
--accelerator auto|gpu|tpu
--hardware-profile auto|rtx3060|rtx5090|kaggle_p100|kaggle_t4|kaggle_t4x2
--score-source auto|holdout|cv|test
--holdout-frac FLOAT
--cv-folds INT
--max-iterations INT
--max-total-min INT
--patience INT
--min-improvement FLOAT
--force-submit
--internet auto|off|on
--verify-cmd "..."
--strict-accelerator
--resume-run-id RUN_ID
--resume-latest
```

`--hardware-profile` is passed into strategy prompts and staged kernel environments. For `local_gpu`,
`--hardware-profile rtx3060` means RTX3060-class accuracy-first planning: keep high-ceiling OCR/VLM/transformer paths
enabled where feasible, then scale batch size, chunks, precision, folds/seeds, or candidate ordering to fit 12GB VRAM.
Switch to `--hardware-profile rtx5090` when the same `kernel.py` should scale up on a larger GPU via
`plan.json`/environment knobs.

## Watch Flags

```text
watch --once
watch --submit-policy improved|none
watch --max-total-min INT      # default: no wall-clock limit
watch --max-iterations INT     # default 5; long heavy local_gpu runs may be capped to 3
watch --allow-slug SLUG
watch --block-slug SLUG
watch --self-improvement-interval-hours FLOAT
watch --self-improvement-codex/--no-self-improvement-codex
watch --self-improvement-publish/--no-self-improvement-publish
```

`--submit-policy improved` disables the initial contract-probe submit and only submits when an artifact improves over
a previously submitted checkpoint. Use `--submit-policy none` for artifact generation without live submissions.

Selection priority first favors entered competitions with no local autopilot run history, then competitions with no local
submission history, monetary prizes, and submitted competitions with poor current rank percentile where there is more
leaderboard headroom.
Because `watch` only reads Kaggle's entered-competition group, a passed new-entrant deadline does not exclude a
competition; only a passed submission deadline does. Unfamiliar competition types are passed through to autopilot
instead of being filtered out at selection time; complex simulation/reasoning/optimization tasks receive larger
training-time estimates so lightweight sidecars can still make capacity-aware choices.

`watch` also runs a periodic self-improvement loop. The loop scans recent runs, submission outcomes, top1 gaps,
diagnostics, and submit failures under `artifacts/`, writes `_self_improvement/latest.json`, `latest.md`,
`strategy_context.md`, `experiment_backlog.json`, and normalized `outcomes.jsonl`, then calls Codex to implement the
best top1-oriented repo improvement when the git worktree is clean. That improvement may include architectural changes
to planner, runner, evaluation, strategy, knowledge, model-search, or self-improvement boundaries when the report shows
the current architecture is blocking leaderboard progress. The generated strategy context is injected into
future bootstrap/planning prompts and live `knowledge_hints.txt`; generated playbooks are written under
`knowledge/playbooks/`. Use `--self-improvement-interval-hours 0` to disable it or `--no-self-improvement-codex` to
write reports without invoking Codex. `--self-improvement-publish` additionally verifies, commits, and pushes Codex
repo changes after success; it is off by default and still requires global `--force`. Manual runs are available
through `kagglebot self-improve [--publish]`.

## Artifacts

Key files:

- `artifacts/<slug>/plan.json`
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/research_storage.json`
- `artifacts/<slug>/context/method_scout_queries.json`
- `artifacts/<slug>/context/source_registry.json`
- `artifacts/<slug>/context/method_registry.json`
- `artifacts/<slug>/context/validation_registry.json`
- `artifacts/<slug>/context/validation_lab_report.json`
- `artifacts/<slug>/context/win_contract.json`
- `artifacts/<slug>/context/private_robustness_report.json`
- `artifacts/<slug>/context/top1_exhaustion_report.json`
- `artifacts/<slug>/context/reference_reproduction_report.json`
- `artifacts/<slug>/context/experiment_graph.json`
- `artifacts/<slug>/context/campaign_outcomes.jsonl`
- `artifacts/<slug>/runs/<run-id>/iter-*/portfolio_plan.json`
- `artifacts/<slug>/runs/<run-id>/iter-*/blend_report.json`
- `artifacts/<slug>/runs/<run-id>/iter-*/allocator_decision.json`
- `artifacts/<slug>/runs/<run-id>/iter-*/portfolio_optimizer_report.json`
- `artifacts/<slug>/runs/<run-id>/iter-*/graph_execution_report.json`
- `artifacts/<slug>/runs/<run-id>/candidates/<candidate-id>/candidate_manifest.json`
- `artifacts/<slug>/runs/<run-id>/candidates/<candidate-id>/metrics.json`
- `artifacts/<slug>/context/agent/brief_for_strategy.md`
- `artifacts/<slug>/context/agent/strategy_plan.md`
- `artifacts/<slug>/context/agent/codex_instructions.md`
- `artifacts/<slug>/runs/<run-id>/submit_failure_context.json`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/metrics.json`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/diagnostics.md`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/submission_manifest.json`
- `artifacts/_self_improvement/latest.json`
- `artifacts/_self_improvement/latest.md`
- `artifacts/_self_improvement/strategy_context.md`
- `artifacts/_self_improvement/experiment_backlog.json`
- `artifacts/_self_improvement/outcomes.jsonl`
- `knowledge/playbooks/*.md`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (authoritative persistence)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (authoritative persistence)

## Notes

- Rules acceptance is always manual in browser.
- Submission artifact resolution is manifest-first. Tabular runs can keep using `submission.csv`, but non-tabular single-file artifacts, bundles, and multi-file zip submissions are described through `submission_manifest.json`.
- Submit failures now persist a structured `submit_failure_context.json` snapshot so `submit_autofix` can distinguish between submission-file repairs, submit-mode/kernel fixes, platform issues, and manual blockers such as missing rules acceptance or credentials.
- For local kernel training (`local_gpu`), terminal logs show elapsed/ETA and stage progress (`seed i/N`, `fold j/K`, `step s/T`) when patterns are detectable from kernel output.
- For Kaggle kernel training, execution and logs are tracked through kernel run artifacts.
- If autopilot crashes, restart with `--resume-run-id <run-id>` or `--resume-latest`.
