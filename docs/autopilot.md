# Autopilot

Autopilot is a non-interactive Kaggle loop with readiness-score iteration control.
It always follows this high-level path:

1. Bootstrap competition context
2. Plan and implement initial kernel via `codex -> oracle(latest-pro) -> codex(sol-ultra)`; unavailable Oracle calls use the configured Codex fallback
3. Train/evaluate per iteration
4. Improve if needed
5. Submit and use submission outcomes as secondary guardrails

## Quick Start

```bash
uv run kagglebot --force autopilot https://www.kaggle.com/competitions/<slug> \
  --compute local_gpu
```

`autopilot` is non-interactive and submit-enabled. Normal execution therefore requires the global `--force` flag
before the `autopilot` subcommand; use global `--dry-run` instead for a side-effect-free preview.

For long-running operation across competitions already entered by the Kaggle account:

```bash
uv run kagglebot --force watch --compute local_gpu
```

`watch` selects from `group=entered`, runs one competition at a time, records its loop state under
`artifacts/_watch/`, and then selects again. It does not accept rules, join competitions, or consider
unentered competitions. Entered-list slugs are normalized to lowercase while preserving Kaggle's supported hyphens
and underscores. Selection uses the lexicographic priority tier `unsubmitted -> monetary prize -> awards points/medal`
before the recorded score (deadline urgency, rank headroom, category, and competition size). This prevents a large
score bonus from moving a submitted or non-prize competition ahead of a higher priority tier.
`--dry-run watch --once` still performs the read-only entered-list lookup and prints the competition it would select.
It also performs read-only leaderboard lookup when a recorded submission score needs rank-percentile enrichment, so
the preview uses the same priority inputs as a real watch cycle.

Supported compute values:
- `local_gpu`
- `kaggle_gpu`
- `kaggle_tpu`

## Persistent systemd Service

Install and immediately start the repository-managed user services from any clone location:

```bash
./scripts/kagglebot-systemd install
```

The installer creates a stable symlink to the current clone and registers the unit files in `deploy/systemd/`.
`kagglebot-watch.service` directly runs `uv run kagglebot --force watch`; there is no separate service-specific
autopilot implementation. Pulling repository code therefore updates the implementation used on the next service
restart. The installer also enables `kagglebot-oracle-update.timer`, which checks npm every 15 minutes and before
`watch` starts. It installs the latest `@steipete/oracle` release through the dedicated Node 24 npm prefix. An update is
deferred while an Oracle or Oracle MCP process is active, then retried on the next timer cycle. Set
`KAGGLEBOT_ORACLE_AUTO_UPDATE=0` in `watch.env` to disable it. The primary, Oracle, and Oracle-follow-up model identities
remain centralized in `[tool.kagglebot.agent]` in `pyproject.toml`, not duplicated in the systemd unit.

Use `./scripts/kagglebot-systemd start|stop|restart|status|uninstall` for lifecycle management. Optional per-machine
settings belong in `~/.config/kagglebot-autopilot/watch.env`; start from `deploy/systemd/watch.env.example`. Keep API
tokens out of the repository. Enabling user lingering with `loginctl enable-linger "$USER"` makes enabled user services
start at boot without waiting for an interactive login.

## Discord Status Notifications

Run the notification worker separately from `watch`:

```bash
uv run kagglebot discord-notifier --interval-sec 300 --heartbeat-sec 1800
```

Configure `KAGGLEBOT_DISCORD_EVENT_API_URL`, `KAGGLEBOT_DISCORD_EVENT_API_TOKEN`, and optionally
`KAGGLEBOT_DISCORD_EVENT_ACCOUNT`. The worker monitors the local and sidecar watch scopes under
`artifacts/_watch/`.

Each run uses a `discord_update_key` containing its `run_id`, so switching to a new competition/run creates a new
Discord message while heartbeat updates for the same run edit that run's status card. Completion and failure use
separate event keys. The worker also consumes `started`, `finished`, and `failed` records from each watch
`ledger.jsonl` with a persisted byte offset. It advances the offset only after the event API reports at least one
matched route, so transient API failures and missing route configuration are retried instead of silently discarded.
On first upgrade, the cursor starts at the end of the existing ledger to avoid replaying historical notifications;
the current watch snapshot is still emitted normally.

Each Event API attempt is appended to `_watch/discord_delivery_receipts.jsonl` before the watch-ledger cursor
advances. The receipt records the client Event ID, matched route count, server acknowledgement/status, dedupe key,
and ledger offsets. `matched_routes > 0` proves route acceptance; the recorded server status is the downstream
delivery audit evidence when a Discord message is not visible.

For an active run, the watch state's status and phase are authoritative. An inner stage may leave `run.json` at a
failure status while Oracle/Codex autofix or recovery is already running; notification snapshots report that active
recovery phase and preserve the inner-stage value separately as `run_record_status` for diagnostics.

## Planning Flow (codex -> oracle(latest-pro) -> codex(sol-ultra))

Autopilot planning is fixed to:

Codex model and reasoning defaults are configured in `[tool.kagglebot.agent]` in `pyproject.toml`. Override them for
one run with `KAGGLEBOT_PRIMARY_MODEL` and `KAGGLEBOT_PRIMARY_REASONING_EFFORT` when needed.
Oracle uses the `gpt-5-pro` rolling alias, which selects the current ChatGPT Pro model with the default browser engine.
Set `KAGGLEBOT_ORACLE_MODEL` to pin a specific model. API engine users must select an API-supported model explicitly
when the rolling alias does not map to the desired release.

1. Codex: reads local context and writes a brief.
2. Oracle with the latest Pro model: performs strategy planning with the brief, local context bundle, and live web search when available.
3. Codex with the `sol-ultra` profile: implements kernel code from frozen instructions.

Autopilot attempts Oracle first. If Oracle is not installed as `oracle`, set
`KAGGLEBOT_ORACLE_COMMAND`, for example `KAGGLEBOT_ORACLE_COMMAND="npx -y @steipete/oracle"`. Extra Oracle flags can
be supplied with `KAGGLEBOT_ORACLE_ARGS`, such as `--browser-manual-login`. Kagglebot defaults Oracle to
`--engine browser --wait` so planning uses ChatGPT Pro browser access instead of unexpectedly spending API credits;
set `KAGGLEBOT_ORACLE_ENGINE=api` or `KAGGLEBOT_ORACLE_ENGINE=auto` to override that default.
Global `--dry-run` skips both model calls, writes a deterministic preview plan, resolves the run configuration and
guardrails, then stops before kernel preflight, training, evaluation, and submission.
Oracle strategy calls have an independent two-hour (`7200` second) outer timeout. On timeout, launch failure, a
non-zero Oracle exit, or an empty response, Kagglebot preserves the Oracle diagnostics and retries the same prompt
with Codex `gpt-5.6-sol` at `ultra` reasoning. Configure this with `KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC`,
`KAGGLEBOT_ORACLE_FALLBACK_CODEX`, `KAGGLEBOT_ORACLE_FALLBACK_CODEX_MODEL`, and
`KAGGLEBOT_ORACLE_FALLBACK_CODEX_REASONING_EFFORT`. Oracle's own browser wait remains 24 hours, but the outer timeout
takes precedence.
Oracle operational failures are handled by the dedicated strategy fallback rather than the generic Codex autofix
loop. When Oracle has already written a current response, Kagglebot validates that response even if the CLI later
reports a cleanup error. Chat archival is attempted and reported separately, so an archival verification warning
cannot replace or invalidate a usable Oracle answer.
Resume skips planning only when the current run contains `planning_complete.json`, written after Oracle planning,
sol-ultra implementation, and repository verification all succeed. A plan or kernel left by an older or incomplete
run cannot bypass the required Oracle attempt or its configured Codex fallback.
Iteration improvement, kernel-fix, and autofix flows also persist `oracle_workflow_state.json` in the run directory.
If systemd, SIGTERM, or an intentional source-reload restart interrupts Oracle or the following Codex pass, resume
re-runs that complete Oracle-to-Codex workflow before starting planning or another kernel. A corrupt or unsupported
pending workflow blocks resume instead of being silently skipped. Ordinary handled failures are recorded as failed
and continue to use the existing error policy rather than being retried forever.
The periodic repository self-improvement transaction uses the same ordering guarantee at the watch boundary:
`oracle_running` is re-executed in its existing transaction, while `codex_running` resumes from the saved Oracle
response, validated plan, implementation prompt, and repository baseline. This recovery runs before competition
selection or an active-run kernel resume.

Every implementation pass that consumes an Oracle response uses the single `[tool.kagglebot.agent]`
`oracle_implementation_*` profile. This covers initial kernel implementation, improvement iterations, kernel/error
autofix, the `implement` CLI command, and repository self-improvement. Pre-Oracle brief extraction and an explicitly
selected legacy `codex` strategy do not use `sol-ultra`.

When browser Oracle is selected and no explicit browser route is configured, Kagglebot bootstraps Chrome automatically
and appends `--remote-chrome 127.0.0.1:<port>` to the Oracle call. This lets SSH/TTY autopilot loops run without the
current shell having `DISPLAY`; Kagglebot first reuses an existing DevTools port, then tries the current `DISPLAY`, then
discoverable X displays such as `:1` with `/run/user/<uid>/gdm/Xauthority`, and finally headless Chrome if no display is
usable. The default port is `9222`.

While Oracle uploads files, Kagglebot also runs a DevTools compatibility watcher. It recognizes attachment controls by
the exact attached filename and supplies Oracle CLI's expected `Remove <filename>` accessibility label when ChatGPT is
rendered in Japanese or another locale. This changes only the temporary page accessibility attribute; Oracle remains
responsible for uploading the files, inserting the prompt, and submitting the consultation. Filename matching also
accepts ChatGPT's duplicate-upload suffixes such as `(1)`, which can occur on an Oracle quality-gate retry.

Useful overrides:

- `KAGGLEBOT_ORACLE_BROWSER_BOOTSTRAP=0` disables automatic Chrome bootstrap.
- `KAGGLEBOT_ORACLE_BROWSER_PORT=9333` changes the remote-debugging port.
- `KAGGLEBOT_ORACLE_DISPLAY=:1` and `KAGGLEBOT_ORACLE_XAUTHORITY=/run/user/1000/gdm/Xauthority` force a GUI session.
- `KAGGLEBOT_ORACLE_CHROME_COPY_PROFILE=~/.config/google-chrome` chooses the signed-in profile tree to copy into a
  temporary Oracle Chrome profile.
- `KAGGLEBOT_ORACLE_CHROME_USER_DATA_DIR=/path/to/profile` uses a persistent dedicated Chrome profile instead of a
  temporary copied profile.
- Auto-bootstrapped remote Chrome always uses model picker `select` and thinking time `extended`, so the `gpt-5-pro`
  rolling alias is applied instead of silently keeping the browser's current model. For diagnostics, override either
  explicitly through `KAGGLEBOT_ORACLE_ARGS`, for example `--browser-model-strategy current`.
- Browser Oracle attaches the rendered strategy prompt by default instead of pasting a potentially large prompt into
  the ChatGPT composer. API Oracle remains inline by default. Set `KAGGLEBOT_ORACLE_INLINE_PROMPT=1` to force inline
  delivery. Kagglebot also attaches `oracle_context_manifest.md` and `oracle_canonical_context.md`. The latter
  losslessly consolidates the complete canonical rules, overview, data description, submission format, dataset
  profile, sample submission, code/models/discussion snapshots, ranked Kaggle ecosystem discovery, and their indexes
  when available. Improvement and autofix calls additionally
  include the current `plan.json`, `kernel.py`, run state, iteration metrics, diagnostics, evaluation reports, text
  logs, and saved error transcripts. Repository self-improvement calls include `latest.json`, `latest.md`, strategy
  context, experiment backlog, skill candidates, and outcome history. Runtime context files are capped at 4 MiB each
  and 16 MiB total so a runaway log cannot break Oracle delivery. Full bundle contents are authoritative over capped
  inline excerpts. Improvement calls also freeze `iteration_evidence.json`, `iteration_evidence.md`,
  `kernel_before_improvement.py`, and `plan_before_improvement.json` in the completed iteration directory. The
  evidence bundle joins each Oracle hypothesis and Codex implementation report to the next iteration's trusted
  score, runtime errors, candidate outcomes, submission attempts, and actual kernel/plan diff. Score deltas are
  computed only when metric, direction, score source, and evaluation trust are comparable; external reference or
  unfaithful scores remain visible but cannot be labeled as iteration gains.
- `KAGGLEBOT_ORACLE_INLINE_PROMPT=0` explicitly sends the rendered prompt and context bundle through Oracle `--file`
  attachments. In browser mode, Oracle may still inline small text files, so they may not appear as visible ChatGPT
  file chips even though the model receives the content.
- `KAGGLEBOT_ORACLE_BROWSER_ATTACHMENTS=never|auto|always` controls browser file delivery for auto-bootstrapped remote
  Chrome when `--file` attachments are enabled.
- `KAGGLEBOT_ORACLE_COMPETITION_DATA=auto|never|owner-authorized` controls raw competition package delivery. `auto` is
  the default and attaches the canonical downloaded archive only when the competition Rules do not restrict third-party data
  transmission and the package fits `KAGGLEBOT_ORACLE_DATA_ATTACHMENT_MAX_BYTES` (100 MiB by default). Explicit Rules
  restrictions win in `auto` mode. `owner-authorized` records the operator's decision to use their authenticated Oracle
  session as an owner-controlled processing tool rather than a redistribution target. Kagglebot records the exact
  attachment or omission reason in the context manifest. For remote-Chrome browser runs, archives over 15 MiB are
  split into ordered `.zip` parts so every upload remains below Oracle's 20 MiB data-transfer ceiling. The manifest
  records the original SHA-256 and exact byte-concatenation order; Oracle is instructed to reconstruct and verify the
  original archive before analysis. Up to the configured 100 MiB default is therefore delivered without truncation.
- `KAGGLEBOT_REFRESH_EVALUATION_SPEC=1` explicitly refreshes the Oracle-generated evaluation specification. The global
  autopilot `--force` flag is reserved for intended side effects and no longer causes this expensive advisor call on
  every watch cycle; valid frozen specifications are reused by default.
- `KAGGLEBOT_ORACLE_BROWSER_INPUT_TIMEOUT=3600s` and `KAGGLEBOT_ORACLE_BROWSER_TIMEOUT=24h` are the defaults for
  Oracle's browser input and overall browser waits for slower remote Chrome sessions and long GPT Pro answers.
- `KAGGLEBOT_ORACLE_BROWSER_HEADLESS=0` refuses the headless fallback when ChatGPT/Cloudflare requires a real display.
- `KAGGLEBOT_ORACLE_FORCE=0` disables Kagglebot's default Oracle `--force`. The default is on so a stale timed-out
  Oracle session does not block the next autopilot strategy run with Oracle's duplicate-prompt guard.

The GPT stage now produces and the pipeline persists:
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/kaggle_discovery.json` and `kaggle_discovery.md` (ranked public Kaggle Datasets, Models,
  Code, hot Discussions, Game Arena, and Benchmarks context; cached for 24 hours)
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/method_scout_queries.json`
- `artifacts/<slug>/context/source_registry.json`

Kaggle ecosystem discovery is enabled unless `KAGGLEBOT_KAGGLE_DISCOVERY=0`. Adjust its cache with
`KAGGLEBOT_KAGGLE_DISCOVERY_REFRESH_HOURS` and the per-surface shortlist size with
`KAGGLEBOT_KAGGLE_DISCOVERY_MAX_ITEMS_PER_SURFACE`.
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
4. If not top1-tier and iterations remain, freeze an iteration evidence bundle and run Oracle improvement

Before Oracle chooses the next change, the evidence bundle records the current score provenance, the latest/best
public leaderboard score and improvement-direction delta versus the prior public best, fold/readiness evidence,
diagnostics and deduplicated error signatures, portfolio/graph outcomes, submission failures, and frozen
kernel/plan snapshots. From iteration 2 onward it adds the actual kernel diff, plan field changes, prior Oracle
hypothesis, Codex implementation report, and the observed outcome for each transition. Oracle is required to return
an improvement contract containing evidence citations, one falsifiable hypothesis, a material delta from failed or
no-op approaches, an attribution-safe validation plan, expected observations, stop/rollback criteria, and a fallback.
Interrupted Oracle workflows persist the evidence path and SHA-256 and refuse to resume from a modified bundle.

Kernel source preflight persists a redacted, structured diagnostic at
`runs/<run_id>/autofix/attempt-<n>/kernel-preflight-error.txt` before invoking a repair. Repair success is determined
by rerunning the exact source contract, so a no-diff repair may continue when the contract already passes. Identical
failing regenerations remain bounded by the generated kernel SHA-256.

Submission behavior:
- Default: submit every iteration
- A matched Kaggle outcome is merged into `context/submission_history.json` immediately. Before every next-iteration
  improvement decision, non-dry runs refresh the listing again so scores from manual or out-of-band submissions are
  also included; a fetch failure falls back to the merged cache.
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
- The first valid leaderboard checkpoint is submitted to establish an online checkpoint when submit is enabled, even with `submit_policy=improved`; a resumed run keeps this obligation until that run records a successful submission
- When daily/rolling submission slots remain for all remaining iterations, spare-slot policy can submit non-improving or soft quality-guarded candidates while still respecting hard safety guards
- Duplicate submission SHA is skipped before file or notebook submit unless explicitly forced
  - Existing zip/tar submission archives are rejected when they contain duplicate member names, matching bundle-build validation
  - Existing zip submission archives reject encrypted or symlink members instead of treating them as ordinary prediction files
  - Existing 7z/rar submission archives are opened and checked for readable members, unsafe member paths, unsupported member types, and duplicate member names
  - submit-only wrapper kernels apply the same zip/tar archive member validation to embedded file submissions before writing the Kaggle output artifact
  - submit-only wrapper generation preflights embedded 7z/RAR submissions locally before packaging the wrapper kernel
  - Markdown rule text such as `five (5) Submissions per day` and rolling 24h limits such as `2 submissions within 24 hours` are parsed as daily limits
- Loop decision uses readiness score (SRS); submission score/rank are secondary guardrails
- Repeated submit-error fingerprints are aborted safely
- Submit-failure repair classification recognizes submission artifact filenames from the shared asset-modality/tabular/archive suffix registry, including compound and compressed suffixes, instead of a stale hand-written extension list
- Local GPU runs use a 1440-minute safety ceiling unless a lower operator/rule limit applies. Bug-like stagnation is
  controlled separately through repeated-error fingerprints, no-improvement patience, and same-config loop guards.
- Training is not assumed to be mandatory. A plan may skip local fitting only when it explicitly states that local training
  is optional, provides a numeric estimate of at least 1440 minutes (cost labels alone do not qualify), and identifies an implemented pretrained/reference/solver/search/simulation/
  optimization/rule-based path plus a supported validation method. Code competitions then execute that path as a Kaggle
  notebook and continue through the usual output-contract, semantic, Codex evidence-review, quota, duplicate, and ledger
  guards. Missing evidence, unimplemented ideas, sample-template copying, dummy output, or absent runtime proof keeps the
  normal train-and-validate route.
- `deliverable_mode` is canonicalized to `leaderboard|writeup`; legacy `csv` values are accepted for backward compatibility
- writeup runs resolve required notebook outputs from `plan.json`, `context/evaluation_spec.json`, and requirement wording
  in the persisted official competition pages. Named outputs such as `features.csv` satisfy the kernel source/output
  contract without being treated as leaderboard prediction files. The report records and hashes every required output.
  When a private notebook is required and submission is enabled with global `--force`, autopilot publishes the notebook,
  downloads its outputs, requires their hashes to match the locally validated artifacts, adds the private notebook link,
  checks participation/rules, and submits through Kaggle's authenticated Projects/Writeup UI. Started, submitted, and
  ambiguous payload hashes are never retried automatically.
- `submit_mode` is resolved separately as `file|notebook`, with notebook-only rules able to force notebook submit without changing `deliverable_mode`
- notebook submissions with tiny public `test`/`sample_submission` fixtures are treated as hidden/full-test code competitions and use inference-mode notebook submit instead of embedding a local public-test artifact in a wrapper kernel
- completed Code Competition notebooks are not sent directly: Codex first reviews immutable Notebook/model/output/runtime-log evidence, then a deterministic guard re-hashes the evidence and rechecks expected output, known quota, exact Notebook-version duplicate identity, and ledger before the API executor runs. Restored scores with zero current evaluation rows and no full model-backed runtime evidence, fallback-only real predictions, repeated runtime exceptions, and persisted dependency/cache output trees fail closed even if Codex says approve. Hosted gateways are validated separately: `run_local_gateway` and hidden-only `serve()` are rejected, `serve()` must run before the visible fallback, and the tiny visible placeholder is not misclassified as the hidden evaluator's prediction set. A successful API call records the exact kernel/version/output identity immediately.
- file/wrapper submissions use the competition-independent semantic preflight. It rejects an unchanged sample template, exact row-constant predictions, identical multi-output heads, placeholder text, metrics-declared fallback output, selected/emitted pipeline drift, and metrics-vs-artifact row-count/filename/hash mismatches. Decisions are persisted in `runs/<run_id>/submission_semantic_preflight.json`. Hosted Code outputs instead use the server-lifecycle and runtime-attestation contract because their visible commit artifact is not the hidden prediction set.
- inference-mode packages stage `plan.json` beside `kernel.py` and inject `KAGGLEBOT_SELECTED_PIPELINE`, `KAGGLEBOT_SELECTED_OFFLINE_SCORE`, and `KAGGLEBOT_REQUIRE_REFERENCE_PATH` when the local qualification metrics provide them; kernels may use this contract to skip repeated CV and reproduce the selected hidden-test inference path. Because Kaggle script push uploads only the configured code file, the runtime recorder and exact expected contract are embedded into `kernel.py`, explicitly flushed before output collection, and compared with downloaded output/metrics before submit.
- GPU notebooks resolve an exact machine shape in this order: `KAGGLEBOT_SUBMIT_KERNEL_MACHINE_SHAPE`, `plan.json`'s `submit_machine_shape`/`runtime_budget.submit_machine_shape`, mapped Kaggle hardware profile, competition policy, then the broadly compatible `NvidiaTeslaT4` default. Gateway/inference code submissions always require GPU and reject CPU overrides or capacity fallback before push. ARC-AGI-3's competition policy selects the official `NvidiaRtxPro6000` CLI ID and forces internet off for that competition's reserved RTX pool. Other competitions do not receive RTX automatically; an explicit environment, plan, or hardware-profile selection is preserved when Kaggle grants that competition access to the same machine ID. Notebook metadata casing such as `nvidiaTeslaT4` is accepted only as input and canonicalized. The resolved shape is included in both the source fingerprint and notebook slug, preventing corrected hardware from reusing an older notebook's persisted accelerator configuration. Set the environment value to `default` to deliberately omit an exact shape.
- when a plan opts into required reference reproduction, dataset/kernel/model inputs from that reference notebook's metadata are inherited by the staged notebook; referenced models are also treated as required local assets so local runs fail with a cache command instead of silently scoring a weak fallback
- after the remote submit notebook finishes, its `metrics.json` is compared with the selected local candidate; a missing/invalid report, a still-blocked reference reproduction report, an omitted selected pipeline/model/score, pipeline or metric changes, newly missing dependencies/sources, loss of a required reference path, and material score regressions abort before the Kaggle submission API is called
- submit inference caches default to `/tmp/kagglebot-cache`; packages that explicitly persist dependency cache/site-package/wheelhouse trees under `/kaggle/working` are rejected before push
- remote submit output retrieval requests only the exact expected submission filename plus `metrics.json` (and only log files while polling), using Kaggle's maximum supported output page size so dependency caches and diagnostic artifacts cannot hide the real deliverable or exhaust output-listing requests
- static wrapper submit kernels fail fast for detected code competitions when the embedded artifact has only tiny public-test rows, preventing accidental 3-row notebook submissions; the row-count guard recognizes compressed and zip-wrapped single-table binary artifacts such as Parquet
- submit-only wrapper kernels also re-check runtime test files; when Kaggle exposes a hidden/full test set with more rows than the tiny public sample submission, the wrapper expands the output to the runtime test ids and fills unknown ids with a deterministic fallback
- submit-only wrapper kernels preserve non-CSV tabular output names and can align against runtime CSV/TSV/TAB/PSV/TXT, JSON/JSONL/JSONLINES/NDJSON, YAML/YML, Parquet/PARQ/PQ/Avro/HDF5/AnnData `.h5ad` obs/X matrices/Loom `.loom` col_attrs/matrix stores/GeoPackage `.gpkg`/`.geopackage` attribute tables/Shapefile `.shp` and DBF `.dbf` attribute tables/KML-KMZ `.kml`/`.kmz` placemarks, including compressed `.kml` inputs, NetCDF `.nc`/`.netcdf`/`.cdf`/`.nc4` table-like variables/FITS `.fits`/`.fit`/`.fts` binary tables/NumPy `.npy`/`.npz` tabular arrays/Feather/FTR/Arrow IPC, Stata, XML, Excel/XLSM, Pickle, compressed tabular, zip-wrapped single-table inputs such as `.csv.zip`/`.psv.zip`/`.jsonl.zip` plus binary table members like Parquet/Feather/Avro/ORC/Pickle/Excel/XLSB/Stata/SAS/SPSS, SQLite-derived sample/test files, DuckDB `.duckdb`/`.ddb` table inputs, and RDS/RData `.rds`/`.rda`/`.rdata` table inputs; NumPy matrix inputs can use adjacent `*_columns.txt`, `*.schema.json`, `columns.txt`, or `schema.json` sidecars when their width matches
- submit-only wrapper kernels package directory submissions, such as array-store outputs, Hugging Face/PEFT model directories, MLflow model directories, TensorFlow SavedModel directories, and TensorFlow checkpoint directories, into a deterministic zip before emitting the final Kaggle working-directory artifact
- submission writers and fold-intermediate artifact writers expand tiny public sample templates to the actual runtime test ids during notebook reruns, so completed folds can produce row-count-valid `submission_<candidate>_fold<N>.<suffix>` artifacts
- local submission validation rejects tiny static submissions when context identifies a hidden/full-test notebook/code competition, so public 3-row placeholder outputs do not pass preflight
- heuristic `writeup` inference is conservative and ignores negative mentions such as `not a judged/writeup competition`
- leaderboard runs default to `target_medal=winner` and `target_rank_percentile=0.001`; until that near-first-place band is reached, autopilot will not collapse into `minor_tuning`
- an observed bottom-decile result (with at least 20 teams), bottom-two-percent estimate plus independent score-collapse evidence, or an online score below 2% of a positive top score is treated as a probable implementation anomaly rather than ordinary model weakness. The next iteration enters `implementation_audit`: it traces hidden-test discovery, model/assets, runtime fallbacks, prediction variance, ID/order alignment, output filename/schema, metric scale, and exact Notebook output selection before model-family tuning. Existing hash/daily-limit guards still prevent unchanged or excessive resubmissions
- for large tabular binary datasets with meaningful categoricals, planning quality gates require multi-family search plus at least one OOF blend candidate
- required reference notebooks emit `context/reference_inputs_manifest.json`; with `--download`, referenced datasets/competitions are staged under `context/reference_inputs/`
- if pseudo-labeling fully fails or an external/original-data feature path collapses to constants, the next iteration gets explicit repair targets instead of silently accepting the degraded path
- if CV improves but public LB regresses, autopilot treats that as a validation mismatch first and forces validation redesign with group/time/leak/proxy split candidates before model-only changes
- when kernels report multiple candidate pipelines, autopilot blocks CV-only winners whose holdout/validation score is materially worse than another candidate, especially if their test/submission prediction distribution collapses to sparse or constant-like outputs; detection candidates are also blocked when the selected output has at least 5x the candidate-median prediction count while mean confidence is at most 0.10
- long-running multi-fold candidates are instructed to persist fold-level OOF/test predictions, metadata, and a valid `submission_<candidate>_fold<N>.<suffix>` after each completed fold so an interrupted run can still submit the completed-fold candidate

## Important Defaults

- `--max-iterations`: default runtime behavior is 12 unless overridden by CLI
- `--internet`: default `on` for autopilot, but forced to `off` when captured competition rules ban notebook internet access
- Submission in autopilot is enabled by default
- Data bootstrap checks existing competition files and skips re-download when local file count/size already matches
- `--agent` and `--submit` are not part of autopilot CLI
- RNA sequence/structure datasets with residue-level coordinate submissions are profiled as `rna_structure` instead of generic tabular when the schema matches that family.
- Analyzer metadata also routes RNA sequence/structure layouts to `task=rna_structure`, preserving residue anchors and coordinate triplets for RMSE-based strategy planning.
- Local kernel runs default to conservative worker/runtime guards: `KAGGLEBOT_NUM_WORKERS=0`, torch shared-memory fallback `file_system`, a best-effort higher `RLIMIT_NOFILE`, and a local stall watchdog so `watch` can fail and resume cleanly instead of showing a stale `local kernel running` state forever.
- Submission schema handling is flexible:
  - supports ID-based alignment when an ID column exists
  - falls back to row-order alignment when no reliable ID column exists
  - supports multi-target submission columns at I/O/validation layer
  - coordinate target columns such as `x/y/z` or `latitude/longitude` are treated as structured coordinate regression
    with coordinate-axis strategy defaults instead of generic multi-output regression
  - local I/O inference can treat a single train-only label column as the training target when the sample submission contains multiple numeric class-probability columns
  - analyzer metadata treats that class-probability-column pattern as `prediction_kind=probability_columns` with logloss direction
  - analyzer metadata reuses solver target/prediction-kind inference, so probability-like binary names such as `isFraud` and low-cardinality numeric regression names such as `sales` do not fall back to stale generic classification assumptions
  - analyzer strategies use `prediction_kind` when choosing defaults, adding probability calibration/renormalization candidates for probability submissions and ordered-threshold/QWK candidates for ordinal targets
  - dataset profiles record the same `probability_columns` pattern for downstream planning and knowledge lookup
  - text-like target columns such as translation/answer/caption, and generic string targets whose training values look like natural language, are tracked as `prediction_kind=text` and `task=text`; analyzer/profile strategy uses text baselines instead of forcing numeric/class models
  - dataset profiles tag text-generation targets such as translation, summary, answer, caption, and response as `target_semantics=text_generation`; method scouting adds retrieval/seq2seq/semantic-similarity queries and candidates
  - generic string targets such as `target` are routed to text baselines when training values look like natural language, even if `sample_submission` contains short placeholders such as `placeholder`
  - local and generated-notebook text baselines use TF-IDF nearest-neighbor retrieval over text feature columns, with deterministic constant-text fallback when no usable text signal exists
  - generated notebook baselines add lightweight metadata features for file-reference columns (`filename`, `path`, `image`, `audio`, `video`, `scan`, `array`, `point`), including existence, size, suffix, image dimensions/channels/pixels/aspect ratio, image frame count for multi-page images, thumbnail intensity statistics, wav/audio duration, audio sample rate/channels/frame count, and video width/height/fps/frame count/duration when available
  - generated notebook image/audio/video metadata suffix gates and medical-imaging metadata parser groups are injected from the shared asset-modality suffix registry instead of stale template-local sets
  - submission-format media, signal/waveform, medical-imaging, point-cloud/3D, geospatial, bio/sequence/structure, graph, scientific-array, annotation, and model-artifact keyword detection is generated from the shared suffix registries, with only canonical prose/context aliases such as JPEG/TIFF/AIFF/NIfTI/DICOM/OFF mesh/Shapefile/GeoPackage/PDB/Matrix Market/NetCDF/FITS/EDF/WFDB/NWB/TensorFlow Lite/COCO/LabelMe/YOLO kept as explicit mappings
  - generated notebook baselines add lightweight mask-image metadata for image file references, including nonzero pixel count, foreground coverage, and single-channel label count for segmentation sidecars
  - generated notebook baselines also add best-effort medical-imaging metadata features for references, including DICOM rows/columns/pixel spacing, NIfTI shape/voxel spacing, and NRRD/MHA/MHD header dimensions/spacing when the files are readable; gzip/bzip2/xz/zstd-compressed DICOM/IMA/NIfTI/NRRD/MHA/MHD and cryo-EM MRC/CCP4 files are supported for these lightweight header features, and Analyze `.hdr`/`.img` pairs plus microscopy formats such as ND2/LIF/LSM/DM3/DM4 are recognized by the shared medical suffix registry
- generated notebook baselines add lightweight Alembic/Blender/DAE/PLY/OBJ/X3D/USD/XYZ/VTK/Gmsh/STEP/IFC-style point-cloud metadata features, including gzip/bzip2/xz/zstd-compressed text-style point-cloud sidecars and point/face counts when headers or text records expose them; LAS/LAZ outputs preserve adjacent projection/index sidecars such as `.prj`, `.wkt`, `.lax`, `.lasx`, and `.aux.xml` when present; PLY outputs preserve `TextureFile` sidecars before submit, COLLADA `.dae` outputs preserve referenced local image sidecars before submit, X3D outputs preserve local `url` sidecars before submit, and USD ASCII outputs preserve local `@asset@` sidecars before submit
  - generated notebook baselines add lightweight graph metadata features for GraphML/GEXF/GML/Matrix Market/edge-list references, while the shared graph registry also recognizes knowledge-graph/linked-data formats such as RDF, Turtle, JSON-LD, N-Triples/N-Quads, OWL, and TriG, including gzip/bzip2/xz/zstd-compressed graph sidecars and submission-format prose such as "gzip-compressed Turtle RDF"; node/edge counts are added when the structure can be read cheaply
  - generated notebook baselines add lightweight geospatial metadata features for GeoJSON/KML/KMZ references, including compressed GeoJSON/KML sidecars, feature/placemark counts, and coordinate bounding boxes when the files can be read cheaply; georeferenced raster image outputs preserve world-file/GDAL sidecars such as `.tfw`, `.pgw`, `.jgw`, `.wld`, `.prj`, `.aux.xml`, and `.ovr` when present; elevation/compressed raster formats such as HGT/DEM/DTED and ECW/MrSID are routed as geospatial assets; GDAL VRT `.vrt` outputs preserve safe relative `SourceFilename`/`SourceDataset` raster sources; ENVI `.hdr` raster outputs preserve same-stem or `data file =` sidecars, MapInfo TAB outputs preserve same-stem `.dat`/`.map`/`.id` sidecars before submit, and MapInfo Interchange `.mif` outputs preserve same-stem `.mid` sidecars
  - generated notebook baselines add lightweight document metadata features for PDF/DOC/DOCX/EPUB/OpenDocument/PowerPoint/HTML/Markdown/LaTeX/reStructuredText/AsciiDoc/RTF/plain-text/subtitle references, including compressed text-style sidecars, page counts, and text character/word/paragraph counts when the files can be read cheaply
  - generated notebook baselines add lightweight NumPy/Zarr/OME-Zarr/N5 and scientific-array metadata features for NetCDF/FITS/MAT/HDF5-family references, including gzip/bzip2/xz/zstd-compressed FITS files, array count, rank, leading dimensions, and element count without loading full array payloads when headers expose shapes; OME-Zarr and N5 directory stores can use nested `zarr.json` or `attributes.json` metadata
  - generated notebook baselines add lightweight model-artifact metadata features for ONNX, safetensors, CatBoost `.cbm`, XGBoost `.ubj`/`.xgb`/`.bst`, PMML `.pmml`, CoreML `.mlmodel`/`.mlpackage`/`.mlmodelc`, skops `.skops`, TensorRT `.engine`/`.plan`, OpenVINO `.blob`, Rockchip RKNN `.rknn`, Hailo `.hef`, and Qualcomm/SNPE `.dlc` references, including graph node/input/output counts, tensor count, and parameter count when safe metadata APIs can read them; archive format hints that mention these model suffixes are treated as model bundles; Hugging Face/PEFT model directories with config/tokenizer metadata plus weights and MLflow directories with `MLmodel` plus payloads are packaged before submit, TensorFlow SavedModel output directories are detected via `saved_model.pb`/`saved_model.pbtxt`, and TensorFlow checkpoint directories or indexes such as `model.ckpt.index` preserve matching `.data-*-of-*` shards before submit
  - local output discovery also accepts plain non-empty `submission/`, `predictions/`, `masks/`, and similar final-output directories as multi-file archive candidates when no explicit submission file exists
  - generated notebook baselines add lightweight bio file metadata features for FASTA/FASTQ and PDB/mmCIF/SDF/MOL2 references, while the shared bio registry also recognizes genomics formats such as VCF/SAM/BAM/CRAM/GFF/GTF/BED, molecular text formats such as SMILES `.smi`/`.smiles`, InChI `.inchi`, SELFIES `.selfies`, and reaction `.rxn` files; common gzip/bzip2/xz/zstd-compressed sequence, genomics text, molecular-text, and structure files are recognized for modality and submission-format routing, with sequence length statistics and structure atom/residue/bond/molecule counts when the files can be read cheaply
  - generated notebook baselines add lightweight detection/segmentation annotation metadata features for COCO/LabelMe-style JSON references, Pascal VOC/CVAT/Label Studio-style XML annotations, and YOLO-style text label sidecars, including image, annotation, category, bbox, and segmentation counts when the files can be read cheaply; path-aware dataset modality detection treats common `annotations/`, `labels/`, `bboxes/`, and segmentation sidecars as image/detection assets without reclassifying generic `train.json` tabular data
  - dataset profiles detect image/audio/video/signal assets from local file extensions so planning can route heavy modalities more appropriately, including gzip/bzip2/xz/zstd-compressed DICOM/IMA/NIfTI/Analyze `.hdr`/`.img` pair/NRRD/MHA/MHD/MRC/CCP4 medical-imaging files, microscope formats such as ND2/LIF/LSM/DM3/DM4, and signal/waveform files such as EDF/BDF/WFDB headers/NWB/TDMS/ABF plus BrainVision, EEGLAB, FIF, CNT, and TRC neurophysiology formats
  - generated notebook file-reference detection also recognizes stem-only bio/geospatial/graph/array/document/text/detection columns such as `protein_id`, `geo_id`, `network_id`, `embedding_id`, `matrix_id`, `text_id`, `transcript_id`, `subtitle_id`, `transcription_id`, `mask_id`, `annotation_id`, and `bbox_id` when matching files exist
  - generated notebook file-reference resolution maps `test` asset references through common inference split directories such as `eval/`, `public/`, `private/`, `validation/`, `scoring/`, `leaderboard/`, and `inference/`; its asset index also keeps role-relative keys like `training/case_id` and `public/case_id` for unknown parent directory names
  - local asset-table synthesis resolves duplicate train/test asset stems with split-aware aliases such as `training/`, `public/`, and `eval/`, and treats asset IDs case-insensitively so label/sample rows do not silently bind to the wrong split
  - asset collection directories include bio/geospatial/graph/document/text/embedding/matrix/detection names such as `proteins/`, `molecules/`, `sequences/`, `geojson/`, `graphs/`, `documents/`, `texts/`, `transcripts/`, `subtitles/`, `embeddings/`, `masks/`, `labels/`, `annotations/`, and `bboxes/`, so duplicate stems can still resolve by split
  - analyzer metadata preserves tabular asset-reference and domain profile modalities such as image/audio/video, medical-imaging, array, point-cloud, geospatial, graph, bio/RNA, and multimodal for domain-specific strategy defaults instead of flattening them to plain tabular classification/regression
  - analyzer and method scouting medical-imaging plans now refer to generic medical header metadata/windowing so DICOM/IMA/NIfTI/Analyze `.hdr`/`.img` pair/NRRD/MHA/MHD competitions are not planned as DICOM/NIfTI-only workflows
  - analyzer metadata preserves text-feature classification/regression layouts with TF-IDF and embedding strategy defaults instead of treating natural-language feature columns as ordinary categorical tabular columns
  - local submission validation allows text values in text-like target columns even when the sample file has empty placeholder values
  - generated notebook baselines can expand a single training label column into multiple class-probability submission columns
  - class-probability sample columns can use either prefixes or suffixes, such as `class_cat`, `cat_probability`, `dog_proba`, or `class_0_probability`, and are mapped back to observed class labels by name
  - local and generated-notebook baselines route delimiter-based single-column multi-label targets such as `cat dog` or `cat,dog` through text-style retrieval output, even when the sample submission contains non-empty placeholder strings
  - local and generated-notebook baselines can also expand delimiter-based multi-label targets into one-vs-rest label probability columns when `sample_submission` has one numeric column per label
  - analyzer metadata also recognizes delimiter-based multi-label targets with F1 metric hints and one-vs-rest/per-label-threshold strategy defaults
  - analyzer/profile metadata recognizes multi-label indicator matrices with several binary label columns and routes them to one-vs-rest multi-label planning instead of generic multi-target classification
  - analyzer metadata also recognizes generic multi-output regression, multi-target classification, and mixed multi-task target layouts with per-target head strategy defaults
  - local binary classification baselines emit probability submissions for probability metrics such as AUC/logloss even when `sample_submission` has integer `0` placeholders
  - local and generated-notebook binary baselines also treat probability-like single target names such as `isFraud`, `risk`, `score`, or `probability` as probability submissions despite integer placeholders
  - local and generated-notebook baselines treat ordinal integer targets such as `severity`/`grade`/`stage` as ordered continuous scores, then round and clip predictions back to valid label values
  - local and generated-notebook baselines keep numeric target names such as `sales`, `price`, `amount`, or `count` on the regression path even when early data has only a few integer values
  - analyzer/profile metadata recognizes non-negative integer count targets such as `count`, `demand`, `quantity`, `trips`, or `orders` as `count_regression` with RMSLE and Poisson/log1p strategy hints; local and generated-notebook tabular baselines fit these targets with `log1p` and clip count predictions to non-negative values
  - analyzer/profile metadata recognizes bounded rate/ratio targets such as `conversion_rate`, `win_probability`, or `target_ratio` as `bounded_regression`; local and generated-notebook tabular baselines clip clear 0..1 or 0..100 predictions back to the target range
  - analyzer/profile metadata recognizes strongly right-skewed non-negative value targets such as `SalePrice`, `revenue`, or `amount` as `positive_skew_regression` with RMSLE/log1p strategy hints; local and generated-notebook tabular baselines fit these targets with `log1p` and clip predictions to non-negative values
  - shared tabular ensemble submission validation and prediction-range metadata preserve explicit regression predictions instead of always probability-clipping them, while still applying count/bounded/positive-skew clipping when that target structure is known
  - shared tabular blend helpers keep logit/rank blends for probability outputs and use weighted raw-scale blends for explicit regression outputs, with RMSE/RMSLE-compatible regression blend scoring and generic score-based seed support checks
  - local and generated-notebook tabular baselines expand a single continuous training target into quantile, prediction-interval, or generic continuous submission columns such as `p10/p50/p90`, `lower/upper`, and repeated numeric prediction heads, preserving monotonic column order where the format implies one
  - analyzer metadata also recognizes quantile and prediction-interval submission columns with pinball/interval-score metric hints and non-crossing output strategy defaults
  - local and generated-notebook tabular baselines use chronological holdouts when train/test expose a future datetime or ordinal time column such as `date_block_num`
  - local and generated-notebook tabular baselines derive lightweight calendar features from parseable datetime feature columns, such as year/month/day/day-of-week, while leaving numeric ordinal time keys intact and exposing the derived columns in run summaries
  - local and generated-notebook submissions can align composite sample IDs such as `row_id=user_id_item_id`, `user_id:item_id`, `user_id.item_id`, or concatenated `user_iditem_id` from multiple test columns when the composite ID column is absent from `test.csv`
  - local and generated-notebook baselines can emit lightweight unsupervised anomaly scores when train/test have no label column but the sample submission asks for a numeric `anomaly_score`, `outlier_score`, or `risk_score`
  - analyzer schema inference also accepts no-label anomaly-score layouts by falling back to solver-inferred schema metadata instead of requiring the sample target column to exist in train
  - analyzer metadata also recognizes user-item CTR/recommender layouts with calibrated probability or rating-score strategy defaults
  - local and generated-notebook baselines collapse survival event/time targets such as `efs` and `efs_time` into a single numeric risk score when the sample submission asks for one `prediction`/score column
  - analyzer metadata also recognizes that survival event/time plus single-score layout as `task=survival` with a concordance-index metric and censoring-aware strategy defaults
  - analyzer metadata also recognizes pairwise matchup layouts as `task=pairwise` with probability calibration and pair-difference strategy defaults
  - local and generated-notebook baselines treat query/document `relevance` or rank targets as continuous ranking scores, even when the sample submission uses integer zero placeholders
  - analyzer metadata also recognizes query/document relevance layouts as `task=learning_to_rank` with NDCG and grouped-ranking strategy defaults
  - analyzer metadata also recognizes detection/segmentation submission columns such as `prediction_string` or `EncodedPixels` with mAP/Dice metric hints and prediction-string/RLE strategy defaults
  - analyzer metadata also recognizes future temporal layouts as `task=forecasting` with chronological validation and lag/rolling-feature strategy defaults
  - fold-intermediate tabular runtime artifacts can also write multi-column class-probability submissions from 2D prediction matrices
  - discovers and validates sample/submission files across CSV/TSV/TAB/PSV/TXT, space-delimited DAT/TXT, fixed-width FWF inputs, JSON/JSONL/JSONLINES/NDJSON, YAML/YML, Parquet/PARQ/PQ/Avro/HDF5 dataset groups/AnnData `.h5ad` obs/X matrices/Loom `.loom` col_attrs/matrix stores/GeoPackage `.gpkg`/`.geopackage` attribute tables/Shapefile `.shp` and DBF `.dbf` attribute tables/KML-KMZ `.kml`/`.kmz` placemarks plus compressed `.kml` inputs, NetCDF `.nc`/`.netcdf`/`.cdf`/`.nc4` table-like variables/FITS `.fits`/`.fit`/`.fts` binary tables/NumPy `.npy`/`.npz` tabular arrays with optional adjacent column/schema sidecars/Feather/FTR/Arrow IPC, Stata, MATLAB MAT, RDS/RData `.rds`/`.rda`/`.rdata`, SAS/SPSS, XML, HTML table inputs, Excel/XLSM/ODS plus XLSB inputs, ARFF/OpenML-style inputs, LibSVM/SVMLight inputs, Pickle, compressed tabular variants, zip-wrapped single-table inputs including binary members for profiling/row-count guards, SQLite/DB and DuckDB `.duckdb`/`.ddb` table inputs, zip/tar/tgz/tar.bz2/tbz2/tar.xz/txz/tar.zst/tzst/7z/RAR, and supported code archives
  - discovery, bootstrap/context caching, local, fold-runtime, generated-notebook, submit-wrapper, validation, and autofix tabular read/write paths stabilize missing, blank, pandas-generated `Unnamed: ...`, MultiIndex, and duplicate column names so file selection, validation, writing, and baseline feature selection do not fail on format-specific column quirks
  - top-level and shallow nested zip/tar/tgz/tar.bz2/tbz2/tar.xz/txz/tar.zst/tzst/7z/RAR archives are safely extracted before file discovery/profile building; existing extracted files are preserved outside the explicit download-unzip path
  - data archive extraction rejects unsafe paths, duplicate extraction targets, and unsupported member types such as links, zip symlinks, or encrypted zip members before writing files
  - generated Kaggle notebooks also extract zip/tar/tgz/tar.bz2/tbz2/tar.xz/txz/tar.zst/tzst/7z/RAR inputs into `/kaggle/working/extracted_input` with the same duplicate-target and unsupported-member checks before train/test/sample discovery, preserving `/kaggle/input` read-only semantics
  - submit-only wrapper kernels use the same safe extraction checks before runtime sample/test alignment, so archived sample/test files still guard static notebook submissions
  - generated Kaggle notebooks can synthesize train/test tables from asset files plus `train_labels.*` and sample IDs when a competition omits `test.csv`
  - generated Kaggle notebooks can synthesize a sample submission from context format hints (`submission_format.md`, `overview.md`, `data.md`, etc.) plus discovered tabular test IDs or asset split IDs when no usable sample file is bundled, preserving supported tabular suffixes such as `.jsonl`, `.jsonlines`, `.ndjson`, `.yaml`, `.html`, and compressed variants like `.jsonl.gz`, `.ndjson.zst`, and `.csv.zst`
  - local kernels and submit-only wrapper kernels resolve hardcoded CSV-style reads against non-CSV tabular siblings, including fixed-width FWF, MATLAB MAT, RDS/RData `.rds`/`.rda`/`.rdata`, SAS/SPSS, ARFF/OpenML-style inputs, LibSVM/SVMLight, YAML/YML, HTML tables, SQLite, DuckDB `.duckdb`/`.ddb`, compressed text/JSON variants, and zip-wrapped single-table Parquet/Feather/Avro/ORC/Pickle/Excel/XLSB/Stata/SAS/SPSS inputs
  - local full-data metric guards recognize `sample_submission.*` suffix variants when checking known complete dataset layouts
  - baseline submission writers honor the output filename suffix for supported tabular outputs instead of forcing `submission.csv`
  - tabular ensemble runtime uses `KAGGLEBOT_SUBMISSION_FILENAME` to mirror fold submissions and manifests with the requested tabular suffix
  - generated Kaggle notebook baselines also honor `KAGGLEBOT_SUBMISSION_FILENAME` for supported tabular output suffixes
  - local kernels, generated Kaggle notebooks, and submit-only notebooks honor explicit output filenames from `submission_format.md`, such as `answers.nii.gz`, `predictions.zarr`, `labels.ome.zarr`, or `volumes.n5`, instead of always rewriting them to `submission<suffix>`
  - generated tabular baselines no longer write CSV payloads directly into non-tabular filenames; when they cannot produce the requested artifact class, they emit a tabular fallback plus `submission_manifest.json` instead of disguising the file contents
  - local initial-model baselines follow the same rule: tabular predictions targeting non-tabular artifact names are written as a `*.tabular.csv` fallback with `requested_output_path` recorded in the manifest
  - solver tabular fallbacks for non-tabular requested outputs use the requested artifact stem, such as `answers.tabular.csv`, instead of a shared `submission.csv`, avoiding collisions when multiple fallback artifacts are emitted
  - generated Kaggle notebooks and tabular ensemble runtimes use the same requested-stem `*.tabular.csv` fallback naming for non-tabular requested outputs
  - local tabular ensemble runtimes keep the emitted file tabular when a non-tabular output name is requested, including SQLite `.db`/`.sqlite`/`.sqlite3` single-file database outputs that the tabular writer cannot emit directly, and record the requested filename in `submission_manifest.json` for downstream repair/routing
  - manifest resolution preserves both the emitted `submission_path` and requested/expected output aliases such as `requested_output_path` or `expectedOutputFile`
  - stored submission artifacts keep a run-specific manifest next to the copied file, so requested-output metadata is not lost when artifacts move into `submissions/`
  - when the primary artifact is `submission_manifest.json` itself, it is stored as a run-specific manifest and remains directly usable for bundle preparation
  - direct manifest submissions must declare a usable `submission_path`, `staging_dir`, or `members/files` reference; metadata-only manifests are rejected instead of submitting the JSON manifest itself
  - stored primary manifests copy their referenced staging directory or members into a run-specific bundle directory while preserving member layout, so relative manifest paths remain valid after moving into `submissions/`
  - manifest member lists accept single-string/glob values such as `{ "files": "bundle/*.tif" }`, single-object entries such as `{ "sourcePath": "bundle/a.tif", "targetPath": "masks/a.tif" }`, object path values such as `{ "sourcePath": { "path": "bundle/a.tif" } }`, plus source/destination alias keys and unambiguous `source path -> archive path` mappings from common tooling
  - manifest staging/member source paths reject absolute paths and `..` traversal, preventing bundle manifests from accidentally packaging files outside the run output
  - recursive manifest globs keep their relative layout under archive-directory targets, avoiding collisions such as `fold1/mask.tif` and `fold2/mask.tif` both becoming `masks/mask.tif`
  - submission preparation refuses to submit a manifest-declared tabular fallback when the manifest says the requested output was non-tabular, forcing a real artifact implementation or manifest correction
  - non-tabular bundle submissions can be prepared and detected as zip/tar/tgz/tar.bz2/tbz2/tar.xz/txz/tar.zst/tzst archives when the competition format requires them
  - local kernel output discovery opens candidate zip/tar/tar.zst/7z/RAR submission archives before selecting them, so invalid, empty, unsafe, or duplicate-member archives do not shadow a valid fallback artifact
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
--hardware-profile auto|rtx3060|rtx5090|kaggle_p100|kaggle_t4|kaggle_t4x2|kaggle_rtx_pro_6000
--score-source holdout|cv
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

Selection priority is `unsubmitted`, then `monetary prize`, then Kaggle `awardsPoints`/medal eligibility, then all other
eligible entered competitions. Score and rank headroom order candidates only after those tier keys.
Because `watch` only reads Kaggle's entered-competition group, a passed new-entrant deadline does not exclude a
competition; only a passed submission deadline does. Unfamiliar competition types are passed through to autopilot
instead of being filtered out at selection time; complex simulation/reasoning/optimization tasks receive larger
training-time estimates so lightweight sidecars can still make capacity-aware choices.

`watch` also runs a periodic and incident-driven self-improvement loop. The loop scans up to 500 historical runs by
default, including submission outcomes, top1 gaps, diagnostics, submit failures, explicitly applied reusable skills,
and `_watch/**/ledger.jsonl` failures under
`artifacts/`. Watch failures are included even when discovery or dataset profiling failed before a run directory was
created. A new normalized watch-failure fingerprint bypasses the periodic interval once; handled fingerprints are
checkpointed so the same incident does not cause an unbounded Oracle loop. New watch failures also persist a
secret-free incident artifact with phase, traceback, run-existence, and repository-root evidence, allowing Oracle/Codex
to repair failures that previously had only a one-line ledger message. It writes
`_self_improvement/latest.json`, `latest.md`, `strategy_context.md`, `experiment_backlog.json`,
`skill_candidates.json`, and normalized `outcomes.jsonl`, then asks the Oracle/GPT strategy adviser for the highest
value improvement brief and calls Codex to implement it only after the repository is clean, committed, pushed to its
configured upstream, and bound to an exact repository URL and commit SHA. Oracle must echo that baseline in a validated
JSON plan; the baseline is revalidated before the `sol-ultra` Codex profile runs. That improvement may include
architectural changes to planner, runner, evaluation, strategy, knowledge, model-search, or self-improvement boundaries
when the report shows the current architecture is blocking leaderboard progress.

The loop also consolidates reusable knowledge: raw report and lesson events go into `knowledge/kb.sqlite`, failure
lessons are searchable through the FTS-backed event store, skill candidates are written to `knowledge/skills/*.md`,
and only skills recorded as `implemented` or `verified` in `applied_knowledge.json` update fitness in
`skill_evaluations`. Suggested skills remain separate and cannot receive outcome credit. The generated strategy context is injected
into future bootstrap/planning prompts and live `knowledge_hints.txt`; generated playbooks are written under
`knowledge/playbooks/`. Use `--self-improvement-interval-hours 0` to disable it or `--no-self-improvement-codex` to
write reports without invoking the Oracle/Codex implementation step. `--self-improvement-publish` additionally
verifies, commits, and pushes repo changes after success; it is enabled by default in `watch` and still requires global `--force`.
Optional absent repository files do not break the publish transaction. `watch` tracks the repository HEAD loaded by the
current process and re-executes itself between competition cycles whenever it changes, including a pre-publish repair
commit made during a self-improvement transaction. The next competition therefore cannot continue with stale source.
If the triggering incident failed before its run directory was created, a successful verified/published repair schedules
one immediate preflight retry with the original slug/run ID; later identical fingerprints fall back to the normal
interval and cooldown instead of looping indefinitely.
Near-last leaderboard outcomes are also normalized into stable per-competition anomaly fingerprints. A new fingerprint
bypasses the periodic interval, is prioritized as a repository/runtime-fidelity repair, and after a verified push
schedules one fresh guarded run. Repeated runs with the same anomaly signals reuse the handled fingerprint, so they do
not create an automatic repair/resubmit loop; subsequent submissions remain subject to duplicate-hash and Kaggle-limit
guards.
Historical outcomes without recorded rank are enriched from a cached full leaderboard. Rank ordering also overrides a
stale or misclassified metric direction, so minimize competitions are not audited as maximize competitions. Kaggle
outcome polling refuses stale nonmatching descriptions and only permits a description-less Code row within the bounded
request-time window, preventing an old score from being recorded against a newly failed submission.
When the CLI is launched from an artifact-only directory with the default `--workdir .`, agent edits and source-change
detection resolve to the repository containing the imported `kagglebot` package while artifacts remain in their
configured location; explicit embedding roots are preserved.
Manual runs are available through `kagglebot self-improve [--publish]`.

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
- `artifacts/_self_improvement/skill_candidates.json`
- `artifacts/_self_improvement/outcomes.jsonl`
- `knowledge/kb.sqlite`
- `knowledge/playbooks/*.md`
- `knowledge/skills/*.md`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (authoritative persistence)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (authoritative persistence)

## Notes

- Rules acceptance is always manual in browser.
- Submission artifact resolution is manifest-first. Tabular runs can keep using `submission.csv`, but non-tabular single-file artifacts, bundles, and multi-file zip submissions are described through `submission_manifest.json`.
- File submissions only associate adjacent run-specific manifests or same-directory manifests that reference the submitted file; nested manifests are reserved for directory/bundle submissions so unrelated artifacts do not affect plain file validation.
- Run-specific file manifests can omit `submission_path`; their filename association is enough to apply requested-output metadata to the submitted file.
- When associated file manifests omit `artifact_class`, the artifact class is inferred from the submitted file/hints before requested-output guardrails run.
- Requested-output guardrails also inspect the emitted file suffix, so a CSV/TSV/etc. cannot bypass non-tabular artifact checks by declaring `artifact_class=single_file`.
- Run-specific file manifests that do declare `submission_path` must point back to the submitted file; mismatches are treated as stale/corrupt metadata and rejected.
- Associated submission manifests must be valid JSON objects; malformed run-specific metadata is rejected rather than silently ignored.
- Manifest path fields accept common aliases and object values such as `{ "path": "predictions.zarr" }`, `{ "sourcePath": "answers.nii.gz" }`, and `folderPath`, so hand-written or agent-generated manifests do not have to use one exact schema.
- Submit failures now persist a structured `submit_failure_context.json` snapshot so `submit_autofix` can distinguish between submission-file repairs, submit-mode/kernel fixes, platform issues, and manual blockers such as missing rules acceptance or credentials.
- For local kernel training (`local_gpu`), terminal logs show elapsed/ETA and stage progress (`seed i/N`, `fold j/K`, `step s/T`) when patterns are detectable from kernel output. Local kernels have a 1440-minute hard wall-clock timeout by default, enforced by the parent process rather than relying on generated kernel code. Override it with `--time-budget-min` or `KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN` (`0` keeps the existing explicit unlimited escape hatch).
- Historical ETA is used only for the exact staged-kernel source fingerprint. Once that median is exceeded, status reports `eta=unknown` instead of `eta~0s`. If the same exact source times out twice, or has at least two completed runs whose median exceeds the configured hard timeout, preflight rejects it for runtime-contract repair; it does not automatically remove models, folds, seeds, resolution, or other accuracy-bearing work.
- Discord status reports the active iteration's submit state. While an enabled run is still training it shows `pending`; it does not copy `disabled` or another terminal submit state from the previous completed iteration.
- Local-to-Kaggle GPU handoff is a last resort. A timeout, exhausted CUDA-OOM retry, host-memory guard, stall, or
  SIGKILL-style exit must repeat three times with the same resource-failure class after the Oracle-to-Codex repair
  loop before handoff is considered. A missing local GPU is immediately eligible, but every handoff still requires
  fresh quota data with at least 900 available minutes (and at least the configured run time budget). Stale, missing,
  or insufficient quota data keeps the run on `local_gpu`. The handoff is committed only after Kaggle accepts a kernel;
  package preparation is reported separately as `kaggle_kernel_preparing`. State is stored in
  `runs/<run-id>/compute_handoff.json` and `iter-<n>/compute_handoff.json`. Set
  `KAGGLEBOT_LOCAL_TO_KAGGLE_GPU=0` to disable handoff,
  `KAGGLEBOT_LOCAL_TO_KAGGLE_GPU_MIN_RESOURCE_FAILURES=<n>` to change the repeat threshold,
  `KAGGLEBOT_KAGGLE_GPU_HANDOFF_MIN_AVAILABLE_MINUTES=<n>` to change the quota floor, or
  `KAGGLEBOT_KAGGLE_GPU_HANDOFF_PROFILE=kaggle_t4` to override the default `kaggle_p100` profile.
- For Kaggle kernel training, execution and logs are tracked through kernel run artifacts.
- If autopilot crashes, restart with `--resume-run-id <run-id>` or `--resume-latest`.
