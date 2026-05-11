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
- `submission_gate` is activated only when rules indicate submission-count limits
- Duplicate submission SHA is skipped before file or notebook submit unless explicitly forced
- Rolling 24h submission limits such as `2 submissions within 24 hours` are parsed as daily limits
- Loop decision uses readiness score (SRS); submission score/rank are secondary guardrails
- Repeated submit-error fingerprints are aborted safely
- `deliverable_mode` is canonicalized to `leaderboard|writeup`; legacy `csv` values are accepted for backward compatibility
- `submit_mode` is resolved separately as `file|notebook`, with notebook-only rules able to force notebook submit without changing `deliverable_mode`
- heuristic `writeup` inference is conservative and ignores negative mentions such as `not a judged/writeup competition`
- leaderboard runs default to `target_medal=bronze` and `target_rank_percentile=0.10`; until that band is reached, autopilot will not collapse into `minor_tuning`
- for large tabular binary datasets with meaningful categoricals, planning quality gates require multi-family search plus at least one OOF blend candidate
- required reference notebooks emit `context/reference_inputs_manifest.json`; with `--download`, referenced datasets/competitions are staged under `context/reference_inputs/`
- if pseudo-labeling fully fails or an external/original-data feature path collapses to constants, the next iteration gets explicit repair targets instead of silently accepting the degraded path
- if CV improves but public LB regresses, autopilot treats that as a major online-mismatch signal and forces broader family/blend exploration next iteration

## Important Defaults

- `--max-iterations`: default runtime behavior is 3 unless overridden by CLI
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

## Watch Flags

```text
watch --once
watch --submit-policy improved|none
watch --max-total-min INT      # default: no wall-clock limit
watch --max-iterations INT     # default 12
watch --allow-slug SLUG
watch --block-slug SLUG
```

`--submit-policy improved` disables the initial contract-probe submit and only submits when an artifact improves over
a previously submitted checkpoint. Use `--submit-policy none` for artifact generation without live submissions.

Selection priority favors entered competitions with monetary prizes, competitions with no local submission history, and
submitted competitions with poor current rank percentile where there is more leaderboard headroom.

## Artifacts

Key files:

- `artifacts/<slug>/plan.json`
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/research_storage.json`
- `artifacts/<slug>/context/agent/brief_for_strategy.md`
- `artifacts/<slug>/context/agent/strategy_plan.md`
- `artifacts/<slug>/context/agent/codex_instructions.md`
- `artifacts/<slug>/runs/<run-id>/submit_failure_context.json`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/metrics.json`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/diagnostics.md`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/submission_manifest.json`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (authoritative persistence)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (authoritative persistence)

## Notes

- Rules acceptance is always manual in browser.
- Submission artifact resolution is manifest-first. Tabular runs can keep using `submission.csv`, but non-tabular single-file artifacts, bundles, and multi-file zip submissions are described through `submission_manifest.json`.
- Submit failures now persist a structured `submit_failure_context.json` snapshot so `submit_autofix` can distinguish between submission-file repairs, submit-mode/kernel fixes, platform issues, and manual blockers such as missing rules acceptance or credentials.
- For local kernel training (`local_gpu`), terminal logs show elapsed/ETA and stage progress (`seed i/N`, `fold j/K`, `step s/T`) when patterns are detectable from kernel output.
- For Kaggle kernel training, execution and logs are tracked through kernel run artifacts.
- If autopilot crashes, restart with `--resume-run-id <run-id>` or `--resume-latest`.
