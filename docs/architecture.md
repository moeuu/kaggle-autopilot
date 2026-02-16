# Architecture

This document describes the current autopilot execution architecture.

## Execution Overview

`kagglebot autopilot` executes in three major phases:

1. Bootstrap context
2. Agent pipeline (`codex -> gpt -> codex`)
3. Iterative train/evaluate/improve/submit loop

## 1) Bootstrap Context

Bootstrap prepares `artifacts/<slug>/context/`:
- rules URL and rules text
- overview/data/submission-format summaries
- dataset profile
- sample submission snapshot
- top1 public score snapshot

## 2) Agent Pipeline (`codex -> gpt -> codex`)

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
