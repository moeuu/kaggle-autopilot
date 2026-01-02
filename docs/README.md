# Kagglebot Documentation

Welcome to the kagglebot documentation! This directory contains comprehensive guides, specifications, and reference materials for using and developing kagglebot.

---

## Quick Navigation

### 🚀 Getting Started

- **[../README.md](../README.md)** - Minimal quick start guide with single command example
- **[autopilot.md](autopilot.md)** - Autopilot mode walkthrough and examples
- **[AUTOPILOT_SUMMARY.md](AUTOPILOT_SUMMARY.md)** - Summary of autopilot behavior and features

### 📐 Architecture & Design

- **[spec_autopilot.md](spec_autopilot.md)** - Complete production-grade specification
  - CLI contract, artifacts schema, 5-iteration loop
  - Metric inference, Top1 heuristic, compute modes
  - Knowledge base design, git guardrails, terminal UX
  - **START HERE** for understanding the full system design

- **[architecture.md](architecture.md)** - System architecture and control flow
  - Autopilot control flow diagrams
  - Safety gates (score-based, patience, submission)
  - Hard caps and non-interactive design
  - Evaluation strategy and error handling

- **[architecture_final.md](architecture_final.md)** - Full architecture reference
  - Component inventory and data flow
  - Safety gates, kernel runner, autopilot mode
  - Known failure modes and future work

### 🛡️ Safety & Compliance

- **[guardrails_checklist.md](guardrails_checklist.md)** - Comprehensive safety checklist
  - Submission safety, rules and ethics
  - Secret protection, non-interactive design
  - Correctness, resource limits, git safety
  - Pre-commit, pre-release, and incident response checklists

- **[safety/submission_checklist.md](safety/submission_checklist.md)** - Pre-submission checklist
- **[safety/failure_modes.md](safety/failure_modes.md)** - Known failure modes and mitigations

### 🧠 Knowledge Base

- **[knowledge.md](knowledge.md)** - Knowledge base design and usage
  - SQLite schema, tagging system
  - Similarity search and retrieval
  - Cross-competition learning

- **[taxonomy.md](taxonomy.md)** - Controlled vocabulary for tags
  - Data modality, problem type, dataset scale
  - Characteristics, model insights
  - Tag usage guidelines

### 🎯 Use Cases

- **[AUTOPILOT_SINGLE_SUBMIT.md](AUTOPILOT_SINGLE_SUBMIT.md)** - Single submission workflow

---

## Documentation Structure

### Repository Layout

```
kaggle-autopilot/
├── README.md                      # Minimal quick start (single command)
├── CLAUDE.md                      # Instructions for Claude Code
├── docs/                          # All detailed documentation (you are here)
│   ├── README.md                  # Documentation index
│   ├── spec_autopilot.md          # Complete production spec ⭐
│   ├── architecture.md            # Architecture and control flow
│   ├── architecture_final.md      # Full architecture reference
│   ├── guardrails_checklist.md    # Safety checklist
│   ├── autopilot.md               # Autopilot walkthrough
│   ├── knowledge.md               # Knowledge base design
│   ├── taxonomy.md                # Tag taxonomy
│   ├── AUTOPILOT_SUMMARY.md       # Autopilot summary
│   ├── AUTOPILOT_SINGLE_SUBMIT.md # Single submit workflow
│   ├── safety/                    # Submission safety references
│   ├── compute/                   # Compute switching specs/plans
│   ├── notebook_runner/           # Kaggle notebook runner design
│   └── agents/                    # Agent-specific guides
├── src/
│   └── kagglebot/
│       ├── cli.py                 # CLI entry point
│       ├── autopilot.py           # Autopilot orchestration
│       ├── solver/                # Generic ML pipelines
│       ├── knowledge/             # KB management
│       ├── git_utils.py           # Main-only git workflow
│       └── templates/             # Agent prompt templates
│           ├── baseline_plan_and_implement.md
│           ├── improve_iteration.md
│           └── postmortem_and_kb_update.md
├── tests/                         # pytest test suite
├── artifacts/                     # Competition-specific runs (gitignored)
│   ├── diffs/                     # Git diffs from autopilot runs
│   └── <slug>/
│       ├── meta.json              # Competition metadata
│       ├── plan.json              # Agent-defined targets
│       ├── context/               # Rules, dataset profile, top1, sample
│       ├── prompts/               # Generated agent prompts
│       ├── runs/<run-id>/         # Per-run artifacts
│       │   ├── run.json           # Run config + status
│       │   ├── iter-<k>/          # Per-iteration artifacts
│       │   │   ├── metrics.json
│       │   │   ├── diagnostics.md
│       │   │   └── submission.csv
│       │   └── agent/             # Agent interactions
│       └── submissions/
│           └── ledger.jsonl       # Submission deduplication log
└── knowledge/                     # Cross-competition learning (gitignored)
    ├── kb.sqlite                  # SQLite database
    └── taxonomy.yml               # Controlled tag vocabulary
```

---

## Document Descriptions

### Core Specifications

#### spec_autopilot.md ⭐

**The** authoritative specification for the kagglebot autopilot system. Read this first if you want to understand:
- What the system does and how it works
- CLI interface and required arguments
- Artifacts schema and file formats
- Metric inference and direction detection
- Top1 heuristic comparison rule
- Knowledge base design with SQL schema
- Compute modes (local_gpu, kaggle_gpu, kaggle_tpu)
- Git workflow guardrails
- Terminal UX design
- Error handling and security

**Audience**: Developers, contributors, architects
**Length**: ~800 lines
**Status**: Production-grade, complete

#### architecture.md

Detailed control flow diagrams and safety gate descriptions:
- Bootstrap & plan generation
- Iteration loop (verify → train → evaluate → check → improve)
- Safety gates (score-based, patience, submission guardrails)
- Hard caps (max iterations, time, submissions)
- Evaluation strategies (holdout, CV, labeled test)
- Artifacts layout and schemas
- Knowledge base integration
- Error handling with exit codes

**Audience**: Developers implementing features
**Length**: ~370 lines
**Status**: Complete

#### guardrails_checklist.md

Comprehensive safety and correctness checklist covering:
- Submission safety (dry-run, deduplication, rate limiting, validation)
- Rules and ethics (no automation, no scraping, no circumvention)
- Secret protection (no commits, no logging, no prompts)
- Non-interactive design (no user prompts, all pre-configured)
- Correctness (validation, evaluation, score-based gating)
- Resource limits (hard caps, compute modes, GPU/TPU utilization)
- Git safety (main-only, auto-stash, no force push)
- Agent prompts (context, acceptance criteria, no secrets)
- Error handling (exit codes, actionable messages)
- Knowledge base (metadata only, SQL injection safe)
- Testing (coverage, CI/CD, no real submissions)
- Documentation (minimal README, help text, error messages)
- Pre-commit, pre-release, and incident response checklists

**Audience**: All contributors, reviewers
**Length**: ~500 lines
**Status**: Complete

### User Guides

#### autopilot.md

Step-by-step walkthrough of autopilot mode:
- How to run autopilot with minimal args
- What happens during bootstrap, iteration, submission
- How to interpret metrics and diagnostics
- How to customize behavior with flags
- Common workflows and examples

**Audience**: End users
**Status**: User-friendly guide

#### AUTOPILOT_SUMMARY.md

High-level summary of autopilot features and behavior:
- Default settings and safety features
- Iteration loop overview
- Submission policies
- Top1 heuristic explanation

**Audience**: End users, quick reference
**Status**: Summary document

#### AUTOPILOT_SINGLE_SUBMIT.md

Specific workflow for single submission use case:
- When to use single submit vs full autopilot
- Command examples
- Expected behavior

**Audience**: End users with specific use case
**Status**: Workflow guide

### Knowledge Base

#### knowledge.md

Design and implementation of the knowledge base:
- SQLite schema with tables (competitions, tags, runs, iterations, improvements)
- Tagging system for similarity search
- Retrieval strategy (max tag overlap)
- Update procedures (post-run summaries)
- Privacy considerations (no raw data)

**Audience**: Developers working on KB features
**Status**: Design document

#### taxonomy.md

Controlled vocabulary for competition tags:
- Data modality (tabular, text, image, timeseries, multi_modal)
- Problem type (regression, binary, multiclass, ranking, forecasting)
- Dataset scale (tiny, small, medium, large, huge)
- Characteristics (missingness, cardinality, imbalance, skewness)
- Model insights (feature engineering, regularization, ensemble, GPU, etc.)

**Audience**: Developers, users customizing tags
**Status**: Reference document

---

## Agent Prompt Templates

Located in `src/kagglebot/templates/`:

### baseline_plan_and_implement.md

Template for iteration 0 (plan + baseline):
- Competition overview and context
- Compute environment and constraints
- Available context files and KB hints
- Step-by-step plan creation guidance
- Baseline implementation guide (data loading, preprocessing, modeling, evaluation, submission)
- Safety rules and quality checklist
- Acceptance criteria (tests pass, offline score, submission valid, GPU utilized)

**Usage**: Formatted at runtime with competition-specific values
**Audience**: Codex agent (LLM)

### improve_iteration.md

Template for iterations 1-5 (improvements):
- Current performance analysis
- Iteration history and previous attempts
- Compute environment and constraints
- Root cause diagnosis (underfitting, overfitting, features, hyperparameters, etc.)
- Targeted improvement strategies with code examples
- Validation and safety checks
- Acceptance criteria

**Usage**: Formatted at runtime with current metrics, diagnostics, and history
**Audience**: Codex agent (LLM)

### postmortem_and_kb_update.md

Template for post-run analysis:
- Run summary and iteration history
- What worked (successful strategies with quantified impact)
- What didn't work (failed strategies with reasons)
- Competition-specific insights and learnings
- Knowledge base update format (JSON schema)
- Tag assignment from controlled taxonomy
- Acceptance criteria for KB update

**Usage**: Formatted at runtime with full run results
**Audience**: Codex agent (LLM)

---

## Reading Paths

### For End Users

1. **[../README.md](../README.md)** - Quick start
2. **[autopilot.md](autopilot.md)** - Detailed walkthrough
3. **[AUTOPILOT_SUMMARY.md](AUTOPILOT_SUMMARY.md)** - Features overview

### For Contributors

1. **[spec_autopilot.md](spec_autopilot.md)** - Complete system spec ⭐
2. **[architecture.md](architecture.md)** - Control flow and safety gates
3. **[guardrails_checklist.md](guardrails_checklist.md)** - Safety checklist
4. **[../CLAUDE.md](../CLAUDE.md)** - Claude Code instructions

### For Researchers/Architects

1. **[spec_autopilot.md](spec_autopilot.md)** - Full design rationale
2. **[knowledge.md](knowledge.md)** - KB design and learning system
3. **[taxonomy.md](taxonomy.md)** - Controlled vocabulary
4. Template files in `src/kagglebot/templates/` - Agent prompts

---

## Contributing to Docs

### Documentation Principles

1. **Minimal README**: Main README.md has single command example only
2. **Details in docs/**: All comprehensive guides live here
3. **No duplication**: Each fact lives in one place, cross-reference with links
4. **Actionable errors**: Error messages include what, why, and how to fix
5. **Examples everywhere**: Show, don't just tell

### Adding New Documentation

1. Create file in `docs/` directory
2. Add entry to this README.md (under appropriate section)
3. Cross-reference from related docs
4. Update CHANGELOG.md if user-facing
5. Run: `uv run ruff format docs/`

### Updating Existing Docs

1. Edit the canonical source (don't duplicate)
2. Update "Last Updated" date if present
3. Check all cross-references still valid
4. Update CHANGELOG.md if significant
5. Review with guardrails checklist

---

## Help & Support

- **CLI Help**: `uv run kagglebot --help` or `uv run kagglebot <command> --help`
- **Issues**: https://github.com/anthropics/kagglebot/issues (if public repo)
- **Discussions**: Check GitHub Discussions for Q&A

---

**Last Updated**: 2026-01-02
**Maintainer**: kagglebot contributors
**License**: See LICENSE file in root directory
