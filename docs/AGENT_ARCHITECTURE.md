# Agent Architecture

## Overview

The `kagglebot.agents` package provides a framework for autonomous code generation using a three-stage "Codex → Claude → Codex" pipeline. This architecture enables automated competition strategy development and kernel implementation with strong safety guardrails.

## Three-Stage Pipeline

### Stage 1: Codex Brief Extraction
**Purpose**: Extract key competition facts from context files

**Inputs**:
- `context/overview.md` - Competition overview
- `context/data.md` - Dataset description
- `context/rules.md` - Competition rules
- `context/dataset_profile.json` - Dataset statistics
- `context/sample_submission_head.txt` - Sample submission format
- `context/top1_public.json` - Current leaderboard leader

**Outputs**:
- `context/agent/brief.md` - Human-readable brief
- `context/agent/brief.json` - Structured brief data

**Allowlist**: Only `context/agent/brief.*` files are writable

### Stage 2: Claude Strategy Generation
**Purpose**: Deep strategy research and implementation planning

**Inputs**:
- All context files from Stage 1
- Brief from Codex

**Outputs**:
- `context/agent/strategy.md` - Human-readable strategy
- `context/agent/codex_instructions.md` - Step-by-step implementation guide
- `context/agent/references.md` - Research references and links

**Delimiter Format**:
```markdown
===CLAUDE_STRATEGY===
<strategy content>
===CODEX_IMPLEMENTATION_INSTRUCTIONS===
<implementation instructions>
===REFERENCES===
<references and citations>
```

**Allowlist**: Only `context/agent/*.md` files are writable

### Stage 3: Codex Kernel Implementation
**Purpose**: Implement competition kernel following Claude's instructions

**Inputs**:
- `context/agent/codex_instructions.md` from Stage 2

**Outputs**:
- `kernel/kernel.py` - Main kernel script
- `kernel/**/*` - Supporting modules and utilities

**Allowlist**: Only `kernel/**` is writable (most restrictive!)

## Safety Guardrails

### 1. Write Allowlists
File write restrictions using glob patterns prevent agents from modifying forbidden files.

```python
from kagglebot.agents import WriteAllowlist

allowlist = WriteAllowlist(base_dir=Path("artifacts/my-competition"))
allowlist.allow("kernel/**")  # Recursive: all files under kernel/
allowlist.allow("*.md")         # Top-level only: .md files in base_dir
allowlist.allow("runs/*.json")  # One level: .json in runs/

# Check if path is allowed
if allowlist.is_allowed(Path("artifacts/my-competition/kernel/train.py")):
    # Write permitted
    ...
```

**Pattern Syntax**:
- `kernel/**` - All files recursively under kernel/
- `*.py` - Python files in base directory only
- `data/*.csv` - CSV files in data/ directory
- `context/agent/brief.md` - Specific file

### 2. File Snapshots
Detect all file changes using mtime + SHA256 hashing.

```python
from kagglebot.agents import FileSnapshot

# Before agent execution
snapshot_pre = FileSnapshot.create(competition_dir)

# Run agent
run_agent()

# After agent execution
snapshot_post = FileSnapshot.create(competition_dir)

# Find violations (changes not allowed by allowlist)
violations = snapshot_pre.diff(snapshot_post, allowlist)
if violations:
    raise AllowlistViolationError(violations)
```

**Change Detection**:
- New files created
- Existing files modified (content changed)
- Files deleted

### 3. Exception Handling
Dedicated exit codes for agent errors:

```python
from kagglebot.agents import (
    AllowlistViolationError,  # Exit 20: Modified forbidden files
    AgentOutputError,         # Exit 21: Missing required outputs
    AgentTimeoutError,        # Exit 22: Execution timeout
    DelimiterParseError,      # Exit 23: Malformed Claude output
    AgentExecutionError,      # Exit 30: Agent process failed
)
```

## Usage

### Basic Usage

```python
from pathlib import Path
from kagglebot.agents import run_agent_pipeline

# Run full three-stage pipeline
outputs = run_agent_pipeline(
    slug="titanic",
    artifacts_dir=Path("artifacts"),
    dry_run=False,
)

# Access outputs
print(f"Brief: {outputs['brief']}")
print(f"Strategy: {outputs['strategy']}")
print(f"Kernel: {outputs['kernel']}")
```

### Advanced Usage - Custom Workflow

```python
from pathlib import Path
from kagglebot.agents import (
    WriteAllowlist,
    FileSnapshot,
    parse_claude_strategy_output,
    verify_outputs_exist,
)
from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.claude_runner import run_claude

# Stage 1: Codex Brief
allowlist = WriteAllowlist(base_dir=competition_dir)
allowlist.allow("context/agent/brief.md")
allowlist.allow("context/agent/brief.json")

snapshot_pre = FileSnapshot.create(competition_dir)
result = run_codex(prompt_path, output_dir)
snapshot_post = FileSnapshot.create(competition_dir)

violations = snapshot_pre.diff(snapshot_post, allowlist)
if violations:
    raise AllowlistViolationError(violations)

verify_outputs_exist(agent_dir, ["brief.md", "brief.json"])

# Stage 2: Claude Strategy
result = run_claude(prompt_path, output_dir)
parsed = parse_claude_strategy_output(result.stdout)

# Write parsed sections
(agent_dir / "strategy.md").write_text(parsed.strategy)
(agent_dir / "codex_instructions.md").write_text(parsed.codex_instructions)
(agent_dir / "references.md").write_text(parsed.references)
```

## Testing

All agent infrastructure is tested using `tmp_path` fixtures to avoid touching real artifacts.

```bash
# Run agent tests
uv run pytest tests/test_agents_*.py -v

# Test allowlist patterns
uv run pytest tests/test_agents_allowlist.py -v

# Test snapshot change detection
uv run pytest tests/test_agents_snapshot.py -v

# Test delimiter parsing
uv run pytest tests/test_agents_base.py -v
```

### Writing Agent Tests

```python
def test_my_agent_feature(tmp_path: Path):
    """Test agent feature using tmp_path."""
    # Create temporary competition structure
    competition_dir = tmp_path / "my-competition"
    competition_dir.mkdir()

    # Run agent operations on tmp_path
    allowlist = WriteAllowlist(base_dir=competition_dir)
    allowlist.allow("kernel/**")

    # Assertions
    assert allowlist.is_allowed(competition_dir / "kernel/train.py")
```

**Critical**: NEVER use real `artifacts/` directories in tests. Always use `tmp_path` fixtures.

## Architecture Diagram

```
artifacts/
└── <slug>/
    ├── context/           # READONLY for agents
    │   ├── overview.md
    │   ├── data.md
    │   ├── rules.md
    │   ├── dataset_profile.json
    │   ├── sample_submission_head.txt
    │   ├── top1_public.json
    │   └── agent/         # WRITABLE (stage-specific)
    │       ├── brief.md              # Stage 1 output
    │       ├── brief.json            # Stage 1 output
    │       ├── strategy.md           # Stage 2 output
    │       ├── codex_instructions.md # Stage 2 output
    │       └── references.md         # Stage 2 output
    ├── kernel/            # WRITABLE (Stage 3 only)
    │   ├── kernel.py      # Main kernel script
    │   └── **/*          # Supporting modules
    └── runs/              # WRITABLE (autopilot logs)
        └── <run_id>/
            ├── 01_codex_brief/
            │   ├── codex_exec.jsonl
            │   └── codex_last_message.txt
            ├── 02_claude_strategy/
            │   ├── claude_exec.txt
            │   └── claude_last_message.txt
            └── 03_codex_implementation/
                ├── codex_exec.jsonl
                └── codex_last_message.txt
```

## Prompt Templates

Templates are located in `src/kagglebot/prompts/`:

1. **codex_brief.md** - Stage 1 prompt for Codex (reads local context file paths and summarizes)
2. **claude_strategy.md** - Stage 2 prompt for Claude
3. **codex_kernel_impl.md** - Stage 3 prompt for Codex

### Template Variables

Templates use `{{variable}}` syntax for substitution:

```python
from kagglebot.agents import render_prompt_template

prompt_text = render_prompt_template(
    template_path=Path("prompts/claude_strategy.md"),
    variables={
        "slug": "titanic",
        "metric": "accuracy",
        "target": "0.80",
    }
)
```

## Best Practices

### 1. Always Use Allowlists
Never run agents without allowlist enforcement:

```python
# ✅ GOOD: Allowlist + snapshot verification
allowlist = WriteAllowlist(base_dir=competition_dir)
allowlist.allow("kernel/**")
snapshot_pre = FileSnapshot.create(competition_dir)
run_codex(...)
snapshot_post = FileSnapshot.create(competition_dir)
violations = snapshot_pre.diff(snapshot_post, allowlist)
if violations:
    raise AllowlistViolationError(violations)

# ❌ BAD: No allowlist verification
run_codex(...)  # Agent can modify any file!
```

### 2. Verify Outputs
Always check that required outputs were created:

```python
# ✅ GOOD: Explicit verification
verify_outputs_exist(output_dir, ["kernel.py", "plan.json"])

# ❌ BAD: Assume files exist
kernel_path = output_dir / "kernel.py"
kernel_path.read_text()  # May raise FileNotFoundError
```

### 3. Use tmp_path in Tests
Never touch real artifacts in tests:

```python
# ✅ GOOD: Use tmp_path fixture
def test_feature(tmp_path: Path):
    competition_dir = tmp_path / "test-competition"
    ...

# ❌ BAD: Touch real artifacts
def test_feature():
    competition_dir = Path("artifacts/real-competition")  # DON'T!
    ...
```

### 4. Handle Delimiters Correctly
Claude output must use exact delimiter markers:

```python
# ✅ GOOD: Proper delimiter format
output = """
===CLAUDE_STRATEGY===
Strategy content here
===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Implementation steps here
===REFERENCES===
References here
"""
parsed = parse_claude_strategy_output(output)

# ❌ BAD: Missing or misspelled delimiters
output = """
=== STRATEGY ===  # Wrong! Missing "CLAUDE_" prefix and wrong spacing
...
"""
```

## Troubleshooting

### AllowlistViolationError
**Cause**: Agent modified files outside allowed paths

**Solutions**:
- Check allowlist patterns are correct (use `**` for recursive)
- Verify agent isn't writing to parent directories
- Review agent logs to identify unauthorized writes

### DelimiterParseError
**Cause**: Claude output missing required delimiter sections

**Solutions**:
- Check Claude prompt includes delimiter format examples
- Verify Claude's response wasn't truncated
- Review raw Claude output in `runs/<run_id>/02_claude_strategy/claude_exec.txt`

### AgentOutputError
**Cause**: Required output files weren't created

**Solutions**:
- Check agent error logs for execution failures
- Verify file paths in prompt templates are correct
- Review agent's final message for errors

## Implementation Checklist

When implementing a new agent workflow:

- [ ] Define allowlist patterns for writable files
- [ ] Create prompt template with variable placeholders
- [ ] Set up pre/post snapshots around agent execution
- [ ] Enforce allowlist violations (hard fail)
- [ ] Verify required outputs exist
- [ ] Write tests using `tmp_path` fixtures
- [ ] Document expected inputs/outputs
- [ ] Add error handling with appropriate exceptions
- [ ] Log agent prompts and responses for debugging
- [ ] Test with `dry_run=True` first

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project-level guidance
- [AGENTS.md](../AGENTS.md) - Agent-specific implementation notes
- [README.md](../README.md) - Getting started guide
