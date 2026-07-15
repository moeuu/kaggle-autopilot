# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- ai-agent-security-multi-step-tool-attacks 20260715T160516Z-7045e102: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: reason=submit_aborted fingerprint=3be5105ab5a5014fa8f4
- arc-prize-2026-arc-agi-2 20260629T151254Z-4a6d2fa8: reason=RuntimeError fingerprint=80636502f0494ff1b437
- arc-prize-2026-arc-agi-2 20260612T132534Z-2b1b9842: reason=RuntimeError fingerprint=92217d94e261b01df814

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
