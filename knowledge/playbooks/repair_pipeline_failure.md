# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- scripture-in-new-frontiers 20260716T010915Z-7b12de3d: reason=OracleStrategyError fingerprint=4a943231f253606b92a7
- skill-lift 20260716T001150Z-1038a59e: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- soccer-feature-engineering-hackathon 20260715T235055Z-2864ef78: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- cohort-x-task-2 20260715T211756Z-824e1427: reason=OracleStrategyError fingerprint=56655e6b2c92738ada85
- arc-prize-2026-arc-agi-2 20260715T191406Z-29a80a2b: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- ai-agent-security-multi-step-tool-attacks 20260715T160516Z-7045e102: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: reason=submit_aborted fingerprint=3be5105ab5a5014fa8f4
- arc-prize-2026-arc-agi-2 20260629T151254Z-4a6d2fa8: reason=RuntimeError fingerprint=80636502f0494ff1b437
- arc-prize-2026-arc-agi-2 20260612T132534Z-2b1b9842: reason=RuntimeError fingerprint=92217d94e261b01df814

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
