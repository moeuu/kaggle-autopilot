# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- playground-series-s6e7 20260719T155759Z-ddb4b9f1: reason=RuntimeError fingerprint=92217d94e261b01df814
- cuhk-x-competition-large-model-track 20260719T144303Z-1f8ef9d3: reason=RuntimeError fingerprint=92217d94e261b01df814
- soccer-feature-engineering-hackathon 20260719T100737Z-031b686d: reason=RuntimeError fingerprint=92217d94e261b01df814
- scripture-in-new-frontiers 20260718T095023Z-c876dbbc: reason=RuntimeError fingerprint=a7804a994fcb95dc5ce2
- ai-agent-security-multi-step-tool-attacks 20260717T061040Z-4aad47f3: reason=RuntimeError fingerprint=92217d94e261b01df814
- cuhk-x-competition-large-model-track 20260716T025336Z-83f794ce: reason=OracleStrategyError fingerprint=65df22ce55cf6bb51d0e
- cuhk-x-competition-small-model-track 20260716T024240Z-edb451b6: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- scripture-in-new-frontiers 20260716T010915Z-7b12de3d: reason=OracleStrategyError fingerprint=4a943231f253606b92a7
- skill-lift 20260716T001150Z-1038a59e: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f
- soccer-feature-engineering-hackathon 20260715T235055Z-2864ef78: reason=OracleStrategyError fingerprint=e8f53ed69d7cb5e4140f

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
