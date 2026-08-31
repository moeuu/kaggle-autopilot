# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- cuhk-x-competition-small-model-track 20260831T061503Z-895db5ea: reason=OracleStrategyError fingerprint=d47dca38b1550fbcb1bd
- kaggriculture 20260831T060924Z-65329ad7: reason=OracleStrategyError fingerprint=c44f6fd1d93bc3519c5e
- autonomous-agent-prediction-beta 20260727T124942Z-d9b3525c: reason=OracleStrategyError fingerprint=a63a761b94b857bf966a
- playground-series-s6e7 20260727T121034Z-a02d37d5: reason=RuntimeError fingerprint=92217d94e261b01df814
- cuhk-x-competition-small-model-track 20260727T005939Z-5205009c: reason=KaggleBotError fingerprint=e3780906cedd3cd49430
- playground-series-s6e7 20260719T155759Z-ddb4b9f1: reason=RuntimeError fingerprint=92217d94e261b01df814
- cuhk-x-competition-large-model-track 20260719T144303Z-1f8ef9d3: reason=RuntimeError fingerprint=92217d94e261b01df814
- soccer-feature-engineering-hackathon 20260719T100737Z-031b686d: reason=RuntimeError fingerprint=92217d94e261b01df814
- scripture-in-new-frontiers 20260718T095023Z-c876dbbc: reason=RuntimeError fingerprint=a7804a994fcb95dc5ce2
- ai-agent-security-multi-step-tool-attacks 20260717T061040Z-4aad47f3: reason=RuntimeError fingerprint=92217d94e261b01df814

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
