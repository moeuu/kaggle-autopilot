# Playbook: resource_or_capacity

Recommended action: Add cheaper smoke tests and resource-aware model schedules before expensive runs.

## Signals
- museumscat-specimen-collection-annotation-task 20260803T191248Z-c9ddc8c9: gap=0.09738
- soccer-feature-engineering-hackathon 20260719T100737Z-031b686d: gap=None
- kaggriculture 20260831T060924Z-65329ad7: reason=OracleStrategyError fingerprint=c44f6fd1d93bc3519c5e
- autonomous-agent-prediction-beta 20260727T124942Z-d9b3525c: reason=OracleStrategyError fingerprint=a63a761b94b857bf966a

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
