# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- 3rd-wear-dataset-challenge-hasca-2026 20260918T044235Z-869dbb11: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- biohub-cell-tracking-during-development 20260918T034618Z-f339cb2e: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- arc-prize-2026-arc-agi-3 20260918T025926Z-ae124719: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- arc-prize-2026-arc-agi-2 20260918T024243Z-339c9274: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- kaggriculture 20260918T022434Z-4799081d: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- playground-series-s6e8 20260831T061651Z-0e715f3e: reason=OracleStrategyError fingerprint=fd3dddf7ddd6d7fe392b
- cuhk-x-competition-small-model-track 20260831T061503Z-895db5ea: reason=OracleStrategyError fingerprint=d47dca38b1550fbcb1bd
- kaggriculture 20260831T060924Z-65329ad7: reason=OracleStrategyError fingerprint=c44f6fd1d93bc3519c5e
- autonomous-agent-prediction-beta 20260727T124942Z-d9b3525c: reason=OracleStrategyError fingerprint=a63a761b94b857bf966a
- playground-series-s6e7 20260727T121034Z-a02d37d5: reason=RuntimeError fingerprint=92217d94e261b01df814

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
