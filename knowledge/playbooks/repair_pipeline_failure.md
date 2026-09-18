# Playbook: repair_pipeline_failure

Recommended action: Harden Oracle/Codex verification and source reload so verified fixes reach the active process.

## Signals
- spaceship-titanic 20260918T053312Z-aca28cd1: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- filament-segmentation-2026 20260918T052020Z-412fefee: reason=OracleStrategyError fingerprint=7b5c76fafe9d12ec172e
- umud-challenge-muscle-architecture-in-ultrasound-data 20260918T050142Z-0cb422c8: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- 3rd-wear-dataset-challenge-hasca-2026 20260918T044235Z-869dbb11: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- biohub-cell-tracking-during-development 20260918T034618Z-f339cb2e: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- arc-prize-2026-arc-agi-3 20260918T025926Z-ae124719: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- arc-prize-2026-arc-agi-2 20260918T024243Z-339c9274: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- kaggriculture 20260918T022434Z-4799081d: reason=OracleStrategyError fingerprint=97ba42b8116635bddbb7
- playground-series-s6e8 20260831T061651Z-0e715f3e: reason=OracleStrategyError fingerprint=fd3dddf7ddd6d7fe392b
- cuhk-x-competition-small-model-track 20260831T061503Z-895db5ea: reason=OracleStrategyError fingerprint=d47dca38b1550fbcb1bd

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
