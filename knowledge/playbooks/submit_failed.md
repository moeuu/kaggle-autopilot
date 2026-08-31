# Playbook: submit_failed

Recommended action: Improve submit fallback diagnostics, notebook/file mode inference, and retry classification.

## Signals
- museumscat-specimen-collection-annotation-task 20260803T191248Z-c9ddc8c9: gap=0.09738
- cuhk-x-competition-large-model-track 20260727T053217Z-5763c450: gap=0.53508
- cuhk-x-competition-small-model-track 20260831T061503Z-895db5ea: reason=OracleStrategyError fingerprint=d47dca38b1550fbcb1bd
- kaggriculture 20260831T060924Z-65329ad7: reason=OracleStrategyError fingerprint=c44f6fd1d93bc3519c5e
- cuhk-x-competition-small-model-track 20260803T170033Z-e983c0ce: reason=missing_competition_data fingerprint=a076e59a55142aa47dab
- autonomous-agent-prediction-beta 20260727T124942Z-d9b3525c: reason=OracleStrategyError fingerprint=a63a761b94b857bf966a
- cuhk-x-competition-small-model-track 20260727T005939Z-5205009c: reason=KaggleBotError fingerprint=e3780906cedd3cd49430
- scripture-in-new-frontiers 20260718T095023Z-c876dbbc: reason=RuntimeError fingerprint=a7804a994fcb95dc5ce2

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
