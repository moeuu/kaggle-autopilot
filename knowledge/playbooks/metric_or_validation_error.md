# Playbook: metric_or_validation_error

Recommended action: Tighten metric contract validation and fail earlier when scoring is untrusted.

## Signals
- playground-series-s6e7 20260731T170652Z-e6cedcfd: gap=0.0036599999999999966
- scripture-in-new-frontiers 20260729T081158Z-399ad2dd: gap=None
- soccer-feature-engineering-hackathon 20260719T100737Z-031b686d: gap=None
- autonomous-agent-prediction-beta 20260727T124942Z-d9b3525c: reason=OracleStrategyError fingerprint=a63a761b94b857bf966a
- cuhk-x-competition-small-model-track 20260727T005939Z-5205009c: reason=KaggleBotError fingerprint=e3780906cedd3cd49430
- filament-segmentation-2026 20260716T032330Z-5b754aa5: reason=submit_aborted fingerprint=e69185e5bed70e4bac2c

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
