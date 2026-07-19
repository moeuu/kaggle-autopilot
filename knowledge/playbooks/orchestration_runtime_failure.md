# Playbook: orchestration_runtime_failure

Recommended action: Classify supervisor/runtime errors centrally and add reusable recovery instead of per-competition patches.

## Signals
- cuhk-x-competition-small-model-track 20260719T030527Z-69497fe0: reason=KaggleCliError fingerprint=5ec60cecf216ae30aaf0
- soccer-feature-engineering-hackathon 20260718T074715Z-870c671b: reason=KernelFailedError fingerprint=d171de413b1f36180141
- biohub-cell-tracking-during-development 20260707T063753Z-169cfcdf: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
