# Playbook: orchestration_runtime_failure

Recommended action: Classify supervisor/runtime errors centrally and add reusable recovery instead of per-competition patches.

## Signals
- museumscat-specimen-collection-annotation-task 20260803T191248Z-c9ddc8c9: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- autonomous-agent-prediction-beta 20260803T180718Z-e82efebb: reason=training_data_detection_error fingerprint=d6ca998851993ecbdd93
- cuhk-x-competition-small-model-track 20260719T030527Z-69497fe0: reason=KaggleCliError fingerprint=5ec60cecf216ae30aaf0
- soccer-feature-engineering-hackathon 20260718T074715Z-870c671b: reason=KernelFailedError fingerprint=d171de413b1f36180141

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
