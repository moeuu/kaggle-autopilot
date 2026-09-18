# Playbook: orchestration_runtime_failure

Recommended action: Classify supervisor/runtime errors centrally and add reusable recovery instead of per-competition patches.

## Signals
- arc-prize-2026-arc-agi-2 20260831T062244Z-9623e205: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- museumscat-specimen-collection-annotation-task 20260803T191248Z-c9ddc8c9: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- autonomous-agent-prediction-beta 20260803T180718Z-e82efebb: reason=training_data_detection_error fingerprint=d6ca998851993ecbdd93

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
