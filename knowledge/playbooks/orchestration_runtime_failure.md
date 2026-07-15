# Playbook: orchestration_runtime_failure

Recommended action: Classify supervisor/runtime errors centrally and add reusable recovery instead of per-competition patches.

## Signals
- biohub-cell-tracking-during-development 20260707T063753Z-169cfcdf: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- amia-public-challenge-2026 20260610T103310Z-965204a0: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- cohort-x-task-2 20260602T031415Z-7cec3334: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
