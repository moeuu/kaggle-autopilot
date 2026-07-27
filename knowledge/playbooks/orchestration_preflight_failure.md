# Playbook: orchestration_preflight_failure

Recommended action: Promote pre-run discovery/profile failures into typed autofix incidents with regression tests.

## Signals
- None None: reason=KaggleCliError fingerprint=cba6662e9b2e00191cee
- None None: reason=KaggleCliError fingerprint=cba6662e9b2e00191cee
- autonomous-agent-prediction-beta 20260719T171322Z-21ec36c9: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
