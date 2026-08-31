# Playbook: orchestration_preflight_failure

Recommended action: Promote pre-run discovery/profile failures into typed autofix incidents with regression tests.

## Signals
- None None: reason=KaggleCliError fingerprint=cba6662e9b2e00191cee
- museumscat-specimen-collection-annotation-task 20260727T132941Z-e6928de5: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- playground-series-s6e7 20260727T121034Z-a02d37d5: reason=RuntimeError fingerprint=92217d94e261b01df814
- None None: reason=KaggleCliError fingerprint=cba6662e9b2e00191cee
- None None: reason=KaggleCliError fingerprint=cba6662e9b2e00191cee
- autonomous-agent-prediction-beta 20260719T171322Z-21ec36c9: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- playground-series-s6e7 20260719T155759Z-ddb4b9f1: reason=RuntimeError fingerprint=92217d94e261b01df814
- scripture-in-new-frontiers 20260718T095023Z-c876dbbc: reason=RuntimeError fingerprint=a7804a994fcb95dc5ce2

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
