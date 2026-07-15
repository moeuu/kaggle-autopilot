# Playbook: orchestration_preflight_failure

Recommended action: Promote pre-run discovery/profile failures into typed autofix incidents with regression tests.

## Signals
- arc-prize-2026-arc-agi-2 20260714T155630Z-4d48ff70: reason=ValueError fingerprint=5c43de77f69e42d474ab
- None None: reason=KaggleCliError fingerprint=8c19465d6ca635817391
- ai-agent-security-multi-step-tool-attacks 20260624T165941Z-2c71599e: reason=stale_watch_state fingerprint=e44bf52e9df39debfccc
- 5-day-ai-agents-intensive-vibecoding-course-with-google 20260612T160909Z-38caeb78: reason=KaggleCliError fingerprint=4747c91c6a2518655dee
- arc-prize-2026-arc-agi-2 20260610T102231Z-573a2dad: reason=TypeError fingerprint=f44916ad92425df07a28
- arc-prize-2026-arc-agi-2 20260610T102223Z-5017de1c: reason=TypeError fingerprint=f44916ad92425df07a28

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
