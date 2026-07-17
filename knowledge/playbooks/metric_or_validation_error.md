# Playbook: metric_or_validation_error

Recommended action: Tighten metric contract validation and fail earlier when scoring is untrusted.

## Signals
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: gap=None
- cohort-x-task-1 20260629T021700Z-c2183268: gap=0.29333
- filament-segmentation-2026 20260716T032330Z-5b754aa5: reason=submit_aborted fingerprint=e69185e5bed70e4bac2c
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: reason=submit_aborted fingerprint=3be5105ab5a5014fa8f4
- arc-prize-2026-arc-agi-2 20260714T155630Z-4d48ff70: reason=ValueError fingerprint=5c43de77f69e42d474ab

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
