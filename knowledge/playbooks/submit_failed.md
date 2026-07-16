# Playbook: submit_failed

Recommended action: Improve submit fallback diagnostics, notebook/file mode inference, and retry classification.

## Signals
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: gap=None
- arc-prize-2026-arc-agi-3 20260714T060514Z-134b7f94: gap=None
- cohort-x-task-1 20260629T021700Z-c2183268: gap=0.29333
- biohub-cell-tracking-during-development 20260714T160327Z-60b94f2d: reason=submit_aborted fingerprint=3be5105ab5a5014fa8f4
- arc-prize-2026-arc-agi-3 20260714T060514Z-134b7f94: reason=submit_aborted fingerprint=05752cc26d065e7d7ebf
- ai-agent-security-multi-step-tool-attacks 20260628T172512Z-e4b70727: reason=submit_aborted fingerprint=3f0a0e8be0c3e5adc3e5
- arc-prize-2026-arc-agi-2 20260627T040552Z-1c7b2617: reason=submit_aborted fingerprint=05752cc26d065e7d7ebf
- arc-prize-2026-arc-agi-3 20260627T010924Z-72d6b99d: reason=submit_aborted fingerprint=05752cc26d065e7d7ebf
- ai-agent-security-multi-step-tool-attacks 20260626T124420Z-b7d4597a: reason=submit_aborted fingerprint=3f0a0e8be0c3e5adc3e5

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
