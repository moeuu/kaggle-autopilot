# Playbook: submit_failed

Recommended action: Improve submit fallback diagnostics, notebook/file mode inference, and retry classification.

## Signals
- llm-classification-finetuning 20260613T190226Z-ede144f1: gap=0.1937899999999999
- maze-crawler 20260612T161419Z-06f97fbd: gap=1479.9
- arc-prize-2026-arc-agi-2 20260612T132534Z-2b1b9842: gap=None
- cohort-x-task-1 20260601T200941Z-e6678bc5: gap=None
- cohort-x-task-3 20260601T185407Z-db451a4d: gap=None
- neurogolf-2026 20260601T041044Z-20da4209: gap=2202.2299999999996
- handwritten-to-data 20260526T185134Z-1f9533e5: gap=0.35822999999999994
- orbit-wars 20260526T155707Z-9af0287b: gap=None
- nvidia-nemotron-model-reasoning-challenge 20260526T055122Z-8d55c17d: gap=0.33999999999999997
- arc-prize-2026-arc-agi-3 20260525T053929Z-900672fe: gap=None

## Next Experiment
- Pick one reusable orchestration, diagnostics, validation, or strategy-prompt improvement.
- If a local fix would only mask the issue, change the responsible architecture boundary instead.
- Add focused tests proving the behavior on synthetic artifacts.
