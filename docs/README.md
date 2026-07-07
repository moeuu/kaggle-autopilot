# Kagglebot Documentation

Primary docs:

- `../README.md`: quick start and core commands
- `autopilot.md`: end-to-end autopilot usage
- `spec_autopilot.md`: current implementation specification
- `architecture.md`: control flow and safety gates
- `guardrails_checklist.md`: safety/correctness checklist
- `AUTOPILOT_SINGLE_SUBMIT.md`: submission-loop workflow
- `knowledge.md`: knowledge base design and usage
- `taxonomy.md`: tag taxonomy reference
- `safety/submission_checklist.md`: pre-submission checks
- `safety/failure_modes.md`: known failure modes and mitigations

Notes:

- Legacy planning/task docs were removed to reduce maintenance overhead.
- The files above are the supported documentation set.
- Planning pipeline is `codex -> oracle(gpt-5.5-pro) -> codex` when Oracle is available and persists research artifacts to `knowledge/research/<problem_type>/<slug>/` (with working copies under `artifacts/<slug>/context/`).
