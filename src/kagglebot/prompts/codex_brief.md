# Codex Brief Extraction

You are Codex. Produce a concise, human-readable brief for Claude from local context files.
Do not modify any files or code. Output only the brief text.
Summarize; do not paste raw data, long quotes, or large tables.
Focus on: task/metric, submission format, data schema & file types, and rules.

Competition: {{slug}}
URL: {{competition_url}}

Rules URL:
{{rules_url}}

Read and summarize these local files (paths). Do not paste raw content:
- Overview: {{overview_path}}
- Data description: {{data_path}}
- Rules: {{rules_path}}
- Dataset profile (JSON): {{dataset_profile_path}}
- Submission format: {{submission_format_path}}
- Sample submission head: {{sample_submission_head_path}}
- Sample submission file (full, if needed): {{sample_submission_path}}

If a file is missing or empty, note it explicitly.

Brief requirements (use short headings):
- Task type and evaluation metric (with confidence)
- Target column guess and ID column (if any)
- Submission format (columns, delimiter, row expectations)
- Data structure (file types, keys, sizes, missingness, categorical/high-cardinality notes)
- Constraints or rules that impact modeling (no external data, etc.)
- Any uncertainties or missing info
