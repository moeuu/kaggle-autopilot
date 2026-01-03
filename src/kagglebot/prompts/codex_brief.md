# Codex Brief Extraction

You are Codex. Produce a concise brief for Claude from the context below.
Do not modify any files or code. Output only the brief text.

Competition: {{slug}}
URL: {{competition_url}}

Rules URL:
{{rules_url}}

Overview (content):
<<<
{{overview_content}}
>>>

Data description (content):
<<<
{{data_content}}
>>>

Rules (content):
<<<
{{rules_content}}
>>>

Dataset profile (JSON):
<<<
{{dataset_profile}}
>>>

Sample submission (CSV):
<<<
{{sample_submission}}
>>>

Brief requirements:
- Task type and likely metric
- Target column guess and ID column (if any)
- Submission format
- Data size, missingness, categorical/high-cardinality notes
- Constraints or rules that impact modeling
- Any uncertainties
