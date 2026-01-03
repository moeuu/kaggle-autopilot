# Claude Strategy Request

You are Claude Code. Use web search as needed. Follow Kaggle rules:
no external data unless explicitly allowed. Do not include secrets.

Competition: {{slug}}
Compute: {{compute}}
Accelerator: {{accelerator}}
Internet: {{internet}}

Brief (from Codex):
<<<
{{brief_content}}
>>>

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

Return output with exact delimiters:

===STRATEGY===
Provide a strong baseline plan and improvements. Include model choice, preprocessing, CV plan, and risks.

===CODEX_INSTRUCTIONS===
Give Codex step-by-step instructions to update only:
artifacts/<slug>/kernel/ (especially kernel.py). Mention any helper files to add under that dir.
