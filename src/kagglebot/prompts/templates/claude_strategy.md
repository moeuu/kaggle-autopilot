# Claude Strategy Request

You are Claude Code. Use web search as needed to propose a strong solution plan.
Follow Kaggle rules: no external data unless explicitly allowed. No secrets.

Brief content (read this first):
{{brief_content}}

Brief JSON (structured):
{{brief_json_content}}

Context files (for reference):
- Brief (Markdown): {{brief_md}}
- Brief (JSON): {{brief_json}}
- Overview: {{overview_md}}
- Data description: {{data_md}}
- Rules: {{rules_md}}
- Dataset profile: {{dataset_profile}}
- Sample submission head: {{sample_submission_head}}
- Top1 public snapshot: {{top1_public}}

Compute:
- Mode: {{compute}}
- Accelerator: {{accelerator}}
- Internet: {{internet}}

Return output with these exact section markers:

===CLAUDE_STRATEGY===
Provide a deep solution strategy. Include model choices, preprocessing, CV plan, and risk notes.

===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Give Codex a step-by-step implementation plan with exact file paths to modify and acceptance criteria.
Include explicit instructions to update `artifacts/<slug>/plan.json` with:
target_metric, target_score, target_direction, score_source, holdout_frac, cv_folds, seed, and any other required defaults.

===REFERENCES===
List papers, repos, blog posts, and links used to justify the plan.
