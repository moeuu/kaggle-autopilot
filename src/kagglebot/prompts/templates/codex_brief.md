# {{brief_agent_name}} Brief for Strategy

You are {{brief_agent_name}}. Read the context files and write a concise brief for Strategy.

Files to read:
- {{overview_md}}
- {{data_md}}
- {{rules_md}}
- {{dataset_profile}}
- {{sample_submission_head}}
- {{top1_public}}
- {{rules_url}}

Task:
- Extract key facts: task type, evaluation metric, target column guess, id column, submission format.
- Summarize dataset size, missingness, categorical/high-cardinality notes.
- List constraints and rule reminders.
- Note any uncertainty explicitly.

IMPORTANT:
- Do NOT modify source code or plan.json.
- Only write the outputs below.

Outputs (must write both):
- {{brief_md}}
- {{brief_json}}

The JSON must include keys:
slug, task_type, metric, target_column, id_column, submission_columns,
score_source_guess, constraints, top1_public_score, notes
