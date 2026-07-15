# Strategy Plan Request

You are Strategy Code. Use web search as needed to propose a strong solution plan.
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
- Sample submission preview (required format): {{sample_submission_head}}
- Top1 public snapshot: {{top1_public}}

Compute:
- Mode: {{compute}}
- Accelerator: {{accelerator}}
- Internet: {{internet}}
- Hardware profile: {{hardware_profile}}

Hardware execution budget:
{{hardware_constraints}}

Return output with these exact section markers:

===STRATEGY_PLAN===
Provide a deep solution strategy. Include model choices, preprocessing, CV plan, and risk notes.
Treat the sample submission preview as the required output format, not as proof that the artifact must be CSV.
For local_gpu, keep each iteration under about 24 hours when possible, but accuracy is the priority. For image/video/audio/text/document/medical-imaging/array/point-cloud/3D/geospatial/bio/sequence/graph/signal/annotation/model-artifact, avoid wasteful full seed x fold x model-family multiplication; keep the strongest feasible pretrained/OCR/VLM/geometric/geospatial/structure-feature path alive and scale it with smaller batches, chunking, quantization, cached embeddings, TTA, OOF blends, or lightweight heads before dropping it.
Make RTX3060-class execution accuracy-first rather than a hard cap. Expose plan.json/env scale knobs so a stronger GPU profile such as RTX5090 can increase batch size, folds/seeds, candidate count, or image size without rewriting kernel.py.
Do not assume model fitting is mandatory. When the explicit local-training estimate is at least 1440 minutes and a completed pretrained/reference/solver/search/simulation/optimization/rule-based path can generate the real hidden-test output, specify `local_training_required=false`, the numeric estimated local duration, and a concrete `non_training_submission` implementation and validation contract. A cost-class label alone is insufficient. Never use sample-submission copying, dummy predictions, or an unimplemented proposal for this route; keep all output and submission guards enabled.

===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Give {{implementation_agent_name}} a step-by-step implementation plan with exact file paths to modify and acceptance criteria.
Include explicit instructions to update `artifacts/<slug>/plan.json` with:
target_metric, target_score, target_direction, score_source, holdout_frac, cv_folds, seed, time_budget_min, hardware_profile, runtime_budget, non_training_submission (object or null), and any other required defaults.

===REFERENCES===
List papers, repos, blog posts, and links used to justify the plan.
