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
Learning is mandatory for every competition. Set `local_training_required=true` and `non_training_submission=null`; runtime cost must never switch the plan to a static diagnostic, frozen inference, packaging-only, or rule-only shortcut. If Kaggle does not bundle labeled training rows, build a competition-faithful learning corpus from rules-permitted public/reference data, the official practice evaluator, or leakage-safe generated training examples. Fit or update at least one parameterized component such as a proposer, reranker, calibrator, adapter, policy, or search distribution, validate the learned result against a fixed baseline, and emit `training_performed=true`. Prompt generation, deterministic compilation, and evaluation without an update step do not count as learning.

===CODEX_IMPLEMENTATION_INSTRUCTIONS===
Give {{implementation_agent_name}} a step-by-step implementation plan with exact file paths to modify and acceptance criteria.
Include explicit instructions to update `artifacts/<slug>/plan.json` with:
target_metric, target_score, target_direction, score_source, holdout_frac, cv_folds, seed, time_budget_min, hardware_profile, runtime_budget, non_training_submission (object or null), and any other required defaults.

===REFERENCES===
List papers, repos, blog posts, and links used to justify the plan.
