# Codex Kernel Implementation

You are Codex. Implement the kernel for this competition.

IMPORTANT:
- Modify ONLY files under: {{kernel_dir}}
- Primary entrypoint: {{kernel_path}}
- Do NOT modify any files outside the kernel directory.
- Do NOT access secrets or Kaggle credentials.
- Do NOT perform open-ended web search in this phase.
- Follow the frozen plan in `{{plan_path}}` and research summary in `{{research_summary_path}}`.

This stage is Prompt 3/5 -> Prompt 4/5 -> Prompt 5/5 only.
You must execute the shortlisted pipelines exactly as frozen in plan.json.
If a dependency is unavailable in Kaggle runtime, use the fallback defined in plan.json.

Instructions from Strategy:
<<<
{{instructions}}
>>>

Strategy from Strategy:
<<<
{{strategy}}
>>>

Blocked modules (do NOT import; not available on Kaggle runtime):
<<<
{{blocked_modules}}
>>>
If a blocked module appears in previous code, remove it and replace with Kaggle-default libraries
(lightgbm, xgboost, catboost, torch, transformers, sklearn). If unsure, prefer these defaults.

Implementation contract for `kernel.py`:
- Top-level knobs:
  - `N_FOLDS`, `SEEDS`, `FAST_DEV`
  - plan-driven pipeline toggles (do not hardcode a single model family)
  - `GPU_DEVICE`
- Unified execution contract:
  - The exact same `kernel.py` must run on `local_gpu` and `kaggle_gpu`
  - Only execution location/runtime differs; algorithm path must be shared
- Robust I/O:
  - Read from `/kaggle/input/<competition_slug>/` (or local mirror safely)
  - Read `train`, `test`, and `sample_submission`
- Strict target handling:
  - Infer/match target robustly and assert target column validity
- Feature alignment helper:
  - Implement `align_features(train_df, test_df, feature_cols)`
  - Add missing cols, drop extras, reorder consistently
- Leak-safe encoders:
  - Fit on train fold only; apply to validation/test
  - Support only the encoders/transforms selected in plan.json
- Per-pipeline artifacts:
  - Save `oof_preds_<name>.npy` and `test_preds_<name>.npy`
  - Save under `/kaggle/working` and local output dir when available
- Evaluation:
  - CV with deterministic seeds where feasible
  - Print per-pipeline CV summary for the plan primary metric
- Ensemble:
  - Implement only ensemble methods listed in plan.json
  - Choose final method by plan primary metric (tie-breaker: simpler/faster)
- Submission:
  - Write `/kaggle/working/submission.csv`
  - Validate columns and row count against sample submission
  - Ensure no NaN/inf in predictions; clip to safe bounds when needed
- Modality coverage:
  - Add dataset modality detection (tabular/image/video/text/audio/other)
  - Keep tabular path robust by default
  - For non-tabular tasks, provide/maintain `custom_main()` route in the same `kernel.py`
- Pretrained assets:
  - Provide a helper to download/cache pretrained checkpoints when useful
  - Respect internet/rules constraints and include fallback when download is unavailable
  - Optional manifest: `pretrained_models.json` (list of `{name,url}` objects) next to `kernel.py`

Required safety details:
- Fit-on-train, apply-to-test for ALL feature stats/encoders/bins.
- Never recompute train statistics on test.
- Handle train/test column mismatches safely.
- Handles categorical missing values safely:
  - If any column is pandas Categorical, add "Unknown" before fillna:
    `col = col.cat.add_categories(["Unknown"]).fillna("Unknown")`
  - Or cast to string/object before fillna.
  - Do not call `fillna("Unknown")` directly on categoricals (raises TypeError).
