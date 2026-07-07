# {{implementation_agent_name}} Kernel Implementation

You are {{implementation_agent_name}}. Implement the kernel for this competition.

IMPORTANT:
- Default edit scope: {{kernel_dir}}
- Dependency exception: you may also edit repo-root `pyproject.toml` and `uv.lock` only when
  a required package is missing and must be added via `uv add <package>`.
- Primary entrypoint: {{kernel_path}}
- Do NOT modify any other files outside the allowed scope above.
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
(lightgbm, xgboost, catboost, torch, timm, torchvision, opencv, transformers, ultralytics,
tabicl, sklearn). Prefer the strongest available backend and avoid silent fallback to weak baselines
when these imports succeed.
Favor maximum score ceiling over guaranteed submitability: keep higher-capacity model paths alive unless
they are clearly invalid or dominated.
If a required package is genuinely missing, add it with `uv add <package>` instead of deleting the
entire model path.
If `KAGGLEBOT_DISABLE_XGBOOST=1`, force-disable XGBoost paths even if code previously enabled them.

Implementation contract for `kernel.py`:
- Top-level knobs:
  - `N_FOLDS`, `SEEDS`, `FAST_DEV`
  - plan-driven pipeline toggles (do not hardcode a single model family)
  - `GPU_DEVICE`
- Training and validation are mandatory for every iteration. Ignore or override plan/env settings
  that disable training, disable validation, request packaging-only/noop/identity fallback, or
  produce unscored/proxy/public-anchor metrics.
- Unified execution contract:
  - The exact same `kernel.py` must run on `local_gpu` and `kaggle_gpu`
  - Only execution location/runtime differs; algorithm path must be shared
- Robust I/O:
  - Read from `/kaggle/input/<competition_slug>/` (or local mirror safely)
  - Read `train`, `test`, and `sample_submission`
  - Resolve output dirs dynamically; write to `/kaggle/working` only when writable
  - Never print noisy permission warnings for expected unwritable paths during local runs
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
  - Save under local output dir always, and mirror to `/kaggle/working` only when writable
  - For any multi-fold or long-running candidate, save an intermediate candidate after each completed fold:
    `oof_preds_<name>_fold<N>.npy`, `test_preds_<name>_fold<N>.npy`,
    `preds_<name>_fold<N>_metadata.json`, `candidate_<name>_fold<N>.json`, and a fold-level
    `submission_<name>_fold<N>.<suffix>` using the final tabular submission suffix when inferable.
    The fold-level submission must be valid against the required submission format, and when public `sample_submission.*`
    is tiny/header-only/dummy for a hidden/full-test notebook rerun, it must expand to runtime test ids before
    writing. It must be usable if the remaining folds are stopped. Do not keep completed-fold predictions only in
    memory.
- Evaluation:
  - CV with deterministic seeds where feasible
  - Print per-pipeline CV summary for the plan primary metric
  - Use the same metric implementation/path for epoch model-selection and final offline scoring
    (do not optimize on a proxy metric that differs from final reported score)
  - `metrics.json` must describe a real trained candidate with a competition-faithful CV/holdout
    score from train data. Do not write placeholder, proxy, public-anchor, packaging-only, or
    unscored scores as the primary score.
  - When multiple candidates exist, log each candidate's CV score, holdout/validation score,
    and test/submission prediction distribution. Do not choose the final submission by CV alone
    if another candidate has materially better holdout/validation or the best-CV candidate has
    collapsed/suspicious predictions.
- Ensemble:
  - Implement only ensemble methods listed in plan.json
  - If plan.json contains 2 or more model pipelines, build at least one explicit blend candidate from OOF predictions
    (for example weighted mean, rank blend, or logit blend) unless the plan explicitly forbids ensembling
  - Choose final method by competition-faithful validation first, then plan primary metric
    (tie-breaker: simpler/faster)
  - Include at least one simple baseline evaluated on the same folds/windows; never choose a final
    pipeline that is worse than the baseline on the primary offline metric
- Submission:
  - Resolve the required output artifact from `KAGGLEBOT_SUBMISSION_FILENAME`, `sample_submission.*`,
    `submission_format.md`/`overview.md`, or `submission_manifest.json`; use CSV only when no other format is required
  - Write the supported submission artifact into a writable output dir
  - Mirror the chosen artifact to `/kaggle/working/<submission filename>` only when writable
  - For tabular outputs, validate columns against sample submission
  - For tabular outputs, validate row count and id order against runtime test ids when `sample_submission.*` is tiny/header-only/dummy
    for a hidden/full-test notebook rerun; never emit a 3-row public placeholder submission on Kaggle hidden/full test
  - Ensure no NaN/inf in predictions; clip to safe bounds when needed
  - If `KAGGLEBOT_LOCAL_KERNEL=1`, avoid hard-failing on `/kaggle/working` writes
- Optional model backends:
  - Use XGBoost/CatBoost only if import succeeds and runtime toggles allow it
  - If torch/timm/torchvision are importable, keep deep vision backbones enabled unless the plan
    explicitly disables them
  - If `KAGGLEBOT_DISABLE_LGBM_GPU=1`, force LightGBM to CPU (no GPU retry loops)
- Modality coverage:
  - Add dataset modality detection (tabular/image/video/text/audio/document/medical-imaging/point-cloud/3D/geospatial/bio/sequence/graph/signal/annotation/array/model-artifact/other)
  - Keep tabular path robust by default
  - For non-tabular tasks, provide/maintain `custom_main()` route in the same `kernel.py`
  - Preserve requested non-tabular artifact suffixes, sidecars, directory bundles, and `submission_manifest.json`
    instead of disguising tabular CSV bytes as the requested artifact
- Pretrained assets:
  - Provide a helper to download/cache pretrained checkpoints when useful
  - Respect internet/rules constraints and include fallback when download is unavailable
  - Optional manifest: `pretrained_models.json` (list of `{name,url}` objects) next to `kernel.py`

Required safety details:
- Fit-on-train, apply-to-test for ALL feature stats/encoders/bins.
- Never recompute train statistics on test.
- Handle train/test column mismatches safely.
- If plan-driven pipeline lookup is used (e.g. `get_pipeline_cfg`), missing pipeline names must NOT raise.
  Return a safe default config and continue.
- Handles categorical missing values safely:
  - If any column is pandas Categorical, add "Unknown" before fillna:
    `col = col.cat.add_categories(["Unknown"]).fillna("Unknown")`
  - Or cast to string/object before fillna.
  - Do not call `fillna("Unknown")` directly on categoricals (raises TypeError).
