# Codex Kernel Implementation

You are Codex. Implement the kernel for this competition.

IMPORTANT:
- Modify ONLY files under: {{kernel_dir}}
- Primary entrypoint: {{kernel_path}}
- Do NOT modify any files outside the kernel directory.
- Do NOT access secrets or Kaggle credentials.
- Prefer strong, high-capacity models over baselines. Use GPU/TPU acceleration when available.
- Expose training-intensity knobs (epochs/iterations/model size) near the top of kernel.py.

Instructions from Claude:
<<<
{{instructions}}
>>>

Strategy from Claude:
<<<
{{strategy}}
>>>

Blocked modules (do NOT import; not available on Kaggle runtime):
<<<
{{blocked_modules}}
>>>
If a blocked module appears in previous code, remove it and replace with Kaggle-default libraries
(lightgbm, xgboost, catboost, torch, transformers, sklearn). If unsure, prefer these defaults.

Ensure kernel.py:
- Reads data from /kaggle/input/<competition_slug>/
- Writes /kaggle/working/submission.csv
- Writes /kaggle/working/metrics.json with offline split metric
- Logs whether GPU/TPU is actually used (device + training time)
 - Handles categorical missing values safely:
   - If any column is pandas Categorical, add "Unknown" before fillna:
     `col = col.cat.add_categories(["Unknown"]).fillna("Unknown")`
   - Or cast to string/object before fillna.
   - Do not call `fillna("Unknown")` directly on categoricals (raises TypeError).

Embedding/model safety:
- If you use an encoder-decoder model (e.g. ProtT5/T5), load the encoder-only variant:
  - Prefer `T5EncoderModel.from_pretrained(...)`, or `AutoModel(...).get_encoder()`
  - Call with `input_ids`/`attention_mask` only (no decoder_input_ids)
  - This avoids `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`
