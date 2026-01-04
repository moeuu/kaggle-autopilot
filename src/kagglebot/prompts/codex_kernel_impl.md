# Codex Kernel Implementation

You are Codex. Implement the kernel for this competition.

IMPORTANT:
- Modify ONLY files under: {{kernel_dir}}
- Primary entrypoint: {{kernel_path}}
- Do NOT modify any files outside the kernel directory.
- Do NOT access secrets or Kaggle credentials.

Instructions from Claude:
<<<
{{instructions}}
>>>

Strategy from Claude:
<<<
{{strategy}}
>>>

Ensure kernel.py:
- Reads data from /kaggle/input/<competition_slug>/
- Writes /kaggle/working/submission.csv
- Writes /kaggle/working/metrics.json with offline split metric

Embedding/model safety:
- If you use an encoder-decoder model (e.g. ProtT5/T5), load the encoder-only variant:
  - Prefer `T5EncoderModel.from_pretrained(...)`, or `AutoModel(...).get_encoder()`
  - Call with `input_ids`/`attention_mask` only (no decoder_input_ids)
  - This avoids `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`
