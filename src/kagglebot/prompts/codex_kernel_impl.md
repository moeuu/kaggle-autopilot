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

Ensure kernel.py:
- Reads data from /kaggle/input/<competition_slug>/
- Writes /kaggle/working/submission.csv
- Writes /kaggle/working/metrics.json with offline split metric
