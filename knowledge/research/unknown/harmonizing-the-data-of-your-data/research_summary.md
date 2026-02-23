# research_summary.md

## Ranked candidate pipelines (2–4)

### 1) **LLM-RAG Schema Filler (Recommended)**
- **Leak-free features/encodings:** TF-IDF (char+word ngrams) retrieval over train PubText; per-column value vocab from train SDRFs only; optional rapidfuzz substring hits over manuscript/raw names.
- **Models + key hyperparameters:** Local instruct LLM (7B-class) in 4-bit; `temperature=0.0`, `max_new_tokens=800–1400`, 1–3 self-consistency passes; top-k retrieval `k=3–5`; per-column candidate cap `<=10`.
- **Runtime/memory:** Retrieval seconds; LLM dominates (15 PXDs). Fits 16GB GPU with 4-bit; CPU fallback slower.
- **Leakage risk:** Low if retrieval/vocab built from train only during CV; no test labels used.
- **Fallback if dependency unavailable:** Drop LLM; output retrieval-only candidates with conservative thresholds.

### 2) **Candidate Retrieval + LLM Verifier (Precision-first)**
- **Leak-free features/encodings:** Same retrieval/vocab; for each column propose candidates (top frequency + fuzzy match) then ask LLM to select supported ones with evidence.
- **Models + key hyperparameters:** Smaller LLM or same LLM with short prompts; `temperature=0`, strict “choose-from-list-only”.
- **Runtime/memory:** Faster than full generation; smaller context.
- **Leakage risk:** Low.
- **Fallback:** Deterministic selection by score thresholds.

### 3) **Retrieval-Only Lexical/Semantic Baseline**
- **Leak-free features/encodings:** Column-wise vocab + TF-IDF similarity of value strings to document windows; rapidfuzz partial_ratio thresholds; optionally sentence-transformer embeddings if available.
- **Models + key hyperparameters:** No generative model; thresholds per column; top-N values per column (often 0–3).
- **Runtime/memory:** Very fast; low memory.
- **Leakage risk:** Low.
- **Fallback:** Pure frequency priors from train SDRFs.

Notes: For set-based scoring, distribute predicted sets across rows per PXD and avoid mixing `"Not Applicable"` with real values in the same (PXD, column).
