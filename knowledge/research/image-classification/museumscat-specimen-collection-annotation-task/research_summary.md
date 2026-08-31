## Ranked shortlist

### 1. Qwen3-VL-8B QLoRA + OCR candidates + OOF risk reranking (recommended final)

**Leak-free features/encodings:** fold-held-out Qwen candidates; frozen OCR outputs; pHash/layout features; candidate consensus NED; token log-probability, entropy, ratio/tail features; crop quality; format/confounder flags; and char 3–5 gram target-similarity fit only on each fold’s training targets. **Models/hyperparameters:** Qwen3-VL-8B-Instruct, NF4 4-bit, LoRA rank 16/alpha 32/dropout 0.05, four epochs, LR `8e-5`, batch 1, grad accumulation 16, 786,432 pixels; CatBoost NED regressors depth 4, 400 iterations, LR 0.03; isotonic/rank confidence; at most five candidates and 900 high-risk reruns. **Runtime/memory:** about 1,050–1,260 minutes end-to-end, peak 11–11.8GB. **Leakage risk:** medium if candidates or target vocabularies are not cross-fitted; enforce fold provenance. **Fallback:** Qwen3-VL-4B, then required Qwen2.5-VL-7B; HistGradientBoosting if CatBoost is absent; GLM-OCR if HunyuanOCR is unavailable.

### 2. Qwen3-VL-8B multi-view standalone

**Leak-free features/encodings:** unlabeled label geometry, full/union/contact-sheet views, deterministic prompt variants, no target-derived postprocessing except fold-fit format diagnostics. **Models/hyperparameters:** same 4-bit QLoRA adapter, combined JSON output for all images and field-specific prompt only on uncertain validation/test examples. **Runtime/memory:** roughly 700–950 minutes, 11GB peak. **Leakage risk:** low, but validation variance is high with 200 labels. **Fallback:** lower max pixels and rank-8 LoRA before moving to Qwen3-VL-4B.

### 3. Required Qwen2.5-VL-7B collector-filtered baseline

**Leak-free features/encodings:** reference crop/contact-sheet logic, target-only system prompt, deterministic parsing, forbidden-token rules, and OOF confidence fitted from its own validation errors. **Models/hyperparameters:** Qwen2.5-VL-7B 4-bit, batch 1, greedy decoding, two views. **Runtime/memory:** 250–450 minutes, about 9–11GB. **Leakage risk:** low. **Fallback:** Qwen2.5-VL-3B or frozen inference plus a learned confidence head. This pipeline is mandatory for comparison but is not the accuracy target.

### 4. OCR-specialist candidate system

**Leak-free features/encodings:** HunyuanOCR/GLM-OCR text from rectified labels, detector confidence, connected-component geometry, OCR agreement, date regex plausibility, coordinate patterns, and fold-fit candidate selection. **Models/hyperparameters:** HunyuanOCR-1.5 native Transformers, sequential crop inference; GLM-OCR 0.9B fallback; EasyOCR only for boxes. **Runtime/memory:** 180–360 minutes, typically below 10GB when run sequentially. **Leakage risk:** low unless train target strings are used for unrestricted correction. **Fallback:** OpenCV + EasyOCR detection and Qwen-only transcription.

The recommended promotion order is: first verify the exact metric and mandatory baseline; then establish Qwen3 OOF; then add OCR candidate diversity; finally add the learned risk ordering and selective second pass. Qwen3-VL’s official OCR/multi-image capabilities, HunyuanOCR’s OCR specialization, and generation-confidence research support this order. ([GitHub][2])
