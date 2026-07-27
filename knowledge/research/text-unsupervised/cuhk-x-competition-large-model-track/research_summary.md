## Ranked shortlist

### 1. Category-aware multimodal ensemble — recommended final

**Leak-free features/encodings:** Cross-fitted Qwen candidate logits, VideoMAE full-clip and quarter-window logits, modality-presence flags, deterministic motion/clip metadata, and fold-fit option-frequency, answer-cardinality, co-occurrence, and precedence priors. Calibration temperatures and blend weights are learned only from other folds’ OOF predictions. Candidate text is encoded independently; no test distribution or sample-submission prediction values are fit.

**Models and concrete configuration:** Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B QLoRA (rank 32, alpha 32, dropout 0.05, two epochs, learning rate 2e-5, six hard negatives); VideoMAE base (16×224 frames, last two blocks unfrozen, four epochs, backbone/head learning rates 1e-5/3e-4); Qwen3-VL-8B-Instruct NF4 hard-case rerank (eight 448-pixel composites, top eight legal candidates, deterministic generation). Cross-fitted blend step is 0.05 with component weights constrained to 0.10–0.80.

**Expected runtime/memory:** About 1,260 minutes end to end on one RTX3060; peak 11.8GB VRAM and about 14GB host RAM. The VLM is loaded sequentially after releasing training models.

**Leakage risk:** Medium if subject/clip grouping or cross-fitted blending is implemented incorrectly; otherwise low. The same clip can have multiple questions, so grouping by QA row is forbidden.

**Fallback:** Reduce candidate count, frames, image size, and TTA before substituting Qwen3-VL-4B for 8B. If the Qwen embedding repository is unavailable, use 4B hidden-state/candidate likelihood scores with the same exact decoder. This hybrid is best supported by CUHK-X evidence that non-RGB transfer and cross-subject shift are difficult while reasoning helps HARn. ([arXiv][5])

### 2. Qwen3-VL 2B candidate matcher/reranker — strongest single branch

**Leak-free features/encodings:** Legal candidate enumeration per category; synchronized modality frames; question and option text; source/category/option-count flags; fold-fit priors. For multi, enumerate all nonempty subsets. For sequence, enumerate valid permutations and score the complete ordered candidate, avoiding post-hoc letter heuristics.

**Models and concrete configuration:** Official Qwen3-VL-Embedding-2B for cached video/question and candidate representations, followed by Qwen3-VL-Reranker-2B QLoRA. Use 12 frames at 224 pixels, NF4, batch one, accumulation 16, two epochs, maximum 24 legal candidates and top-eight cross-encoder reranking. The official source documents video input, 2B variants, 2,048-dimensional embeddings, and rank/alpha 32 LoRA. ([GitHub][2])

**Expected runtime/memory:** Roughly 720 minutes, peak 11.5GB VRAM, 12GB host RAM with cached media and candidate embeddings.

**Leakage risk:** Low when hard negatives and priors are created inside each fold. Risk rises if the action vocabulary or calibration is built from all train rows before CV.

**Fallback:** Freeze the Qwen backbone and train a small listwise MLP; if PEFT/bitsandbytes is absent, use the Qwen3-VL-4B-Instruct local likelihood scorer. No external API is needed.

### 3. VideoMAE structured temporal branch — diversity and sequence specialist

**Leak-free features/encodings:** Two deterministic pseudo-RGB views from depth/IR/motion and Depth_Color/Thermal/motion; full-clip embedding; four quarter-window embeddings; candidate text embeddings; train-fold-only cardinality, co-occurrence, and precedence statistics. Global normalization is fit on the fold train clips; per-clip robust normalization is deterministic.

**Models and concrete configuration:** `MCG-NJU/videomae-base-finetuned-kinetics`, 16 frames at 224, batch two, accumulation eight, four epochs, last two transformer blocks unfrozen, learning rates 1e-5 and 3e-4, weight decay 0.05, label smoothing 0.05. A listwise candidate head handles all categories; a quarter-window alignment term supplies explicit order evidence for sequence questions. VideoMAE’s released base model uses 16-frame processing and supports downstream feature extraction. ([Hugging Face][3])

**Expected runtime/memory:** About 420 minutes; peak 8.5GB VRAM and 10GB host RAM.

**Leakage risk:** Low with subject and clip grouping. The main modeling risk is domain transfer from RGB/Kinetics rather than label leakage.

**Fallback:** Freeze the full backbone and train only the head on cached embeddings. If the checkpoint cannot be resolved, use `torchvision.models.video.r3d_18` under the same grouped CV, candidate enumeration, and decoder.

### 4. Local reference VLM — required sanity baseline, not submission choice

**Leak-free features/encodings:** Uniform/motion-sampled frames and only the row’s question/options. No retrieved validation answer, test label, or sample-submission value enters the prompt.

**Models and concrete configuration:** Qwen3-VL-4B-Instruct in 4-bit mode, eight composite frames, strict category prompt, temperature zero, maximum 96 generated tokens, one parse-repair attempt.

**Expected runtime/memory:** Approximately 120–240 minutes for a representative grouped validation subset; 7–10GB VRAM.

**Leakage risk:** Low, but prompt iteration on the same validation fold can overfit. Freeze prompts after the first ablation stage.

**Fallback:** Use Qwen3-VL-2B-Instruct or the already-installed local LLaVA path while preserving frame order and exact output validation. The mandatory canonical notebook’s 0.400 evidence makes this useful for contract testing, but not competitive against the 0.86842 target.
