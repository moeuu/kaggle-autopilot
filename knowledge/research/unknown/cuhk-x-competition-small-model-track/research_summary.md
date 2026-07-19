## Ranked shortlist

### 1. Shared-TSM reliability-gated multimodal model — recommended final

**Leak-free features/encodings:** construct the train manifest by action, subject, and trial. Fit visual means/stds, IMU/radar/skeleton median-IQR scalers, schema mappings, imputation rules, class weights, and reliability calibration on each training fold only. Use root-relative skeleton coordinates, fold-fit body scale, velocities, recognized bone vectors, radar masks/counts/extents/radial moments, IMU derivatives/magnitudes, sequence lengths, parser flags, and modality availability. Never fit on test or use path/subject as predictive input.

**Models and hyperparameters:** scratch shared ResNet18-GroupNorm with per-visual-modality stems and TSM; 12 frames at 160×160; BiGRU attention; IMU depthwise 1D-CNN plus two Transformer layers; radar PointNet-style encoder plus BiGRU; skeleton per-joint MLP/temporal CNN plus two Transformer layers; six-token, two-layer 256-d reliability-gated fusion Transformer. Use staged specialist training and end-to-end fine-tuning, AdamW, batch 2, accumulation 8, label smoothing 0.08, modality dropout 0.20, MixStyle 0.50, auxiliary loss 0.18, consistency loss 0.08, subject-adversarial weight 0.03, EMA 0.9997, three TTA views, and OOF distillation weight 0.15. ([arXiv][9])

**Runtime/memory:** approximately 16–19M parameters, 64–76 MB FP32, 9–11.5 GB VRAM, and 18–22 hours for promotion, three folds, refit, and inference on RTX3060. **Leakage risk:** severe if subjects cross folds or statistics see validation/test; otherwise low. **Fallback:** isolate parser failures, then batch 2→1 with accumulation 16, frames 12→10, image 160→144, radar points 96→64, and fusion width 256→192.

### 2. Shared-TSM visual specialist — strongest lower-complexity challenger

**Leak-free features/encodings:** Depth_Color, IR, Thermal, clip lengths, valid-frame masks, parser flags, and availability normalized only from train-fold frames. **Model:** shared scratch ResNet18-GroupNorm + TSM, 16 frames at 176×176, BiGRU attention, gated visual fusion, MixStyle, modality dropout, AdamW 3e-4, up to 40 epochs, batch 2, accumulation 8, EMA, and three temporal views.

**Runtime/memory:** approximately 12–14M parameters, 48–56 MB FP32, 8–11 GB VRAM, 3–4 hours for one promotion fold, or 8–11 hours if promoted to three folds. **Leakage risk:** subject, clothing, and room shortcuts. **Fallback:** use 160×160 and 12 frames without changing the family; OpenCV may fall back to PIL.

### 3. Residual 3D-CNN + BiGRU reference — mandatory real-data sanity route

**Leak-free features/encodings:** 16 uniformly sampled Depth_Color frames at 112×112, fold-fit intensity normalization, clip length, and valid-frame mask. **Model:** residual 3D-CNN channels 32–64–128, two-layer BiGRU hidden 192, AdamW 3e-4, batch 4, accumulation 4, at most 30 epochs, label smoothing 0.08, and two temporal views.

**Runtime/memory:** approximately 3M parameters, under 15 MB FP32, 5–8 GB VRAM, and 1.5–3 hours for the required promotion fold. **Leakage risk:** low under grouped splitting, but public synthetic/sample behavior is not score evidence. **Fallback:** use only for loader, label, CV, size, OOF, and CSV validation; never copy sample predictions.

Promotion and final selection use complete pooled three-fold subject-disjoint OOF accuracy. A tie within 0.001 goes to the smaller/faster candidate. The deployed artifact is one refit checkpoint under 95 MiB plus deterministic TTA, not a fold ensemble.
