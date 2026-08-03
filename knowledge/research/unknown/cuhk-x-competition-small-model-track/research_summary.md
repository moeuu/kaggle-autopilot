## Ranked shortlist

### 1. Shared TDN–TSM reliability-gated multimodal model — recommended final candidate

**Leak-free features/encodings.** Build an outer-joined clip manifest keyed by `(action_id, subject_id, trial_id)` for train and `path` for test. Within every fold, fit visual mean/std, IMU and Radar robust scaling, Radar histogram edges, Skeleton schema/root/body-scale fallback, reliability thresholds, class sampling, and any calibration on fold-train only. Visual streams use one fixed clip crop, normalized-time sampling, native appearance channels, signed temporal differences, valid-frame masks, and crop-quality flags. IMU uses named channel aliases, magnitudes, differences, device-location groups, masks, and parser confidence. Radar uses point aliases, range–azimuth/range–Doppler histograms, and robust statistics. Skeleton uses dynamic pelvis roots, body-scale normalization, confidence, bones, velocity, acceleration, and joint masks.

**Models and key hyperparameters.** Scratch shared ResNet18-like GroupNorm trunk with modality-specific stems, TSM `shift_div=8`, TDN-lite excitation, 12 active frames at 160 pixels; visual BiGRU hidden 128 per direction; IMU width 192 with two Transformer layers; Radar 32 frames/96 points plus 16×16 histograms and BiGRU hidden 128; Skeleton 64 steps, width 192, two Transformer layers; six 256-dimensional slots; two-layer/four-head fusion Transformer; shared target-conditioned proxy generator. AdamW, batch 2, accumulation 8, visual LR `2.5e-4`, sensor/fusion LR `5e-4`, weight decay 0.03, label smoothing 0.06, square-root class sampling, modality dropout 0.20, MixStyle 0.35, EMA 0.9997, and three temporal TTA views. TSM/TDN, Transformer missing-modality fusion, and compact bottleneck slots are supported by primary research. ([arXiv][3])

**Runtime/memory.** Approximately 18–20M deploy parameters, 70–80 MB FP32, 9.5–11.8 GB VRAM, and 18–23 hours total on RTX3060 for promotion, three full folds, targeted ablations, refit, and inference. **Leakage risk:** high if users cross folds, fold statistics see validation/test, crop thresholds are tuned globally, stale promotion predictions are reused as full-CV OOF, or a teacher scores rows it trained on. **Fallback:** quarantine a failing modality parser; batch `2→1`, accumulation `8→16`; visual chunk `8→4`; frames `12→10`; image `160→144`; Radar points `96→64`; remove proxy/MixStyle/SupCon/IMU/Radar only after a negative common-fold ablation. Dependency fallback uses only PyTorch, NumPy, pandas, sklearn, OpenCV/Pillow.

### 2. Shared TSM visual–Skeleton challenger — strongest lower-complexity pipeline

**Leak-free features/encodings.** Depth_Color, IR, Thermal, and numeric Skeleton with fold-fitted normalization, one synchronized crop, temporal differences, valid-frame/joint masks, clip lengths, modality availability, all-zero/unreadable flags, and parser/crop quality. Spatial and temporal augmentation must be clip-consistent; any horizontal flip must also mirror Skeleton coordinates and swap left/right joints.

**Models and key hyperparameters.** Shared scratch ResNet18-like GroupNorm visual trunk, TSM at 16 frames × 176 pixels, visual BiGRU hidden 128 per direction, Skeleton temporal CNN plus two Transformer layers at width 192, and a two-layer 256-dimensional reliability-gated fusion Transformer. AdamW `3e-4`, batch 2, accumulation 8, maximum 24 epochs, label smoothing 0.06, modality dropout 0.15, MixStyle 0.35, supervised contrastive weight 0.03 after epoch 4, EMA 0.9997, and three temporal TTA views.

**Runtime/memory.** Approximately 14–16M parameters, 52–65 MB FP32, 8–11.5 GB VRAM; 3–5 hours for promotion or 10–13 hours for three folds. **Leakage risk:** subject/room shortcuts, global pose scale, or crops selected using validation behavior. **Fallback:** frames `16→12`, image `176→160`, preserve missing-Skeleton tokens, and disable Skeleton only after schema/finite-value audits fail. This candidate is strongly motivated by official modality benchmark ordering, though those benchmark accuracies are not Kaggle leaderboard estimates. ([OpenAI OT Lab][4])

### 3. Scratch Depth+IR R(2+1)D–BiGRU — mandatory reference and fail-safe

**Leak-free features/encodings.** Sixteen uniformly sampled Depth_Color and IR frames, one data-derived fixed crop, fold-fitted normalization, signed temporal differences, clip length, valid-frame masks, and missing-stream masks. No YOLO, external detector, pretrained video backbone, sample-label copying, or path feature.

**Models and key hyperparameters.** Scratch factorized 3D residual network with channels `32-64-128-192`, one-layer bidirectional GRU hidden 192, attentive pooling, and a 40-class head. Image size 128, batch 4, accumulation 4, AdamW `3e-4`, weight decay 0.02, label smoothing 0.06, maximum 18 epochs, EMA 0.9997, and two temporal TTA views.

**Runtime/memory.** Approximately 7.1M parameters, 27–32 MB FP32, 5–8 GB VRAM, and 2–4 hours for one promotion fold. **Leakage risk:** low if subject grouping and fold transforms are correct; the main compliance risk is accidentally retaining the public notebook’s external detector or pretrained weights. **Fallback:** explicit zero input plus missing masks, full-frame crop when quality fails, and use this model only as the real-data archive/cache/CV/submission contract before promoting a stronger candidate.

## Promotion rule

Evaluate all three on the same fold-0 subject holdout for ten epochs. The full model must improve on the visual–Skeleton challenger by at least `0.002` absolute accuracy; otherwise choose the better simpler model. Within `0.001`, tie-break by serialized checkpoint bytes and then runtime. Run full three-fold CV only for the promoted family, and never reuse promotion-fold artifacts as full-CV OOF.
