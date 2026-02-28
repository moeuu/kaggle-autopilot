# research_summary.md (Beyond Visible Spectrum AI for Agriculture 2026)

Ranked shortlist of candidate pipelines (practical, leak-free, Kaggle-friendly). Primary selection metric: macro-F1 (maximize), computed from CV OOF only.

## 1) Tri-branch ConvNeXt fusion + HS PCA (Recommended)
- Leak-free features/encodings: per-fold HS trimming (drop edge bands), per-fold normalization (mean/std), per-fold PCA on HS (K=24/32) fit on train split only; optional modality dropout.
- Models + key hyperparameters: timm ConvNeXt-S/B for RGB (pretrained), ConvNeXt-T/S for MS with `in_chans=5`, HS branch ConvNeXt-T with `in_chans=K`; fuse pooled embeddings with attention/MLP; CE or focal loss; cosine LR (2e-4) with warmup; AMP; batch size 32–128 depending on VRAM.
- Expected runtime/memory: ~5-fold × 3 seeds × 30 epochs ≈ 1–3 hours on a single consumer GPU; moderate VRAM (8–16GB) with AMP.
- Leakage risk: low if PCA/scalers are fit per fold only.
- Fallbacks: if PCA too slow, replace HS with band-mean vector + small MLP; if timm unavailable, use torchvision resnet50 and custom first conv.

## 2) Early fusion projector into pretrained ConvNeXt
- Leak-free features/encodings: concatenate RGB+MS+HS_PCA(K) channels; fold-fit normalization + PCA; 1×1 conv projector to 3ch.
- Models + key hyperparameters: single ConvNeXt-B pretrained; stronger aug (randaugment, cutmix/mixup); label smoothing 0.05–0.1; dropout/stochastic depth.
- Expected runtime/memory: fastest deep option; similar VRAM to RGB-only model.
- Leakage risk: low (same fold-fit rules).
- Fallbacks: if pretrained weights disallowed, run from scratch with stronger regularization and longer schedule.

## 3) Tabular spectral indices + LightGBM blend
- Leak-free features/encodings: MS indices (NDVI, NDRE), per-band pooled stats, HS PCA on pooled spectra; standard scaler fit on train split only; align_features for train/test mismatch safety.
- Models + key hyperparameters: LightGBM multiclass (num_leaves 31–127, min_data_in_leaf 5–30, feature_fraction 0.7–1.0) with early stopping; blend probabilities with deep model (simple mean).
- Expected runtime/memory: minutes; tiny RAM/VRAM.
- Leakage risk: low if transforms fit on train split only.
- Fallbacks: sklearn HistGradientBoostingClassifier or LogisticRegression on scaled features.

## 4) HS HybridSN-style 3D→2D CNN + RGB/MS model ensemble
- Leak-free features/encodings: HS cube standardization per fold; optional focal loss.
- Models + key hyperparameters: 3D conv blocks over spectral dimension then 2D conv; ensemble average with RGB/MS ConvNeXt.
- Expected runtime/memory: higher VRAM/time; sensitive to patch size and spectral depth.
- Leakage risk: low if fold-fit normalization used.
- Fallbacks: replace with HS PCA+2D CNN branch (Pipeline #1).
