# research_summary.md (content)

## Ranked shortlist (practical, leak-free)

### 1) **SimMIM-pretrain SwinV2-Tiny (12ch) → supervised finetune + blend**
- Leak-free features/encodings:
  - Stack 12 raw bands; resize each band to `IMG_SIZE=256`; normalize with **train-fold-only** per-band mean/std (or robust p5/p95).
  - Optional: band dropout (randomly zero 1–2 bands) as augmentation (no label leakage).
- Models + key hyperparameters:
  - Encoder: `timm` `swinv2_tiny_window16_256`, `in_chans=12`, `pretrained=True`
  - SSL: SimMIM-style masking ratio ~0.5–0.7, patch size 16–32, MSE reconstruction, 20–50k steps (cap by time)
  - Finetune: epochs ~30, batch 16–32, lr ~5e-4 with cosine, weight decay ~0.05, AMP on, label smoothing 0.05, class-weighted CE or focal
- Expected runtime/memory:
  - Pretrain: 1–3 hours on a single consumer GPU (depends on sampling); finetune: ~30–60 min for 5-fold single-seed at 256px
- Leakage risk: low if pretrain uses only unlabeled data and finetune CV is strict.
- Fallbacks:
  - If SSL too slow: skip pretrain; finetune only.
  - If `timm` unavailable: `torchvision` ResNet50 with expanded conv1.

### 2) **Supervised SwinV2-Tiny (12ch) + strong aug (no SSL)**
- Leak-free features/encodings: same resizing + train-fit normalization.
- Models + key hyperparameters:
  - `swinv2_tiny_window16_256`, `in_chans=12`, mixup/cutmix on, focal loss for imbalance, EMA optional
- Runtime/memory: fast; good for iteration and CV stability.
- Leakage risk: low.
- Fallback: ConvNeXt-Tiny / EfficientNetV2-S with `in_chans=12`.

### 3) **Hybrid: deep model + tabular spectral indices (small blend)**
- Leak-free features/encodings:
  - Compute NDVI/NDWI/red-edge indices after resizing; aggregate mean/std/percentiles; fit scaler on train fold only.
- Models + key hyperparameters:
  - LightGBM (or CatBoost) on indices + stats; blend with CNN probabilities (simple weighted average).
- Runtime/memory: very fast; minimal GPU; good as ensemble stabilizer.
- Leakage risk: low if all stats/scalers are train-fit only.
- Fallback: if boosting libs missing, use sklearn LogisticRegression/MLP on stats.
