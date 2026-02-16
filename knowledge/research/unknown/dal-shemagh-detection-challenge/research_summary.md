# research_summary.md (ranked shortlist)

## 1) Pipeline A (recommended): Dual-detector + classifier + WBF
- Leak-free features/encodings:
  - Vision-only (pixels). Optional meta-features from detector outputs (`n_boxes`, `max_score`, `sum_scores`, `max_area`, classwise counts) fit per fold; use `align_features()` before any meta-model.
  - If computing mean/std: compute on fold-train only; apply to val/test.
- Models + key hyperparameters:
  - Detector1: YOLOv8-L, `imgsz=1024`, 150–250 epochs, cosine LR, EMA, strong aug.
  - Detector2: YOLOv8-X (or YOLOv8-L at different imgsz/aug), same schedule.
  - Classifier: ConvNeXt (timm), 384px, weighted BCE / focal, 15–30 epochs.
  - Inference: TTA (hflip + scale), per-class WBF (IoU~0.55–0.65), cap max dets/image.
- Expected runtime/memory:
  - 1 GPU: detectors are the main cost (hours each depending on GPU); classifier <1–2h.
  - Inference: minutes + WBF overhead on CPU.
- Leakage risk:
  - Medium only if ROI crops are generated from non-fold-safe predictions; mitigate by using GT crops for train and fold-val predictions for val, and test predictions for test.
- Fallbacks:
  - If `ultralytics` unavailable: YOLOv5/YOLOv7 repo-based training; keep submission formatting identical.
  - If WBF lib not available: Soft-NMS / standard NMS + score calibration.

## 2) Pipeline B: Detector + geometry/meta right_place (fast iteration)
- Leak-free features/encodings:
  - Meta-features from detector predictions only; fit logreg on fold-train, apply to val/test.
- Models + key hyperparameters:
  - Single strong detector (YOLOv8-X) with higher imgsz + longer schedule.
  - Right_place: logistic regression / light MLP on meta-features; threshold tuned per fold.
- Runtime/memory:
  - Faster than Pipeline A (one detector + tiny meta-model).
- Leakage risk:
  - Low if meta-model is fit per fold.
- Fallback:
  - If meta-model unstable, revert to fixed heuristic threshold on max detection score.

## 3) Pipeline C: Unified multi-task (highest engineering, optional)
- Leak-free features/encodings:
  - Shared backbone; no tabular encodings.
- Models + key hyperparameters:
  - MMDetection/Lightning custom head; joint loss balancing for det + cls.
- Runtime/memory:
  - Highest; more tuning needed.
- Leakage risk:
  - Low, but higher implementation risk.
- Fallback:
  - Revert to Pipeline A.
