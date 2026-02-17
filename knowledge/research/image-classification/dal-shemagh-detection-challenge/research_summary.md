# research_summary.md (dal-shemagh-detection-challenge)

## Ranked shortlist (candidate pipelines)

### 1) 2× YOLO detector ensemble + WBF + geometry right_place (Recommended)
- Leak-free features/encodings: detector fit on train folds only; WBF is deterministic at inference; `right_place` computed from predicted boxes (best IoU pair + area ratio + confidence gating).
- Models + key hyperparameters:
  - Detectors: Ultralytics YOLO (e.g., `yolo11m` + `yolo11l`), `imgsz=896–1024`, `epochs=200–300`, `patience=30–50`, AMP on, strong aug.
  - Fusion: WBF `iou_thr=0.6–0.75`, `skip_box_thr=1e-4`, per-class.
  - right_place: rule thresholds tuned on OOF (grid over `conf_thres`, `t_iou`, `t_area`).
- Expected runtime/memory: ~2–6 GPU-hours depending on model sizes; modest CPU overhead for WBF.
- Leakage risk: low if thresholds are tuned only on OOF; avoid any pseudo-label training by default.
- Fallback if dependency unavailable: if no `ensemble_boxes`, replace WBF with per-class NMS + score-weighted box averaging.

### 2) Single strong YOLO + geometry right_place (Fast baseline with strong ceiling)
- Leak-free features/encodings: same rule; no fusion.
- Models + key hyperparameters: 1 YOLO model (bigger than “n”), `imgsz=896`, `epochs=250`, early stop; tune `conf_thres/t_iou/t_area` on OOF.
- Expected runtime/memory: ~1–3 GPU-hours.
- Leakage risk: low.
- Fallback: if no Ultralytics, use torchvision FasterRCNN and keep geometry rule.

### 3) Ensemble + meta-classifier on geometry features (Ablation if F1 is stuck)
- Leak-free features/encodings: build per-image geometry features from OOF predictions; fit scaler+LogReg on fold-train only; apply to fold-val/test; threshold tuned on OOF.
- Models + key hyperparameters: `LogisticRegression(C=0.1..1.0, class_weight='balanced')`, conservative feature set (counts, max conf, best IoU, area ratios).
- Expected runtime/memory: tiny compared to detector training.
- Leakage risk: medium if OOF separation is mishandled; keep strict fold boundaries.
- Fallback: disable meta-classifier and revert to rule-only.
