# research_summary.md (dal-shemagh-detection-challenge)

## Ranked shortlist (practical, Kaggle-runtime oriented)

### 1) Detector ensemble + WBF + geometry-derived right_place (Recommended)
- Leak-free features/encodings: raw RGB images → detector boxes; geometry features (IoU, containment, center distance, relative y-position, area ratios, counts) computed from **OOF** detector predictions only; fit scaler/classifier on train-fold, apply to val/test.
- Models + key hyperparameters: YOLO-style detector (from scratch) with img=640 (optionally 768), epochs=200–300, strong augmentation; ensemble 3–5 runs (seeds/sizes); WBF iou_thr≈0.55, skip_box_thr≈0.001; geometry classifier = logistic regression (C tuned) or small MLP.
- Expected runtime/memory: local GPU ~30–60 min per detector run (varies by GPU); inference minutes for 842 images; low RAM.
- Leakage risk: low if right_place classifier is trained only on OOF predictions and thresholds chosen per-fold.
- Fallback if dependency unavailable: if WBF lib missing, do per-class NMS then “best-box per class” geometry; if YOLO framework unavailable, use a torchvision detector (slower) with the same post-processing.

### 2) Two-stage: detector → ROI crop classifier
- Leak-free features/encodings: crop around union(top head box, top shemagh box) from detector outputs; train crop-classifier with fold-wise crops.
- Models + key hyperparameters: detector as above; crop-classifier = small CNN (e.g., ResNet18-like from scratch), input 224–320, heavy augmentation, class-balanced sampling, focal/BCE with pos_weight; threshold tuned for F1.
- Expected runtime/memory: extra 20–40 min per fold for classifier; modest VRAM.
- Leakage risk: medium if crops are generated from a detector trained on all data; must generate OOF crops only.
- Fallback: drop crop stage and use geometry classifier.

### 3) Multi-task single model (shared backbone, 2 heads)
- Leak-free features/encodings: raw images only.
- Models + key hyperparameters: shared backbone; detection loss + image-level BCE/focal; loss weights tuned; longer schedule needed.
- Expected runtime/memory: highest complexity; longer training.
- Leakage risk: low (end-to-end), but higher overfit risk on rare positives.
- Fallback: revert to pipeline #1.
