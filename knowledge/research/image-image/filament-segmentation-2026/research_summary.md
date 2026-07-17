## Ranked shortlist

### 1. Edge-conditioned ConvNeXt-Tiny multitask network + embedding-aware watershed — primary

**Leak-free features/encodings:** Four per-image channels: robust disk-normalized grayscale, Gaussian dark residual, Sobel magnitude, and radial coordinate. Fold normalization is fit only on physical training files. A complete annotator instance set is selected coherently per physical image and epoch. Boundary bands are rasterized per instance, supplied spines become soft heatmaps, and an eight-dimensional pixel embedding is supervised from instance labels.

**Models + key hyperparameters:** ImageNet-pretrained ConvNeXt-Tiny; FPN/U-Net decoder; 256-dimensional, 8-head edge-conditioned bottleneck attention; foreground, boundary, spine, seed-distance and 8-D embedding heads. Train 20×700 steps/fold, 768 tiles, batch 1, accumulation 8, AdamW `2e-4`, encoder LR multiplier `0.25`, AMP FP16, three grouped folds, one seed. Loss weights: foreground `0.58`, boundary `0.10`, spine `0.08`, soft-clDice `0.08`, embedding `0.11`, seed-distance `0.05`. Inference uses 1024 tiles/256 overlap, 768 global pass, four TTAs, watershed and embedding graph merge; candidate refiner is 384-pixel ResNet18 U-Net.

**Expected runtime/memory:** Roughly 1,090–1,340 minutes including full CV, OOF calibration, deferred test inference and a promoted refiner. Batch 1, activation checkpointing and FP16 target 12GB.

**Leakage risk:** Duplicate physical observations, fold-contaminated normalization, per-fold post-process tuning, and training a refiner on its validation proposals. Enforce filename groups, fold-fit transforms, leave-one-fold-out calibration, and OOF-only refiner training.

**Fallback:** Reduce tile to 640 then 576, increase accumulation, reduce TTA, and skip the refiner before dropping the backbone or instance heads. Use torchvision ConvNeXt-Tiny if `timm` or its weights are unavailable. Edge guidance, topology loss and embedding supervision are supported by competition-specific and proposal-free instance-segmentation research. ([arXiv][2])

### 2. RF-DETR-Seg-Medium marker specialist — conditional promotion

**Leak-free features/encodings:** Export one-class COCO data from disjoint physical filenames, choose one deterministic coherent annotator set per image, and assert no train/validation filename overlap. Use RF-DETR masks only as marker hypotheses for the main high-resolution maps.

**Models + key hyperparameters:** Official `RFDETRSegMedium`, resolution 432, batch 2, accumulation 4, 30 epochs, LR `1e-4`, weight decay `1e-4`, confidence `0.18`, maximum 40 instances, fold 0 only. Promote at `+0.003` Dice, or a material split/merge reduction with no more than `0.001` Dice loss.

**Expected runtime/memory:** About 120–180 minutes on RTX3060 12GB, run only if at least 260 minutes remain after main OOF evidence.

**Leakage risk:** Annotator-qualified IDs can disguise duplicate JPEGs; confidence or marker rules can overfit fold 0; COCO AP is not the competition metric.

**Fallback:** Skip the dependency and retain distance, spine and embedding markers from pipeline 1. Never dynamically install RF-DETR during the run. Official documentation confirms COCO segmentation fine-tuning and pretrained model support. ([Roboflow][8])

### 3. Required grouped ResNet34 U-Net — measured reference

**Leak-free features/encodings:** Single grayscale channel; fold-0 split by physical filename; semantic target from one coherent annotator set or union only in the explicit ablation.

**Models + key hyperparameters:** ResNet34 U-Net, 512 pixels, batch 4, 5 epochs, 200 steps/epoch, AdamW `1e-3`, Dice+BCE, threshold `0.50`, connected components and minimum-area filtering.

**Expected runtime/memory:** About 30–60 minutes and below 12GB.

**Leakage risk:** Low after grouping, but severe modeling risk: 512 downsampling erases barbs and connected components cannot reliably distinguish touching filaments. The canonical context requires this notebook as a baseline but records no numeric score. 

**Fallback:** Run the same short model from random initialization if pretrained weights are unavailable, record the limitation, and never substitute its popularity for OOF evidence.

### 4. Dark-residual morphology — contract and hard-negative sanity path

**Leak-free features/encodings:** Per-image solar-disk mask, Gaussian background subtraction, residual quantile, and component shape/area features; no learned statistics.

**Models + key hyperparameters:** OpenCV only; Gaussian sigma `31`, residual quantile `0.965`, closing radius `2`, area range `10–50,000`, maximum 40 instances.

**Expected runtime/memory:** Under 20 minutes and under 2GB RAM.

**Leakage risk:** Minimal statistical leakage, but high false-positive/false-negative risk from sunspots, limb artifacts, clouds, and local contrast variation.

**Fallback:** Use for I/O, RLE round-trip, hard-negative mining and an explicit learned-model floor. It must never silently produce the scored final submission after a learned-route failure.
