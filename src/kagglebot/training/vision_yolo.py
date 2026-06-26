from __future__ import annotations

import ast
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.env_utils import env_flag

OFFICIAL_COMBINED_METRIC = "0.5 * mAP@[0.5:0.95] + 0.5 * F1-Score"


@dataclass(frozen=True)
class DetectorBundle:
    backend: str
    model: Any
    device: str
    nms_iou: float = 0.5
    imgsz: int = 640
    weights_path: str | None = None


def detect_yolo_submission_task(data_dir: Path, sample_df: pd.DataFrame) -> bool:
    required = {"filename", "right_place", "prediction_string"}
    if set(sample_df.columns) != required:
        return False
    return all(
        [
            (data_dir / "images" / "train").is_dir(),
            (data_dir / "images" / "test").is_dir(),
            (data_dir / "labels" / "train").is_dir(),
        ]
    )


def train_val_split(
    train_labels_df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.2,
) -> tuple[list[str], list[str]]:
    from sklearn.model_selection import train_test_split

    frame = train_labels_df[["filename", "right_place"]].dropna().copy()
    frame["filename"] = frame["filename"].astype(str)
    y = frame["right_place"].astype(int)

    if frame.empty:
        return [], []
    if len(frame) < 2:
        return frame["filename"].tolist(), []

    val_count = max(1, int(round(len(frame) * val_frac)))
    if val_count >= len(frame):
        val_count = len(frame) - 1

    train_count = len(frame) - val_count
    can_stratify = (
        y.nunique() >= 2 and y.value_counts().min() >= 2 and val_count >= y.nunique() and train_count >= y.nunique()
    )
    stratify = y if can_stratify else None

    train_part, val_part = train_test_split(
        frame,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_part["filename"].tolist(), val_part["filename"].tolist()


def prepare_ultralytics_dataset(
    root_out: Path,
    data_dir: Path,
    train_files: list[str],
    val_files: list[str],
) -> Path:
    ds_root = root_out / "yolo_ds"
    images_train = ds_root / "images" / "train"
    images_val = ds_root / "images" / "val"
    labels_train = ds_root / "labels" / "train"
    labels_val = ds_root / "labels" / "val"

    for directory in [images_train, images_val, labels_train, labels_val]:
        directory.mkdir(parents=True, exist_ok=True)

    src_images = data_dir / "images" / "train"
    src_labels = data_dir / "labels" / "train"

    for filename in sorted({str(item) for item in train_files}):
        _link_or_copy(src_images / filename, images_train / filename)
        label_name = f"{Path(filename).stem}.txt"
        _link_or_copy_or_empty(src_labels / label_name, labels_train / label_name)

    for filename in sorted({str(item) for item in val_files}):
        _link_or_copy(src_images / filename, images_val / filename)
        label_name = f"{Path(filename).stem}.txt"
        _link_or_copy_or_empty(src_labels / label_name, labels_val / label_name)

    class_ids = _discover_class_ids(src_labels)
    names = _dataset_names_for_class_ids(class_ids)
    dataset_yaml = ds_root / "dataset.yaml"
    yaml_text = "\n".join(
        [
            f"path: {ds_root}",
            "train: images/train",
            "val: images/val",
            f"names: {names}",
            "",
        ]
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")
    return dataset_yaml


def train_detector(
    dataset_yaml: Path,
    *,
    seed: int,
    device: str,
    time_budget_min: int | None,
) -> Any:
    ultralytics_available = _ultralytics_available()
    prefer_ultralytics = ultralytics_available
    if prefer_ultralytics:
        try:
            return _train_ultralytics_detector(
                dataset_yaml=dataset_yaml,
                seed=seed,
                device=device,
                time_budget_min=time_budget_min,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[vision] ultralytics unavailable/failing; falling back to torchvision detector: {exc}", flush=True)

    return _train_torchvision_detector(
        dataset_yaml=dataset_yaml,
        seed=seed,
        device=device,
        time_budget_min=time_budget_min,
    )


def predict_detector(model: Any, image_paths: list[Path]) -> dict[str, np.ndarray]:
    if isinstance(model, DetectorBundle) and model.backend == "ultralytics":
        return _predict_with_ultralytics(model, image_paths)
    if isinstance(model, DetectorBundle) and model.backend == "torchvision":
        return _predict_with_torchvision(model, image_paths)

    if hasattr(model, "predict"):
        bundle = DetectorBundle(backend="ultralytics", model=model, device="cpu")
        return _predict_with_ultralytics(bundle, image_paths)

    raise TypeError(f"Unsupported detector model type: {type(model)!r}")


def format_prediction_string(dets: np.ndarray, score_thr: float) -> str:
    arr = np.asarray(dets, dtype=float)
    if arr.size == 0:
        return "-"
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 6:
        raise ValueError(f"Expected detections with shape (N,6), got {arr.shape}.")

    arr = arr[arr[:, 1] >= float(score_thr)]
    if arr.size == 0:
        return "-"

    arr[:, 2:] = np.clip(arr[:, 2:], 0.0, 1.0)
    order = np.argsort(-arr[:, 1])
    arr = arr[order]

    tokens: list[str] = []
    for row in arr:
        tokens.append(str(int(round(float(row[0])))))
        tokens.append(f"{float(row[1]):.6f}")
        tokens.append(f"{float(row[2]):.6f}")
        tokens.append(f"{float(row[3]):.6f}")
        tokens.append(f"{float(row[4]):.6f}")
        tokens.append(f"{float(row[5]):.6f}")
    return " ".join(tokens)


def derive_right_place(
    dets: np.ndarray,
    *,
    head_cls: int,
    shemagh_cls: int,
    params: dict[str, float] | None,
) -> int:
    thresholds = {
        "head_score_thr": 0.15,
        "shemagh_score_thr": 0.15,
        "iou_thr": 0.02,
        "containment_thr": 0.08,
        "center_dist_thr": 1.35,
        "top_k": 5.0,
    }
    if params:
        thresholds.update({k: float(v) for k, v in params.items() if isinstance(v, (int, float))})

    arr = np.asarray(dets, dtype=float)
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 6:
        return 0

    top_k = max(1, int(round(thresholds["top_k"])))
    head_candidates = _topk_detections_for_class(
        arr=arr,
        class_id=head_cls,
        min_score=thresholds["head_score_thr"],
        top_k=top_k,
    )
    shemagh_candidates = _topk_detections_for_class(
        arr=arr,
        class_id=shemagh_cls,
        min_score=thresholds["shemagh_score_thr"],
        top_k=top_k,
    )
    if head_candidates.size == 0 or shemagh_candidates.size == 0:
        return 0

    best_pair: tuple[float, float, float] | None = None
    for head in head_candidates:
        for shemagh in shemagh_candidates:
            iou, containment, center_dist = _placement_pair_stats(head=head, shemagh=shemagh)
            dist_margin = 1.0 - min(center_dist / max(thresholds["center_dist_thr"], 1e-9), 1.0)
            placement_score = (0.50 * containment) + (0.35 * iou) + (0.15 * dist_margin)
            if best_pair is None or placement_score > best_pair[0]:
                best_pair = (placement_score, iou, containment, center_dist)
    if best_pair is None:
        return 0
    _score, iou, containment, center_dist = best_pair

    if iou < thresholds["iou_thr"]:
        return 0
    if containment < thresholds["containment_thr"]:
        return 0
    if center_dist > thresholds["center_dist_thr"]:
        return 0
    return 1


def tune_right_place_params(
    val_truth_df: pd.DataFrame,
    val_dets_by_file: dict[str, np.ndarray],
    *,
    head_cls: int = 0,
    shemagh_cls: int = 1,
) -> dict[str, float]:
    truth = val_truth_df[["filename", "right_place"]].copy()
    truth["filename"] = truth["filename"].astype(str)
    y_true = truth["right_place"].astype(int).to_numpy()

    best_params = {
        "head_score_thr": 0.1,
        "shemagh_score_thr": 0.1,
        "iou_thr": 0.0,
        "containment_thr": 0.0,
        "center_dist_thr": 1.5,
        "top_k": 5.0,
    }
    best_f1 = -1.0

    for head_thr in [0.05, 0.1, 0.2, 0.3]:
        for shemagh_thr in [0.05, 0.1, 0.2, 0.3]:
            for iou_thr in [0.0, 0.02, 0.05, 0.1]:
                for containment_thr in [0.0, 0.05, 0.1, 0.2]:
                    for center_dist_thr in [0.9, 1.1, 1.3, 1.5]:
                        for top_k in [3.0, 5.0, 8.0]:
                            params = {
                                "head_score_thr": head_thr,
                                "shemagh_score_thr": shemagh_thr,
                                "iou_thr": iou_thr,
                                "containment_thr": containment_thr,
                                "center_dist_thr": center_dist_thr,
                                "top_k": top_k,
                            }
                            y_pred = [
                                derive_right_place(
                                    val_dets_by_file.get(filename, np.empty((0, 6), dtype=float)),
                                    head_cls=head_cls,
                                    shemagh_cls=shemagh_cls,
                                    params=params,
                                )
                                for filename in truth["filename"].tolist()
                            ]
                            score = _positive_class_f1(y_true=y_true, y_pred=np.asarray(y_pred, dtype=int))
                            if score > best_f1:
                                best_f1 = score
                                best_params = params

    best_params = {
        **best_params,
        "f1": best_f1,
        "prediction_score_thr": min(best_params["head_score_thr"], best_params["shemagh_score_thr"]),
    }
    return best_params


def evaluate_combined_metric(
    val_truth_df: pd.DataFrame,
    val_dets_by_file: dict[str, np.ndarray],
    detector_map50_95: float,
    *,
    head_cls: int = 0,
    shemagh_cls: int = 1,
    tuned_params: dict[str, float] | None = None,
) -> dict[str, object]:
    params = tuned_params or tune_right_place_params(
        val_truth_df,
        val_dets_by_file,
        head_cls=head_cls,
        shemagh_cls=shemagh_cls,
    )
    truth = val_truth_df[["filename", "right_place"]].copy()
    truth["filename"] = truth["filename"].astype(str)
    y_true = truth["right_place"].astype(int).to_numpy()
    y_pred = [
        derive_right_place(
            val_dets_by_file.get(filename, np.empty((0, 6), dtype=float)),
            head_cls=head_cls,
            shemagh_cls=shemagh_cls,
            params=params,
        )
        for filename in truth["filename"].tolist()
    ]
    f1 = _positive_class_f1(y_true=y_true, y_pred=np.asarray(y_pred, dtype=int))
    combined = 0.5 * float(detector_map50_95) + 0.5 * f1
    return {
        "map50_95": float(detector_map50_95),
        "f1": f1,
        "combined": float(combined),
        "thresholds": params,
    }


def infer_head_shemagh_classes(labels_dir: Path) -> tuple[int, int, dict[str, object]]:
    class_areas: dict[int, list[float]] = defaultdict(list)
    class_aspects: dict[int, list[float]] = defaultdict(list)

    for label_file in sorted(labels_dir.glob("*.txt")):
        rows = _read_yolo_label_rows(label_file)
        for cls, _cx, _cy, w, h in rows:
            class_id = int(cls)
            class_areas[class_id].append(float(max(w, 0.0) * max(h, 0.0)))
            if h > 0:
                class_aspects[class_id].append(float(w / h))

    if not class_areas:
        return 0, 1, {"reason": "no_labels", "class_stats": {}}

    area_medians = {cid: float(np.median(vals)) for cid, vals in class_areas.items() if vals}
    if not area_medians:
        return 0, 1, {"reason": "empty_labels", "class_stats": {}}

    sorted_by_area = sorted(area_medians.items(), key=lambda item: item[1])
    head_cls = int(sorted_by_area[0][0])

    other_classes = [cid for cid in area_medians if cid != head_cls]
    if other_classes:
        shemagh_cls = int(max(other_classes, key=lambda cid: area_medians[cid]))
    else:
        shemagh_cls = int(head_cls + 1)

    class_stats = {}
    for cid in sorted(area_medians):
        aspect_values = class_aspects.get(cid, [])
        class_stats[int(cid)] = {
            "median_area": float(area_medians[cid]),
            "median_aspect": float(np.median(aspect_values)) if aspect_values else None,
            "count": int(len(class_areas[cid])),
        }

    return head_cls, shemagh_cls, {"class_stats": class_stats}


def compute_map50_95(
    *,
    data_dir: Path,
    filenames: list[str],
    dets_by_file: dict[str, np.ndarray],
) -> float:
    labels_dir = data_dir / "labels" / "train"
    gt_by_file: dict[str, np.ndarray] = {}
    class_ids: set[int] = set()

    for filename in filenames:
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        rows = _read_yolo_label_rows(label_path)
        if rows:
            arr = np.asarray(rows, dtype=float)
        else:
            arr = np.empty((0, 5), dtype=float)
        gt_by_file[str(filename)] = arr
        if arr.size:
            class_ids.update(arr[:, 0].astype(int).tolist())

    if not class_ids:
        return 0.0

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    threshold_maps: list[float] = []
    for iou_thr in iou_thresholds:
        ap_values: list[float] = []
        for class_id in sorted(class_ids):
            ap = _average_precision_for_class(
                class_id=class_id,
                iou_thr=float(iou_thr),
                gt_by_file=gt_by_file,
                dets_by_file=dets_by_file,
            )
            if ap is not None:
                ap_values.append(ap)
        if ap_values:
            threshold_maps.append(float(np.mean(ap_values)))

    if not threshold_maps:
        return 0.0
    return float(np.mean(threshold_maps))


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _train_ultralytics_detector(
    *,
    dataset_yaml: Path,
    seed: int,
    device: str,
    time_budget_min: int | None,
) -> DetectorBundle:
    from ultralytics import YOLO

    use_pretrain = _use_pretrained_backbone()
    base_model = "yolov8m.pt" if use_pretrain else "yolov8n.yaml"
    epochs = _resolve_training_epochs(time_budget_min=time_budget_min, default_epochs=90, cap_epochs=150)
    patience = min(35, max(10, epochs // 5))
    imgsz = 960 if (time_budget_min is not None and time_budget_min >= 90) else 768

    yolo_device = "cpu"
    if device == "cuda":
        yolo_device = "0"
    elif device == "mps":
        yolo_device = "mps"

    model = YOLO(base_model)
    run_dir = dataset_yaml.parent.parent / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(dataset_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=-1,
        amp=True,
        patience=patience,
        seed=seed,
        deterministic=True,
        device=yolo_device,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    weights_path = run_dir / "train" / "weights" / "best.pt"
    resolved_weights = str(weights_path) if weights_path.exists() else None
    final_model = YOLO(resolved_weights) if resolved_weights is not None else model
    return DetectorBundle(
        backend="ultralytics",
        model=final_model,
        device=yolo_device,
        imgsz=imgsz,
        weights_path=resolved_weights,
    )


def _train_torchvision_detector(
    *,
    dataset_yaml: Path,
    seed: int,
    device: str,
    time_budget_min: int | None,
) -> DetectorBundle:
    import torch
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    _seed_everything(seed)

    ds_spec = _read_dataset_spec(dataset_yaml)
    ds_root = Path(ds_spec["path"])
    train_images = sorted((ds_root / ds_spec["train"]).glob("*"))
    train_labels = [ds_root / "labels" / "train" / f"{item.stem}.txt" for item in train_images]

    if not train_images:
        raise RuntimeError("No training images found for detection dataset.")

    max_class_id = 1
    for label_path in train_labels:
        rows = _read_yolo_label_rows(label_path)
        for cls, *_ in rows:
            max_class_id = max(max_class_id, int(cls))

    num_classes = int(max_class_id + 2)
    use_pretrain = _use_pretrained_backbone()

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if use_pretrain else None
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrain else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=backbone_weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    resolved_device = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        resolved_device = "cuda"
    elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        resolved_device = "mps"

    torch_device = torch.device(resolved_device)
    model.to(torch_device)

    dataset = _YoloTorchvisionDataset(image_paths=train_images, label_paths=train_labels)
    batch_size = 2 if resolved_device == "cuda" else 1
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_detection_batch,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)

    epochs = _resolve_training_epochs(time_budget_min=time_budget_min, default_epochs=4, cap_epochs=8)
    deadline = None
    if time_budget_min is not None and time_budget_min > 0:
        deadline = time.monotonic() + (time_budget_min * 60.0)

    model.train()
    for _epoch in range(epochs):
        if deadline is not None and time.monotonic() > deadline:
            break
        for images, targets in loader:
            if deadline is not None and time.monotonic() > deadline:
                break
            batch_images = [img.to(torch_device) for img in images]
            batch_targets = [{k: v.to(torch_device) for k, v in t.items()} for t in targets]
            loss_dict = model(batch_images, batch_targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    return DetectorBundle(backend="torchvision", model=model, device=resolved_device)


def _predict_with_ultralytics(bundle: DetectorBundle, image_paths: list[Path]) -> dict[str, np.ndarray]:
    if not image_paths:
        return {}
    results = bundle.model.predict(
        source=[str(path) for path in image_paths],
        conf=0.001,
        iou=0.7,
        imgsz=bundle.imgsz,
        device=bundle.device,
        verbose=False,
    )

    out: dict[str, np.ndarray] = {}
    for path, result in zip(image_paths, results, strict=False):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            out[path.name] = np.empty((0, 6), dtype=float)
            continue
        cls = boxes.cls.detach().cpu().numpy().astype(float)
        score = boxes.conf.detach().cpu().numpy().astype(float)
        xywhn = boxes.xywhn.detach().cpu().numpy().astype(float)
        dets = np.column_stack([cls, score, xywhn]).astype(float)
        dets = _sanitize_detections(dets)
        dets = _apply_classwise_nms(dets, iou_thr=bundle.nms_iou)
        out[path.name] = dets
    return out


def _predict_with_torchvision(bundle: DetectorBundle, image_paths: list[Path]) -> dict[str, np.ndarray]:
    import torch
    from torchvision.transforms.functional import to_tensor

    out: dict[str, np.ndarray] = {}
    if not image_paths:
        return out

    model = bundle.model
    device = torch.device(bundle.device)
    model.eval()

    with torch.inference_mode():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            tensor = to_tensor(image).to(device)
            prediction = model([tensor])[0]

            labels = prediction["labels"].detach().cpu().numpy().astype(int) - 1
            scores = prediction["scores"].detach().cpu().numpy().astype(float)
            boxes_abs = prediction["boxes"].detach().cpu().numpy().astype(float)

            if boxes_abs.size == 0:
                out[path.name] = np.empty((0, 6), dtype=float)
                continue

            boxes_norm = boxes_abs.copy()
            boxes_norm[:, [0, 2]] /= max(width, 1)
            boxes_norm[:, [1, 3]] /= max(height, 1)
            cxcywh = _xyxy_to_cxcywh(boxes_norm)
            dets = np.column_stack([labels.astype(float), scores, cxcywh]).astype(float)
            dets = _sanitize_detections(dets)
            dets = _apply_classwise_nms(dets, iou_thr=bundle.nms_iou)
            out[path.name] = dets

    return out


def _average_precision_for_class(
    *,
    class_id: int,
    iou_thr: float,
    gt_by_file: dict[str, np.ndarray],
    dets_by_file: dict[str, np.ndarray],
) -> float | None:
    gt_boxes: dict[str, np.ndarray] = {}
    total_gt = 0
    for filename, arr in gt_by_file.items():
        if arr.size == 0:
            gt_boxes[filename] = np.empty((0, 4), dtype=float)
            continue
        class_mask = arr[:, 0].astype(int) == class_id
        boxes = arr[class_mask, 1:5]
        boxes_xyxy = (
            np.asarray([_cxcywh_to_xyxy(box) for box in boxes], dtype=float) if len(boxes) else np.empty((0, 4))
        )
        gt_boxes[filename] = boxes_xyxy
        total_gt += len(boxes_xyxy)

    if total_gt == 0:
        return None

    predictions: list[tuple[float, str, np.ndarray]] = []
    for filename, dets in dets_by_file.items():
        arr = np.asarray(dets, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        class_rows = arr[arr[:, 0].astype(int) == class_id]
        for row in class_rows:
            predictions.append((float(row[1]), filename, _cxcywh_to_xyxy(row[2:6])))

    if not predictions:
        return 0.0

    predictions.sort(key=lambda item: item[0], reverse=True)
    matched = {filename: np.zeros(len(boxes), dtype=bool) for filename, boxes in gt_boxes.items()}

    tp = np.zeros(len(predictions), dtype=float)
    fp = np.zeros(len(predictions), dtype=float)

    for idx, (_score, filename, pred_box) in enumerate(predictions):
        gts = gt_boxes.get(filename, np.empty((0, 4), dtype=float))
        if len(gts) == 0:
            fp[idx] = 1.0
            continue

        ious = np.asarray([_iou_xyxy(pred_box, gt_box) for gt_box in gts], dtype=float)
        best_i = int(np.argmax(ious)) if len(ious) else -1
        best_iou = float(ious[best_i]) if best_i >= 0 else 0.0
        if best_i >= 0 and best_iou >= iou_thr and not matched[filename][best_i]:
            tp[idx] = 1.0
            matched[filename][best_i] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(float(total_gt), 1e-12)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return _integrate_precision_recall(precision=precision, recall=recall)


def _integrate_precision_recall(*, precision: np.ndarray, recall: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def _sanitize_detections(dets: np.ndarray) -> np.ndarray:
    arr = np.asarray(dets, dtype=float)
    if arr.size == 0:
        return np.empty((0, 6), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[:, :6].copy()
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    arr[:, 2:] = np.clip(arr[:, 2:], 0.0, 1.0)
    arr[:, 4:] = np.clip(arr[:, 4:], 1e-6, 1.0)
    return arr


def _apply_classwise_nms(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    arr = np.asarray(dets, dtype=float)
    if arr.size == 0:
        return np.empty((0, 6), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    kept_indices: list[int] = []
    for class_id in np.unique(arr[:, 0].astype(int)):
        indices = np.where(arr[:, 0].astype(int) == class_id)[0]
        class_rows = arr[indices]
        boxes = np.asarray([_cxcywh_to_xyxy(row[2:6]) for row in class_rows], dtype=float)
        scores = class_rows[:, 1]
        keep_local = _nms_indices(boxes=boxes, scores=scores, iou_thr=iou_thr)
        kept_indices.extend(indices[idx] for idx in keep_local)

    if not kept_indices:
        return np.empty((0, 6), dtype=float)
    kept = arr[np.asarray(kept_indices, dtype=int)]
    order = np.argsort(-kept[:, 1])
    return kept[order]


def _nms_indices(*, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    try:
        import torch
        from torchvision.ops import nms

        t_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        t_scores = torch.as_tensor(scores, dtype=torch.float32)
        keep = nms(t_boxes, t_scores, iou_thr)
        return keep.detach().cpu().numpy().astype(int).tolist()
    except Exception:  # noqa: BLE001
        order = np.argsort(-scores)
        keep: list[int] = []
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            ious = np.asarray([_iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=float)
            order = rest[ious <= iou_thr]
        return keep


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "cudnn", None):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        return


def _use_pretrained_backbone() -> bool:
    return env_flag("KAGGLEBOT_YOLO_PRETRAIN", default=True)


def _resolve_training_epochs(*, time_budget_min: int | None, default_epochs: int, cap_epochs: int) -> int:
    env_override = os.getenv("KAGGLEBOT_YOLO_EPOCHS")
    if env_override:
        try:
            return max(1, min(int(env_override), cap_epochs))
        except ValueError:
            pass

    epochs = default_epochs
    if time_budget_min is None:
        return max(1, min(epochs, cap_epochs))
    if time_budget_min <= 20:
        epochs = min(epochs, 30)
    elif time_budget_min <= 40:
        epochs = min(epochs, 50)
    elif time_budget_min <= 70:
        epochs = min(epochs, 80)
    elif time_budget_min <= 100:
        epochs = min(epochs, 110)
    else:
        epochs = min(cap_epochs, max(epochs, 130))
    return max(1, min(epochs, cap_epochs))


def _read_dataset_spec(dataset_yaml: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw in dataset_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()

    path = payload.get("path")
    train = payload.get("train")
    val = payload.get("val")
    if not path or not train or not val:
        raise ValueError(f"Invalid dataset yaml at {dataset_yaml}.")

    names_raw = payload.get("names", "[]")
    try:
        names = ast.literal_eval(names_raw)
    except Exception:  # noqa: BLE001
        names = []

    return {
        "path": path,
        "train": train,
        "val": val,
        "names": str(names),
    }


def _dataset_names_for_class_ids(class_ids: list[int]) -> list[str]:
    if class_ids == [0, 1] or class_ids == [0] or class_ids == [1]:
        return ["head", "shemagh"]
    if not class_ids:
        return ["head", "shemagh"]
    upper = max(class_ids)
    names = [f"class_{idx}" for idx in range(max(upper + 1, 2))]
    names[0] = "head"
    if len(names) > 1:
        names[1] = "shemagh"
    return names


def _discover_class_ids(labels_dir: Path) -> list[int]:
    classes: set[int] = set()
    for path in labels_dir.glob("*.txt"):
        for row in _read_yolo_label_rows(path):
            classes.add(int(row[0]))
    return sorted(classes)


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:  # noqa: BLE001
        copy_artifact_if_needed(source=src, destination=dst)


def _link_or_copy_or_empty(src: Path, dst: Path) -> None:
    if src.exists():
        _link_or_copy(src, dst)
        return
    dst.write_text("", encoding="utf-8")


def _read_yolo_label_rows(path: Path) -> list[tuple[float, float, float, float, float]]:
    if not path.exists() or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    rows: list[tuple[float, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls, cx, cy, w, h = (float(item) for item in parts)
        except ValueError:
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


def _topk_detections_for_class(arr: np.ndarray, *, class_id: int, min_score: float, top_k: int) -> np.ndarray:
    class_mask = arr[:, 0].astype(int) == int(class_id)
    candidates = arr[class_mask]
    if candidates.size == 0:
        return np.empty((0, 6), dtype=float)
    candidates = candidates[candidates[:, 1] >= float(min_score)]
    if candidates.size == 0:
        return np.empty((0, 6), dtype=float)
    ordered = np.argsort(-candidates[:, 1])
    return candidates[ordered][: max(1, int(top_k))]


def _placement_pair_stats(*, head: np.ndarray, shemagh: np.ndarray) -> tuple[float, float, float]:
    head_box = _cxcywh_to_xyxy(head[2:6])
    shemagh_box = _cxcywh_to_xyxy(shemagh[2:6])
    iou = _iou_xyxy(head_box, shemagh_box)
    inter = _intersection_area_xyxy(head_box, shemagh_box)
    shemagh_area = _box_area_xyxy(shemagh_box)
    containment = inter / max(shemagh_area, 1e-9)
    head_diag = np.hypot(max(head[4], 1e-9), max(head[5], 1e-9))
    center_dist = float(np.hypot(head[2] - shemagh[2], head[3] - shemagh[3]) / max(head_diag, 1e-9))
    return float(iou), float(containment), center_dist


def _best_detection_for_class(arr: np.ndarray, *, class_id: int, min_score: float) -> np.ndarray | None:
    class_mask = arr[:, 0].astype(int) == int(class_id)
    candidates = arr[class_mask]
    if candidates.size == 0:
        return None
    candidates = candidates[candidates[:, 1] >= float(min_score)]
    if candidates.size == 0:
        return None
    best_idx = int(np.argmax(candidates[:, 1]))
    return candidates[best_idx]


def _positive_class_f1(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))


def _cxcywh_to_xyxy(box: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    cx, cy, w, h = [float(x) for x in box]
    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)
    return np.clip(np.asarray([x1, y1, x2, y2], dtype=float), 0.0, 1.0)


def _xyxy_to_cxcywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes_xyxy, dtype=float)
    out[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    out[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    out[:, 2] = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 1e-6)
    out[:, 3] = np.maximum(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 1e-6)
    return np.clip(out, 0.0, 1.0)


def _intersection_area_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _box_area_xyxy(box: np.ndarray) -> float:
    w = max(0.0, float(box[2]) - float(box[0]))
    h = max(0.0, float(box[3]) - float(box[1]))
    return w * h


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    inter = _intersection_area_xyxy(a, b)
    union = _box_area_xyxy(a) + _box_area_xyxy(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


class _YoloTorchvisionDataset:
    def __init__(self, *, image_paths: list[Path], label_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        import torch
        from torchvision.transforms.functional import to_tensor

        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = to_tensor(image)

        rows = _read_yolo_label_rows(label_path)
        if not rows:
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        else:
            arr = np.asarray(rows, dtype=np.float32)
            boxes_norm = np.asarray([_cxcywh_to_xyxy(row[1:5]) for row in arr], dtype=np.float32)
            boxes_xyxy = boxes_norm.copy()
            boxes_xyxy[:, [0, 2]] *= float(max(width, 1))
            boxes_xyxy[:, [1, 3]] *= float(max(height, 1))
            labels = arr[:, 0].astype(np.int64) + 1

        boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def _collate_detection_batch(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
