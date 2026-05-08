from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.compute import Compute, detect_local_gpu
from kagglebot.exceptions import GPUNotAvailableError
from kagglebot.rna_structure import (
    build_coordinate_baseline_predictions,
    detect_rna_structure_task,
    evaluate_coordinate_predictions,
    extract_target_id,
    load_rna_structure_task,
    write_rna_structure_submission,
)
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.solver.io import load_competition_data
from kagglebot.training import (
    OFFICIAL_COMBINED_METRIC,
    compute_map50_95,
    derive_right_place,
    detect_yolo_submission_task,
    evaluate_combined_metric,
    format_prediction_string,
    infer_head_shemagh_classes,
    predict_detector,
    prepare_ultralytics_dataset,
    train_detector,
    train_val_split,
    tune_right_place_params,
)


@dataclass(frozen=True)
class TrainingOutcome:
    submission_path: Path
    evaluation: EvaluationResult
    model_name: str
    model_summary: dict[str, object]
    accelerator: str


def train_evaluate_and_predict(
    *,
    data_dir: Path,
    output_path: Path,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    plan_score_source: str | None,
    target_override: str | None,
) -> TrainingOutcome:
    del score_source, metric, direction, holdout_frac, cv_folds, plan_score_source

    if detect_rna_structure_task(data_dir):
        return _train_rna_structure_submission(
            data_dir=data_dir,
            output_path=output_path,
            seed=seed,
        )

    # Force vision routing first for the shemagh detection schema so target overrides cannot bypass detector training.
    base_data = load_competition_data(data_dir)
    if detect_yolo_submission_task(data_dir=data_dir, sample_df=base_data.sample):
        return _train_vision_yolo_submission(
            data_dir=data_dir,
            sample_df=base_data.sample,
            output_path=output_path,
            compute=compute,
            strict_accelerator=strict_accelerator,
            seed=seed,
        )

    if target_override is not None:
        load_competition_data(data_dir, target_column_override=target_override)

    raise RuntimeError(
        "Legacy src local trainer has been removed. "
        "Use artifacts/<slug>/kernel/kernel.py via `kagglebot train` or `kagglebot autopilot`."
    )


def _train_vision_yolo_submission(
    *,
    data_dir: Path,
    sample_df: pd.DataFrame,
    output_path: Path,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
) -> TrainingOutcome:
    labels_path = data_dir / "train_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing training labels for vision task: {labels_path}")

    train_labels = pd.read_csv(labels_path)
    required = {"filename", "right_place"}
    missing = required - set(train_labels.columns)
    if missing:
        raise ValueError(f"train_labels.csv missing required columns: {sorted(missing)}")

    train_labels = train_labels[["filename", "right_place"]].copy()
    train_labels["filename"] = train_labels["filename"].astype(str)
    train_labels["right_place"] = train_labels["right_place"].astype(int)

    train_files, val_files = train_val_split(train_labels_df=train_labels, seed=seed)
    if not train_files or not val_files:
        raise RuntimeError("Unable to build non-empty train/validation split for vision detector.")

    dataset_yaml = prepare_ultralytics_dataset(
        root_out=output_path.parent,
        data_dir=data_dir,
        train_files=train_files,
        val_files=val_files,
    )
    detector = train_detector(
        dataset_yaml=dataset_yaml,
        seed=seed,
        device=_select_device(compute=compute, strict_accelerator=strict_accelerator),
        time_budget_min=None,
    )

    val_paths = [data_dir / "images" / "train" / name for name in val_files]
    val_dets = predict_detector(detector, val_paths)

    test_filenames = sample_df["filename"].astype(str).tolist()
    test_paths = [data_dir / "images" / "test" / name for name in test_filenames]
    test_dets = predict_detector(detector, test_paths)

    head_cls, shemagh_cls, class_details = infer_head_shemagh_classes(data_dir / "labels" / "train")
    val_truth = train_labels[train_labels["filename"].isin(set(val_files))].copy()

    map50_95 = compute_map50_95(
        data_dir=data_dir,
        filenames=val_truth["filename"].astype(str).tolist(),
        dets_by_file=val_dets,
    )
    tuned = tune_right_place_params(
        val_truth_df=val_truth,
        val_dets_by_file=val_dets,
        head_cls=head_cls,
        shemagh_cls=shemagh_cls,
    )
    eval_payload = evaluate_combined_metric(
        val_truth_df=val_truth,
        val_dets_by_file=val_dets,
        detector_map50_95=map50_95,
        head_cls=head_cls,
        shemagh_cls=shemagh_cls,
        tuned_params=tuned,
    )

    prediction_score_thr = float(tuned.get("prediction_score_thr", 0.1))
    rows: list[dict[str, object]] = []
    for filename in test_filenames:
        dets = test_dets.get(filename, np.empty((0, 6), dtype=float))
        right_place = derive_right_place(
            dets=dets,
            head_cls=head_cls,
            shemagh_cls=shemagh_cls,
            params=tuned,
        )
        rows.append(
            {
                "filename": filename,
                "right_place": int(right_place),
                "prediction_string": format_prediction_string(dets, score_thr=prediction_score_thr),
            }
        )

    submission = pd.DataFrame(rows, columns=["filename", "right_place", "prediction_string"])
    submission["prediction_string"] = submission["prediction_string"].fillna("-").replace("", "-")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    model_summary = {
        "model": "vision_yolo",
        "backend": str(getattr(detector, "backend", "unknown")),
        "device": str(getattr(detector, "device", "cpu")),
        "head_cls": int(head_cls),
        "shemagh_cls": int(shemagh_cls),
        "class_inference": class_details,
        "map50_95": float(eval_payload["map50_95"]),
        "f1": float(eval_payload["f1"]),
        "combined": float(eval_payload["combined"]),
        "thresholds": eval_payload["thresholds"],
    }
    print(
        "[local train] model=vision_yolo "
        f"backend={model_summary['backend']} "
        f"map50_95={model_summary['map50_95']:.6f} "
        f"f1={model_summary['f1']:.6f} "
        f"combined={model_summary['combined']:.6f}",
        flush=True,
    )

    return TrainingOutcome(
        submission_path=output_path,
        evaluation=EvaluationResult(
            score_source="holdout",
            metric=OFFICIAL_COMBINED_METRIC,
            direction="maximize",
            value=float(eval_payload["combined"]),
            std=0.0,
            train_score=None,
            val_score=float(eval_payload["combined"]),
            fold_scores=[float(eval_payload["combined"])],
        ),
        model_name="vision_yolo",
        model_summary=model_summary,
        accelerator=_accelerator_label(str(getattr(detector, "device", "cpu"))),
    )


def _select_device(*, compute: Compute, strict_accelerator: bool) -> str:
    if compute != Compute.local_gpu:
        return "cpu"
    availability = detect_local_gpu()
    if availability.cuda:
        return "cuda"
    if availability.mps:
        return "mps"
    if strict_accelerator:
        raise GPUNotAvailableError(
            "No local GPU detected for --compute local_gpu. Disable --strict-accelerator to fall back to CPU."
        )
    return "cpu"


def _train_rna_structure_submission(
    *,
    data_dir: Path,
    output_path: Path,
    seed: int,
) -> TrainingOutcome:
    task = load_rna_structure_task(data_dir)
    label_id_column = task.label_id_column
    sample_id_column = task.sample_id_column

    target_ids = sorted({str(value).strip() for value in task.train_sequences[task.sequence_id_column].astype(str)})
    if len(target_ids) < 2:
        raise RuntimeError("RNA structure baseline requires at least two labeled targets for holdout evaluation.")

    rng = np.random.default_rng(seed)
    shuffled = list(target_ids)
    rng.shuffle(shuffled)
    split_index = max(1, int(round(len(shuffled) * 0.8)))
    if split_index >= len(shuffled):
        split_index = len(shuffled) - 1
    train_ids = set(shuffled[:split_index])
    valid_ids = set(shuffled[split_index:])
    if not train_ids or not valid_ids:
        raise RuntimeError("Unable to create a non-empty RNA structure train/validation split.")

    train_labels = task.train_labels[
        task.train_labels[label_id_column].astype(str).map(extract_target_id).isin(train_ids)
    ].copy()
    valid_labels = task.train_labels[
        task.train_labels[label_id_column].astype(str).map(extract_target_id).isin(valid_ids)
    ].copy()
    if train_labels.empty or valid_labels.empty:
        raise RuntimeError("RNA structure baseline split produced empty train or validation labels.")

    valid_sample = task.sample_submission[
        task.sample_submission[sample_id_column].astype(str).map(extract_target_id).isin(valid_ids)
    ].copy()
    valid_predictions = build_coordinate_baseline_predictions(
        train_labels=train_labels,
        sample_submission=valid_sample,
        label_id_column=label_id_column,
    )
    valid_submission_path = output_path.parent / "validation_submission.csv"
    write_rna_structure_submission(
        sample_submission=valid_sample,
        predictions_by_target=valid_predictions,
        output_path=valid_submission_path,
    )
    valid_submission = pd.read_csv(valid_submission_path)
    val_rmse = evaluate_coordinate_predictions(
        truth=valid_labels,
        predictions=valid_submission,
        id_column=sample_id_column,
    )

    submission_predictions = build_coordinate_baseline_predictions(
        train_labels=task.train_labels,
        sample_submission=task.sample_submission,
        label_id_column=label_id_column,
    )
    write_rna_structure_submission(
        sample_submission=task.sample_submission,
        predictions_by_target=submission_predictions,
        output_path=output_path,
    )

    model_summary = {
        "model": "rna_coordinate_mean_baseline",
        "task": task.target_kind,
        "sequence_id_column": task.sequence_id_column,
        "sample_id_column": task.sample_id_column,
        "label_id_column": task.label_id_column,
        "coordinate_triplets": len(task.sample_coordinate_triplets),
        "train_targets": len(train_ids),
        "valid_targets": len(valid_ids),
        "val_rmse": float(val_rmse),
    }
    print(
        "[local train] model=rna_coordinate_mean_baseline "
        f"val_rmse={float(val_rmse):.6f} "
        f"targets={len(train_ids) + len(valid_ids)}",
        flush=True,
    )

    return TrainingOutcome(
        submission_path=output_path,
        evaluation=EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=float(val_rmse),
            std=0.0,
            train_score=None,
            val_score=float(val_rmse),
            fold_scores=[float(val_rmse)],
        ),
        model_name="rna_coordinate_mean_baseline",
        model_summary=model_summary,
        accelerator="cpu",
    )


def _accelerator_label(device: str) -> str:
    if device in {"0", "cuda"}:
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"
