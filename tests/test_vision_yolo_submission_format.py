from __future__ import annotations

import numpy as np

from kagglebot.training.vision_yolo import (
    derive_right_place,
    format_prediction_string,
    infer_head_shemagh_classes,
    infer_pairwise_object_classes,
)


def test_format_prediction_string_empty() -> None:
    dets = np.empty((0, 6), dtype=float)
    assert format_prediction_string(dets, score_thr=0.1) == "-"


def test_format_prediction_string_single_detection() -> None:
    dets = np.asarray([[1.0, 0.8765432, 0.4, 0.5, 0.2, 0.3]], dtype=float)
    text = format_prediction_string(dets, score_thr=0.1)
    tokens = text.split()

    assert len(tokens) == 6
    assert tokens[0] == "1"
    assert tokens[1] == "0.876543"


def test_format_prediction_string_clips_coordinates() -> None:
    dets = np.asarray([[0.0, 0.9, -0.3, 1.8, 1.4, -0.2]], dtype=float)
    text = format_prediction_string(dets, score_thr=0.1)
    parts = text.split()
    coords = [float(parts[idx]) for idx in [2, 3, 4, 5]]

    assert all(0.0 <= value <= 1.0 for value in coords)


def test_derive_right_place_positive_and_negative_cases() -> None:
    positive_dets = np.asarray(
        [
            [0.0, 0.95, 0.5, 0.5, 0.6, 0.6],
            [1.0, 0.92, 0.5, 0.5, 0.4, 0.4],
        ],
        dtype=float,
    )
    negative_dets = np.asarray(
        [
            [0.0, 0.95, 0.2, 0.2, 0.2, 0.2],
            [1.0, 0.92, 0.8, 0.8, 0.2, 0.2],
        ],
        dtype=float,
    )
    params = {
        "head_score_thr": 0.2,
        "shemagh_score_thr": 0.2,
        "iou_thr": 0.01,
        "containment_thr": 0.01,
        "center_dist_thr": 1.5,
    }

    assert derive_right_place(positive_dets, head_cls=0, shemagh_cls=1, params=params) == 1
    assert derive_right_place(negative_dets, head_cls=0, shemagh_cls=1, params=params) == 0


def test_derive_right_place_uses_topk_pairing() -> None:
    dets = np.asarray(
        [
            [0.0, 0.99, 0.1, 0.1, 0.10, 0.10],  # best head score, but wrong spatial pairing
            [0.0, 0.70, 0.5, 0.5, 0.60, 0.60],  # lower score, correct spatial pairing
            [1.0, 0.98, 0.9, 0.9, 0.12, 0.12],  # best shemagh score, wrong spatial pairing
            [1.0, 0.65, 0.5, 0.5, 0.40, 0.40],  # lower score, correct spatial pairing
        ],
        dtype=float,
    )
    params = {
        "head_score_thr": 0.2,
        "shemagh_score_thr": 0.2,
        "iou_thr": 0.01,
        "containment_thr": 0.01,
        "center_dist_thr": 1.5,
        "top_k": 5.0,
    }
    assert derive_right_place(dets, head_cls=0, shemagh_cls=1, params=params) == 1


def test_infer_head_shemagh_classes_prefers_smaller_box_for_head(tmp_path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)

    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("0 0.4 0.4 0.18 0.18\n1 0.6 0.6 0.65 0.55\n", encoding="utf-8")

    head_cls, shemagh_cls, details = infer_head_shemagh_classes(labels_dir)

    assert head_cls == 0
    assert shemagh_cls == 1
    assert "class_stats" in details


def test_infer_pairwise_object_classes_uses_generic_roles(tmp_path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)

    (labels_dir / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("7 0.4 0.4 0.18 0.18\n3 0.6 0.6 0.65 0.55\n", encoding="utf-8")

    inferred = infer_pairwise_object_classes(labels_dir, primary_role="anchor", secondary_role="container")

    assert inferred.primary_cls == 7
    assert inferred.secondary_cls == 3
    assert inferred.details["roles"] == {"primary": "anchor", "secondary": "container"}
    assert inferred.details["selection"] == "smallest_median_area_vs_largest_other_median_area"


def test_infer_pairwise_object_classes_reads_nested_label_files(tmp_path) -> None:
    labels_dir = tmp_path / "labels" / "train"
    nested = labels_dir / "fold_a"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    (nested / "b.txt").write_text("7 0.4 0.4 0.18 0.18\n3 0.6 0.6 0.65 0.55\n", encoding="utf-8")

    inferred = infer_pairwise_object_classes(labels_dir, primary_role="anchor", secondary_role="container")

    assert inferred.primary_cls == 7
    assert inferred.secondary_cls == 3
    assert inferred.details["class_stats"][7]["count"] == 2
