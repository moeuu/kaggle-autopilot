from __future__ import annotations

from kagglebot.metric_matching import infer_metric_direction_for_mismatch, metrics_equivalent


def test_metrics_equivalent_accepts_umud_metric_alias() -> None:
    assert metrics_equivalent(
        "UMUD normalized MAE",
        "UMUD Score: normalized mean absolute error across pa_deg, fl_mm, and mt_mm",
    )


def test_metrics_equivalent_accepts_balanced_accuracy_alias() -> None:
    assert metrics_equivalent("balanced_acc", "Balanced Accuracy")


def test_infer_metric_direction_for_mismatch_uses_known_and_text_hints() -> None:
    assert infer_metric_direction_for_mismatch("AURC", "maximize") == ("minimize", True)
    assert infer_metric_direction_for_mismatch("RMSE", "maximize") == ("minimize", True)
    assert infer_metric_direction_for_mismatch("custom error", "maximize") == ("minimize", True)
    assert infer_metric_direction_for_mismatch("custom MAP", "minimize") == ("maximize", True)
    assert infer_metric_direction_for_mismatch("custom metric", "maximize") == ("maximize", False)
