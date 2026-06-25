from __future__ import annotations

from kagglebot.iteration_signals import detect_online_mismatch_signal as _detect_online_mismatch_signal
from kagglebot.iteration_signals import extract_missing_ensemble_signal as _extract_missing_ensemble_signal
from kagglebot.iteration_signals import extract_orig_proba_signal as _extract_orig_proba_signal
from kagglebot.iteration_signals import extract_original_data_unused_signal as _extract_original_data_unused_signal
from kagglebot.iteration_signals import extract_pseudo_label_failure_signal as _extract_pseudo_label_failure_signal
from kagglebot.iteration_signals import extract_same_family_plateau_signal as _extract_same_family_plateau_signal
from kagglebot.iteration_signals import requires_tabular_multi_family_policy as _requires_tabular_multi_family_policy
from kagglebot.scalar_utils import parse_finite_float, parse_int
from kagglebot.score_utils import should_update_best_score as _update_best_score

__all__ = [
    "_detect_online_mismatch_signal",
    "_extract_missing_ensemble_signal",
    "_extract_orig_proba_signal",
    "_extract_original_data_unused_signal",
    "_extract_pseudo_label_failure_signal",
    "_extract_same_family_plateau_signal",
    "_requires_tabular_multi_family_policy",
    "_to_float",
    "_to_int",
    "_update_best_score",
]


def _to_float(value: object) -> float | None:
    return parse_finite_float(value, allow_commas=True)


def _to_int(value: object) -> int | None:
    return parse_int(value, allow_commas=True, allow_float=True, require_integral_float=False)
