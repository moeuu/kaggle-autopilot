"""Backward-compatible notebook submission-fidelity adapter.

The mode-neutral contract and quarantine implementation lives in
kagglebot.submission_fidelity. Existing imports remain valid here so notebook
runners and downstream integrations do not need a flag-day migration.
"""

from kagglebot.submission_fidelity import (
    EXPECTED_FILE_NAME,
    REPORT_FILE_NAME,
    RUNTIME_FILE_NAME,
    build_submission_fidelity_report,
    build_submit_fidelity_expected_contract,
    build_submit_runtime_env,
    load_expected_submit_metrics_snapshot,
    stage_submit_fidelity_expected_contract,
    validate_reference_submission_readiness,
    validate_submit_kernel_runtime_fidelity,
)

__all__ = [
    "EXPECTED_FILE_NAME",
    "REPORT_FILE_NAME",
    "RUNTIME_FILE_NAME",
    "build_submission_fidelity_report",
    "build_submit_fidelity_expected_contract",
    "build_submit_runtime_env",
    "load_expected_submit_metrics_snapshot",
    "stage_submit_fidelity_expected_contract",
    "validate_reference_submission_readiness",
    "validate_submit_kernel_runtime_fidelity",
]
