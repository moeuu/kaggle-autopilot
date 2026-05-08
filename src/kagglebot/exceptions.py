from __future__ import annotations


class KaggleBotError(Exception):
    """Base error for kagglebot commands."""

    exit_code = 1


class RulesNotAcceptedError(KaggleBotError):
    """Competition rules were not accepted by the user."""

    exit_code = 2


class KaggleCliError(KaggleBotError):
    """Kaggle CLI returned a non-zero exit code."""

    exit_code = 4

    def __init__(
        self,
        message: str,
        command: list[str] | None = None,
        exit_code: int | None = None,
        output: str = "",
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.command = command
        self.exit_code = exit_code or 4
        self.output = output
        self.stdout = stdout
        self.stderr = stderr


class KaggleNetworkError(KaggleCliError):
    """Kaggle CLI failed due to network connectivity issues."""

    exit_code = 16


class KernelCapacityError(KaggleCliError):
    """Kaggle GPU session capacity limit reached."""

    exit_code = 15


class ValidationError(KaggleBotError):
    """Submission validation failed."""

    exit_code = 6


class SubmissionValidationError(ValidationError, ValueError):
    """Submission validation failed before calling Kaggle CLI."""


class SubmissionCliError(KaggleCliError):
    """Kaggle CLI submit command failed."""


class DuplicateSubmissionError(KaggleBotError):
    """Submission hash already exists in the local ledger."""

    exit_code = 8


class SubmissionRateLimitError(KaggleBotError):
    """Local rate limit exceeded for submissions."""

    exit_code = 9


class MaxSubmissionsError(KaggleBotError):
    """Max submissions quota exceeded for autopilot run."""

    exit_code = 14


class SubmitAbortedError(KaggleBotError):
    """Submission flow aborted and must not be retried in the same run."""

    exit_code = 4


class GPUNotAvailableError(KaggleBotError):
    """Requested GPU is not available locally."""

    exit_code = 10


class KernelTimeoutError(KaggleBotError):
    """Kaggle kernel did not complete within the timeout."""

    exit_code = 11


class KernelStillRunningError(KernelTimeoutError):
    """Kaggle kernel exceeded the local wait budget but is still running remotely."""


class KernelFailedError(KaggleBotError):
    """Kaggle kernel completed with a failure status."""

    exit_code = 12
