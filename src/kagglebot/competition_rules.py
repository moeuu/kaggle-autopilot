from __future__ import annotations

import re
from dataclasses import dataclass

from kagglebot.paths import CompetitionPaths

_NUMBER_WORD_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_NOTEBOOK_SUBMISSION_ONLY_RULE_PATTERNS = (
    r"submissions?\s+to\s+this\s+competition\s+must\s+be\s+made\s+through\s+notebooks?",
    r"submissions?\s+must\s+be\s+made\s+through\s+notebooks?",
    r"only\s+accepts?\s+submissions?\s+from\s+notebooks?",
)
_INTERNET_DISABLED_RULE_PATTERNS = (
    r"internet\s+access\s+disabled",
    r"internet\s+access\s+(?:is\s+)?(?:disabled|disallowed|not\s+allowed|forbidden|prohibited|unavailable)",
    r"cannot\s+use\s+internet\s+access(?:\s+in\s+this\s+competition)?",
    r"notebooks?\s+cannot\s+use\s+internet\s+access(?:\s+in\s+this\s+competition)?",
    r"disable\s+internet\s+in\s+the\s+notebook\s+editor",
    r"turn\s+off\s+internet(?:\s+access)?",
    r"internet\s+must\s+be\s+(?:disabled|off)",
    r"no\s+internet\s+access",
    r"enable[_\s-]?internet\s*[:=]\s*false",
)
_UNRESTRICTED_SUBMISSION_ATTEMPT_PATTERNS = (
    r"submit\s+without\s+restriction\s+as\s+to\s+the\s+number\s+of\s+attempts",
    r"submissions?\s+(?:are\s+)?unlimited",
    r"unlimited\s+submissions?",
    r"no\s+limit\s+on\s+(?:the\s+number\s+of\s+)?submissions?",
)


@dataclass(frozen=True)
class CompetitionRuleConstraints:
    notebook_submissions_only: bool = False
    internet_must_be_off: bool = False
    submission_limit_detected: bool = False
    submission_limit_per_day: int | None = None
    cpu_runtime_limit_min: int | None = None
    gpu_runtime_limit_min: int | None = None


def matches_any_rule_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def extract_submission_limit_per_day(lowered_rules_text: str) -> int | None:
    """Extract an explicit daily or rolling-24h submission limit from rules text."""
    candidates: list[int] = []
    normalized_rules_text = re.sub(r"[*_`]+", " ", lowered_rules_text)
    normalized_rules_text = re.sub(r"\s+", " ", normalized_rules_text)

    for match in re.finditer(r"\((\d+)\)\W+submissions?\s+per\s+day", normalized_rules_text):
        candidates.append(int(match.group(1)))

    numeric_patterns = (
        r"\b(\d+)\s+submissions?\s+per\s+day\b",
        r"\bmaximum\s+of\s+(\d+)\s+submissions?\s+per\s+day\b",
        r"\b(\d+)\s+submissions?\s+within\s+24\s+hours?\b",
        r"\b(\d+)\s+submissions?\s+per\s+24\s*h(?:ours?)?\s*(?:interval|window)?\b",
        r"\b(\d+)\s+submissions?\s+per\s+24\s+hours?\s*(?:interval|window)?\b",
    )
    for pattern in numeric_patterns:
        for match in re.finditer(pattern, normalized_rules_text):
            candidates.append(int(match.group(1)))

    word_patterns = (
        r"\b([a-z]+)(?:\s+\(\d+\))?\s+submissions?\s+per\s+day\b",
        r"\bmaximum\s+of\s+([a-z]+)(?:\s+\(\d+\))?\s+submissions?\s+per\s+day\b",
        r"\b([a-z]+)\s+submissions?\s+within\s+24\s+hours?\b",
        r"\b([a-z]+)\s+submissions?\s+per\s+24\s*h(?:ours?)?\s*(?:interval|window)?\b",
        r"\b([a-z]+)\s+submissions?\s+per\s+24\s+hours?\s*(?:interval|window)?\b",
    )
    for pattern in word_patterns:
        for match in re.finditer(pattern, normalized_rules_text):
            number_word = match.group(1)
            if number_word in _NUMBER_WORD_TO_INT:
                candidates.append(_NUMBER_WORD_TO_INT[number_word])

    for match in re.finditer(
        r"submission\s+limit\s+is\s+(\d+)\s+submissions?\s+within\s+24\s+hours?", normalized_rules_text
    ):
        candidates.append(int(match.group(1)))

    for match in re.finditer(
        r"submission\s+limit\s+is\s+([a-z]+)\s+submissions?\s+within\s+24\s+hours?",
        normalized_rules_text,
    ):
        number_word = match.group(1)
        if number_word in _NUMBER_WORD_TO_INT:
            candidates.append(_NUMBER_WORD_TO_INT[number_word])

    positive = [value for value in candidates if value > 0]
    if not positive:
        return None
    return min(positive)


def load_competition_rule_constraints(paths: CompetitionPaths) -> CompetitionRuleConstraints:
    text_parts: list[str] = []
    for path in (paths.rules_md_path, paths.rules_html_path):
        if not path.exists():
            continue
        try:
            text_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    if not text_parts:
        return CompetitionRuleConstraints()
    text = "\n".join(text_parts)
    lowered = text.lower()

    notebook_submissions_only = matches_any_rule_pattern(lowered, _NOTEBOOK_SUBMISSION_ONLY_RULE_PATTERNS)
    internet_must_be_off = matches_any_rule_pattern(lowered, _INTERNET_DISABLED_RULE_PATTERNS)
    submission_limit_per_day = extract_submission_limit_per_day(lowered)
    unrestricted_submission_attempts = submission_limit_per_day is None and matches_any_rule_pattern(
        lowered, _UNRESTRICTED_SUBMISSION_ATTEMPT_PATTERNS
    )
    submission_limit_detected = (not unrestricted_submission_attempts) and bool(
        re.search(r"maximum\s+number\s+of\s+submissions", lowered)
        or re.search(r"submission\s+limit", lowered)
        or re.search(r"\bmax(?:imum)?\s+submissions?\b", lowered)
        or re.search(r"\b\d+\s+submissions?\s+per\s+day\b", lowered)
        or re.search(r"\bdaily\s+submissions?\b", lowered)
        or submission_limit_per_day is not None
    )

    cpu_runtime_limit_min: int | None = None
    gpu_runtime_limit_min: int | None = None
    for match in re.finditer(
        r"\b(cpu|gpu)\s+notebook\s*<=?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b",
        lowered,
    ):
        device = match.group(1)
        hours = float(match.group(2))
        minutes = max(1, int(round(hours * 60)))
        if device == "cpu":
            cpu_runtime_limit_min = minutes if cpu_runtime_limit_min is None else min(cpu_runtime_limit_min, minutes)
        else:
            gpu_runtime_limit_min = minutes if gpu_runtime_limit_min is None else min(gpu_runtime_limit_min, minutes)

    return CompetitionRuleConstraints(
        notebook_submissions_only=notebook_submissions_only,
        internet_must_be_off=internet_must_be_off,
        submission_limit_detected=submission_limit_detected,
        submission_limit_per_day=submission_limit_per_day,
        cpu_runtime_limit_min=cpu_runtime_limit_min,
        gpu_runtime_limit_min=gpu_runtime_limit_min,
    )


def runtime_limit_for_compute(*, constraints: CompetitionRuleConstraints, compute: str) -> int | None:
    normalized = str(compute or "").strip().lower()
    if normalized == "kaggle_gpu":
        return constraints.gpu_runtime_limit_min or constraints.cpu_runtime_limit_min
    if normalized == "kaggle_tpu":
        limits = [value for value in (constraints.gpu_runtime_limit_min, constraints.cpu_runtime_limit_min) if value]
        if limits:
            return min(limits)
        return None
    return None
