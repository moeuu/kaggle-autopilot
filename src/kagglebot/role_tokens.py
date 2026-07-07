from __future__ import annotations

TRAIN_ROLE_ALIASES = frozenset({"train", "training"})
TEST_DIRECT_ROLE_ALIASES = frozenset({"test", "testing"})
TEST_INFERENCE_ROLE_ALIASES = frozenset(
    {
        "blind",
        "challenge",
        "eval",
        "evaluation",
        "final",
        "holdout",
        "inference",
        "leaderboard",
        "predict",
        "prediction",
        "private",
        "public",
        "score",
        "scoring",
        "unlabeled",
        "val",
        "valid",
        "validation",
    }
)
TEST_ROLE_ALIASES = TEST_DIRECT_ROLE_ALIASES | TEST_INFERENCE_ROLE_ALIASES
ROLE_ALIASES = {
    "train": TRAIN_ROLE_ALIASES,
    "test": TEST_ROLE_ALIASES,
}
TEST_INFERENCE_ROLE_TOKENS = TEST_ROLE_ALIASES | frozenset({"submit", "submission"})
ROLE_TRAILING_PREFIXES = frozenset({"x", "public", "private", "final", "eval", "validation", "valid", "val"})
