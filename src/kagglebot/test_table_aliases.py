from __future__ import annotations

from kagglebot.role_tokens import TEST_DIRECT_ROLE_ALIASES, TEST_INFERENCE_ROLE_ALIASES, TRAIN_ROLE_ALIASES

TEST_TABLE_STEMS = (
    "test",
    "testing",
    "test_features",
    "test_metadata",
    "eval",
    "evaluation",
    "eval_features",
    "evaluation_features",
    "validation",
    "valid",
    "val",
    "validation_features",
    "valid_features",
    "val_features",
    "holdout",
    "holdout_features",
    "scoring",
    "score",
    "scoring_features",
    "score_features",
    "predict",
    "prediction",
    "predict_features",
    "prediction_features",
    "unlabeled",
    "unlabeled_features",
    "public_test",
    "private_test",
    "public",
    "private",
    "leaderboard",
    "inference",
)

TEST_TABLE_COMPACT_ALIASES = frozenset(value.replace("_", "") for value in TEST_TABLE_STEMS)
TEST_TABLE_EXCLUDE_TOKENS = TRAIN_ROLE_ALIASES | frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
    }
)
STRONG_TEST_TABLE_TOKENS = TEST_DIRECT_ROLE_ALIASES | frozenset(
    {
        "eval",
        "evaluation",
        "validation",
        "valid",
        "val",
        "scoring",
        "score",
    }
)
WEAK_TEST_TABLE_TOKENS = TEST_INFERENCE_ROLE_ALIASES - STRONG_TEST_TABLE_TOKENS
