from __future__ import annotations

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_push_validation import (
    extract_invalid_kernel_push_sources,
    raise_for_invalid_kernel_push_sources,
)


def test_extract_invalid_kernel_push_sources_deduplicates_refs() -> None:
    output = "\n".join(
        [
            "The following are not valid dataset sources and could not be added to the kernel: "
            "['alice/missing-dataset', 'alice/missing-dataset']",
            "The following are not valid model sources and could not be added to the kernel: ['bob/model']",
        ]
    )

    assert extract_invalid_kernel_push_sources(output) == {
        "dataset": ["alice/missing-dataset"],
        "model": ["bob/model"],
    }


def test_raise_for_invalid_kernel_push_sources_mentions_metadata_path(tmp_path) -> None:
    output = "The following are not valid kernel sources and could not be added to the kernel: ['user/kernel']"

    with pytest.raises(KernelFailedError, match="kernel-metadata.json"):
        raise_for_invalid_kernel_push_sources(output, kernel_dir=tmp_path / "kernel")
