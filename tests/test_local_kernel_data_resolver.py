from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.local_kernel_data_resolver import inject_data_dir_resolver


def test_inject_data_dir_resolver_rewrites_candidate_presence_check(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def locate_data_dir(slug: str) -> Path:",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:data_resolver" in updated
    assert "_KB_TABULAR_SUFFIXES" in updated
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "_kb_find_file(data_dir, 'test.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated
    assert "for cand in sorted(input_root.iterdir(), key=lambda p: p.name):" in updated


def test_inject_data_dir_resolver_upgrades_existing_marker(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "# kagglebot:data_resolver",
                "from pathlib import Path as _KBPath",
                "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
                "    return base / name",
                "",
                "def locate_data_dir(slug: str):",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir):",
                "    return data_dir / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert updated.count("# kagglebot:data_resolver") == 1
    assert "_KB_TABULAR_SUFFIXES" in updated
    assert "_KB_ROLE_ALIASES" in updated
    assert "_KB_TEST_DIRECT_ROLE_ALIASES" in updated
    assert "requested_stem_lower = requested_stem.lower()" in updated
    assert "_kb_file_match_score(path, requested_stem_lower, requested_compact)" in updated
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated


def test_injected_data_dir_resolver_finds_non_csv_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.tsv").write_text("id\ttarget\n1\t0\n", encoding="utf-8")
    (data_dir / "test.tsv").write_text("id\n2\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.tsv"
    assert test_path == data_dir / "test.tsv"


def test_injected_data_dir_resolver_maps_test_request_to_validation_features(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "validation_features.csv").write_text("id,feature\n2,20\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.csv"
    assert test_path == data_dir / "validation_features.csv"


def test_injected_data_dir_resolver_maps_test_request_to_leaderboard_features(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "leaderboard_features.csv").write_text("id,feature\n2,20\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.csv"
    assert test_path == data_dir / "leaderboard_features.csv"


def test_injected_data_dir_resolver_maps_test_request_to_holdout_parquet(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    pd.DataFrame({"id": [2], "feature": [20]}).to_parquet(data_dir / "holdout_features.parquet", index=False)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.csv"
    assert test_path == data_dir / "holdout_features.parquet"


def test_injected_data_dir_resolver_finds_case_variant_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Train.CSV").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "Test.CSV").write_text("id\n2\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "Train.CSV"
    assert test_path == data_dir / "Test.CSV"


def test_injected_data_dir_resolver_finds_case_variant_alternate_suffix_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "target": [0]}).to_parquet(data_dir / "Train.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_feather(data_dir / "Test.feather")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "Train.parquet"
    assert test_path == data_dir / "Test.feather"


def test_injected_data_dir_resolver_finds_zip_wrapped_binary_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    pd.DataFrame({"id": [1], "target": [0]}).to_parquet(payload, index=False)
    with zipfile.ZipFile(data_dir / "train.parquet.zip", "w") as archive:
        archive.writestr("nested/train.parquet", payload.getvalue())
    (data_dir / "test.csv").write_text("id\n2\n", encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.parquet', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.parquet.zip"
    assert test_path == data_dir / "test.csv"


def test_injected_data_dir_resolver_finds_role_alias_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "target": [0]}).to_parquet(data_dir / "TrainingSet.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_json(data_dir / "PublicTest.jsonl", orient="records", lines=True)
    pd.DataFrame({"id": [2], "target": [0]}).to_excel(data_dir / "SampleSubmission.xlsx", index=False)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return (",
                "        data_dir / 'train.csv',",
                "        data_dir / 'test.csv',",
                "        data_dir / 'sample_submission.csv',",
                "    )",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path, sample_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "TrainingSet.parquet"
    assert test_path == data_dir / "PublicTest.jsonl"
    assert sample_path == data_dir / "SampleSubmission.xlsx"


def test_injected_data_dir_resolver_finds_compressed_and_excel_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0\n")
    pd.DataFrame({"id": [2]}).to_excel(data_dir / "test.xlsx", index=False)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.csv.gz"
    assert test_path == data_dir / "test.xlsx"


def test_injected_data_dir_resolver_finds_zstd_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor()
    (data_dir / "train.csv.zst").write_bytes(compressor.compress(b"id,target\n1,0\n"))
    (data_dir / "test.csv.zst").write_bytes(compressor.compress(b"id\n2\n"))
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.csv.zst"
    assert test_path == data_dir / "test.csv.zst"


def test_injected_data_dir_resolver_finds_sqlite_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.sqlite").write_bytes(b"sqlite placeholder")
    (data_dir / "test.sqlite").write_bytes(b"sqlite placeholder")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(compile(updated, str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert "'.sqlite'" in updated
    assert train_path == data_dir / "train.sqlite"
    assert test_path == data_dir / "test.sqlite"


def test_injected_data_dir_resolver_finds_feather_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "target": [0]}).to_feather(data_dir / "train.feather")
    pd.DataFrame({"id": [2]}).to_feather(data_dir / "test.feather")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.feather"
    assert test_path == data_dir / "test.feather"


def test_injected_data_dir_resolver_finds_orc_and_hdf_tabular_files(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "target": [0]}).to_orc(data_dir / "train.orc", index=False)
    pd.DataFrame({"id": [2]}).to_hdf(data_dir / "test.hdf5", key="data", mode="w", format="table", index=False)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path, test_path = namespace["load_competition_frames"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.orc"
    assert test_path == data_dir / "test.hdf5"


def test_inject_data_dir_resolver_upgrades_old_suffix_marker_without_helpers(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "# kagglebot:data_resolver",
                "from pathlib import Path as _KBPath",
                "_KB_TABULAR_SUFFIXES = ('.csv', '.tsv', '.txt', '.json', '.jsonl', '.parquet')",
                "",
                "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
                "    return base / name",
                "",
                "def load_competition_frames(data_dir):",
                "    return data_dir / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "def _kb_tabular_suffix(" in updated
    assert "'.csv.gz'" in updated
    assert "'.dta'" in updated
    assert "'.xml'" in updated
    assert "'.pkl.gz'" in updated
    assert "requested_stem = _kb_tabular_stem(requested)" in updated


def test_injected_data_dir_resolver_rewrites_data_root_paths(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    data_dir = tmp_path / "demo" / "data"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text('{"id":1,"target":0}\n', encoding="utf-8")
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def load_train(data_root: Path):",
                "    return data_root / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)
    namespace: dict[str, object] = {}
    exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), namespace)
    train_path = namespace["load_train"](data_dir)  # type: ignore[index,operator]

    assert train_path == data_dir / "train.jsonl"


def test_inject_data_dir_resolver_does_not_rewrite_attribute_data_dir(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def sample_paths(data):",
                "    data_dir = data.data_dir",
                "    return data.data_dir / 'sample_submission.csv', data_dir / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "data.data_dir / 'sample_submission.csv'" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "data._kb_find_file" not in updated


def test_inject_data_dir_resolver_rewrites_absolute_kaggle_input_path_calls(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "TRAIN = Path('/kaggle/input/demo/train.csv')",
                "TEST = Path('/kaggle/input/demo/nested/test.csv')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:data_resolver" in updated
    assert "TRAIN = _kb_resolve_file_literal('/kaggle/input/demo/train.csv')" in updated
    assert "TEST = _kb_resolve_file_literal('/kaggle/input/demo/nested/test.csv')" in updated


def test_inject_data_dir_resolver_rewrites_absolute_kaggle_input_reader_literals(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                "train = pd.read_csv('/kaggle/input/demo/train.csv')",
                "test = pd.read_parquet('/kaggle/input/demo/test.parquet')",
                "archive = pd.read_orc('/kaggle/input/demo/archive.orc')",
                "store = pd.read_hdf('/kaggle/input/demo/train.h5')",
                "sample = pd.read_json('/kaggle/input/demo/sample_submission.jsonl', lines=True)",
                "features = pd.read_pickle('/kaggle/input/demo/features.pkl')",
                "metadata = pd.read_stata('/kaggle/input/demo/metadata.dta')",
                "records = pd.read_xml('/kaggle/input/demo/records.xml')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "pd.read_csv(_kb_resolve_file_literal('/kaggle/input/demo/train.csv'))" in updated
    assert "pd.read_parquet(_kb_resolve_file_literal('/kaggle/input/demo/test.parquet'))" in updated
    assert "pd.read_orc(_kb_resolve_file_literal('/kaggle/input/demo/archive.orc'))" in updated
    assert "pd.read_hdf(_kb_resolve_file_literal('/kaggle/input/demo/train.h5'))" in updated
    assert "pd.read_json(_kb_resolve_file_literal('/kaggle/input/demo/sample_submission.jsonl'), lines=True)" in updated
    assert "pd.read_pickle(_kb_resolve_file_literal('/kaggle/input/demo/features.pkl'))" in updated
    assert "pd.read_stata(_kb_resolve_file_literal('/kaggle/input/demo/metadata.dta'))" in updated
    assert "pd.read_xml(_kb_resolve_file_literal('/kaggle/input/demo/records.xml'))" in updated


@pytest.mark.parametrize(
    ("reader", "filename"),
    [
        ("read_html", "sample_submission.html"),
        ("read_fwf", "train.fwf"),
        ("read_sas", "metadata.sas7bdat"),
        ("read_spss", "labels.sav"),
    ],
)
def test_inject_data_dir_resolver_rewrites_extended_absolute_kaggle_input_reader_literals(
    tmp_path: Path,
    reader: str,
    filename: str,
) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                f"frame = pd.{reader}('/kaggle/input/demo/{filename}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inject_data_dir_resolver(kernel_dir)

    updated = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:data_resolver" in updated
    assert f"pd.{reader}(_kb_resolve_file_literal('/kaggle/input/demo/{filename}'))" in updated


def test_inject_data_dir_resolver_does_not_inject_for_non_tabular_absolute_assets(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    original = "from pathlib import Path\nIMAGE = Path('/kaggle/input/demo/images/case_001.png')\n"
    kernel_path.write_text(original, encoding="utf-8")

    inject_data_dir_resolver(kernel_dir)

    assert kernel_path.read_text(encoding="utf-8") == original
