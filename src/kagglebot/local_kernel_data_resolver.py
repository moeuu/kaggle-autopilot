from __future__ import annotations

import re
from pathlib import Path

from kagglebot import kernel_bootstrap
from kagglebot.role_tokens import ROLE_ALIASES, ROLE_TRAILING_PREFIXES, TEST_DIRECT_ROLE_ALIASES
from kagglebot.submission_sample_discovery import ROLE_SUFFIXES, TABULAR_INPUT_SUFFIXES_ORDERED

KERNEL_DATA_RESOLVER_MARKER = "# kagglebot:data_resolver"
DATA_DIR_LOCATE_FALLBACK_MARKER = "# kagglebot:data-dir-fallback-scan"
TABULAR_SUFFIX_MARKER = "_KB_TABULAR_SUFFIXES"
TABULAR_SUFFIX_FUNCTION_MARKER = "def _kb_tabular_suffix("
PATH_LITERAL_FUNCTION_MARKER = "def _kb_resolve_file_literal("
FILE_MATCH_SCORE_FUNCTION_MARKER = "def _kb_file_match_score("
TABULAR_SUFFIXES = TABULAR_INPUT_SUFFIXES_ORDERED
ROLE_ALIAS_LITERALS = {role: tuple(sorted(aliases)) for role, aliases in sorted(ROLE_ALIASES.items())}
ROLE_SUFFIX_LITERALS = tuple(sorted(ROLE_SUFFIXES))
TEST_DIRECT_ROLE_ALIAS_LITERALS = tuple(sorted(TEST_DIRECT_ROLE_ALIASES))
ROLE_TRAILING_PREFIX_LITERALS = tuple(sorted(ROLE_TRAILING_PREFIXES))

_DATA_DIR_JOIN_RE = re.compile(r"(?<![\w.])(data_dir|data_root)\s*/\s*(['\"])([^'\"]+)\2")
_KAGGLE_INPUT_PATH_CALL_RE = re.compile(
    r"\b(?:Path|_KBPath)\(\s*(?P<quote>['\"])(?P<path>/kaggle/input/[^'\"]+)(?P=quote)\s*\)"
)
_PANDAS_TABULAR_READER_NAMES = (
    "csv",
    "excel",
    "feather",
    "fwf",
    "hdf",
    "html",
    "json",
    "orc",
    "parquet",
    "pickle",
    "sas",
    "spss",
    "stata",
    "table",
    "xml",
)
_KAGGLE_INPUT_READER_LITERAL_RE = re.compile(
    rf"(?P<prefix>\b(?:pd\.)?read_(?:{'|'.join(_PANDAS_TABULAR_READER_NAMES)})\(\s*)"
    r"(?P<quote>['\"])(?P<path>/kaggle/input/[^'\"]+)(?P=quote)"
)
_DATA_DIR_REQUIRED_RE = re.compile(r"all\(\(cand\s*/\s*name\)\.exists\(\)\s*for\s*name\s*in\s*required\)")
_DATA_DIR_RAISE_RE = re.compile(
    r"^\s*raise FileNotFoundError\(f\"Could not find required csv files for slug='\{slug\}'\"\)\s*$",
    re.MULTILINE,
)


def inject_data_dir_resolver(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if not _needs_data_resolver(text):
        return
    lines = text.splitlines()
    if KERNEL_DATA_RESOLVER_MARKER not in text:
        resolver_block = [
            KERNEL_DATA_RESOLVER_MARKER,
            "from pathlib import Path as _KBPath",
            *_canonical_tabular_suffix_lines(),
            *_canonical_find_file_function_lines(),
        ]
        insert_at = kernel_bootstrap.find_bootstrap_block_end(lines)
        if insert_at is None:
            insert_at = kernel_bootstrap.find_bootstrap_insertion_index(lines)
        lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if KERNEL_DATA_RESOLVER_MARKER in updated and (
        TABULAR_SUFFIX_MARKER not in updated
        or TABULAR_SUFFIX_FUNCTION_MARKER not in updated
        or PATH_LITERAL_FUNCTION_MARKER not in updated
        or FILE_MATCH_SCORE_FUNCTION_MARKER not in updated
    ):
        updated = _upgrade_existing_find_file_resolver(updated)
    updated = _DATA_DIR_JOIN_RE.sub(r"_kb_find_file(\1, '\3')", updated)
    updated = _rewrite_kaggle_input_path_calls(updated)
    updated = _rewrite_kaggle_input_reader_literals(updated)
    updated = _DATA_DIR_REQUIRED_RE.sub(
        "all(_kb_find_file(cand, name).exists() for name in required)",
        updated,
    )
    if DATA_DIR_LOCATE_FALLBACK_MARKER not in updated:
        fallback_block = (
            "    input_root = _KBPath('/kaggle/input')\n"
            "    if input_root.exists() and input_root.is_dir():\n"
            f"        {DATA_DIR_LOCATE_FALLBACK_MARKER}\n"
            "        for cand in sorted(input_root.iterdir(), key=lambda p: p.name):\n"
            "            if not cand.is_dir():\n"
            "                continue\n"
            "            if all(_kb_find_file(cand, name).exists() for name in required):\n"
            "                return cand\n"
            "    raise FileNotFoundError(f\"Could not find required data files for slug='{slug}'\")"
        )
        updated = _DATA_DIR_RAISE_RE.sub(fallback_block, updated, count=1)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _needs_data_resolver(text: str) -> bool:
    return (
        _DATA_DIR_JOIN_RE.search(text) is not None
        or _has_rewritable_kaggle_input_path_call(text)
        or _has_rewritable_kaggle_input_reader_literal(text)
    )


def _has_rewritable_kaggle_input_path_call(text: str) -> bool:
    return any(
        _is_rewritable_tabular_literal(match.group("path")) for match in _KAGGLE_INPUT_PATH_CALL_RE.finditer(text)
    )


def _has_rewritable_kaggle_input_reader_literal(text: str) -> bool:
    return any(
        _is_rewritable_tabular_literal(match.group("path")) for match in _KAGGLE_INPUT_READER_LITERAL_RE.finditer(text)
    )


def _rewrite_kaggle_input_path_calls(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        literal = match.group("path")
        if not _is_rewritable_tabular_literal(literal):
            return match.group(0)
        return f"_kb_resolve_file_literal({literal!r})"

    return _KAGGLE_INPUT_PATH_CALL_RE.sub(replace, text)


def _rewrite_kaggle_input_reader_literals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        literal = match.group("path")
        if not _is_rewritable_tabular_literal(literal):
            return match.group(0)
        return f"{match.group('prefix')}_kb_resolve_file_literal({literal!r})"

    return _KAGGLE_INPUT_READER_LITERAL_RE.sub(replace, text)


def _is_rewritable_tabular_literal(value: str) -> bool:
    path = Path(value)
    return path.name != "" and _literal_tabular_suffix(path) in TABULAR_SUFFIXES


def _literal_tabular_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(TABULAR_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _upgrade_existing_find_file_resolver(text: str) -> str:
    lines = text.splitlines()
    upgraded: list[str] = []
    inserted_suffixes = False
    skipping_suffix_prelude = False
    skip_existing_function = False
    for line in lines:
        if line.startswith("_KB_TABULAR_SUFFIXES = "):
            if not inserted_suffixes:
                upgraded.extend(_canonical_tabular_suffix_lines())
                inserted_suffixes = True
            skipping_suffix_prelude = True
            continue
        if skipping_suffix_prelude:
            if line.startswith("def _kb_tabular_suffix(") or line.startswith("def _kb_tabular_stem("):
                skip_existing_function = True
                skipping_suffix_prelude = False
                continue
            skipping_suffix_prelude = False
        if line.startswith("def _kb_find_file("):
            upgraded.extend(_canonical_find_file_function_lines())
            skip_existing_function = True
            continue
        if skip_existing_function:
            if line and (line.startswith(" ") or line.startswith("\t")):
                continue
            skip_existing_function = False
            if not line:
                continue
        upgraded.append(line)
        if not inserted_suffixes and line.strip() == "from pathlib import Path as _KBPath":
            upgraded.extend(_canonical_tabular_suffix_lines())
            inserted_suffixes = True
            continue
    return "\n".join(upgraded)


def _canonical_tabular_suffix_lines() -> list[str]:
    return [
        f"_KB_TABULAR_SUFFIXES = {TABULAR_SUFFIXES!r}",
        "",
        "def _kb_tabular_suffix(path: _KBPath) -> str:",
        "    name = path.name.lower()",
        "    for suffix in sorted(_KB_TABULAR_SUFFIXES, key=len, reverse=True):",
        "        if name.endswith(suffix):",
        "            return suffix",
        "    return path.suffix.lower()",
        "",
        "def _kb_tabular_stem(path: _KBPath) -> str:",
        "    suffix = _kb_tabular_suffix(path)",
        "    name = path.name",
        "    if suffix and name.lower().endswith(suffix):",
        "        return name[: -len(suffix)]",
        "    return path.stem",
        "",
        "def _kb_compact_name(value: str) -> str:",
        "    return ''.join(ch for ch in str(value).lower() if ch.isalnum())",
        "",
        "def _kb_name_tokens(value: str) -> set[str]:",
        "    spaced = []",
        "    previous = ''",
        "    for ch in str(value):",
        "        if ch.isupper() and previous and (previous.islower() or previous.isdigit()):",
        "            spaced.append(' ')",
        "        spaced.append(ch)",
        "        previous = ch",
        "    normalized = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in spaced)",
        "    return {token for token in normalized.split() if token}",
        "",
        f"_KB_ROLE_ALIASES = {ROLE_ALIAS_LITERALS!r}",
        f"_KB_ROLE_SUFFIXES = {ROLE_SUFFIX_LITERALS!r}",
        f"_KB_TEST_DIRECT_ROLE_ALIASES = {TEST_DIRECT_ROLE_ALIAS_LITERALS!r}",
        f"_KB_ROLE_TRAILING_PREFIXES = {ROLE_TRAILING_PREFIX_LITERALS!r}",
        "",
        "def _kb_role_aliases(role: str) -> set[str]:",
        "    return set(_KB_ROLE_ALIASES.get(role, (role,)))",
        "",
        "def _kb_component_mentions_role(value: str, role: str) -> bool:",
        "    tokens = _kb_name_tokens(value)",
        "    aliases = _kb_role_aliases(role)",
        "    train_aliases = _kb_role_aliases('train')",
        "    if role == 'test':",
        "        direct = set(_KB_TEST_DIRECT_ROLE_ALIASES)",
        "        if tokens & direct:",
        "            return True",
        "        if tokens & (aliases - direct) and not tokens & train_aliases:",
        "            return True",
        "    elif tokens & aliases:",
        "        return True",
        "    compact = _kb_compact_name(value)",
        "    if compact in aliases:",
        "        return True",
        "    for alias in aliases:",
        "        if compact.startswith(alias) and compact[len(alias):] in _KB_ROLE_SUFFIXES:",
        "            return True",
        "    if role == 'train' and compact.startswith('train'):",
        "        return compact[len('train'):] in set(_KB_ROLE_SUFFIXES) | {'ing', 'set'}",
        "    if role == 'test' and compact.startswith('test'):",
        "        return compact[len('test'):] in set(_KB_ROLE_SUFFIXES) | {'ing', 'set'}",
        "    if compact.endswith(role):",
        "        return compact[:-len(role)] in _KB_ROLE_TRAILING_PREFIXES",
        "    return False",
        "",
        "def _kb_path_mentions_role(path: _KBPath, role: str) -> bool:",
        "    stem = _kb_tabular_stem(path)",
        "    if _kb_component_mentions_role(stem, role):",
        "        return True",
        "    if role == 'test' and _kb_component_mentions_role(stem, 'train'):",
        "        return False",
        "    aliases = _kb_role_aliases(role)",
        "    return any(str(part).lower() in aliases for part in path.parts)",
        "",
        "def _kb_file_match_score(path: _KBPath, requested_stem_lower: str, requested_compact: str) -> int:",
        "    path_stem = _kb_tabular_stem(path).lower()",
        "    path_compact = _kb_compact_name(path_stem)",
        "    if path_stem == requested_stem_lower:",
        "        return 5",
        "    if path_compact == requested_compact:",
        "        return 4",
        "    if requested_compact == 'samplesubmission' and path_compact in {",
        "        'answertemplate', 'outputtemplate', 'predictiontemplate', 'sampleanswer',",
        "        'sampleoutput', 'sampleprediction', 'samplesolution', 'submissionsample',",
        "    }:",
        "        return 3",
        "    if requested_compact in {'train', 'training'}:",
        "        if _kb_path_mentions_role(path, 'test'):",
        "            return 0",
        "        return 3 if _kb_path_mentions_role(path, 'train') else 0",
        "    if requested_compact in {'test', 'testing'}:",
        "        if _kb_path_mentions_role(path, 'train'):",
        "            return 0",
        "        return 3 if _kb_path_mentions_role(path, 'test') else 0",
        "    return 0",
        "",
        "def _kb_resolve_file_literal(path: str) -> _KBPath:",
        "    requested = _KBPath(path)",
        "    if str(requested).startswith('/kaggle/input/') and _kb_tabular_suffix(requested) in _KB_TABULAR_SUFFIXES:",
        "        return _kb_find_file(requested.parent, requested.name)",
        "    return requested",
        "",
    ]


def _canonical_find_file_function_lines() -> list[str]:
    return [
        "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
        "    candidate = base / name",
        "    if candidate.exists():",
        "        return candidate",
        "    requested_lower = name.lower()",
        "    try:",
        "        direct_case_matches = [path for path in base.iterdir() if path.name.lower() == requested_lower]",
        "    except Exception:",
        "        direct_case_matches = []",
        "    if direct_case_matches:",
        "        return sorted(direct_case_matches, key=lambda p: str(p))[0]",
        "    requested = _KBPath(name)",
        "    requested_suffix = _kb_tabular_suffix(requested)",
        "    requested_stem = _kb_tabular_stem(requested)",
        "    requested_stem_lower = requested_stem.lower()",
        "    requested_compact = _kb_compact_name(requested_stem_lower)",
        "    if requested_suffix in _KB_TABULAR_SUFFIXES:",
        "        for suffix in _KB_TABULAR_SUFFIXES:",
        "            alt = base / f'{requested_stem}{suffix}'",
        "            if alt.exists():",
        "                return alt",
        "    try:",
        "        matches = list(base.rglob(name))",
        "    except Exception:",
        "        matches = []",
        "    if matches:",
        "        return matches[0]",
        "    if requested_suffix in _KB_TABULAR_SUFFIXES:",
        "        try:",
        "            alt_matches = [",
        "                (_kb_file_match_score(path, requested_stem_lower, requested_compact), path)",
        "                for path in base.rglob('*')",
        "                if _kb_tabular_suffix(path) in _KB_TABULAR_SUFFIXES",
        "            ]",
        "        except Exception:",
        "            alt_matches = []",
        "        alt_matches = [(score, path) for score, path in alt_matches if score > 0]",
        "        if alt_matches:",
        "            return max(alt_matches, key=lambda item: (item[0], -len(item[1].parts), str(item[1])))[1]",
        "    return candidate",
        "",
    ]
