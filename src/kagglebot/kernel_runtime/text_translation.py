from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

try:
    import sacrebleu
except Exception:
    sacrebleu = None


TARGET_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=(?:[A-Z<\"']))")


@dataclass(frozen=True)
class ConstraintMemories:
    exact_source_memory: dict[str, str]
    entity_memory: dict[str, str]
    quantity_memory: dict[str, str]
    unit_memory: dict[str, str]


@dataclass
class RetrievalModel:
    char_vectorizer: Any
    char_train_matrix: Any
    word_vectorizer: Any | None
    word_train_matrix: Any | None
    train_sources: np.ndarray
    train_targets: np.ndarray
    exact_lookup: dict[str, str]
    default_target: str


def normalize_source_text(text: Any, *, strip_braces: bool = False) -> str:
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKC", value)
    if strip_braces:
        value = value.replace("{", "").replace("}", "")
    return _norm_spaces(value)


def normalize_target_text(text: Any) -> str:
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKC", value)
    return _norm_spaces(value)


def split_target_sentences(text: str) -> list[str]:
    raw = normalize_target_text(text)
    if not raw:
        return [""]
    parts = [segment.strip() for segment in TARGET_SENTENCE_SPLIT_RE.split(raw) if segment.strip()]
    return parts or [raw]


def split_source_for_target_sentences(
    source_text: str,
    target_sentences: Sequence[str],
    *,
    boundary_tokens: Sequence[str] = (),
    lexicon_map: dict[str, str] | None = None,
) -> list[str]:
    tokens = normalize_source_text(source_text).split()
    n_sentences = len(target_sentences)
    if n_sentences <= 1 or len(tokens) < max(4, n_sentences * 2):
        return [_norm_spaces(source_text)]

    lexicon_map = lexicon_map or {}
    source_weights = _weighted_source_tokens(tokens, boundary_tokens=boundary_tokens, lexicon_map=lexicon_map)
    target_weights = [max(1.0, float(len(str(sentence).split()))) for sentence in target_sentences]
    total_source = float(sum(source_weights))
    total_target = float(sum(target_weights))
    n_tokens = len(tokens)
    cumulative_source = np.cumsum(np.asarray(source_weights, dtype=float))
    target_cumulative = np.cumsum(np.asarray(target_weights, dtype=float))
    dp = np.full((n_sentences + 1, n_tokens + 1), np.inf, dtype=float)
    back = np.full((n_sentences + 1, n_tokens + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for sent_idx in range(1, n_sentences + 1):
        remaining = n_sentences - sent_idx
        desired_source = (target_cumulative[sent_idx - 1] / max(1e-6, total_target)) * total_source
        approx_end = int(np.searchsorted(cumulative_source, desired_source, side="left")) + 1
        min_end = sent_idx
        max_end = n_tokens - remaining
        window = max(10, int(n_tokens / max(1, n_sentences)) + 8)
        for end in range(max(min_end, approx_end - window), min(max_end, approx_end + window) + 1):
            for prev_end in range(sent_idx - 1, end):
                prev_cost = dp[sent_idx - 1, prev_end]
                if not np.isfinite(prev_cost):
                    continue
                seg_cost = _segment_alignment_cost(
                    tokens=tokens,
                    token_weights=source_weights,
                    start=prev_end,
                    end=end,
                    target_weight=float(target_weights[sent_idx - 1]),
                    total_source_weight=total_source,
                    total_target_weight=total_target,
                    boundary_tokens=boundary_tokens,
                    lexicon_map=lexicon_map,
                )
                total_cost = prev_cost + seg_cost
                if total_cost < dp[sent_idx, end]:
                    dp[sent_idx, end] = total_cost
                    back[sent_idx, end] = prev_end

    if not np.isfinite(dp[n_sentences, n_tokens]):
        return [_norm_spaces(source_text)]

    boundaries = [n_tokens]
    cursor = n_tokens
    for sent_idx in range(n_sentences, 0, -1):
        prev_end = int(back[sent_idx, cursor])
        if prev_end < 0:
            return [_norm_spaces(source_text)]
        boundaries.append(prev_end)
        cursor = prev_end
    boundaries.reverse()

    segments = [_norm_spaces(" ".join(tokens[start:end])) for start, end in zip(boundaries[:-1], boundaries[1:])]
    return segments if len(segments) == n_sentences and all(segments) else [_norm_spaces(source_text)]


def build_sentence_pairs(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    group_col: str,
    doc_target_col: str | None = None,
    boundary_tokens: Sequence[str] = (),
    lexicon_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lexicon_map = lexicon_map or {}
    doc_target_col = doc_target_col or target_col
    for doc_index, row in enumerate(df.itertuples(index=False), start=1):
        source_text = str(getattr(row, source_col))
        target_text = str(getattr(row, target_col))
        group_value = str(getattr(row, group_col))
        doc_target = str(getattr(row, doc_target_col))
        target_sentences = split_target_sentences(target_text)
        source_sentences = split_source_for_target_sentences(
            source_text,
            target_sentences,
            boundary_tokens=boundary_tokens,
            lexicon_map=lexicon_map,
        )
        if len(source_sentences) != len(target_sentences):
            source_sentences = [source_text]
            target_sentences = [target_text]
        for sentence_index, (source_sentence, target_sentence) in enumerate(zip(source_sentences, target_sentences)):
            rows.append(
                {
                    "pair_id": f"{group_value}__{sentence_index}",
                    "group_id": group_value,
                    "doc_index": int(doc_index),
                    "sentence_index": int(sentence_index),
                    "source": _norm_spaces(source_sentence),
                    "source_norm": _lexicon_normalize_text(source_sentence, lexicon_map),
                    "target": normalize_target_text(target_sentence),
                    "doc_target": normalize_target_text(doc_target),
                }
            )
    if not rows:
        raise RuntimeError("Sentence pair builder produced zero rows.")
    return pd.DataFrame(rows)


def build_exact_memory(
    pair_df: pd.DataFrame,
    *,
    source_col: str = "source_norm",
    target_col: str = "target",
) -> dict[str, str]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for source_text, target_text in zip(pair_df[source_col].tolist(), pair_df[target_col].tolist(), strict=False):
        source_key = _norm_spaces(str(source_text))
        target_value = normalize_target_text(target_text)
        if not source_key or not target_value:
            continue
        counters[source_key][target_value] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counters.items() if counter}


def compute_text_metrics(
    pair_df: pd.DataFrame,
    predictions: Sequence[str],
    *,
    target_col: str = "target",
    group_col: str = "group_id",
    order_col: str = "sentence_index",
    doc_target_col: str = "doc_target",
) -> tuple[dict[str, float], dict[str, float]]:
    references = pair_df[target_col].tolist()
    sentence_metric = compute_bleu_chrf_gmean(references, predictions)
    if not {group_col, order_col, doc_target_col}.issubset(pair_df.columns):
        return sentence_metric, sentence_metric
    compare = pair_df[[group_col, order_col, doc_target_col]].copy()
    compare["prediction"] = [normalize_target_text(x) for x in predictions[: len(compare)]]
    compare = compare.sort_values([group_col, order_col], kind="stable")
    pred_docs = compare.groupby(group_col, sort=False)["prediction"].apply(lambda s: _norm_spaces(" ".join(s))).tolist()
    ref_docs = compare.groupby(group_col, sort=False)[doc_target_col].first().map(normalize_target_text).tolist()
    return sentence_metric, compute_bleu_chrf_gmean(ref_docs, pred_docs)


def compute_bleu_chrf_gmean(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if sacrebleu is not None:
        refs, hyps = _safe_text_pairs(references, predictions)
        try:
            bleu = float(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", use_effective_order=True).score)
            chrfpp = float(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)
            return {"bleu": bleu, "chrfpp": chrfpp, "gmean": float(math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0)))}
        except Exception:
            pass
    bleu = _fallback_corpus_bleu(references, predictions)
    chrfpp = _fallback_corpus_chrfpp(references, predictions)
    return {"bleu": bleu, "chrfpp": chrfpp, "gmean": float(math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0)))}


def candidate_mbr_utility_score(a: str, b: str) -> float:
    metric = compute_bleu_chrf_gmean([a], [b])
    return float(metric["gmean"] - _candidate_style_penalty(a, b))


def select_mbr_candidate(candidates: Sequence[str]) -> str:
    if not candidates:
        return ""
    if len(candidates) == 1:
        return str(candidates[0])
    scored: list[tuple[float, str]] = []
    for i, cand_i in enumerate(candidates):
        sims = [candidate_mbr_utility_score(str(cand_i), str(cand_j)) for j, cand_j in enumerate(candidates) if i != j]
        scored.append((float(np.mean(sims)) if sims else 0.0, str(cand_i)))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return scored[0][1]


def build_retrieval_model(
    train_sources: Sequence[str],
    train_targets: Sequence[str],
    *,
    char_ngram_range: tuple[int, int] = (3, 5),
    char_min_df: int = 1,
    word_weight: float = 0.25,
    word_min_df: int = 1,
) -> RetrievalModel:
    from sklearn.feature_extraction.text import TfidfVectorizer

    source_values = [str(x) for x in train_sources]
    target_values = [str(x) for x in train_targets]
    default_target = Counter(target_values).most_common(1)[0][0] if target_values else ""
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        lowercase=False,
        min_df=max(1, int(char_min_df)),
    )
    char_train_matrix = char_vectorizer.fit_transform(source_values)
    word_vectorizer = None
    word_train_matrix = None
    if word_weight > 0:
        try:
            word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=False,
                min_df=max(1, int(word_min_df)),
            )
            word_train_matrix = word_vectorizer.fit_transform(source_values)
        except ValueError:
            word_vectorizer = None
            word_train_matrix = None
    return RetrievalModel(
        char_vectorizer=char_vectorizer,
        char_train_matrix=char_train_matrix,
        word_vectorizer=word_vectorizer,
        word_train_matrix=word_train_matrix,
        train_sources=np.array(source_values, dtype=object),
        train_targets=np.array(target_values, dtype=object),
        exact_lookup=build_exact_memory(pd.DataFrame({"source_norm": source_values, "target": target_values})),
        default_target=default_target,
    )


def retrieval_candidate_pools(
    model: RetrievalModel,
    infer_sources: Sequence[str],
    *,
    k: int = 8,
    min_score: float = 0.1,
    max_candidates: int = 4,
    word_weight: float = 0.25,
) -> tuple[list[list[tuple[str, str, int]]], int]:
    from sklearn.metrics.pairwise import linear_kernel

    infer_values = [str(x) for x in infer_sources]
    if len(model.train_targets) == 0:
        return [[] for _ in infer_values], len(infer_values)
    char_infer_matrix = model.char_vectorizer.transform(infer_values)
    char_scores = linear_kernel(char_infer_matrix, model.char_train_matrix)
    word_scores = None
    combined_scores = char_scores.copy()
    if model.word_vectorizer is not None and model.word_train_matrix is not None and word_weight > 0:
        word_infer_matrix = model.word_vectorizer.transform(infer_values)
        word_scores = word_weight * linear_kernel(word_infer_matrix, model.word_train_matrix)
        combined_scores = combined_scores + word_scores
    pools: list[list[tuple[str, str, int]]] = []
    low_score_count = 0
    top_k = max(1, min(int(k), len(model.train_targets)))
    for row_idx, row_scores in enumerate(combined_scores):
        exact_target = model.exact_lookup.get(_norm_spaces(infer_values[row_idx]))
        if exact_target:
            pools.append([(exact_target, "retrieval_exact", 1)])
            continue
        idxs = np.argpartition(-row_scores, top_k - 1)[:top_k] if top_k < row_scores.size else np.argsort(-row_scores)
        ranked: list[tuple[float, int]] = []
        for idx in idxs.tolist():
            char_score = float(char_scores[row_idx, idx]) if char_scores.size else 0.0
            word_score = float(word_scores[row_idx, idx]) if word_scores is not None else 0.0
            lexical = _lexical_overlap_score(infer_values[row_idx], str(model.train_sources[idx]))
            edit_ratio = float(SequenceMatcher(a=infer_values[row_idx], b=str(model.train_sources[idx])).ratio())
            total_score = char_score + word_score + 0.9 * lexical + 0.6 * edit_ratio
            ranked.append((total_score, int(idx)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or ranked[0][0] < min_score:
            pools.append([])
            low_score_count += 1
            continue
        row_pool: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for rank, (_score, idx) in enumerate(ranked, start=1):
            candidate = str(model.train_targets[idx])
            if candidate in seen:
                continue
            seen.add(candidate)
            row_pool.append((candidate, "retrieval", rank))
            if len(row_pool) >= max(1, int(max_candidates)):
                break
        pools.append(row_pool)
    return pools, low_score_count


def apply_consistency_postprocess(
    source_texts: Sequence[str],
    predictions: Sequence[str],
    *,
    group_values: Sequence[str] | None = None,
    exact_source_memory: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    outputs = [normalize_target_text(x) for x in predictions]
    stats = {"memory_rewrites": 0, "consistency_rewrites": 0}
    if exact_source_memory:
        for idx, source_text in enumerate(source_texts):
            candidate = exact_source_memory.get(_norm_spaces(str(source_text)))
            if candidate and outputs[idx] != candidate:
                outputs[idx] = candidate
                stats["memory_rewrites"] += 1
    if group_values is None:
        return outputs, stats
    group_map: dict[str, Counter[str]] = defaultdict(Counter)
    for source_text, prediction, group in zip(source_texts, outputs, group_values, strict=False):
        if not str(group).strip():
            continue
        group_map[str(group)][_norm_spaces(str(source_text)) + "||" + prediction] += 1
    canonical: dict[tuple[str, str], str] = {}
    for group, counter in group_map.items():
        source_counters: dict[str, Counter[str]] = defaultdict(Counter)
        for key, count in counter.items():
            source_text, prediction = key.split("||", 1)
            source_counters[source_text][prediction] += count
        for source_text, source_counter in source_counters.items():
            canonical[(group, source_text)] = source_counter.most_common(1)[0][0]
    for idx, (source_text, group) in enumerate(zip(source_texts, group_values, strict=False)):
        source_key = _norm_spaces(str(source_text))
        group_key = str(group)
        canonical_value = canonical.get((group_key, source_key))
        if canonical_value and outputs[idx] != canonical_value:
            outputs[idx] = canonical_value
            stats["consistency_rewrites"] += 1
    return outputs, stats


def build_constraint_memories(
    *,
    exact_source_memory: dict[str, str] | None = None,
    entity_entries: Iterable[tuple[str, str]] = (),
    quantity_entries: Iterable[tuple[str, str]] = (),
    unit_entries: Iterable[tuple[str, str]] = (),
    min_freq: int = 1,
    min_share: float = 0.6,
) -> ConstraintMemories:
    return ConstraintMemories(
        exact_source_memory=exact_source_memory or {},
        entity_memory=_build_counter_memory(entity_entries, min_freq=min_freq, min_share=min_share),
        quantity_memory=_build_counter_memory(quantity_entries, min_freq=min_freq, min_share=min_share),
        unit_memory=_build_counter_memory(unit_entries, min_freq=min_freq, min_share=min_share),
    )


def _build_counter_memory(
    entries: Iterable[tuple[str, str]],
    *,
    min_freq: int,
    min_share: float,
) -> dict[str, str]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for source_key, target_value in entries:
        source_norm = _norm_spaces(str(source_key))
        target_norm = normalize_target_text(target_value)
        if not source_norm or not target_norm:
            continue
        counters[source_norm][target_norm] += 1
    out: dict[str, str] = {}
    for key, counter in counters.items():
        top_value, top_count = counter.most_common(1)[0]
        total = int(sum(counter.values()))
        if top_count >= max(1, min_freq) and total > 0 and (top_count / total) >= min_share:
            out[key] = top_value
    return out


def _safe_text_pairs(references: Sequence[str], predictions: Sequence[str]) -> tuple[list[str], list[str]]:
    n = min(len(references), len(predictions))
    return [str(x) for x in references[:n]], [str(x) for x in predictions[:n]]


def _fallback_corpus_bleu(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs, hyps = _safe_text_pairs(references, predictions)
    if not refs or not hyps:
        return 0.0
    match = 0
    total = 0
    for ref, hyp in zip(refs, hyps, strict=False):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        overlap = Counter(ref_tokens) & Counter(hyp_tokens)
        match += int(sum(overlap.values()))
        total += max(1, len(hyp_tokens))
    return float(100.0 * (match / max(1, total)))


def _fallback_corpus_chrfpp(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs, hyps = _safe_text_pairs(references, predictions)
    if not refs or not hyps:
        return 0.0
    scores = [SequenceMatcher(a=ref, b=hyp).ratio() for ref, hyp in zip(refs, hyps, strict=False)]
    return float(100.0 * (sum(scores) / max(1, len(scores))))


def _candidate_style_penalty(a: str, b: str) -> float:
    len_a = max(1, len(normalize_target_text(a).split()))
    len_b = max(1, len(normalize_target_text(b).split()))
    return float(abs(len_a - len_b) / max(len_a, len_b))


def _lexicon_normalize_text(text: str, lexicon_map: dict[str, str]) -> str:
    tokens = [lexicon_map.get(token, token) for token in normalize_source_text(text).split()]
    return _norm_spaces(" ".join(tokens))


def _weighted_source_tokens(
    tokens: Sequence[str],
    *,
    boundary_tokens: Sequence[str],
    lexicon_map: dict[str, str],
) -> list[float]:
    boundary_set = {str(token) for token in boundary_tokens}
    weights: list[float] = []
    for token in tokens:
        weight = 1.0
        if token in boundary_set:
            weight += 1.5
        if lexicon_map.get(token) != token:
            weight += 0.35
        weights.append(weight)
    return weights


def _segment_alignment_cost(
    *,
    tokens: Sequence[str],
    token_weights: Sequence[float],
    start: int,
    end: int,
    target_weight: float,
    total_source_weight: float,
    total_target_weight: float,
    boundary_tokens: Sequence[str],
    lexicon_map: dict[str, str],
) -> float:
    if end <= start:
        return float("inf")
    seg_weight = float(sum(token_weights[start:end]))
    expected_weight = max(1.0, (target_weight / max(1e-6, total_target_weight)) * total_source_weight)
    length_penalty = abs(seg_weight - expected_weight) / expected_weight
    boundary_bonus = 0.0
    boundary_set = {str(token) for token in boundary_tokens}
    if start > 0 and tokens[start] in boundary_set:
        boundary_bonus += 1.0
    if end < len(tokens) and tokens[end - 1] in boundary_set:
        boundary_bonus += 1.0
    if any(lexicon_map.get(token) != token for token in tokens[start:end]):
        boundary_bonus += 0.25
    return float(length_penalty - 0.2 * boundary_bonus)


def _lexical_overlap_score(a: str, b: str) -> float:
    a_tokens = set(normalize_source_text(a).split())
    b_tokens = set(normalize_source_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return float(len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens)))


def _norm_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
