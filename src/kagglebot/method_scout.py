from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.scalar_utils import non_nan_float as _to_float

METHOD_REGISTRY_FILENAME = "method_registry.json"
METHOD_SCOUT_QUERIES_FILENAME = "method_scout_queries.json"
SOURCE_REGISTRY_FILENAME = "source_registry.json"
VALIDATION_REGISTRY_FILENAME = "validation_registry.json"

MethodScoutMode = str
ResearchScoutMode = str

_SOURCE_PRIORITY = {
    "competition_specific": 1.0,
    "official_repo": 0.9,
    "paper": 0.85,
    "official_docs": 0.75,
    "similar_competition": 0.65,
    "discussion": 0.6,
    "blog_writeup": 0.45,
    "generic": 0.35,
}
_UNSAFE_TERMS = (
    "test label lookup",
    "test-label lookup",
    "known-label",
    "oracle",
    "leaderboard proxy",
    "lb proxy",
    "exact test match",
    "exact matching on test",
    "submit spam",
    "submission spam",
    "multi-account",
    "accept rules automatically",
    "bypass",
)
_VALIDATION_TERMS = ("split", "cv", "cross-validation", "groupkfold", "timeseries", "leak", "proxy")


@dataclass(frozen=True)
class MethodCandidate:
    method_id: str
    name: str
    source_ids: list[str]
    source_type: str
    candidate_category: str
    problem_fit: float
    metric_fit: float
    data_fit: float
    expected_gain: float
    implementation_cost: float
    dependency_risk: float
    leakage_risk: float
    runtime_risk: float
    status: str = "active"
    blocked_reason: str | None = None
    fallback: str | None = None
    summary: str = ""
    implementation_adapter: dict[str, object] = field(default_factory=dict)
    dependency_check: dict[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def score(self) -> float:
        return (
            (0.30 * self.expected_gain)
            + (0.20 * self.problem_fit)
            + (0.15 * self.metric_fit)
            + (0.15 * self.data_fit)
            - (0.08 * self.implementation_cost)
            - (0.05 * self.dependency_risk)
            - (0.05 * self.runtime_risk)
            - (0.02 * self.leakage_risk)
        )

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rank_score"] = round(self.score(), 6)
        return payload


def normalize_method_scout_mode(value: str | None) -> MethodScoutMode:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "off", "refresh"}:
        return normalized
    raise ValueError("method_scout must be one of: auto, off, refresh")


def normalize_research_scout_mode(value: str | None) -> ResearchScoutMode:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "off", "refresh"}:
        return normalized
    raise ValueError("research_scout must be one of: auto, off, refresh")


def method_registry_path(context_dir: Path) -> Path:
    return context_dir / METHOD_REGISTRY_FILENAME


def method_scout_queries_path(context_dir: Path) -> Path:
    return context_dir / METHOD_SCOUT_QUERIES_FILENAME


def source_registry_path(context_dir: Path) -> Path:
    return context_dir / SOURCE_REGISTRY_FILENAME


def validation_registry_path(context_dir: Path) -> Path:
    return context_dir / VALIDATION_REGISTRY_FILENAME


def run_method_scout(
    *,
    paths: CompetitionPaths,
    slug: str,
    problem_types: list[str],
    dataset_profile: dict[str, object],
    metric: str | None,
    campaign_state: dict[str, object] | None = None,
    mode: str | None = "auto",
    research_mode: str | None = "auto",
    max_sources: int = 12,
) -> dict[str, object]:
    scout_mode = normalize_method_scout_mode(mode)
    source_mode = normalize_research_scout_mode(research_mode)
    registry_path = method_registry_path(paths.context_dir)
    if scout_mode == "off":
        return _load_existing_registry(registry_path, slug=slug, mode=scout_mode)

    queries = build_method_scout_queries(
        slug=slug,
        problem_types=problem_types,
        dataset_profile=dataset_profile,
        metric=metric,
        campaign_state=campaign_state,
        max_sources=max_sources,
    )
    raw_sources = (
        []
        if source_mode == "off"
        else load_research_sources(paths.context_dir / "research_sources.jsonl", limit=max_sources)
    )
    source_registry = build_source_registry(
        slug=slug,
        queries=queries,
        raw_sources=raw_sources,
        mode=source_mode,
        max_sources=max_sources,
    )
    sources = _active_sources_from_registry(source_registry)
    methods = build_method_candidates(
        slug=slug,
        problem_types=problem_types,
        dataset_profile=dataset_profile,
        metric=metric,
        sources=sources,
        campaign_state=campaign_state,
    )
    validation_registry = build_validation_registry(
        slug=slug,
        problem_types=problem_types,
        campaign_state=campaign_state,
        sources=sources,
    )
    active_methods = [method for method in methods if method.status == "active"]
    blocked_methods = [method for method in methods if method.status == "blocked"]
    payload = {
        "version": 1,
        "slug": slug,
        "mode": scout_mode,
        "research_mode": source_mode,
        "updated_at": datetime.now(UTC).isoformat(),
        "problem_types": problem_types,
        "metric": metric,
        "modality": infer_modality(problem_types, dataset_profile),
        "method_scout_queries": queries,
        "source_count": len(sources),
        "source_registry_path": str(source_registry_path(paths.context_dir)),
        "source_types": _count_source_types(sources, slug=slug),
        "active_method_ids": [method.method_id for method in active_methods[:8]],
        "blocked_method_ids": [method.method_id for method in blocked_methods],
        "validation_priority": bool(validation_registry.get("priority")),
        "active_validation_profile": validation_registry.get("active_profile"),
        "methods": [method.to_payload() for method in methods],
    }
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    method_scout_queries_path(paths.context_dir).write_text(
        json.dumps({"version": 1, "slug": slug, "queries": queries}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    source_registry_path(paths.context_dir).write_text(
        json.dumps(source_registry, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    validation_registry_path(paths.context_dir).write_text(
        json.dumps(validation_registry, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def build_source_registry(
    *,
    slug: str,
    queries: list[dict[str, object]],
    raw_sources: list[dict[str, object]],
    mode: str | None,
    max_sources: int,
) -> dict[str, object]:
    source_mode = normalize_research_scout_mode(mode)
    records: list[dict[str, object]] = []
    for index, source in enumerate(raw_sources[: max(1, int(max_sources))]):
        text = _source_text(source)
        blocked_reason = unsafe_method_reason(text)
        source_id = _source_id(source, index)
        records.append(
            {
                "source_id": source_id,
                "status": "blocked" if blocked_reason else "active",
                "blocked_reason": blocked_reason,
                "source_type": classify_source(source, slug=slug),
                "url": source.get("url"),
                "title": source.get("title"),
                "query": source.get("query"),
                "takeaway": source.get("takeaway"),
                "extracted_technique": source.get("extracted_technique"),
                "attribution": _source_attribution(source, source_id=source_id, slug=slug),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
    if source_mode != "off":
        for index, query in enumerate(queries[: max(1, int(max_sources))], start=len(records) + 1):
            records.append(
                {
                    "source_id": f"planned-query-{index}",
                    "status": "planned",
                    "blocked_reason": None,
                    "source_type": "planned_query",
                    "url": None,
                    "title": str(query.get("query") or "")[:160],
                    "query": query.get("query"),
                    "takeaway": (
                        "Use this query to retrieve competition-specific notebooks, papers, repos, and writeups."
                    ),
                    "extracted_technique": None,
                    "attribution": {"source_id": f"planned-query-{index}", "kind": "query"},
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
    return {
        "version": 1,
        "slug": slug,
        "mode": source_mode,
        "updated_at": datetime.now(UTC).isoformat(),
        "source_count": len(records),
        "active_source_ids": [str(item["source_id"]) for item in records if item.get("status") == "active"],
        "blocked_source_ids": [str(item["source_id"]) for item in records if item.get("status") == "blocked"],
        "planned_query_ids": [str(item["source_id"]) for item in records if item.get("status") == "planned"],
        "sources": records,
    }


def build_method_scout_queries(
    *,
    slug: str,
    problem_types: list[str],
    dataset_profile: dict[str, object],
    metric: str | None,
    campaign_state: dict[str, object] | None = None,
    max_sources: int = 12,
) -> list[dict[str, object]]:
    modality = infer_modality(problem_types, dataset_profile)
    domain_terms = _domain_terms(slug=slug, dataset_profile=dataset_profile, problem_types=problem_types)
    metric_text = str(metric or dataset_profile.get("target_metric") or "metric").strip() or "metric"
    validation_first = _campaign_needs_validation_redesign(campaign_state)
    raw_queries = [
        f"{slug} Kaggle winning solution discussion notebook",
        f"{slug} Kaggle top solution {metric_text}",
        f"{' '.join(domain_terms)} {modality} {metric_text} state of the art 2025 arxiv",
        f"{' '.join(domain_terms)} {modality} official baseline GitHub",
        f"{metric_text} {modality} validation split leakage Kaggle",
        f"{modality} {metric_text} ensemble blending stacking 2025",
    ]
    if validation_first:
        raw_queries.insert(0, f"{slug} validation split public leaderboard mismatch leakage")
        raw_queries.insert(1, f"{' '.join(domain_terms)} group time proxy split validation Kaggle")
    if modality == "tabular":
        raw_queries.extend(
            [
                f"{slug} CatBoost XGBoost LightGBM TabPFN TabM TabICL",
                f"tabular foundation model {metric_text} 2025 TabPFN TabICL TabM",
            ]
        )
    elif modality == "image":
        raw_queries.extend(
            [
                f"{slug} timm ConvNeXt ViT Swin augmentation TTA",
                f"{' '.join(domain_terms)} image classification detection segmentation 2025 arxiv",
            ]
        )
    elif modality == "text":
        raw_queries.extend(
            [
                f"{slug} transformers embedding reranker calibration",
                f"{' '.join(domain_terms)} NLP {metric_text} 2025 arxiv official repo",
            ]
        )
    elif modality == "timeseries":
        raw_queries.extend(
            [
                f"{slug} time series forecasting winning solution lag rolling features",
                f"time series foundation model {metric_text} Chronos TFT 2025 arxiv",
            ]
        )
    elif modality in {"rna", "bio"}:
        raw_queries.extend(
            [
                f"{slug} official evaluator RNA protein structure baseline GitHub",
                f"{' '.join(domain_terms)} bioinformatics {metric_text} 2025 arxiv",
            ]
        )
    seen: set[str] = set()
    queries: list[dict[str, object]] = []
    for query in raw_queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(
            {
                "query": normalized,
                "purpose": "validation_redesign"
                if any(term in key for term in _VALIDATION_TERMS)
                else "method_discovery",
                "modality": modality,
                "metric": metric_text,
            }
        )
        if len(queries) >= max(1, int(max_sources)):
            break
    return queries


def load_research_sources(path: Path, *, limit: int = 12) -> list[dict[str, object]]:
    if not path.exists():
        return []
    sources: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            sources.append(payload)
        if len(sources) >= max(1, int(limit)):
            break
    return sources


def _active_sources_from_registry(registry: dict[str, object]) -> list[dict[str, object]]:
    raw = registry.get("sources")
    if not isinstance(raw, list):
        return []
    sources: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("status") != "active":
            continue
        sources.append(
            {
                "url": item.get("url"),
                "title": item.get("title"),
                "query": item.get("query"),
                "takeaway": item.get("takeaway"),
                "extracted_technique": item.get("extracted_technique"),
                "source_id": item.get("source_id"),
            }
        )
    return sources


def build_method_candidates(
    *,
    slug: str,
    problem_types: list[str],
    dataset_profile: dict[str, object],
    metric: str | None,
    sources: list[dict[str, object]],
    campaign_state: dict[str, object] | None = None,
) -> list[MethodCandidate]:
    modality = infer_modality(problem_types, dataset_profile)
    methods = _seed_methods(modality=modality, metric=metric)
    for index, source in enumerate(sources):
        source_id = _source_id(source, index)
        source_type = classify_source(source, slug=slug)
        text = _source_text(source)
        blocked_reason = unsafe_method_reason(text)
        category = infer_candidate_category(text)
        source_priority = _SOURCE_PRIORITY.get(source_type, 0.35)
        methods.append(
            MethodCandidate(
                method_id=_method_id(f"{source_type}-{source.get('title') or source_id}"),
                name=str(source.get("title") or source.get("url") or source_id)[:120],
                source_ids=[source_id],
                source_type=source_type,
                candidate_category=category,
                problem_fit=_fit_score(text=text, modality=modality, source_priority=source_priority),
                metric_fit=_metric_fit(text=text, metric=metric),
                data_fit=_data_fit(text=text, dataset_profile=dataset_profile),
                expected_gain=min(1.0, 0.45 + (0.45 * source_priority)),
                implementation_cost=_implementation_cost(text),
                dependency_risk=_dependency_risk(text),
                leakage_risk=1.0 if blocked_reason else _leakage_risk(text),
                runtime_risk=_runtime_risk(text),
                status="blocked" if blocked_reason else "active",
                blocked_reason=blocked_reason,
                fallback=_fallback_for_category(category, modality=modality),
                summary=str(source.get("takeaway") or source.get("extracted_technique") or "")[:500],
                implementation_adapter=_implementation_adapter(category=category, modality=modality, text=text),
                dependency_check=_dependency_check(text=text, modality=modality),
            )
        )
    if _campaign_needs_validation_redesign(campaign_state):
        methods.append(
            MethodCandidate(
                method_id="validation-redesign-from-public-regression",
                name="Validation redesign from public-regression signal",
                source_ids=["campaign_state"],
                source_type="competition_specific",
                candidate_category="validation_variant",
                problem_fit=1.0,
                metric_fit=0.9,
                data_fit=0.9,
                expected_gain=0.85,
                implementation_cost=0.35,
                dependency_risk=0.05,
                leakage_risk=0.05,
                runtime_risk=0.2,
                fallback="Run grouped/time/proxy split search before adding new model families.",
                summary=(
                    "Historical/champion public baseline is better than the latest public score; "
                    "prioritize split redesign."
                ),
                implementation_adapter=_implementation_adapter(
                    category="validation_variant",
                    modality=modality,
                    text="validation redesign split search",
                ),
                dependency_check={"required": [], "optional": [], "fallback": "Use sklearn splitters and local data."},
            )
        )
    return _dedupe_and_rank_methods(methods)


def build_validation_registry(
    *,
    slug: str,
    problem_types: list[str],
    campaign_state: dict[str, object] | None,
    sources: list[dict[str, object]],
) -> dict[str, object]:
    priority = _campaign_needs_validation_redesign(campaign_state)
    modality = infer_modality(problem_types, {})
    offline_online_correlation = (campaign_state or {}).get("offline_online_correlation")
    latest = _to_float((campaign_state or {}).get("latest_submission_score"))
    champion = _to_float(
        (campaign_state or {}).get("champion_score") or (campaign_state or {}).get("historical_best_score")
    )
    direction = str((campaign_state or {}).get("direction") or "minimize").lower()
    public_regression = bool(
        latest is not None
        and champion is not None
        and ((direction == "maximize" and latest < champion) or (direction != "maximize" and latest > champion))
    )
    profiles = [
        {
            "profile_id": "default_cv",
            "split_family": "default",
            "reason": "Existing evaluation protocol baseline.",
            "priority": 0.3,
            "run_status": "baseline",
            "offline_online_correlation": offline_online_correlation,
            "adoption_status": "fallback",
        },
        {
            "profile_id": "group_or_proxy_cv",
            "split_family": "group_or_proxy",
            "reason": (
                "Use group-like, entity, source, or proxy columns when public regression suggests split mismatch."
            ),
            "priority": 0.9 if priority else 0.55,
            "run_status": "planned",
            "offline_online_correlation": None,
            "adoption_status": "candidate",
        },
        {
            "profile_id": "time_aware_cv",
            "split_family": "time",
            "reason": "Use past-to-future split when date/order columns or temporal drift are present.",
            "priority": 0.8 if priority else 0.45,
            "run_status": "planned",
            "offline_online_correlation": None,
            "adoption_status": "candidate",
        },
        {
            "profile_id": "leak_safe_cv",
            "split_family": "leak_safe",
            "reason": "Drop or fold-fit suspicious proxy/leak features before ranking model families.",
            "priority": 0.85 if priority else 0.5,
            "run_status": "planned",
            "offline_online_correlation": None,
            "adoption_status": "candidate",
        },
        {
            "profile_id": "adversarial_proxy_cv",
            "split_family": "proxy",
            "reason": "Use adversarial or proxy holdout signals when train/test drift dominates public behavior.",
            "priority": 0.82 if priority else 0.52,
            "run_status": "planned",
            "offline_online_correlation": None,
            "adoption_status": "candidate",
        },
    ]
    if modality in {"image", "text", "rna", "bio"}:
        profiles.append(
            {
                "profile_id": "entity_group_cv",
                "split_family": "group",
                "reason": "Group by subject/entity/file/source to avoid near-duplicate leakage.",
                "priority": 0.75 if priority else 0.5,
                "run_status": "planned",
                "offline_online_correlation": None,
                "adoption_status": "candidate",
            }
        )
    source_hits = [
        _source_id(source, index)
        for index, source in enumerate(sources)
        if any(term in _source_text(source).lower() for term in _VALIDATION_TERMS)
    ]
    profiles = sorted(profiles, key=lambda item: float(item["priority"]), reverse=True)
    return {
        "version": 1,
        "slug": slug,
        "updated_at": datetime.now(UTC).isoformat(),
        "priority": priority,
        "public_regression_signal": public_regression,
        "active_profile": profiles[0]["profile_id"],
        "offline_online_correlation": offline_online_correlation,
        "source_hits": source_hits,
        "profiles": profiles,
        "next_action": ("validation_redesign" if priority else "monitor_offline_online_fit"),
    }


def classify_source(source: dict[str, object], *, slug: str) -> str:
    url = str(source.get("url") or "").lower()
    title = str(source.get("title") or "").lower()
    query = str(source.get("query") or "").lower()
    combined = f"{url} {title} {query}"
    slug_key = slug.lower()
    if "kaggle.com" in url and slug_key in combined:
        if "/discussion" in url or "discussion" in combined:
            return "discussion"
        return "competition_specific"
    if "kaggle.com" in url:
        return "similar_competition"
    if any(domain in url for domain in ("arxiv.org", "nature.com", "openreview.net", "proceedings.mlr.press")):
        return "paper"
    if "github.com" in url or "gitlab" in url:
        return "official_repo"
    if "readthedocs" in url or "docs." in url or "/documentation" in url:
        return "official_docs"
    if any(term in combined for term in ("writeup", "solution", "blog", "qiita", "medium", "hatenablog")):
        return "blog_writeup"
    return "generic"


def unsafe_method_reason(text: str) -> str | None:
    normalized = text.lower()
    for term in _UNSAFE_TERMS:
        if term in normalized:
            return f"unsafe_method_term:{term}"
    if "external" in normalized and "test" in normalized and "label" in normalized:
        return "unsafe_external_test_label_transfer"
    return None


def infer_candidate_category(text: str) -> str:
    normalized = text.lower()
    if any(term in normalized for term in ("blend", "stack", "ensemble", "rank average", "logit average")):
        return "blend"
    if any(term in normalized for term in _VALIDATION_TERMS):
        return "validation_variant"
    if any(term in normalized for term in ("reference", "baseline", "top notebook", "winning solution")):
        return "reference_reproduction"
    if any(term in normalized for term in ("feature", "encoding", "augmentation", "postprocess", "calibration")):
        return "feature_variant"
    return "strong_single"


def infer_modality(problem_types: list[str], dataset_profile: dict[str, object]) -> str:
    text = " ".join(problem_types).lower()
    if "image" in text or "vision" in text:
        return "image"
    if "text" in text or "translation" in text or "nlp" in text:
        return "text"
    if "timeseries" in text or "time" in text:
        return "timeseries"
    if "rna" in text or "protein" in text or "bio" in text:
        return "rna" if "rna" in text else "bio"
    if "audio" in text:
        return "audio"
    if "tabular" in text:
        return "tabular"
    columns = " ".join(str(item).lower() for item in dataset_profile.get("columns", []) if isinstance(item, str))
    if any(term in columns for term in ("image", "path", "filename")):
        return "image"
    if any(term in columns for term in ("text", "sentence", "prompt", "translation")):
        return "text"
    if any(term in columns for term in ("date", "time", "timestamp")):
        return "timeseries"
    return "tabular"


def render_method_registry_for_prompt(registry: dict[str, object], *, max_methods: int = 8) -> str:
    methods = registry.get("methods")
    if not isinstance(methods, list) or not methods:
        return ""
    lines = [
        "Competition-specific method scout is active.",
        f"- registry_mode: {registry.get('mode')}",
        f"- modality: {registry.get('modality')}",
        f"- validation_priority: {registry.get('validation_priority')}",
        f"- active_validation_profile: {registry.get('active_validation_profile')}",
        "- Active method candidates:",
    ]
    active_seen = 0
    for item in methods:
        if not isinstance(item, dict) or item.get("status") != "active":
            continue
        adapter = item.get("implementation_adapter") if isinstance(item.get("implementation_adapter"), dict) else {}
        lines.append(
            "  - "
            f"{item.get('method_id')}: {item.get('candidate_category')} "
            f"score={item.get('rank_score')} source={item.get('source_type')} "
            f"adapter={adapter.get('adapter') or 'default'} "
            f"summary={str(item.get('summary') or item.get('name') or '')[:220]}"
        )
        active_seen += 1
        if active_seen >= max_methods:
            break
    blocked = registry.get("blocked_method_ids")
    if isinstance(blocked, list) and blocked:
        lines.append("- Blocked method ids: " + ", ".join(str(item) for item in blocked[:8]))
    return "\n".join(lines)


def _load_existing_registry(path: Path, *, slug: str, mode: str) -> dict[str, object]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            payload["mode"] = mode
            return payload
    return {"version": 1, "slug": slug, "mode": mode, "methods": [], "active_method_ids": [], "blocked_method_ids": []}


def _seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    seed_specs: dict[str, list[tuple[str, str, str]]] = {
        "tabular": [
            ("tabular-gbdt-portfolio", "GBDT portfolio with CatBoost/XGBoost/LightGBM", "blend"),
            ("tabular-tabpfn-family", "TabPFN/TabICL/pytabkit optional candidate family", "strong_single"),
            ("tabular-tabm-family", "TabM/modern deep-tabular optional family", "strong_single"),
        ],
        "image": [
            (
                "vision-backbone-transfer",
                "timm/torchvision pretrained backbone with competition-specific augment/TTA",
                "strong_single",
            ),
            (
                "vision-detection-segmentation-router",
                "Route to detection/segmentation backbone when sample format requires boxes or masks",
                "strong_single",
            ),
        ],
        "text": [
            (
                "text-transformer-embedding-rerank",
                "Transformer embedding/classifier plus calibration or rerank",
                "strong_single",
            ),
            (
                "text-retrieval-augmented-candidate",
                "Retrieval or nearest-neighbor candidate generation with leak-safe grouping",
                "feature_variant",
            ),
        ],
        "timeseries": [
            (
                "timeseries-lag-gbdt",
                "Lag/rolling/statistical features with GBDT and time-aware validation",
                "strong_single",
            ),
            (
                "timeseries-foundation-optional",
                "Chronos/TFT-style optional branch when runtime and dependencies allow",
                "strong_single",
            ),
        ],
        "rna": [
            (
                "rna-domain-evaluator-postprocess",
                "Official evaluator plus structure/ontology-aware postprocess",
                "feature_variant",
            ),
        ],
        "bio": [
            (
                "bio-domain-evaluator-postprocess",
                "Official evaluator plus domain-aware calibration/postprocess",
                "feature_variant",
            ),
        ],
    }
    specs = seed_specs.get(modality, seed_specs["tabular"])
    methods: list[MethodCandidate] = []
    for method_id, name, category in specs:
        dependency_risk = 0.45 if "optional" in name.lower() else 0.15
        runtime_risk = 0.55 if any(term in name.lower() for term in ("foundation", "deep", "backbone")) else 0.25
        methods.append(
            MethodCandidate(
                method_id=method_id,
                name=name,
                source_ids=["method_scout_seed"],
                source_type="generic",
                candidate_category=category,
                problem_fit=0.72,
                metric_fit=0.65 if metric else 0.5,
                data_fit=0.65,
                expected_gain=0.62,
                implementation_cost=0.35 + dependency_risk,
                dependency_risk=dependency_risk,
                leakage_risk=0.08,
                runtime_risk=runtime_risk,
                fallback=(
                    "Use installed Kaggle-default and repo dependencies first; record unavailable optional methods."
                ),
                summary=name,
                implementation_adapter=_implementation_adapter(category=category, modality=modality, text=name),
                dependency_check=_dependency_check(text=name, modality=modality),
            )
        )
    return methods


def _dedupe_and_rank_methods(methods: list[MethodCandidate]) -> list[MethodCandidate]:
    by_id: dict[str, MethodCandidate] = {}
    for method in methods:
        if method.method_id not in by_id or method.score() > by_id[method.method_id].score():
            by_id[method.method_id] = method
    return sorted(by_id.values(), key=lambda item: (item.status != "active", -item.score(), item.method_id))


def _source_id(source: dict[str, object], index: int) -> str:
    existing = str(source.get("source_id") or "").strip()
    if existing:
        return existing
    url = str(source.get("url") or "").strip()
    if url:
        return f"source-{index + 1}:{url}"
    return f"source-{index + 1}"


def _source_attribution(source: dict[str, object], *, source_id: str, slug: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "url": source.get("url"),
        "title": source.get("title"),
        "kind": classify_source(source, slug=slug),
    }


def _source_text(source: dict[str, object]) -> str:
    parts = [
        source.get("url"),
        source.get("title"),
        source.get("why_relevant"),
        source.get("extracted_technique"),
        source.get("query"),
        source.get("takeaway"),
    ]
    return " ".join(str(part) for part in parts if part)


def _method_id(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized[:80] or "method"


def _fit_score(*, text: str, modality: str, source_priority: float) -> float:
    normalized = text.lower()
    bonus = 0.18 if modality in normalized else 0.0
    if modality == "tabular" and any(
        term in normalized for term in ("catboost", "xgboost", "lightgbm", "tabpfn", "tabm")
    ):
        bonus += 0.16
    if modality == "image" and any(term in normalized for term in ("timm", "convnext", "vit", "swin", "yolo")):
        bonus += 0.16
    if modality == "text" and any(term in normalized for term in ("transformer", "embedding", "rerank", "token")):
        bonus += 0.16
    return min(1.0, 0.45 + bonus + (0.25 * source_priority))


def _metric_fit(*, text: str, metric: str | None) -> float:
    if not metric:
        return 0.5
    normalized = text.lower()
    metric_key = str(metric).lower()
    if metric_key in normalized:
        return 0.9
    if any(alias in normalized for alias in ("auc", "logloss", "rmse", "mae", "accuracy", "f1")):
        return 0.7
    return 0.55


def _data_fit(*, text: str, dataset_profile: dict[str, object]) -> float:
    normalized = text.lower()
    score = 0.55
    n_rows = _to_float(dataset_profile.get("n_rows") or dataset_profile.get("rows"))
    if (
        n_rows is not None
        and n_rows > 100_000
        and any(term in normalized for term in ("gpu", "large", "hist", "batch"))
    ):
        score += 0.2
    if "categorical" in normalized or "encoding" in normalized:
        score += 0.12
    if "missing" in normalized or "imputation" in normalized:
        score += 0.08
    return min(1.0, score)


def _implementation_cost(text: str) -> float:
    normalized = text.lower()
    cost = 0.35
    if any(term in normalized for term in ("foundation", "transformer", "pretrained", "deep", "tft")):
        cost += 0.25
    if any(term in normalized for term in ("install", "dependency", "compile", "custom cuda")):
        cost += 0.25
    if any(term in normalized for term in ("simple", "lightgbm", "xgboost", "catboost", "sklearn")):
        cost -= 0.12
    return max(0.05, min(1.0, cost))


def _dependency_risk(text: str) -> float:
    normalized = text.lower()
    risk = 0.15
    if any(term in normalized for term in ("tabpfn", "tabm", "chronos", "faiss", "install", "pip", "uv add")):
        risk += 0.35
    if any(term in normalized for term in ("xgboost", "lightgbm", "catboost", "torch", "transformers", "timm")):
        risk -= 0.08
    return max(0.05, min(1.0, risk))


def _leakage_risk(text: str) -> float:
    normalized = text.lower()
    risk = 0.08
    if any(term in normalized for term in ("external", "original data", "overlap", "matching", "leak")):
        risk += 0.35
    if any(term in normalized for term in ("fold-safe", "leak-free", "train only")):
        risk -= 0.08
    return max(0.02, min(1.0, risk))


def _runtime_risk(text: str) -> float:
    normalized = text.lower()
    risk = 0.2
    if any(term in normalized for term in ("large model", "foundation", "pretrain", "full fine-tuning", "multi-seed")):
        risk += 0.35
    if any(term in normalized for term in ("fast", "lightweight", "gbdt", "cached", "early stopping")):
        risk -= 0.08
    return max(0.05, min(1.0, risk))


def _fallback_for_category(category: str, *, modality: str) -> str:
    if category == "blend":
        return "Use OOF weighted/rank/logit average over available low-correlation candidates."
    if category == "validation_variant":
        return "Run default, group/proxy, time-aware, and leak-safe split candidates."
    if category == "feature_variant":
        return "Use fold-fit encoders/features with train-fit/test-apply guarantees."
    if modality == "tabular":
        return "Use CatBoost/XGBoost/LightGBM already available in the repo environment."
    if modality == "image":
        return "Use installed torch/timm/torchvision backbone or the repo vision runtime fallback."
    if modality == "text":
        return "Use installed transformers/sklearn TF-IDF or cached embeddings fallback."
    return "Use installed repo dependencies and record skipped optional method details."


def _implementation_adapter(*, category: str, modality: str, text: str) -> dict[str, object]:
    normalized = text.lower()
    if category == "validation_variant":
        return {
            "adapter": "validation_profile_search",
            "contract": "emit split profile id, fold scores, OOF predictions, and public-calibration rationale",
            "candidate_family": category,
        }
    if category == "blend":
        return {
            "adapter": "oof_blend_builder",
            "contract": "consume aligned OOF/test predictions and emit rank/logit/weighted blend candidates",
            "candidate_family": category,
        }
    if category == "reference_reproduction":
        return {
            "adapter": "reference_reproduction",
            "contract": "reproduce attributed source before novelty and explain any score gap",
            "candidate_family": category,
        }
    if modality == "tabular" and any(term in normalized for term in ("tabpfn", "tabicl", "pytabkit")):
        return {
            "adapter": "optional_tabular_foundation_family",
            "contract": "try only when dependencies/runtime permit; otherwise record skipped fallback",
            "candidate_family": category,
        }
    if modality == "tabular" and "tabm" in normalized:
        return {
            "adapter": "optional_deep_tabular_family",
            "contract": "train as an optional low-correlation candidate with GBDT fallback",
            "candidate_family": category,
        }
    return {
        "adapter": f"{modality}_{category}",
        "contract": "emit candidate metrics, OOF/test predictions when feasible, and fallback reason when skipped",
        "candidate_family": category,
    }


def _dependency_check(*, text: str, modality: str) -> dict[str, object]:
    normalized = text.lower()
    optional: list[str] = []
    required: list[str] = []
    if modality == "tabular":
        required.extend(["numpy", "pandas", "scikit-learn"])
    if any(term in normalized for term in ("lightgbm", "xgboost", "catboost")):
        optional.extend(["lightgbm", "xgboost", "catboost"])
    if any(term in normalized for term in ("tabpfn", "tabicl", "pytabkit")):
        optional.extend(["tabpfn", "pytabkit"])
    if "tabm" in normalized:
        optional.extend(["torch", "tabm"])
    if modality in {"image", "text", "timeseries", "rna", "bio"}:
        optional.extend(["torch", "transformers"])
    return {
        "required": sorted(set(required)),
        "optional": sorted(set(optional)),
        "fallback": _fallback_for_category(infer_candidate_category(text), modality=modality),
    }


def _campaign_needs_validation_redesign(campaign_state: dict[str, object] | None) -> bool:
    if not isinstance(campaign_state, dict):
        return False
    latest = _to_float(campaign_state.get("latest_submission_score"))
    champion = _to_float(campaign_state.get("champion_score") or campaign_state.get("historical_best_score"))
    direction = str(campaign_state.get("direction") or "minimize").lower()
    corr = _to_float(campaign_state.get("offline_online_correlation"))
    if corr is not None and corr < 0.25:
        return True
    if latest is None or champion is None:
        return False
    if direction == "maximize":
        return latest < champion
    return latest > champion


def _domain_terms(*, slug: str, dataset_profile: dict[str, object], problem_types: list[str]) -> list[str]:
    terms = [part for part in slug.replace("-", " ").split() if len(part) > 2][:4]
    terms.extend(part.split(":")[-1] for part in problem_types[:2])
    for key in ("domain", "task_type", "target_column"):
        value = dataset_profile.get(key)
        if isinstance(value, str) and value.strip():
            terms.append(value.strip())
    return list(dict.fromkeys(term.lower() for term in terms if term))


def _count_source_types(sources: list[dict[str, object]], *, slug: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        source_type = classify_source(source, slug=slug)
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts
