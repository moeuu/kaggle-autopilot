from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.asset_modality import AUDIO_SUFFIXES, DOCUMENT_SUFFIXES, IMAGE_SUFFIXES, SIGNAL_SUFFIXES, VIDEO_SUFFIXES
from kagglebot.json_utils import load_json_object, load_jsonl_records, write_json_object
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
_IMAGE_MODALITY_HINTS = (
    "image",
    "photo",
    "picture",
    "vision",
    *(suffix.lstrip(".") for suffix in sorted(IMAGE_SUFFIXES)),
    *sorted(IMAGE_SUFFIXES),
)
_AUDIO_MODALITY_HINTS = (
    "audio",
    "sound",
    "speech",
    *(suffix.lstrip(".") for suffix in sorted(AUDIO_SUFFIXES)),
    *sorted(AUDIO_SUFFIXES),
)
_VIDEO_MODALITY_HINTS = (
    "video",
    "clip",
    "frame",
    *(suffix.lstrip(".") for suffix in sorted(VIDEO_SUFFIXES)),
    *sorted(VIDEO_SUFFIXES),
)
_SIGNAL_MODALITY_HINTS = (
    "signal",
    "signals",
    "waveform",
    "waveforms",
    "biosignal",
    "biosignals",
    "ecg",
    "eeg",
    "ekg",
    "wfdb",
    "physionet",
    *(suffix.lstrip(".") for suffix in sorted(SIGNAL_SUFFIXES)),
    *sorted(SIGNAL_SUFFIXES),
)
_TEXT_MODALITY_HINTS = (
    "text",
    "translation",
    "nlp",
    "document classification",
    "document understanding",
    "document qa",
    "pdf",
    "docx",
    "markdown",
)
_TEXT_FILE_REFERENCE_COLUMN_HINTS = (
    "document_path",
    "document_file",
    "doc_path",
    "doc_file",
    "pdf_path",
    "pdf_file",
    "report_path",
    "report_file",
    *(suffix for suffix in sorted(DOCUMENT_SUFFIXES) if suffix not in {".html", ".htm"}),
)
_MODALITY_ALIASES = {
    "3d": "point_cloud",
    "point_cloud_3d": "point_cloud",
    "model_artifact": "artifact",
    "model_output": "artifact",
    "submission_artifact": "artifact",
    "sequence": "bio",
    "structure": "bio",
}


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


def effective_method_scout_mode(*, requested_mode: str | None, campaign_mode: str) -> MethodScoutMode:
    requested = normalize_method_scout_mode(requested_mode)
    if campaign_mode != "top1" and requested == "auto":
        return "off"
    return requested


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


def load_method_registry(path: Path) -> dict[str, object]:
    return _load_registry_payload(path)


def load_source_registry(path: Path) -> dict[str, object]:
    return _load_registry_payload(path)


def load_validation_registry(path: Path) -> dict[str, object]:
    return _load_registry_payload(path)


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
        dataset_profile=dataset_profile,
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
    write_json_object(method_scout_queries_path(paths.context_dir), {"version": 1, "slug": slug, "queries": queries})
    write_json_object(source_registry_path(paths.context_dir), source_registry)
    write_json_object(validation_registry_path(paths.context_dir), validation_registry)
    write_json_object(registry_path, payload)
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
    elif modality == "audio":
        raw_queries.extend(
            [
                f"{slug} audio spectrogram MFCC pretrained encoder Kaggle solution",
                f"{' '.join(domain_terms)} audio classification retrieval {metric_text} 2025 arxiv official repo",
            ]
        )
    elif modality == "video":
        raw_queries.extend(
            [
                f"{slug} video frame sampling temporal pooling TTA Kaggle solution",
                f"{' '.join(domain_terms)} video action recognition {metric_text} 2025 arxiv official repo",
            ]
        )
    elif modality == "signal":
        raw_queries.extend(
            [
                f"{slug} ECG EEG waveform signal processing Kaggle solution",
                (
                    f"{' '.join(domain_terms)} biosignal 1D CNN transformer WFDB EDF "
                    f"{metric_text} 2025 arxiv official repo"
                ),
            ]
        )
    elif modality == "medical_imaging":
        raw_queries.extend(
            [
                f"{slug} DICOM IMA NIfTI NRRD MHA medical imaging preprocessing windowing Kaggle solution",
                f"{' '.join(domain_terms)} medical imaging 3D CNN transformer {metric_text} 2025 arxiv official repo",
            ]
        )
    elif modality == "array":
        raw_queries.extend(
            [
                f"{slug} numpy Zarr OME-Zarr N5 AnnData H5AD array feature extraction Kaggle solution",
                f"{' '.join(domain_terms)} NetCDF GRIB FITS scientific array {metric_text} 2025 official repo",
            ]
        )
    elif modality == "point_cloud":
        raw_queries.extend(
            [
                f"{slug} point cloud lidar voxel projection Kaggle solution",
                f"{' '.join(domain_terms)} point cloud PointNet transformer {metric_text} 2025 arxiv official repo",
            ]
        )
    elif modality == "geospatial":
        raw_queries.extend(
            [
                f"{slug} geospatial GIS GeoJSON shapefile feature engineering Kaggle solution",
                f"{' '.join(domain_terms)} geospatial spatial cross validation {metric_text} 2025 official repo",
            ]
        )
    elif modality == "graph":
        raw_queries.extend(
            [
                f"{slug} graph neural network node edge link prediction Kaggle solution",
                f"{' '.join(domain_terms)} graph features NetworkX GNN {metric_text} 2025 official repo",
            ]
        )
    elif modality == "annotation":
        raw_queries.extend(
            [
                f"{slug} COCO YOLO LabelMe annotation detection segmentation Kaggle solution",
                (
                    f"{' '.join(domain_terms)} annotation conversion RLE COCO YOLO mask submission "
                    f"{metric_text} 2025 official repo"
                ),
            ]
        )
    elif modality == "artifact":
        raw_queries.extend(
            [
                f"{slug} model artifact ONNX safetensors checkpoint submission Kaggle solution",
                (
                    f"{' '.join(domain_terms)} model artifact packaging ONNX TensorFlow Lite CoreML "
                    f"safetensors {metric_text} 2025 official repo"
                ),
            ]
        )
    elif modality == "text":
        raw_queries.extend(
            [
                f"{slug} transformers embedding reranker calibration",
                f"{' '.join(domain_terms)} NLP {metric_text} 2025 arxiv official repo",
            ]
        )
        if _is_document_file_reference_profile(dataset_profile, problem_types):
            raw_queries.extend(
                [
                    f"{slug} PDF DOCX Markdown document classification feature extraction Kaggle solution",
                    (
                        f"{' '.join(domain_terms)} document file metadata text extraction embeddings "
                        f"{metric_text} 2025 official repo"
                    ),
                ]
            )
    elif modality == "multimodal":
        raw_queries.extend(
            [
                f"{slug} multimodal image text fusion CLIP embeddings Kaggle solution",
                (
                    f"{' '.join(domain_terms)} vision language dual encoder late fusion "
                    f"{metric_text} 2025 arxiv official repo"
                ),
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
                f"{slug} official evaluator RNA protein molecule structure baseline GitHub",
                f"{' '.join(domain_terms)} bioinformatics SMILES FASTA PDB {metric_text} 2025 arxiv",
            ]
        )
    if _is_multi_label_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} multi-label classification threshold optimization F1 Kaggle solution",
                (
                    f"{' '.join(domain_terms)} multi-label one-vs-rest classifier "
                    f"calibration threshold tuning {metric_text}"
                ),
            ]
        )
    if _is_multi_output_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} multi-output multi-target modeling Kaggle solution",
                (
                    f"{' '.join(domain_terms)} multi-output regression multi-target classification "
                    f"per-target model chaining {metric_text}"
                ),
            ]
        )
    if _is_quantile_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} quantile regression pinball loss Kaggle solution",
                f"{' '.join(domain_terms)} prediction intervals conformal quantile regression {metric_text}",
            ]
        )
    if _is_ordinal_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} ordinal classification quadratic weighted kappa Kaggle solution",
                f"{' '.join(domain_terms)} ordinal regression threshold optimization QWK {metric_text}",
            ]
        )
    if _is_sample_weight_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} sample weights weighted metric Kaggle solution",
                f"{' '.join(domain_terms)} sample_weight weighted loss validation calibration {metric_text}",
            ]
        )
    if _is_text_generation_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} text generation translation summarization retrieval augmented Kaggle solution",
                f"{' '.join(domain_terms)} seq2seq transformer BLEU ROUGE semantic similarity {metric_text}",
            ]
        )
    if _is_survival_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} survival analysis concordance index Kaggle solution",
                (f"{' '.join(domain_terms)} time-to-event Cox Kaplan-Meier event censoring validation {metric_text}"),
            ]
        )
    if _is_pairwise_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} pairwise matchup ranking probability calibration Kaggle solution",
                (
                    f"{' '.join(domain_terms)} pairwise preference model Bradley Terry Elo "
                    f"feature difference calibration {metric_text}"
                ),
            ]
        )
    if _is_learning_to_rank_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} learning to rank NDCG LambdaMART Kaggle solution",
                f"{' '.join(domain_terms)} query document relevance ranking group k-fold {metric_text}",
            ]
        )
    if _is_anomaly_detection_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} anomaly detection isolation forest autoencoder Kaggle solution",
                f"{' '.join(domain_terms)} unsupervised anomaly score calibration validation {metric_text}",
            ]
        )
    elif _is_unsupervised_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} unsupervised prediction baseline clustering density Kaggle solution",
                f"{' '.join(domain_terms)} no train labels pseudo-label anomaly score validation {metric_text}",
            ]
        )
    if _is_ctr_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} click through rate CTR target encoding calibration Kaggle solution",
                f"{' '.join(domain_terms)} user item ad click prediction GBDT calibration {metric_text}",
            ]
        )
    if _is_recommender_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} recommender system user item matrix factorization Kaggle solution",
                f"{' '.join(domain_terms)} user item rating prediction ALS LightFM GBDT features {metric_text}",
            ]
        )
    if _is_forecasting_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} forecasting horizon lag rolling feature validation Kaggle solution",
                f"{' '.join(domain_terms)} time series forecasting backtesting leakage {metric_text}",
            ]
        )
    if _is_detection_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} object detection prediction_string mAP Kaggle solution",
                f"{' '.join(domain_terms)} YOLO Faster R-CNN detection TTA NMS {metric_text}",
            ]
        )
    if _is_segmentation_profile(dataset_profile, problem_types):
        raw_queries.extend(
            [
                f"{slug} segmentation RLE mask dice Kaggle solution",
                f"{' '.join(domain_terms)} U-Net Mask R-CNN segmentation TTA postprocessing {metric_text}",
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
    return load_jsonl_records(path, limit=max(1, int(limit)))


def _is_multi_label_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"multi_label", "multilabel", "multi-label"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"multi_label", "multilabel", "multi-label"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"multi_label", "multilabel", "multi-label"}
        for problem_type in problem_types
    )


def _is_multi_output_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"multi_output_regression", "multi_target_classification", "multi_task"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"multi_output", "multi_target"} for tag in tags):
        return True
    return any(str(problem_type).strip().lower() in {"multi_output", "multi_target"} for problem_type in problem_types)


def _is_quantile_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"quantile_regression", "prediction_interval", "quantile", "interval_prediction"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"quantile_regression", "prediction_interval", "quantile"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"quantile_regression", "prediction_interval", "quantile"}
        for problem_type in problem_types
    )


def _is_ordinal_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"ordinal_classification", "ordinal", "ordinal_regression"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"ordinal_classification", "ordinal"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"ordinal_classification", "ordinal", "ordinal_regression"}
        for problem_type in problem_types
    )


def _is_survival_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"survival", "time_to_event", "time-to-event"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"survival", "time_to_event"} for tag in tags):
        return True
    return any(str(problem_type).strip().lower() in {"survival", "time_to_event"} for problem_type in problem_types)


def _is_pairwise_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"pairwise", "pairwise_preference", "ranking"}:
        return True
    structure = str(dataset_profile.get("competition_structure") or "").strip().lower()
    if "pairwise" in structure:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"pairwise", "ranking"} for tag in tags):
        return True
    return any(str(problem_type).strip().lower() in {"pairwise", "ranking"} for problem_type in problem_types)


def _is_learning_to_rank_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"learning_to_rank", "learning-to-rank", "ltr", "listwise_ranking"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"learning_to_rank", "learning-to-rank", "ltr"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"learning_to_rank", "learning-to-rank", "ltr", "listwise_ranking"}
        for problem_type in problem_types
    )


def _is_anomaly_detection_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"anomaly_detection", "anomaly-detection", "outlier_detection", "fraud_detection"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"anomaly_detection", "anomaly-detection", "outlier_detection"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"anomaly_detection", "anomaly-detection", "outlier_detection"}
        for problem_type in problem_types
    )


def _is_unsupervised_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"unsupervised", "unsupervised_prediction", "no_train_labels"}:
        return True
    task = str(dataset_profile.get("task") or "").strip().lower()
    if task == "unsupervised":
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() == "unsupervised" for tag in tags):
        return True
    return any(str(problem_type).strip().lower() == "unsupervised" for problem_type in problem_types)


def _is_ctr_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"ctr", "click_through_rate", "click-through-rate", "click_prediction"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"ctr", "click_prediction"} for tag in tags):
        return True
    return any(str(problem_type).strip().lower() in {"ctr", "click_prediction"} for problem_type in problem_types)


def _is_recommender_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"recommender", "recommendation", "recommender_system", "user_item"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"recommender", "recommendation", "user_item"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"recommender", "recommendation", "user_item"}
        for problem_type in problem_types
    )


def _is_forecasting_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"forecasting", "forecast", "time_series_forecast", "time-series-forecast"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"forecasting", "forecast"} for tag in tags):
        return True
    return any(
        str(problem_type).strip().lower() in {"forecasting", "forecast", "time_series_forecast"}
        for problem_type in problem_types
    )


def _is_detection_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"object_detection", "detection", "bbox_detection"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() in {"object_detection", "detection"} for tag in tags):
        return True
    return any(str(problem_type).strip().lower() in {"object_detection", "detection"} for problem_type in problem_types)


def _is_segmentation_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"segmentation", "mask_segmentation", "semantic_segmentation", "instance_segmentation"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() == "segmentation" for tag in tags):
        return True
    return any(str(problem_type).strip().lower() == "segmentation" for problem_type in problem_types)


def _is_sample_weight_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    if str(dataset_profile.get("sample_weight_column_hint") or "").strip():
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(str(tag).strip().lower() == "sample_weighted" for tag in tags):
        return True
    return any(str(problem_type).strip().lower() == "sample_weighted" for problem_type in problem_types)


def _is_text_generation_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"text_generation", "translation", "summarization", "question_answering", "qa"}:
        return True
    task = str(dataset_profile.get("task") or "").strip().lower()
    if task in {"text_generation", "translation", "summarization"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"text_generation", "translation", "summarization", "qa"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"text_generation", "translation", "summarization", "qa"}
        for problem_type in problem_types
    )


def _is_document_file_reference_profile(dataset_profile: dict[str, object], problem_types: list[str]) -> bool:
    columns = dataset_profile.get("columns")
    if isinstance(columns, list):
        column_text = " ".join(str(item).lower() for item in columns if isinstance(item, str))
        if any(term in column_text for term in _TEXT_FILE_REFERENCE_COLUMN_HINTS):
            return True
    target_semantics = str(dataset_profile.get("target_semantics") or "").strip().lower()
    if target_semantics in {"document_classification", "document_understanding", "document_qa"}:
        return True
    tags = dataset_profile.get("tags")
    if isinstance(tags, list) and any(
        str(tag).strip().lower() in {"document", "document_classification", "document_understanding"} for tag in tags
    ):
        return True
    return any(
        str(problem_type).strip().lower() in {"document", "document_classification", "document_understanding"}
        for problem_type in problem_types
    )


def _multi_label_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.75 if metric else 0.6
    return [
        MethodCandidate(
            method_id=f"{modality}-multi-label-one-vs-rest-thresholds",
            name="Multi-label one-vs-rest classifier with per-label threshold optimization",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.68,
            implementation_cost=0.42,
            dependency_risk=0.08,
            leakage_risk=0.08,
            runtime_risk=0.25,
            fallback=(
                "Use sklearn OneVsRest-style heads, tune thresholds on OOF predictions, and fall back to label priors."
            ),
            summary=(
                "Multi-label heads must optimize per-class thresholds against the competition metric instead of argmax."
            ),
            implementation_adapter={
                "adapter": f"{modality}_multi_label_thresholds",
                "contract": (
                    "fit per-label binary heads, emit OOF/test scores, and tune thresholds using only validation folds"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn linear/logistic one-vs-rest heads and prior thresholds.",
            },
        )
    ]


def _text_generation_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.76 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-text-generation-retrieval-seq2seq",
            name="Text-generation retrieval baseline with optional seq2seq reranking",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.84,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.64,
            implementation_cost=0.46,
            dependency_risk=0.16,
            leakage_risk=0.12,
            runtime_risk=0.34,
            fallback=(
                "Use TF-IDF/embedding nearest-neighbor text retrieval and deterministic fallback strings when "
                "seq2seq inference is unavailable."
            ),
            summary=(
                "Text-generation submissions need retrieval or seq2seq candidates plus text-normalized validation, "
                "not categorical label encoding."
            ),
            implementation_adapter={
                "adapter": f"{modality}_text_generation_retrieval_seq2seq",
                "contract": (
                    "build prompt/source text features, validate generated strings with text metrics, "
                    "and preserve exact submission text columns"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["sentence-transformers", "transformers", "torch"],
                "fallback": "Use sklearn TF-IDF nearest-neighbor retrieval and constant-text fallback.",
            },
        )
    ]


def _document_file_reference_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.74 if metric else 0.58
    return [
        MethodCandidate(
            method_id=f"{modality}-document-file-metadata-text-head",
            name="Document file metadata and text-stat feature head",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="feature_variant",
            problem_fit=0.82,
            metric_fit=metric_fit,
            data_fit=0.78,
            expected_gain=0.58,
            implementation_cost=0.28,
            dependency_risk=0.08,
            leakage_risk=0.08,
            runtime_risk=0.16,
            fallback=(
                "Use document byte size, suffix, page count, character/word/paragraph counts, "
                "and optional extracted text embeddings with grouped validation when documents repeat."
            ),
            summary=(
                "Document-reference tables should exploit PDF/DOCX/Markdown metadata and text statistics "
                "before falling back to generic tabular or transformer-only baselines."
            ),
            implementation_adapter={
                "adapter": f"{modality}_document_file_metadata_text_head",
                "contract": (
                    "resolve document path columns, add document metadata/text-stat features, "
                    "and optionally cache text embeddings when dependencies permit"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["pypdf", "python-docx", "transformers", "sentence-transformers"],
                "fallback": "Use built-in PDF/DOCX/Markdown metadata and sklearn heads without optional parsers.",
            },
        )
    ]


def _detection_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.78 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-object-detection-router",
            name="Object detection router with prediction_string formatting and NMS/TTA",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.88,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.7,
            implementation_cost=0.5,
            dependency_risk=0.18,
            leakage_risk=0.1,
            runtime_risk=0.38,
            fallback="Use torchvision detector or YOLO when available; otherwise emit validated empty detections.",
            summary="Detection submissions need box formatting, score thresholds, NMS, and mAP-style validation.",
            implementation_adapter={
                "adapter": f"{modality}_object_detection_router",
                "contract": (
                    "train or load detector, format prediction_string boxes, and validate mAP-compatible output"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["torch", "torchvision", "ultralytics", "opencv-python"],
                "fallback": "Use installed torchvision/YOLO paths first, then safe empty detection formatting.",
            },
        )
    ]


def _segmentation_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.78 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-segmentation-mask-rle",
            name="Segmentation mask model with RLE/PNG/TIFF submission formatting",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.68,
            implementation_cost=0.5,
            dependency_risk=0.18,
            leakage_risk=0.1,
            runtime_risk=0.38,
            fallback=(
                "Use lightweight U-Net/torchvision-style mask head or validated empty masks "
                "when training is infeasible."
            ),
            summary=(
                "Segmentation submissions need mask decoding/encoding, thresholding, "
                "connected components, and dice checks."
            ),
            implementation_adapter={
                "adapter": f"{modality}_segmentation_mask_rle",
                "contract": (
                    "produce aligned masks or RLE strings, validate dimensions, and tune mask thresholds on folds"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["torch", "torchvision", "opencv-python", "timm"],
                "fallback": "Use numpy/OpenCV mask postprocessing and validated empty-mask submission formatting.",
            },
        )
    ]


def _pairwise_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.74 if metric else 0.6
    return [
        MethodCandidate(
            method_id=f"{modality}-pairwise-difference-ranking",
            name="Pairwise feature-difference ranking model with calibrated probabilities",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.84,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.66,
            implementation_cost=0.38,
            dependency_risk=0.08,
            leakage_risk=0.12,
            runtime_risk=0.22,
            fallback=(
                "Build A-vs-B feature differences, include swapped-pair augmentation when valid, "
                "and calibrate pairwise probabilities."
            ),
            summary=("Pairwise tasks should model relative strength instead of treating entity ids as flat labels."),
            implementation_adapter={
                "adapter": f"{modality}_pairwise_difference_ranking",
                "contract": (
                    "derive pairwise features, respect group/time leakage, and emit calibrated matchup probabilities"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn logistic/ridge heads over feature differences plus entity historical rates.",
            },
        )
    ]


def _learning_to_rank_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.82 if metric else 0.64
    return [
        MethodCandidate(
            method_id=f"{modality}-learning-to-rank-lambdamart",
            name="Learning-to-rank LambdaMART with query-group validation",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.88,
            metric_fit=metric_fit,
            data_fit=0.76,
            expected_gain=0.7,
            implementation_cost=0.42,
            dependency_risk=0.12,
            leakage_risk=0.18,
            runtime_risk=0.28,
            fallback=(
                "Use group-aware query splits, per-query normalization, and a GBDT/ridge relevance scorer "
                "when rank objectives are unavailable."
            ),
            summary=(
                "Learning-to-rank tasks need query-group validation and NDCG-aware ranking losses or calibration."
            ),
            implementation_adapter={
                "adapter": f"{modality}_learning_to_rank_lambdamart",
                "contract": (
                    "preserve query groups, train relevance/rank scores, "
                    "and validate with NDCG or grouped ranking metrics"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["lightgbm", "xgboost", "catboost"],
                "fallback": "Use sklearn ranking features plus grouped CV and sort candidates by calibrated relevance.",
            },
        )
    ]


def _anomaly_detection_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.76 if metric else 0.58
    return [
        MethodCandidate(
            method_id=f"{modality}-anomaly-score-ensemble",
            name="Anomaly score ensemble with isolation, robust scaling, and calibration",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.84,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.64,
            implementation_cost=0.4,
            dependency_risk=0.08,
            leakage_risk=0.12,
            runtime_risk=0.24,
            fallback=(
                "Use robust numeric/categorical features, IsolationForest/LOF/PCA scores, "
                "and calibrate score direction against any public/sample hints."
            ),
            summary=(
                "Anomaly detection tasks without train labels need score ensembles and careful direction validation."
            ),
            implementation_adapter={
                "adapter": f"{modality}_anomaly_score_ensemble",
                "contract": "fit unsupervised anomaly scores on train/test features and emit aligned risk scores",
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["pyod", "xgboost", "lightgbm"],
                "fallback": (
                    "Use sklearn IsolationForest, robust covariance, PCA reconstruction, and rank-averaged scores."
                ),
            },
        )
    ]


def _unsupervised_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.58 if metric else 0.5
    return [
        MethodCandidate(
            method_id=f"{modality}-unsupervised-score-baseline",
            name="Unsupervised score baseline with clustering and density features",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="feature_variant",
            problem_fit=0.76,
            metric_fit=metric_fit,
            data_fit=0.68,
            expected_gain=0.56,
            implementation_cost=0.34,
            dependency_risk=0.06,
            leakage_risk=0.12,
            runtime_risk=0.2,
            fallback=(
                "Generate density, distance-to-centroid, reconstruction, or rank-normalized scores "
                "when train labels are unavailable."
            ),
            summary="No-label prediction tasks need explicit unsupervised scores instead of supervised target fitting.",
            implementation_adapter={
                "adapter": f"{modality}_unsupervised_score_baseline",
                "contract": (
                    "derive unsupervised scores from train/test features and align them to sample submission rows"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["scipy"],
                "fallback": "Use sklearn preprocessing, PCA, KMeans distances, and robust rank-normalized scores.",
            },
        )
    ]


def _ctr_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.78 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-ctr-gbdt-calibration",
            name="CTR GBDT model with user-item encodings and probability calibration",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.76,
            expected_gain=0.68,
            implementation_cost=0.4,
            dependency_risk=0.08,
            leakage_risk=0.16,
            runtime_risk=0.24,
            fallback=(
                "Use leak-safe user/item frequency, target encodings, GBDT or logistic heads, "
                "and calibrate click probabilities on validation folds."
            ),
            summary=(
                "CTR tasks need user-item/ad interaction features, grouped validation, and calibrated probabilities."
            ),
            implementation_adapter={
                "adapter": f"{modality}_ctr_gbdt_calibration",
                "contract": (
                    "derive leak-safe user/item encodings, fit a click predictor, and emit calibrated probabilities"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": (
                    "Use sklearn logistic regression over frequency/count encodings and calibrated validation scores."
                ),
            },
        )
    ]


def _recommender_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.76 if metric else 0.6
    return [
        MethodCandidate(
            method_id=f"{modality}-recommender-user-item-features",
            name="User-item recommender baseline with aggregate features and matrix factorization fallback",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.84,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.66,
            implementation_cost=0.42,
            dependency_risk=0.1,
            leakage_risk=0.14,
            runtime_risk=0.28,
            fallback=(
                "Blend global/user/item means with leak-safe interaction counts; add ALS/SVD factors when dependencies "
                "and data density allow."
            ),
            summary=(
                "Recommender tasks should model user-item interactions explicitly instead of treating ids as plain "
                "categoricals only."
            ),
            implementation_adapter={
                "adapter": f"{modality}_recommender_user_item_features",
                "contract": (
                    "build user/item aggregate features, optional matrix factors, "
                    "and aligned rating/relevance predictions"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["scipy", "implicit", "lightfm", "xgboost", "lightgbm", "catboost"],
                "fallback": "Use pandas/sklearn aggregate encodings with ridge/GBDT or global-user-item mean blending.",
            },
        )
    ]


def _forecasting_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.78 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-forecasting-backtest-lag-gbdt",
            name="Forecasting backtest with lag, rolling, and calendar features",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.76,
            expected_gain=0.68,
            implementation_cost=0.38,
            dependency_risk=0.08,
            leakage_risk=0.18,
            runtime_risk=0.24,
            fallback=(
                "Use past-only lag/rolling/calendar features with chronological backtests and a GBDT or ridge head."
            ),
            summary=(
                "Forecasting tasks need horizon-aware validation and past-only feature engineering, not shuffled CV."
            ),
            implementation_adapter={
                "adapter": f"{modality}_forecasting_backtest_lag_gbdt",
                "contract": (
                    "derive past-only temporal features, run chronological backtests, "
                    "and emit future-horizon predictions"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost", "statsmodels"],
                "fallback": "Use pandas lag/rolling features plus sklearn ridge/random forest with time splits.",
            },
        )
    ]


def _survival_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.8 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-survival-risk-ranking",
            name="Survival risk ranking with event-time targets and C-index validation",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.68,
            implementation_cost=0.42,
            dependency_risk=0.12,
            leakage_risk=0.1,
            runtime_risk=0.25,
            fallback=(
                "Use event/censoring-aware risk targets, Kaplan-Meier/Cox-style transforms, "
                "and validate with concordance-index-compatible folds."
            ),
            summary="Preserve censoring/event-time semantics and optimize a risk ranking for c-index style metrics.",
            implementation_adapter={
                "adapter": f"{modality}_survival_risk_ranking",
                "contract": "fit risk scores from event/time targets and report c-index or rank validation",
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["lifelines", "xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn/GBDT regression or classification risk scores with event-time transforms.",
            },
        )
    ]


def _multi_output_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.72 if metric else 0.58
    return [
        MethodCandidate(
            method_id=f"{modality}-multi-output-target-heads",
            name="Multi-output target heads with shared features and per-target validation",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.82,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.64,
            implementation_cost=0.38,
            dependency_risk=0.08,
            leakage_risk=0.08,
            runtime_risk=0.24,
            fallback=(
                "Use sklearn MultiOutputRegressor/Classifier or one model per target, "
                "then validate and blend per target."
            ),
            summary="Preserve each submission target column with per-target models instead of collapsing to one label.",
            implementation_adapter={
                "adapter": f"{modality}_multi_output_heads",
                "contract": (
                    "fit one head per target column, emit aligned 2D predictions, and report per-target metrics"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn multi-output wrappers or independent linear/tree heads per target.",
            },
        )
    ]


def _quantile_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.8 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-quantile-interval-heads",
            name="Quantile and prediction-interval heads with monotonic postprocessing",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.66,
            implementation_cost=0.4,
            dependency_risk=0.08,
            leakage_risk=0.1,
            runtime_risk=0.22,
            fallback=(
                "Train separate quantile heads or conformal intervals, enforce lower<=median<=upper, "
                "and validate with pinball/interval scores."
            ),
            summary=(
                "Quantile submissions need ordered interval outputs and loss functions aligned to requested quantiles."
            ),
            implementation_adapter={
                "adapter": f"{modality}_quantile_interval_heads",
                "contract": (
                    "fit quantile or interval heads, enforce non-crossing outputs, "
                    "and emit all requested sample columns"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["lightgbm", "xgboost", "catboost", "mapie"],
                "fallback": "Use sklearn GradientBoostingRegressor quantile loss or conformal residual intervals.",
            },
        )
    ]


def _ordinal_seed_methods(*, modality: str, metric: str | None) -> list[MethodCandidate]:
    metric_fit = 0.82 if metric else 0.62
    return [
        MethodCandidate(
            method_id=f"{modality}-ordinal-threshold-qwk",
            name="Ordinal classification with threshold tuning for QWK-style metrics",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="strong_single",
            problem_fit=0.86,
            metric_fit=metric_fit,
            data_fit=0.74,
            expected_gain=0.66,
            implementation_cost=0.36,
            dependency_risk=0.08,
            leakage_risk=0.08,
            runtime_risk=0.2,
            fallback=(
                "Fit regression or ordinal class scores, tune monotonic thresholds on validation folds, "
                "and clip predictions to valid ordered labels."
            ),
            summary="Ordinal tasks should optimize ordered thresholds instead of treating labels as unordered classes.",
            implementation_adapter={
                "adapter": f"{modality}_ordinal_threshold_qwk",
                "contract": (
                    "fit ordered class/regression scores, tune thresholds for QWK or ordinal metric, "
                    "and emit valid ordered labels"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn regression/classification scores with validation threshold search.",
            },
        )
    ]


def _sample_weight_seed_methods(
    *, modality: str, metric: str | None, weight_column: str | None
) -> list[MethodCandidate]:
    metric_fit = 0.76 if metric else 0.6
    column_text = f"`{weight_column}`" if weight_column else "the detected sample-weight column"
    return [
        MethodCandidate(
            method_id=f"{modality}-sample-weight-aware-training",
            name="Sample-weight-aware training and validation",
            source_ids=["method_scout_seed"],
            source_type="generic",
            candidate_category="feature_variant",
            problem_fit=0.78,
            metric_fit=metric_fit,
            data_fit=0.72,
            expected_gain=0.58,
            implementation_cost=0.24,
            dependency_risk=0.06,
            leakage_risk=0.08,
            runtime_risk=0.12,
            fallback=(
                f"Use {column_text} only as sample_weight in fit/evaluation, never as a predictive feature, "
                "and fall back to unweighted metrics only when the estimator lacks weight support."
            ),
            summary="Preserve competition-provided row weights in model fitting, validation metrics, and calibration.",
            implementation_adapter={
                "adapter": f"{modality}_sample_weight_aware_training",
                "contract": (
                    "remove the weight column from features, pass it to supported estimators/metrics, "
                    "and report weighted plus unweighted validation diagnostics"
                ),
            },
            dependency_check={
                "required": ["sklearn"],
                "optional": ["xgboost", "lightgbm", "catboost"],
                "fallback": "Use sklearn estimators/metrics that accept sample_weight and log unsupported paths.",
            },
        )
    ]


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
    if _is_multi_label_profile(dataset_profile, problem_types):
        methods.extend(_multi_label_seed_methods(modality=modality, metric=metric))
    if _is_multi_output_profile(dataset_profile, problem_types):
        methods.extend(_multi_output_seed_methods(modality=modality, metric=metric))
    if _is_quantile_profile(dataset_profile, problem_types):
        methods.extend(_quantile_seed_methods(modality=modality, metric=metric))
    if _is_ordinal_profile(dataset_profile, problem_types):
        methods.extend(_ordinal_seed_methods(modality=modality, metric=metric))
    if _is_sample_weight_profile(dataset_profile, problem_types):
        methods.extend(
            _sample_weight_seed_methods(
                modality=modality,
                metric=metric,
                weight_column=str(dataset_profile.get("sample_weight_column_hint") or "").strip() or None,
            )
        )
    if _is_text_generation_profile(dataset_profile, problem_types):
        methods.extend(_text_generation_seed_methods(modality=modality, metric=metric))
    if modality == "text" and _is_document_file_reference_profile(dataset_profile, problem_types):
        methods.extend(_document_file_reference_seed_methods(modality=modality, metric=metric))
    if _is_survival_profile(dataset_profile, problem_types):
        methods.extend(_survival_seed_methods(modality=modality, metric=metric))
    if _is_pairwise_profile(dataset_profile, problem_types):
        methods.extend(_pairwise_seed_methods(modality=modality, metric=metric))
    if _is_learning_to_rank_profile(dataset_profile, problem_types):
        methods.extend(_learning_to_rank_seed_methods(modality=modality, metric=metric))
    if _is_anomaly_detection_profile(dataset_profile, problem_types):
        methods.extend(_anomaly_detection_seed_methods(modality=modality, metric=metric))
    elif _is_unsupervised_profile(dataset_profile, problem_types):
        methods.extend(_unsupervised_seed_methods(modality=modality, metric=metric))
    if _is_ctr_profile(dataset_profile, problem_types):
        methods.extend(_ctr_seed_methods(modality=modality, metric=metric))
    if _is_recommender_profile(dataset_profile, problem_types):
        methods.extend(_recommender_seed_methods(modality=modality, metric=metric))
    if _is_forecasting_profile(dataset_profile, problem_types):
        methods.extend(_forecasting_seed_methods(modality=modality, metric=metric))
    if _is_detection_profile(dataset_profile, problem_types):
        methods.extend(_detection_seed_methods(modality=modality, metric=metric))
    if _is_segmentation_profile(dataset_profile, problem_types):
        methods.extend(_segmentation_seed_methods(modality=modality, metric=metric))
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
    dataset_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    priority = _campaign_needs_validation_redesign(campaign_state)
    dataset_profile = dataset_profile or {}
    modality = infer_modality(problem_types, dataset_profile)
    split_hint = str(dataset_profile.get("split_strategy_hint") or "").strip().lower()
    group_column_hint = str(dataset_profile.get("group_column_hint") or "").strip()
    has_group_hint = bool(group_column_hint or split_hint in {"group_kfold", "group-kfold", "groupkfold"})
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
    if has_group_hint or modality in {"image", "text", "multimodal", "rna", "bio"}:
        reason = "Group by subject/entity/file/source to avoid near-duplicate leakage."
        if group_column_hint:
            reason = f"Group by `{group_column_hint}` to avoid entity leakage between folds."
        profiles.append(
            {
                "profile_id": "entity_group_cv",
                "split_family": "group",
                "reason": reason,
                "priority": 0.88 if has_group_hint else (0.75 if priority else 0.5),
                "run_status": "planned",
                "offline_online_correlation": None,
                "adoption_status": "candidate",
                "group_column_hint": group_column_hint or None,
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
    profile_modality = _normalize_modality_key(dataset_profile.get("modality"))
    if profile_modality:
        return profile_modality
    if "multimodal" in text or "multi modal" in text or "vision language" in text:
        return "multimodal"
    if (
        "medical_imaging" in text
        or "medical imaging" in text
        or "medical image" in text
        or "dicom" in text
        or "ima" in text
        or "nifti" in text
        or "nrrd" in text
        or "nhdr" in text
        or "mha" in text
        or "mhd" in text
    ):
        return "medical_imaging"
    if any(term in text for term in _IMAGE_MODALITY_HINTS):
        return "image"
    if any(term in text for term in _SIGNAL_MODALITY_HINTS):
        return "signal"
    if any(
        term in text
        for term in (
            "array",
            "npy",
            "npz",
            "netcdf",
            "grib",
            "fits",
            "scientific array",
            "zarr",
            "ome-zarr",
            "ome zarr",
            "n5",
            "anndata",
            "h5ad",
            "loom",
            "single-cell",
            "single cell",
        )
    ):
        return "array"
    if "point_cloud" in text or "point cloud" in text or "lidar" in text:
        return "point_cloud"
    if any(term in text for term in ("geospatial", "geojson", "shapefile", "gis", "geopackage", "kml")):
        return "geospatial"
    if any(
        term in text
        for term in (
            "graph",
            "network",
            "node classification",
            "edge prediction",
            "link prediction",
            "knowledge graph",
            "graphml",
            "gexf",
            "edgelist",
            "adjacency",
        )
    ):
        return "graph"
    if any(
        term in text
        for term in (
            "annotation",
            "annotations",
            "coco",
            "yolo",
            "labelme",
            "pascal voc",
            "rle mask",
            "run length encoding",
            "bounding box",
            "keypoint",
        )
    ):
        return "annotation"
    if any(term in text for term in _TEXT_MODALITY_HINTS):
        return "text"
    if "timeseries" in text or "time" in text:
        return "timeseries"
    if any(
        term in text
        for term in (
            "rna",
            "protein",
            "bio",
            "molecule",
            "molecular",
            "smiles",
            "inchi",
            "selfies",
            "smi",
            "fasta",
            "fastq",
            "pdb",
            "mmcif",
        )
    ):
        return "rna" if "rna" in text else "bio"
    if any(term in text for term in _AUDIO_MODALITY_HINTS):
        return "audio"
    if any(term in text for term in _VIDEO_MODALITY_HINTS):
        return "video"
    if "tabular" in text:
        return "tabular"
    columns = " ".join(str(item).lower() for item in dataset_profile.get("columns", []) if isinstance(item, str))
    has_text_column = any(term in columns for term in ("text", "sentence", "prompt", "question", "caption", "review"))
    has_asset_column = any(
        term in columns for term in (*_IMAGE_MODALITY_HINTS, *_AUDIO_MODALITY_HINTS, *_VIDEO_MODALITY_HINTS)
    )
    if has_asset_column and has_text_column:
        return "multimodal"
    if any(term in columns for term in _IMAGE_MODALITY_HINTS):
        return "image"
    if any(term in columns for term in _AUDIO_MODALITY_HINTS):
        return "audio"
    if any(term in columns for term in _VIDEO_MODALITY_HINTS):
        return "video"
    if any(term in columns for term in _SIGNAL_MODALITY_HINTS):
        return "signal"
    if any(term in columns for term in _TEXT_FILE_REFERENCE_COLUMN_HINTS):
        return "text"
    if any(
        term in columns for term in ("dicom", "nifti", "scan", ".dcm", ".ima", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
    ):
        return "medical_imaging"
    if any(term in columns for term in ("point", "lidar", ".ply", ".pcd", ".las", ".laz")):
        return "point_cloud"
    if any(term in columns for term in ("geo", "geometry", "latitude", "longitude", "geojson", ".shp", ".gpkg")):
        return "geospatial"
    if any(
        term in columns
        for term in (
            "rna",
            "protein",
            "molecule",
            "molecular",
            "smiles",
            "inchi",
            "selfies",
            "sequence",
            "fasta",
            "fastq",
            ".pdb",
            ".cif",
            ".mmcif",
            ".sdf",
            ".mol2",
            ".smi",
            ".smiles",
            ".inchi",
            ".selfies",
        )
    ):
        return "rna" if "rna" in columns else "bio"
    if any(
        term in columns
        for term in (
            "graph_id",
            "node_id",
            "edge_id",
            "edge_index",
            "source_node",
            "target_node",
            "adjacency",
            ".graphml",
            ".gexf",
            ".edgelist",
            ".edges",
            ".mtx",
        )
    ):
        return "graph"
    if any(
        term in columns
        for term in (
            "annotation",
            "annotation_path",
            "coco",
            "yolo",
            "labelme",
            "bbox",
            "bounding_box",
            "rle",
            "mask",
            "keypoint",
        )
    ):
        return "annotation"
    if any(
        term in columns
        for term in (
            "array",
            ".npy",
            ".npz",
            "netcdf",
            ".nc",
            "grib",
            ".grib",
            "fits",
            ".fits",
            "ome.zarr",
            "zarr",
            ".zarr",
            ".ome.zarr",
            "n5",
            ".n5",
            "anndata",
            "h5ad",
            ".h5ad",
            "loom",
            ".loom",
        )
    ):
        return "array"
    if any(term in columns for term in ("text", "sentence", "prompt", "question", "caption", "review", "translation")):
        return "text"
    if any(term in columns for term in ("date", "time", "timestamp")):
        return "timeseries"
    return "tabular"


def _normalize_modality_key(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _MODALITY_ALIASES.get(normalized, normalized)


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
    payload = load_json_object(path)
    if payload is not None:
        payload["mode"] = mode
        return payload
    return {"version": 1, "slug": slug, "mode": mode, "methods": [], "active_method_ids": [], "blocked_method_ids": []}


def _load_registry_payload(path: Path) -> dict[str, object]:
    payload = load_json_object(path)
    return payload if isinstance(payload, dict) else {}


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
        "audio": [
            (
                "audio-spectrogram-transfer",
                "Waveform or mel-spectrogram pipeline with pretrained audio/CNN encoder when available",
                "strong_single",
            ),
            (
                "audio-handcrafted-feature-head",
                "MFCC/log-mel/statistical audio features with leak-safe sklearn head",
                "feature_variant",
            ),
        ],
        "video": [
            (
                "video-frame-backbone-transfer",
                "Sampled-frame pretrained image/video backbone with temporal pooling and TTA",
                "strong_single",
            ),
            (
                "video-metadata-motion-feature-head",
                "Clip metadata, frame statistics, and lightweight motion features with sklearn head",
                "feature_variant",
            ),
        ],
        "signal": [
            (
                "signal-statistical-feature-head",
                "Waveform summary, frequency-domain, and peak features with robust tabular head baseline",
                "strong_single",
            ),
            (
                "signal-1d-cnn-optional-branch",
                "Optional 1D CNN/transformer encoder for ECG/EEG/WFDB-style signal assets",
                "strong_single",
            ),
        ],
        "medical_imaging": [
            (
                "medical-imaging-slice-or-volume-transfer",
                "Medical slice/volume preprocessing with pretrained image/3D backbone when available",
                "strong_single",
            ),
            (
                "medical-imaging-windowing-metadata-head",
                "Modality-aware windowing, spacing metadata, and cached embedding head",
                "feature_variant",
            ),
        ],
        "array": [
            (
                "array-statistical-feature-head",
                "Numpy array shape/intensity/statistical features with robust sklearn/GBDT head",
                "strong_single",
            ),
            (
                "array-torch-dataset-encoder",
                "Torch Dataset over npy/npz assets with cached embeddings and lightweight head",
                "feature_variant",
            ),
        ],
        "point_cloud": [
            (
                "point-cloud-geometric-features",
                "Point-cloud geometric/statistical features with robust tabular head baseline",
                "strong_single",
            ),
            (
                "point-cloud-projection-deep-head",
                "Voxel/projection or point encoder branch with cached features and lightweight head",
                "feature_variant",
            ),
        ],
        "geospatial": [
            (
                "geospatial-geometry-feature-head",
                "GeoJSON/shapefile geometry features with spatial joins and robust tabular head baseline",
                "strong_single",
            ),
            (
                "geospatial-spatial-validation",
                "Spatial grouping, distance features, and leak-aware geographic validation profile",
                "validation_variant",
            ),
        ],
        "graph": [
            (
                "graph-topology-feature-head",
                "Graph topology, node degree, community, and edge features with robust tabular head baseline",
                "strong_single",
            ),
            (
                "graph-gnn-optional-branch",
                "Optional GNN or graph embedding branch for node, edge, link, and graph-level prediction tasks",
                "strong_single",
            ),
        ],
        "annotation": [
            (
                "annotation-format-conversion",
                "COCO/YOLO/LabelMe/RLE annotation conversion with strict sample-submission contract checks",
                "feature_variant",
            ),
            (
                "annotation-detection-segmentation-router",
                "Route boxes, masks, keypoints, or labels to detection/segmentation baselines and packaging",
                "strong_single",
            ),
        ],
        "artifact": [
            (
                "artifact-contract-validation",
                "Single-file or bundled artifact contract validation with local byte/shape checks",
                "feature_variant",
            ),
            (
                "artifact-runtime-packaging",
                "Kernel output packaging path for model weights, arrays, archives, or manifest bundles",
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
        "multimodal": [
            (
                "multimodal-vision-language-fusion",
                "Image/audio/video asset embeddings plus text embeddings with late-fusion or dual-encoder head",
                "strong_single",
            ),
            (
                "multimodal-metadata-feature-head",
                "Asset metadata, text statistics, and cached embeddings with leak-safe tabular head",
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
    if modality == "audio" and any(term in normalized for term in ("audio", "spectrogram", "mfcc", "waveform")):
        bonus += 0.16
    if modality == "video" and any(term in normalized for term in ("video", "frame", "clip", "temporal", "motion")):
        bonus += 0.16
    if modality == "medical_imaging" and any(
        term in normalized
        for term in ("dicom", "ima", "nifti", "nrrd", "nhdr", "mha", "mhd", "window", "volume", "slice", "3d")
    ):
        bonus += 0.16
    if modality == "array" and any(
        term in normalized
        for term in (
            "array",
            "npy",
            "npz",
            "netcdf",
            "grib",
            "fits",
            "zarr",
            "ome-zarr",
            "ome zarr",
            "n5",
            "h5ad",
            "loom",
            "torch",
            "embedding",
        )
    ):
        bonus += 0.16
    if modality == "point_cloud" and any(term in normalized for term in ("point", "lidar", "voxel", "projection")):
        bonus += 0.16
    if modality == "geospatial" and any(
        term in normalized for term in ("geo", "gis", "spatial", "shapefile", "geometry", "distance")
    ):
        bonus += 0.16
    if modality == "graph" and any(
        term in normalized for term in ("graph", "node", "edge", "link", "gnn", "networkx", "topology")
    ):
        bonus += 0.16
    if modality == "annotation" and any(
        term in normalized for term in ("annotation", "coco", "yolo", "labelme", "rle", "mask", "box")
    ):
        bonus += 0.16
    if modality == "rna" and any(term in normalized for term in ("rna", "sequence", "structure", "evaluator")):
        bonus += 0.16
    if modality == "bio" and any(
        term in normalized for term in ("bio", "protein", "molecule", "smiles", "fasta", "pdb", "structure")
    ):
        bonus += 0.16
    if modality == "artifact" and any(
        term in normalized for term in ("artifact", "bundle", "archive", "onnx", "weights", "model")
    ):
        bonus += 0.16
    if modality == "text" and any(term in normalized for term in ("transformer", "embedding", "rerank", "token")):
        bonus += 0.16
    if modality == "multimodal" and any(
        term in normalized for term in ("multimodal", "vision language", "clip", "fusion", "dual encoder", "embedding")
    ):
        bonus += 0.18
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
    if modality == "audio":
        return "Use log-mel/MFCC features with sklearn, or torch audio-style encoder paths when available."
    if modality == "video":
        return (
            "Use sampled-frame features with torchvision/OpenCV-style fallbacks and keep temporal pooling lightweight."
        )
    if modality == "medical_imaging":
        return (
            "Use image/volume preprocessing plus installed torch fallback; keep DICOM/IMA/NIfTI/NRRD/MHA/MHD "
            "metadata handling local."
        )
    if modality == "array":
        return "Use numpy statistical features with sklearn/GBDT, or cached torch embeddings when available."
    if modality == "point_cloud":
        return "Use geometric features with sklearn/GBDT, or cached voxel/projection features when torch is available."
    if modality == "geospatial":
        return "Use geometry-derived features, spatial joins, and leak-aware geographic validation with sklearn/GBDT."
    if modality == "graph":
        return "Use topology-derived features, node/edge aggregations, and sklearn/GBDT before optional GNN branches."
    if modality == "annotation":
        return (
            "Use annotation conversion, sample-contract validation, "
            "and lightweight vision baselines before deep branches."
        )
    if modality == "rna":
        return "Use official evaluator contracts, sequence/structure features, and lightweight postprocessing."
    if modality == "bio":
        return (
            "Use sequence, SMILES, or structure-derived features with official evaluator checks "
            "and sklearn/GBDT fallback."
        )
    if modality == "artifact":
        return "Use local artifact validation, manifest-aware packaging, and submit only the checked output path."
    if modality == "text":
        return "Use installed transformers/sklearn TF-IDF or cached embeddings fallback."
    if modality == "multimodal":
        return "Use cached asset/text embeddings plus a sklearn/GBDT fusion head before optional deep dual encoders."
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
    if modality in {
        "image",
        "audio",
        "video",
        "text",
        "multimodal",
        "timeseries",
        "medical_imaging",
        "array",
        "point_cloud",
        "geospatial",
        "graph",
        "annotation",
        "rna",
        "bio",
    }:
        optional.extend(["torch", "transformers"])
    if modality == "graph":
        optional.extend(["networkx", "torch-geometric"])
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
