# kagglebot

Safe, non-interactive automation for Kaggle competitions with readiness-score-driven autopilot.

## Prerequisites

- Python 3.11+
- `uv` installed ([install](https://github.com/astral-sh/uv))
- Kaggle CLI on PATH
- Kaggle credentials (`~/.kaggle/kaggle.json` or env vars)
- **Competition rules accepted manually in browser** (required once per competition)
- Rules/overview/data are fetched from Kaggle during download; you can override rules with `--rules-file` (md/txt/html).

## Install

```bash
uv sync
```

For a persistent user-level autopilot directly backed by this clone:

```bash
./scripts/kagglebot-systemd install
```

The installer runs `uv sync --frozen`, registers the versioned units from `deploy/systemd/`, and enables the continuous
`watch` loop, Discord notifier, and Oracle auto-update timer. The timer checks npm every 15 minutes, installs the latest
`@steipete/oracle` release with the dedicated Node 24 runtime, and defers safely while an Oracle session is active. It
does not copy or fork the autopilot implementation: systemd runs `uv run kagglebot --force watch` from this checkout.
Agent model settings continue to come only from `[tool.kagglebot.agent]` in `pyproject.toml`.

Optional machine-specific watch settings can be copied from `deploy/systemd/watch.env.example` to
`~/.config/kagglebot-autopilot/watch.env`. Discord credentials belong in
`~/.config/kagglebot-autopilot/discord-notifier.env`; never commit either file. Manage the services with
`./scripts/kagglebot-systemd start|stop|restart|status|uninstall`. Use `loginctl enable-linger "$USER"` when the user
services must start at boot before the first login.

## Quick Start (Minimal Args)

Run autopilot with a single command:

```bash
uv run kagglebot --force autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --compute kaggle_gpu
```
Optional: add `--rules-file /path/to/rules.md` (md/txt/html) to override fetched rules text.

This will:
1. **Bootstrap**: Download data, profile dataset, query Knowledge Base for similar competitions
2. **Plan**: Codex summarizes context, Oracle/latest Pro runs research + frozen plan, then the shared `sol-xhigh` Codex implementer applies it; an unavailable or timed-out Oracle call falls back to `gpt-5.6-sol` with xhigh reasoning
3. **Initial model**: Agent implements an initial solution in `artifacts/<slug>/kernel/kernel.py` (all compute modes)
4. **Iterate**: Train → evaluate → diagnose → improve (default 5 iterations; long heavy local GPU runs are capped to 3)
5. **Submit + Decide**: Submit each iteration, wait for Kaggle score, then decide continue/stop
6. **Log**: Print Top1 public score and agent prompt/response to the terminal

Schema/method flexibility:
- Target/schema inference is file-name agnostic (not fixed to exact `train.csv`/`test.csv` naming only); when role names
  are missing, sample-submission columns can be used to infer the train/test pair by schema, and train/test column casing
  is aligned to the sample submission when safe. Feature column casing is also aligned between train/test when safe.
- When no sample file is bundled, context format hints (`submission_format.md`, `overview.md`, `data.md`, etc.) can
  synthesize sample submissions from tabular eval/test IDs or asset split IDs, preserving supported tabular suffixes such
  as `.jsonl` and compressed variants like `.jsonl.gz` / `.csv.zst` when they are specified.
- `submission_format.md` parsing accepts prose header/schema forms such as `CSV header: id,prediction`, not only fenced
  code blocks and markdown tables.
- `submission_format.md` parsing also recognizes non-tabular single-file suffixes such as `.npy`, `.npz`, `.nii.gz`,
  `.tif`, `.wav`, `.mp4`, `.ply`, `.onnx`, and `.bin` for array, medical-imaging, media, point-cloud, and model-artifact submissions.
- Sample synthesis for asset competitions also recognizes common inference image folders such as `test_images/`,
  `eval_images/`, public/private, validation, scoring, final, blind/challenge, and unlabeled directories.
- If `submission_format.md` examples use stem IDs like `case_001`, synthesized asset samples use stems instead of
  blindly emitting `case_001.jpg`.
- Asset-table synthesis resolves IDs written as stems, basenames, or relative paths such as `images/test/case_001.jpg`.
- Asset-table synthesis can use common label table names such as `annotations.csv`, `ground_truth.csv`, and
  `train_targets.csv`, not only `train_labels.csv`; Excel annotation files are accepted and unusable candidates are
  skipped in favor of the next matching file, and label ID aliases such as `filename` vs `image_id` are resolved.
- Separate-label tabular loading uses the same label-table aliases, so `ground_truth.csv` / `annotations.csv` can be
  merged into train features; join/target column casing such as `ID` vs `id` and numeric/string ID dtype mismatches are
  normalized, and ID-like aliases such as `image_id` vs `filename` can be joined.
- Generated/local kernels also rewrite hardcoded tabular `/kaggle/input/.../train.csv` style paths and pandas reader
  literals through the same suffix/case fallback resolver, so `.tsv`/`.tab`/`.psv`, `.parquet`/`.parq`/`.pq`, `.avro`, `.dta`, `.xml`, `.html`, `.yaml`/`.yml`,
  `.xlsx`/`.xlsm`, `.pkl`/`.pickle`, fixed-width `.fwf`, MATLAB `.mat`, SAS/SPSS, ARFF/OpenML-style `.arff`,
  LibSVM/SVMLight, SQLite, DuckDB `.duckdb`/`.ddb`, RDS/RData `.rds`/`.rda`/`.rdata`, compressed variants, and
  zip-wrapped single-table Parquet/Feather/Avro/ORC/Pickle/Excel/XLSB/Stata/SAS/SPSS inputs can be found without editing the notebook source by hand.
- JSON Lines inputs/submissions support `.jsonl`, `.jsonlines`, and `.ndjson` names, including gzip/bzip2/xz/zstd-compressed variants.
- Single-table zip inputs such as `train.csv.zip`, `test.tsv.zip`, `test.psv.zip`, `sample_submission.jsonl.zip`, and `.arff.zip`
  are read directly as tabular inputs without treating generic `.zip` submissions as CSV; binary table members such as
  Parquet, Feather/Arrow IPC, Avro, ORC, Pickle, Excel, XLSB, Stata, SAS, and SPSS are supported when the zip wraps one logical table, including
  dataset profiling and row-count guard paths.
- JSON table inputs can also unwrap common API-style envelopes such as `data`, `records`, `rows`, and `items` arrays,
  so wrapped `train.json`/`test.json`/`sample_submission.json` files still feed local baselines, generated-notebook
  baselines, sample discovery/row-count checks, submission validation/autofix, and submit-only notebook wrappers.
- Discovery, bootstrap/context caching, local, fold-runtime, generated-notebook, submit-wrapper, validation, and autofix tabular
  read/write paths stabilize problematic columns across supported formats: missing, blank, or pandas-generated
  `Unnamed: ...` names get `column_<n>` fallbacks, MultiIndex columns are flattened, and duplicate names receive numeric suffixes.
- Local, generated-notebook, and submit-only wrapper discovery safely extracts top-level and shallow nested
  zip/tar/tgz/tar.bz2/tbz2/tar.xz/txz/tar.zst/tzst/7z/RAR data archives before looking for train/test/sample files.
- Asset modality fallback classifies standalone tabular inputs such as CSV/TSV/TAB/PSV/TXT, JSON/JSONL/JSONLINES/NDJSON, YAML/YML, Parquet/PARQ/PQ,
  Avro Object Container File, HDF5/HDF datasets and column groups, AnnData `.h5ad` obs/X matrices, Loom `.loom` col_attrs/matrix stores, GeoPackage `.gpkg`/`.geopackage` attribute tables, Shapefile `.shp`/DBF `.dbf` attribute tables, KML/KMZ `.kml`/`.kmz` placemarks including compressed `.kml` inputs, NetCDF `.nc`/`.netcdf`/`.cdf`/`.nc4` table-like variables, FITS `.fits`/`.fit`/`.fts` binary tables, tabular-like NumPy `.npy`/`.npz` arrays, Feather/FTR/Arrow IPC, Excel/XLSM/ODS/XLSB, MATLAB MAT, RDS/RData, SAS/SPSS, ARFF, LibSVM/SVMLight, space-delimited DAT/TXT, fixed-width FWF, Stata, XML, HTML tables, Pickle, SQLite, DuckDB `.duckdb`/`.ddb`, zip-wrapped single tables, and compressed tabular variants as `tabular`, while keeping
  image/audio/video/medical/point-cloud/geospatial/graph assets higher priority when label tables are present.
  For 1D/2D NumPy train/test matrices, adjacent column sidecars such as `train_columns.txt`,
  `train.columns.json`, `train.schema.json`, `columns.txt`, or `schema.json` are used when their
  column count matches the array width.
  GeoJSON FeatureCollections can still be read as tabular inputs, while GeoJSON/GeoJSON-compressed,
  GeoJSON Lines/sequence, TopoJSON, OpenStreetMap `.osm`/`.osm.pbf`, MBTiles/PMTiles/MVT, GDAL VRT `.vrt`,
  and ENVI `.bil`/`.bsq`/`.bip` raster assets keep geospatial modality priority.
- Dataset profiling can also infer image/audio/video/medical-imaging/array/point-cloud modalities from table reference
  columns or filename values such as `image_path`, `scan_path`, `audio_path`, and `case_001.jpg`, including compressed
  DICOM/IMA/NIfTI/Analyze pair/NRRD/NHDR/MHA/MHD medical images such as `.dcm.gz`, `.ima.gz`, `.nii.xz`, `.hdr.gz`, `.img.zst`, and `.nrrd.zst`; generated notebooks
  also treat stem-only columns such as `protein_id`, `geo_id`, `network_id`, `embedding_id`, and `matrix_id` as possible
  asset references when matching files exist, and resolve test assets from common split aliases such as `eval/`,
  `public/`, `private/`, `validation/`, `scoring/`, `leaderboard/`, and `inference/`.
  Asset collection directories include image/media/medical names plus bio/geospatial/graph/document/embedding/matrix names
  such as `proteins/`, `molecules/`, `sequences/`, `geojson/`, `graphs/`, `documents/`, and `embeddings/`, plus
  detection/segmentation directories such as `masks/`, `labels/`, `annotations/`, and `bboxes/`.
- Generated notebook baselines add lightweight image metadata features for file references, including width, height,
  channel count, pixel count, aspect ratio, image frame count for multi-page images, and thumbnail intensity statistics.
- Generated notebook baselines also add lightweight mask-image metadata for image file references, including nonzero pixel
  count, foreground coverage, and single-channel label count for segmentation sidecars.
- Generated notebook baselines add best-effort medical-imaging metadata features, such as DICOM rows/columns/pixel spacing,
  NIfTI shape/voxel spacing, and NRRD/MHA/MHD header dimensions/spacing, while the shared medical suffix registry also
  recognizes Analyze `.hdr`/`.img` pairs, cryo-EM MRC/CCP4 volumes, and common microscopy formats such as ND2/LIF/LSM/DM3/DM4.
- Generated notebook baselines also add lightweight audio/video metadata features for file references, such as audio
  duration/sample rate/channels/frame count and video width/height/fps/frame count/duration.
- Signal/waveform modality detection recognizes common ECG/EEG/neurophysiology formats such as EDF/BDF/WFDB headers,
  NWB/TDMS/ABF, BrainVision `.vhdr`/`.vmrk`/`.eeg`, EEGLAB `.set`/`.fdt`, MNE FIF, Neuroscan CNT, and Micromed TRC.
- Generated notebook baselines also add lightweight point-cloud metadata features for Alembic/Blender/DAE/PLY/OBJ/X3D/USD/XYZ/VTK/Gmsh/STEP/IFC-style references,
  including gzip/bzip2/xz/zstd-compressed text-style point-cloud sidecars, such as point and face counts when they can be
  read from headers or text records. LAS/LAZ outputs preserve adjacent projection/index sidecars such as `.prj`, `.wkt`,
  `.lax`, `.lasx`, and `.aux.xml` when present. PLY outputs preserve `TextureFile` sidecars before submit, COLLADA
  `.dae` outputs preserve referenced image sidecars before submit, OBJ outputs preserve referenced material libraries and
  textures before submit, X3D outputs preserve referenced `url` sidecars before submit, USD ASCII outputs preserve
  referenced local `@asset@` sidecars before submit, and glTF outputs preserve
  external buffer and image URI sidecars, including percent-encoded local URI paths.
- Generated notebook baselines also add lightweight graph metadata features for GraphML/GEXF/GML/Matrix Market/edge-list
  references. The shared graph registry also recognizes knowledge-graph/linked-data formats such as RDF, Turtle,
  JSON-LD, N-Triples/N-Quads, OWL, and TriG, including gzip/bzip2/xz/zstd-compressed graph sidecars.
- Generated notebook baselines also add lightweight geospatial metadata features for GeoJSON/GeoJSONL/TopoJSON/OSM/KML/KMZ references, including
  compressed GeoJSON/KML sidecars, such as feature/placemark counts and coordinate bounding boxes. Georeferenced raster
  image outputs preserve adjacent world-file/GDAL sidecars such as `.tfw`, `.pgw`, `.jgw`, `.wld`, `.prj`, `.aux.xml`,
  and `.ovr` when those files are present. The shared geospatial registry also routes elevation/compressed raster formats
  such as HGT/DEM/DTED and ECW/MrSID as geospatial assets.
- Generated notebook baselines also add lightweight document metadata features for PDF/DOCX/HTML/Markdown/RTF/plain-text/subtitle references,
  including compressed text-style sidecars, such as page counts and text character/word/paragraph counts.
- Generated notebook baselines also add lightweight array metadata features for NumPy/Zarr/OME-Zarr/N5 and scientific-array
  references such as NetCDF/FITS/MAT/HDF5-family files, including gzip/bzip2/xz/zstd-compressed FITS files, array count,
  rank, leading dimensions, and element count without loading full array payloads when headers expose shapes; OME-Zarr and
  N5 directory stores can use nested `zarr.json` or `attributes.json` metadata.
- Generated notebook baselines also add lightweight model-artifact metadata features for ONNX, safetensors, and
  Hugging Face/PEFT-style tokenizer, processor, adapter, and config sidecars, such as graph node/input/output counts,
  tensor count, parameter count, index shard sizes, tokenizer vocabulary size, and config key counts when safe metadata
  APIs can read them. Classic ML model artifacts such as CatBoost `.cbm`, XGBoost `.ubj`/`.xgb`/`.bst`, PMML `.pmml`,
  CoreML `.mlmodel`/`.mlpackage`/`.mlmodelc`, skops `.skops`, TensorRT `.engine`/`.plan`, OpenVINO `.blob`,
  Rockchip RKNN `.rknn`, Hailo `.hef`, and Qualcomm/SNPE `.dlc` are treated
  as model submissions, including archive-bundle format hints that mention those model suffixes. Hugging Face/PEFT and
  MLflow model directories with config/tokenizer or `MLmodel` metadata plus
  weight files are packaged as deterministic zip submissions. TensorFlow SavedModel directories are recognized by `saved_model.pb`/`saved_model.pbtxt`
  markers and TensorFlow checkpoint directories or indexes such as `model.ckpt.index` preserve matching `.data-*-of-*`
  shards before deterministic zip submission.
- Generated notebook baselines also add lightweight bio file metadata features for FASTA/FASTQ and PDB/mmCIF/SDF/MOL2
  references, and the shared bio registry recognizes common genomics formats such as VCF/SAM/BAM/CRAM/GFF/GTF/BED.
  Common gzip/bzip2/xz/zstd-compressed sequence, genomics text, and structure files are routed as bio assets, with
  sequence length statistics and structure atom/residue/bond/molecule counts when cheap metadata is available.
- Generated notebook baselines also add lightweight detection/segmentation annotation metadata features for
  COCO/LabelMe-style JSON references, Pascal VOC/CVAT/Label Studio-style XML annotations, and YOLO-style text label
  sidecars, such as image, annotation, category, bbox, and segmentation counts.
- Generated notebook file-reference detection also recognizes stem-only text and transcript columns, such as `text_id`,
  `transcript_id`, `caption_id`, `prompt_id`, `subtitle_id`, and `transcription_id`, and split-aware asset directories
  such as `texts/`, `transcripts/`, `captions/`, and `subtitles/`.
- Dataset profiling recognizes short NLP classification features such as `review`, `question`, `prompt`, and `tweet`
  columns as `text` when their values look like natural language, not categorical codes.
- Text-generation targets such as `translation`, `summary`, `answer`, `caption`, and `response` are tagged separately
  from categorical text labels so method scouting can search retrieval/seq2seq/semantic-similarity approaches.
- Analyzer metadata also preserves text-feature classification/regression layouts with TF-IDF / embedding strategy
  defaults, instead of treating natural-language feature columns as ordinary categorical tabular columns.
- Local and generated-notebook text baselines use TF-IDF nearest-neighbor retrieval over prompt/text feature columns,
  with deterministic constant-text fallback when no usable text signal exists.
- Generic string targets such as `target` are also routed to text baselines when the training values look like natural
  language, even if `sample_submission` contains short placeholders such as `placeholder`.
- Dataset profiling promotes mixed asset-reference plus natural-language tables such as `image_path` + `question` or
  `caption` to `multimodal`, and method scouting adds CLIP/dual-encoder/late-fusion candidates.
- Repeated entity columns such as `patient_id`, `subject_id`, `user_id`, `session_id`, or `case_id` are surfaced as
  `group_column_hint` with `group_kfold` validation guidance; local tabular baselines use group-aware holdouts to avoid
  entity leakage in generic tabular competitions.
- Explicit row-weight columns such as `sample_weight`, `row_weight`, or `observation_weight` are removed from features
  and surfaced as `sample_weight_column_hint`; local tabular baselines and method scouting preserve them for weighted
  fitting and validation metrics.
- Dataset profiling marks delimiter-based multi-label targets, such as `labels` values like `cat dog` or
  `target` values like `cat,dog`, with `target_semantics: multi_label` and a `multi_label` tag.
- Dataset profiling and analyzer metadata also recognize multi-label indicator matrices where `sample_submission`
  exposes several binary label columns such as `toxic`, `obscene`, or `insult`, routing them to one-vs-rest
  multi-label strategy defaults instead of generic multi-target classification.
- Method scouting and implementation prompts carry `target_semantics`, so multi-label tasks get threshold-tuning and
  one-vs-rest candidate guidance instead of silently falling back to single-label classification assumptions.
- Analyzer metadata also recognizes delimiter-based multi-label targets, with F1 metric hints and one-vs-rest /
  per-label-threshold strategy defaults.
- Local and generated-notebook baselines also route delimiter-based single-column multi-label targets through text-style
  retrieval output, so placeholder values in `sample_submission` do not collapse them into single-class labels.
- When `sample_submission` exposes one binary/probability column per label, local and generated-notebook baselines expand
  delimiter-based multi-label targets into independent one-vs-rest label probabilities.
- For local binary classification baselines, metrics that require probabilities such as AUC/logloss emit probability
  submissions even when `sample_submission` uses integer `0` placeholders.
- Probability-like single target columns such as `isFraud`, `risk`, `score`, or `probability` are treated as probability
  submissions for local and generated-notebook binary baselines, even when the sample uses integer placeholders.
- Analyzer planning metadata reuses the same solver target/prediction-kind inference, so `isFraud` probability targets
  and low-cardinality numeric regression targets such as `sales` do not fall back to stale generic classification plans.
- Analyzer strategies also use `prediction_kind`, adding probability calibration/renormalization defaults for probability
  submissions and ordered-threshold/QWK defaults for ordinal targets.
- Class-probability sample columns can use either prefixes or suffixes, such as `class_cat`, `cat_probability`,
  `dog_proba`, or `class_0_probability`, and are mapped back to the observed class labels by name.
- Multi-column targets are profiled as `multi_output_regression` or `multi_target_classification`, and method scouting
  adds per-target head / multi-output wrapper candidates instead of collapsing them to a single target.
- Analyzer metadata also recognizes generic multi-output regression, multi-target classification, and mixed multi-task
  target layouts, with per-target head strategy defaults and MCRMSE/F1/mixed-target metric hints.
- Coordinate target columns such as `x/y/z` or `latitude/longitude` are preserved as structured coordinate regression
  instead of generic multi-output regression, with coordinate-axis validation and RMSE strategy hints.
- Sample submission columns such as `p10/p50/p90`, `q0.1/q0.5/q0.9`, or `lower/upper` are profiled as
  `quantile_regression` or `prediction_interval`, with pinball/interval-loss method scouting.
- Local and generated-notebook tabular baselines expand a single continuous training target into quantile,
  prediction-interval, or generic continuous submission columns such as `p10/p50/p90`, `lower/upper`, and
  repeated numeric prediction heads, preserving monotonic column order where the format implies one.
- Analyzer metadata also recognizes quantile and prediction-interval submission columns, with pinball/interval-score
  metric hints and non-crossing output strategy defaults.
- Ordered targets such as `severity`, `grade`, `stage`, `level`, or `rating` with small integer/category classes are
  profiled as `ordinal_classification`, with QWK/threshold-tuning method scouting.
- Local and generated-notebook baselines treat ordinal integer targets as continuous ordered scores and round/clip
  predictions back to valid label values for submission.
- Numeric target names such as `sales`, `price`, `amount`, or `count` stay on the regression path even when early data has
  only a few integer values.
- Non-negative integer count targets such as `count`, `demand`, `quantity`, `trips`, or `orders` are profiled as
  `count_regression`, with RMSLE metric hints and Poisson/log1p strategy defaults; local and generated-notebook tabular
  baselines fit these targets with `log1p` and clip count predictions to non-negative values.
- Rate/ratio targets such as `conversion_rate`, `win_probability`, `target_ratio`, or `percentage` are profiled as
  `bounded_regression`, with bounded-output strategy defaults; local and generated-notebook tabular baselines clip clear
  0..1 or 0..100 predictions back to the target range.
- Strongly right-skewed non-negative value targets such as `SalePrice`, `revenue`, or `amount` are profiled as
  `positive_skew_regression`, with RMSLE/log1p strategy hints; local and generated-notebook tabular baselines fit these
  targets with `log1p` and clip predictions to non-negative values.
- Shared tabular ensemble submission validation and prediction-range metadata keep probability clipping for probability
  outputs, but preserve explicit regression predictions and apply count/bounded/positive-skew clipping only when that
  target structure is known.
- Shared tabular blend helpers keep logit/rank blends for probability outputs and use weighted raw-scale blends for
  explicit regression outputs, scoring regression blends by RMSE/RMSLE-compatible loss and comparing seed support by
  generic score instead of AUC-only metadata.
- Survival/time-to-event layouts with event/time target pairs such as `efs` and `efs_time` are profiled as
  `target_semantics: survival`, use a concordance-index metric hint, and get event/censoring-aware method scouting.
- Local and generated-notebook baselines collapse survival event/time target pairs into a single numeric risk score when
  the sample submission asks for one `prediction`/score column.
- Analyzer metadata recognizes the same survival event/time plus single-score layout as `task=survival`, with
  concordance-index and censoring-aware strategy defaults.
- Pairwise/ranking layouts with paired feature columns such as `team1/team2`, `home_team/away_team`, or
  `model_a/model_b` are profiled as `target_semantics: pairwise` and get matchup/ranking calibration candidates.
- Analyzer metadata also recognizes those pairwise matchup layouts, with probability calibration and pair-difference
  strategy defaults.
- Search ranking layouts with `query_id` plus `document_id`/candidate columns and `relevance`/rank targets are profiled
  as `target_semantics: learning_to_rank`, with NDCG/LambdaMART-style method scouting.
- Local and generated-notebook baselines treat query/document `relevance` or rank targets as continuous ranking scores,
  even when the sample submission uses integer zero placeholders.
- Analyzer metadata also recognizes those query/document relevance layouts as `task=learning_to_rank`, with NDCG and
  grouped-ranking strategy defaults.
- Unlabeled train/test scoring layouts where `sample_submission` asks for anomaly/outlier/fraud-style scores are profiled
  as `target_semantics: anomaly_detection`, with unsupervised score-ensemble method scouting.
- Local and generated-notebook baselines can emit a lightweight unsupervised anomaly score for no-label layouts whose
  sample submission contains a numeric score column such as `anomaly_score`, `outlier_score`, or `risk_score`.
- Analyzer schema inference also accepts those no-label anomaly-score layouts by falling back to solver-inferred schema
  metadata instead of requiring the sample target column to exist in train.
- User-item interaction layouts such as `user_id` plus `item_id`/`ad_id` with `clicked` or `rating` targets are profiled
  as `target_semantics: ctr` or `recommender`, with click calibration or recommender-specific method candidates.
- Analyzer metadata also recognizes those user-item CTR/recommender layouts, with calibrated probability or rating-score
  strategy defaults.
- Vision submission columns such as `prediction_string` and `EncodedPixels` are profiled as `object_detection` or
  `segmentation`, so method scouting can route to detector or mask/RLE candidates.
- Analyzer metadata also recognizes those detection/segmentation submission columns, with mAP/Dice metric hints and
  prediction-string/RLE strategy defaults.
- Local and generated-notebook initial baselines now route RLE segmentation samples such as `EncodedPixels` to a valid
  empty-mask submission instead of forcing them through the generic tabular regressor/classifier path.
- Dataset profiling promotes tabular forecasting-style data with parsed datetime columns or future ordinal time keys
  such as `date_block_num` to `timeseries`, marks future-holdout targets as `target_semantics: forecasting`, and
  records a `timeseries_split` hint for safer validation.
- Local and generated-notebook tabular baselines use chronological holdouts when a future datetime or ordinal time column
  such as `date_block_num` is detected, reducing leakage in forecasting-style competitions.
- Local and generated-notebook tabular baselines derive lightweight calendar features from parseable datetime feature
  columns, such as year/month/day/day-of-week, while leaving numeric ordinal time keys intact and exposing the derived
  columns in run summaries.
- Analyzer metadata also recognizes those future temporal layouts as `task=forecasting`, with chronological validation and
  lag/rolling-feature strategy defaults.
- Dataset profiling also recognizes tabular geospatial features such as latitude/longitude pairs and WKT geometry
  columns as `geospatial`, even when the competition ships plain CSV/Parquet tables instead of GIS files.
- Dataset profiling recognizes graph/link-prediction tables with `edge_index` or source/destination node columns as
  `graph`, so plain CSV graph competitions can trigger graph-specific research and candidate planning.
- Dataset profiling recognizes tabular molecular and sequence data such as SMILES, protein sequences, and RNA sequences
  as `bio`/`rna`, so chemistry and bioinformatics competitions do not fall back to generic tabular planning.
- Analyzer metadata also preserves tabular asset-reference and domain modalities such as image/audio/video,
  medical-imaging, array, point-cloud, geospatial, graph, bio/RNA, and multimodal for domain-specific strategy defaults
  instead of collapsing them into plain tabular classification/regression.
- Local kernel output discovery accepts safe generic final names such as `predictions.csv`, `answer.parquet`, and
  `results.jsonl` when `submission.*` is absent, plus common aliases such as `preds.csv`, `sub.csv`, and
  `submission_final.csv`, while excluding sample/templates and intermediate artifacts such as OOF, CV, train, fold, and
  validation predictions. Generic non-tabular single-file outputs such as `predictions.npy`, `results.npz`,
  `answers.nii.gz`, and `submission.onnx` are also discovered. Model-artifact final names such as
  `model.safetensors.index.json`, `checkpoint.safetensors`, `adapter_model.safetensors`,
  `pytorch_model.bin.index.json`, and `weights.gguf` are accepted only for model artifact suffixes, so plain tabular
  helper files such as `model.csv` are not promoted to submissions. Sharded model index files preserve referenced
  `weight_map` shard files during local output copying and run artifact storage, and are packaged with their shards before
  submit. Hugging Face/PEFT-style model outputs also preserve adjacent config and tokenizer sidecars such as
  `adapter_config.json`, `config.json`, and `tokenizer_config.json` when those files are present, and model directories
  containing those metadata files plus weights such as `model.safetensors` are zipped as directories before submit. MLflow
  model directories with `MLmodel` plus model payloads, CatBoost `.cbm`, XGBoost `.ubj`/`.xgb`/`.bst`, PMML `.pmml`,
  CoreML `.mlmodel` / `.mlpackage` / `.mlmodelc`, and skops `.skops` outputs are also preserved as model artifacts.
  TensorFlow SavedModel output directories are detected via `saved_model.pb`/`saved_model.pbtxt` and zipped before submit. TensorFlow checkpoint
  output directories or indexes such as `model.ckpt.index` preserve matching `.data-*-of-*`, `.meta`, and `checkpoint`
  sidecars before submit. Plain non-empty `submission/`, `predictions/`, `masks/`, and similar final-output directories
  are also discovered as multi-file archive candidates when no explicit submission file exists. KML `.kml` outputs preserve safe local `<href>` sidecars such as icon and overlay images.
  Analyze/NIfTI pair `.hdr`/`.img` outputs
  preserve their matching same-stem pair file, while MetaImage `.mhd`
  and detached NRRD `.nhdr` outputs preserve referenced raw sidecars such as `raw/volume.raw` and package them before
  submit. If
  `KAGGLEBOT_SUBMISSION_FILENAME` is set explicitly,
  local output discovery honors non-tabular single-file artifacts such as `predictions.bin`, while still excluding
  metrics/planning side artifacts.
- Submit-only notebook wrappers can align tiny/public sample submissions against common runtime test table names such as
  `eval_features.*`, `PublicTest.*`, `scoring.*`, and `inference.*`, not only exact `test.*` files. They also preserve
  non-tabular single-file artifact names and compound suffixes such as `answers.nii.gz`, `predictions.npy`, and `results.npz`.
  NumPy `.npy`/`.npz`, AnnData `.h5ad`, Loom `.loom`, GeoPackage `.gpkg`/`.geopackage`, Shapefile `.shp`/DBF `.dbf`, GeoJSONL/GeoJSON sequence/TopoJSON/OSM/MBTiles/PMTiles/MVT, GDAL VRT `.vrt` raster datasets, ENVI `.hdr` raster headers, MapInfo TAB `.tab` bundles, MapInfo Interchange `.mif`/`.mid` pairs, KML/KMZ `.kml`/`.kmz` plus compressed `.kml` inputs, NetCDF `.nc`/`.netcdf`/`.cdf`/`.nc4`, and FITS `.fits`/`.fit`/`.fts` remain non-tabular submission artifact formats, but they can be loaded
  as tabular inputs when they are used for train/test feature tables; NumPy matrix inputs can use matching adjacent
  column/schema sidecars for stable `id`, feature, and target names.
- Submit-only notebook wrappers validate embedded zip/tar submissions for unsafe paths, duplicate members, symlinks, and
  encrypted members, and preflight embedded 7z/RAR submissions locally before packaging the wrapper kernel.
- Local sample-submission expansion uses the same broader test-table aliases, so placeholder samples can be expanded from
  `eval_features.*`, `scoring.*`, public/private, leaderboard, or inference tables before local validation while keeping
  mirrored `sample_submission.<suffix>` files expanded for suffix-aware kernels.
- Train/test discovery now treats `eval.*`, `scoring.*`, `inference.*`, leaderboard, and public/private tabular files as
  inference/test candidates, including inside generated Kaggle notebook baselines.
- Local kernel execution derives `KAGGLEBOT_SUBMISSION_FILENAME` from the discovered sample suffix when the user has not
  set it explicitly, so generated tabular runtimes emit `submission.jsonl`, `submission.xlsx`, etc. without manual env setup.
- Autopilot iteration output paths also honor explicit final filenames from `submission_format.md`, such as
  `model.safetensors.index.json`, before falling back to `submission<suffix>`.
- Generated Kaggle notebook baselines also derive the output suffix from the selected sample submission when no explicit
  submission filename env is present, avoiding a fallback to `submission.csv` for JSONL, Excel, compressed, or TSV samples.
- Sample-submission discovery accepts common aliases such as `sample_predictions.csv`, `example_submission.csv`,
  `prediction_template.csv`, and `submission_sample.csv` while still preferring canonical `sample_submission.*` files.
- Submission autofix maps common ID/prediction column aliases such as `filename` -> `image_id` and
  `pred_string` -> `prediction_string`, and validates detection-style `prediction_string` values as text.
- Multi-target sample submissions are supported at I/O/validation level
- ID-based and row-order-based submission alignment are both supported
- Local and generated-notebook submissions can align sample IDs composed from multiple test columns, such as
  `row_id=user_id_item_id`, `user_id:item_id`, `user_id.item_id`, or concatenated `user_iditem_id`, even when the
  composite ID column is absent from `test.csv`.
- Metric handling supports a broader set (e.g. AUC, logloss, F1, precision/recall, AP, RMSE/MCRMSE/MAE/MAPE/SMAPE/R2,
  Pearson/Spearman, QWK, pinball/interval score, NDCG, and concordance index)
- CV strategy auto-selection supports `TimeSeriesSplit` / `GroupKFold` / `StratifiedKFold` / `KFold`
- Model family selection is plugin-like with optional families (XGBoost/LightGBM if installed)

**Safe defaults**:
- Default max iterations: 5; long-running heavy `local_gpu` plans are automatically capped to 3
- No training time limit (accuracy-first)
- Internet ON by default, but autopilot forces `--internet off` when captured rules say notebook internet is disabled
- Each iteration submits and waits for Kaggle score before stop/continue decision

## Manual Commands

For more control, use individual commands:

```bash
# Bootstrap competition
uv run kagglebot bootstrap <competition>

# Implement solution
uv run kagglebot implement <competition>

# Train locally
uv run kagglebot train <competition> --compute local_gpu

# Submit manually
uv run kagglebot --force submit <competition> -f <submission-artifact> -m "message"

# Keep selecting entered competitions and running autopilot
uv run kagglebot --force watch --compute local_gpu

# Crawl Kaggle competition submission formats
uv run kagglebot crawl-submission-formats --output-dir artifacts/competition-submission-formats

# Audit only competitions already joined by this Kaggle account
uv run kagglebot crawl-submission-formats --entered-only --output-dir artifacts/entered-submission-formats
```

`crawl-submission-formats` discovers competition slugs from Kaggle list pages plus Kaggle API search sweeps,
then scrapes each competition's Kaggle overview/rules pages with headless Chrome and writes:

- `raw_submission_formats.jsonl`
- `normalized_submission_formats.csv`
- `summary.json`

`summary.json` includes `supported_competition_count` and a fail-closed `review_required` list for formats that could
not be mapped to a supported direct-upload extension, notebook runtime/output, or Writeup workflow.

Use `--max-prefix-depth` and `--max-pages-per-search` to broaden historical discovery coverage.

## Plan Configuration

Autopilot creates `artifacts/<slug>/plan.json` with agent-defined targets:

```json
{
  "target_metric": "rmse",
  "target_score": 0.13,
  "target_direction": "minimize",
  "score_source": "holdout",
  "holdout_frac": 0.2,
  "cv_folds": 5,
  "seed": 42,
  "internet": "on",
  "submit_policy": "always"
}
```

**Edit `plan.json` to override** targets or evaluation settings before re-running autopilot.

## Compute Targets

- `local_gpu` - Local training (GPU preferred; falls back to CPU when unavailable)
- `kaggle_gpu` - Kaggle notebook with GPU (T4)
- `kaggle_tpu` - Kaggle notebook with TPU (v3-8)

Use `--accelerator auto|gpu|tpu` to force specific accelerator.

Local GPU kernels default to a 1440-minute parent-enforced timeout. Runtime preflight is accuracy-preserving: it only rejects an exact source version after repeated measured runtimes prove that it cannot fit the configured timeout; it does not automatically shrink the model, folds, seeds, or input resolution.

Use `--hardware-profile auto|rtx3060|rtx5090|kaggle_p100|kaggle_t4|kaggle_t4x2|kaggle_rtx_pro_6000` to control planning and runtime scale knobs.
`auto` detects the local NVIDIA GPU when possible. The default local target is RTX3060-class accuracy-first execution:
strategies should keep the strongest feasible model families enabled, then scale batch size, chunks, precision,
folds/seeds, or candidate ordering to fit a single 12GB GPU. Stronger GPUs such as RTX5090 should scale through
`plan.json`/environment knobs without rewriting `kernel.py`.

Optional environment knobs:
- `KAGGLEBOT_MODEL_CANDIDATES="catboost,xgboost,lightgbm,torch,extra_trees"` to prioritize/limit model families
- Submission sample stage selection (for competitions that publish Stage 1/2 sample files):
  - Default preference is earlier stage samples (`Stage1` before `Stage2`) when multiple staged files exist.
  - `KAGGLEBOT_SUBMISSION_STAGE=<int>` or `KAGGLEBOT_SAMPLE_SUBMISSION_STAGE=<int>` forces preferred stage.
- Large competition download strategy (auto-enabled):
  - Kagglebot pins the current Kaggle CLI and lists files first. It streams ordinary datasets file-by-file to preserve
    nested paths, but automatically switches to the resumable all-data ZIP when a competition has at least 250 files
    and at least 25 remain. This avoids thousands of authenticated requests and persistent HTTP 429 throttling.
  - Downloads use atomic `.part` files, HTTP Range resume, exact size checks, free-space preflight, and a per-destination process lock.
  - Network disconnects, timeouts, HTTP 429, and HTTP 5xx retry indefinitely by default. Authentication, rules, missing resources, unsafe paths, and insufficient disk fail immediately because waiting cannot repair them.
  - Kaggle `Retry-After` is honored, requests are paced, and authentication is refreshed when a streaming retry opens a new session.
  - `KAGGLEBOT_DOWNLOAD_SINGLE_SHOT_FIRST=1` opts back into the legacy all-data ZIP first path.
  - `KAGGLEBOT_DOWNLOAD_AUTO_BULK=0` disables the high-file-count automatic ZIP switch.
  - `KAGGLEBOT_DOWNLOAD_BULK_FILE_COUNT_THRESHOLD=<int>` sets its total-file threshold (default `250`).
  - `KAGGLEBOT_DOWNLOAD_BULK_REMAINING_FILE_COUNT_THRESHOLD=<int>` sets its incomplete-file threshold (default `25`).
  - `KAGGLEBOT_DOWNLOAD_SPLIT_THRESHOLD_BYTES=<int>` controls the split threshold only when streaming is disabled.
  - `KAGGLEBOT_DOWNLOAD_RETRY_ATTEMPTS=<int>` retry attempts for transient failures (`0` = unlimited; default `0`).
  - `KAGGLEBOT_DOWNLOAD_RATE_LIMIT_RETRY_ATTEMPTS=<int>` retry attempts for HTTP 429 (`0` = unlimited; default `0`).
  - `KAGGLEBOT_DOWNLOAD_RETRY_BACKOFF_SEC=<float>` base retry backoff seconds (default `2.0`).
  - `KAGGLEBOT_DOWNLOAD_RETRY_MAX_BACKOFF_SEC=<float>` max retry backoff seconds (default `120.0`).
  - `KAGGLEBOT_DOWNLOAD_MIN_INTERVAL_SEC=<float>` minimum delay between per-file API requests (default `0.25`).
  - `KAGGLEBOT_DOWNLOAD_DISK_RESERVE_BYTES=<int>` free-space reserve retained during downloads (default `1 GiB`).
- Training progress logging (applies even when kernel code is quiet):
  - `KAGGLEBOT_TRAIN_PROGRESS=1|0` (default `1`) enable/disable forced periodic training logs
  - `KAGGLEBOT_PROGRESS_INTERVAL_SEC=<float>` watchdog silence threshold before emitting "no new logs" status (default `45`)
  - `KAGGLEBOT_MODEL_PROGRESS_INTERVAL_SEC=<float>` baseline per-model `fit()` tick interval (auto-adjusted by method/data size; default `12`)
  - `KAGGLEBOT_BOOSTING_LOG_EVERY=<int>` boosting eval-log period in iterations (`0` = auto, default)
- Custom metric hook: use metric string `custom:<module_or_py_path>:<function>`
- AURC / risk-coverage competitions are recognized as minimize metrics. Shared metric helpers accept one per-sample
  risk value and confidence value, sort by descending confidence, and integrate cumulative risk over coverage.
- Vision YOLO routing: if the sample has an image/file id column plus a detection output column such as
  `prediction_string` and YOLO folders exist (`images/train`, `images/test`, `labels/train`), Kagglebot uses a
  detection pipeline instead of tabular models. Formats with `right_place` still get the combined mAP/F1 baseline;
  generic detection formats fall back to mAP validation and `prediction_string` output. Training label files may use
  common aliases such as `filename`/`image_id` and `right_place`/`target`. Alternate YOLO layouts such as
  `train/images`, `test/images`, `train/labels`, `train_images`, `test_images`, and `train_labels` are also detected.
- RNA structure routing: if competition data looks like RNA sequence tables plus residue-level coordinate labels/sample
  (`train_sequences*`, `test_sequences*`, `train_labels*`, coordinate columns like `x_1,y_1,z_1`), Kagglebot classifies the
  task as `rna_structure` instead of generic tabular and preserves residue-anchor columns during validation.
- Analyzer metadata also routes RNA sequence/structure layouts to `task=rna_structure`, with residue-coordinate
  prediction kind, coordinate RMSE, and triplet/anchor-preserving strategy defaults.
- Vision training knobs:
  - `KAGGLEBOT_YOLO_PRETRAIN=1|0` (default `1`) toggles pretrained detector initialization.
  - `KAGGLEBOT_YOLO_EPOCHS=<int>` overrides detector epochs (time-budget caps still apply).

## Safety Guardrails

- ✅ **Readiness-score-driven loop**: stop/continue uses SRS (offline metric + uncertainty), with optional submission/rank guardrails
- ✅ **24h watch mode stays scoped to entered competitions**: `watch` never accepts rules or joins competitions automatically
- ✅ **Deterministic watch priority**: unsubmitted competitions always come first; within each submission tier the order is prize+medal, prize-only, medal-only, then neither, with the candidate score used only as a tie-breaker
- ✅ **Strict local validation before submit**: Column order, row count, ID integrity, numeric prediction checks
- ✅ **Duplicate prevention**: SHA256 hash check against `submissions/ledger.jsonl`
- ✅ **Rate limiting**: 5-min cooldown between submissions plus a bounded Kaggle submit CLI timeout
- ✅ **Notebook-submit accelerator integrity**: submit notebooks inherit the requested accelerator, so GPU code competitions are not silently staged as CPU. GPU notebooks select an official Kaggle CLI machine-shape ID from `KAGGLEBOT_SUBMIT_KERNEL_MACHINE_SHAPE`, `plan.json`'s `submit_machine_shape`, or the hardware profile, defaulting to `NvidiaTeslaT4`. ARC-AGI-3 instead defaults to `NvidiaRtxPro6000` and forces internet off for that competition's reserved RTX pool. Other competitions never receive RTX automatically, but may select the same official machine ID explicitly when Kaggle grants that competition access. Notebook metadata casing such as `nvidiaTeslaT4` is canonicalized before CLI use. Machine shape is included in the notebook slug so corrected hardware cannot inherit an older notebook's settings. Set `KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR=cpu|gpu|tpu` only for an explicit override. `KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC` bounds how long a remote kernel may remain `QUEUED` (default 1800s, `0` disables).
- ✅ **Notebook runtime fidelity**: code-competition submit packages load the staged plan beside `kernel.py`, inherit dataset/kernel/model sources from the required public reference when the plan enables reference reproduction, carry the locally selected pipeline contract into the inference run, require the exact expected output filename, and block Kaggle submission when remote metrics are missing/invalid, the reference reproduction gate remains blocked, or metrics show pipeline drift, missing assets, loss of a required reference path, or a material score regression. Dependency caches are redirected to `/tmp`; explicit cache trees under `/kaggle/working` are rejected before push.
- ✅ **Codex-reviewed code submission**: after a code-competition Notebook completes, Codex reviews the immutable Notebook/model/output/log evidence and must approve all four checks. A deterministic guard then re-hashes the evidence, rejects restored zero-denominator scores without full model-backed runtime evidence, fallback-only predictions, repeated runtime exceptions, dependency/cache output trees, wrong filenames, known daily-quota exhaustion, and duplicate Notebook-version identities. Only then may the guarded executor call Kaggle; the exact kernel/version/output identity is written to the ledger immediately after API success. Missing or malformed Codex output fails closed.
- ✅ **Competition-independent semantic preflight**: autopilot file/wrapper submissions and completed Code runtime outputs share one fail-closed prediction check. It blocks unchanged sample templates, row-constant predictions, copied multi-output heads, placeholder text, metrics-declared fallback output, selected/emitted pipeline drift, and recorded row-count/filename/hash mismatches. File/wrapper runs persist `submission_semantic_preflight.json` before any Kaggle API call.
- ✅ **No infinite submit loop**: Same submit-error fingerprint aborts the run immediately
- ✅ **Controlled retries**: Transient submit errors retry up to 3 times with backoff; permanent errors abort immediately. Kaggle outcome polling uses the exact normalized message sent to Kaggle and stops after about two hours if no matching result appears.
- ✅ **Terminal submission contract**: A submit-enabled leaderboard run cannot finish as `completed` with zero successful submissions; exact duplicate skips remain valid, while every other unmet submit obligation ends as `submit_failed`
- ✅ **Validated frozen evaluation specs**: Saved evaluation specs are schema- and direction-validated before reuse, and invalid or context-conflicting specs are regenerated
- ✅ **Best-effort dataset profiling**: Optional dataset profiling failures are persisted as structured `profile_error` metadata instead of aborting watch preflight before run-level recovery can start
- ✅ **Data-free kernel contract verification**: Generated kernels compile and export `contract_smoke()`, which Kagglebot calls in an isolated subprocess to validate every frozen pipeline without training or scoring. Explicit writeup deliverables bypass the generic labeled-competition-data gate because judged hackathons may intentionally provide no competition dataset; their project runtime and required attachment contracts still validate their own inputs and artifacts. Other frozen plans that require local training record a resumable data blocker when labeled competition data is unavailable. Contract-only verification creates no submission/OOF/checkpoint artifacts during the smoke and can resume when the declared inputs are available.
- ✅ **No rule automation**: Must accept rules manually in browser
- ✅ **Dry-run mode**: `--dry-run` skips external API calls (Kaggle CLI, Codex)
- ✅ **Conservative competition-mode inference**: `deliverable_mode` is canonicalized to `leaderboard|writeup`, Kaggle `Writeups` wording and legacy `csv` aliases are accepted, and negative mentions like `not a judged/writeup competition` do not disable leaderboard submission
- ✅ **Explicit submit mode**: `submit_mode` is tracked separately as `file|notebook`, so notebook-only leaderboard competitions no longer get conflated with writeup competitions
- ✅ **Winner-mode iteration policy**: leaderboard runs default to a near-first-place target (`target_medal=winner`, `target_rank_percentile=0.001`) so `minor_tuning` is suppressed until the run reaches the target rank band
- ✅ **Top1 campaign state**: autopilot CLI defaults to `--campaign-mode top1`, writes `context/campaign_state.json`, `context/candidate_registry.json`, `context/reference_reproduction_report.json`, `context/experiment_graph.json`, and iteration portfolio/blend/allocator reports; submissions that regress against the historical/champion public baseline are blocked unless they are calibration probes or explicitly forced
- ✅ **Experiment graph execution**: `--portfolio-execution off|serial|parallel|budgeted` turns the top1 portfolio into dependency-aware candidate nodes and writes per-candidate manifests, metrics, diagnostics, and `graph_execution_report.json`
- ✅ **Competition-specific method scout**: `--method-scout auto|off|refresh` builds `method_scout_queries.json`, `method_registry.json`, and `validation_registry.json` from the competition modality, metric, research sources, Kaggle context, and public-regression signals
- ✅ **Research source registry and Validation Lab**: `--research-scout auto|off|refresh` writes attributed source evidence to `source_registry.json`; `--validation-lab auto|off|force` calibrates split profiles and records `validation_lab_report.json`
- ✅ **Exhaustive top1 artifacts**: `--top1-exhaustive` enables safe exhaustive defaults and writes `win_contract.json`, `private_robustness_report.json`, `portfolio_optimizer_report.json`, and `top1_exhaustion_report.json` so the run records explored win paths, submit value, and remaining blockers
- ✅ **High-accuracy tabular planning guardrails**: large tabular binary problems with meaningful categoricals must keep multi-family search active (CatBoost + XGBoost + LightGBM/second variant + OOF blend candidate)
- ✅ **Reference input recovery**: required reference notebooks now emit `context/reference_inputs_manifest.json`, and `--download` stages referenced datasets/competitions into `context/reference_inputs/` for kernels that depend on external/original data
- ✅ **Competition-scoped policy overrides**: optional `artifacts/<slug>/context/competition_policy.json` can tighten notebook selection, reference-input recovery, repair signals, and fallback evaluation without changing defaults for other competitions
  - Policy files can also declare generic `required_capabilities` and `execution_hints`, including local-kernel `kernel_contract` guards, so competition-specific win conditions stay in artifacts while `src/` only gains reusable orchestration/runtime features
- ✅ **Online mismatch guardrails**: when CV improves but public LB regresses, the next iteration prioritizes validation redesign with group/time/leak/proxy split candidates before spending submissions on model-only changes
- ✅ **Candidate-selection guardrails**: when kernels report multiple candidates, autopilot blocks submissions where the selected pipeline wins only on CV while another candidate has materially better holdout/validation, especially when prediction distribution collapses to sparse or constant-like outputs
- ✅ **Oracle-gated `sol-xhigh` implementation**: every Codex edit that follows an Oracle response uses the shared `oracle_implementation_*` settings (`gpt-5.6-sol`, `xhigh`, `sol-xhigh`, semantic `xhigh`). Pre-Oracle brief extraction and explicitly selected non-Oracle legacy flows remain on the normal Codex profile.
- ✅ **Transactional Oracle/Codex self-improvement loop**: `watch` periodically analyzes recent errors, top1 gaps, submit outcomes, and verified reusable-skill usage, and also reacts once to each new normalized watch-failure fingerprint. Preflight failures are promoted even when no run directory exists. It then requires a clean, committed, pushed repository baseline with its GitHub URL and exact SHA before Oracle. Codex starts only after a structured Oracle plan and baseline revalidation; `watch` re-execs between competition cycles whenever repository HEAD changes, including pre-publish repair commits, so the next competition cannot run stale source. Completed browser consultations are archived by Oracle, with authenticated CDP retry and a persistent warning report when archival cannot be verified; an archival bookkeeping failure does not discard a valid Oracle answer.
- ✅ **Reliable full Oracle context delivery**: canonical text sources are losslessly consolidated into one attachment. Browser runs split permitted competition archives into ordered parts below the remote-Chrome 20 MiB transfer ceiling, record source SHA-256 and reconstruction instructions, and never reuse a transcript left by an older run.
- ✅ **Automated judged Writeups**: writeup competitions resolve explicitly required notebook outputs, archive attachments, tracks, and “Attached Public Notebook” requirements from the persisted plan and official competition context, so artifacts such as `features.csv` or `submission.zip` are validated by name instead of being rejected as non-submissions. The bundle seals a 560×280 card image, using a deterministic neutral fallback when no authored card exists. When a private notebook is required, submission-enabled `--force` runs publish it, download and hash-check the required outputs against the locally validated artifacts, and add the private notebook link to the report. Organizer-evaluated archives may remain explicitly unscored locally; they are hash-sealed, attached through Kaggle's authenticated Projects/Writeup UI, and submitted only after the exact track, filename, card image, and enabled 5/5 checklist are observed. The complete report/notebook/artifact/card identity is recorded and duplicate or ambiguous attempts are never resubmitted automatically unless a read-only Kaggle check explicitly proves the project is still a draft. Before an active writeup run is resumed, the same exact-title project-card check can reconcile only Kaggle's explicit `SUBMITTED` state, persist the verified URL, close the local run, and emit a Discord completion event.

## Top1 Public Leaderboard (Reference)

The public leaderboard leader's score is fetched and stored in `context/top1_public.json`.
Top1 campaign mode also stores the current campaign baseline, top1 gap, candidate IDs, and validation correlation in
`context/campaign_state.json`, with per-candidate metadata in `context/candidate_registry.json`.
Autopilot uses this as a reference signal (for diagnostics and rank-based major-overhaul guardrails), while primary loop control uses readiness score (SRS).

## Artifacts Layout

```
artifacts/<slug>/
  meta.json                      # Competition metadata
  plan.json                      # Agent-defined targets (editable)
  context/
    dataset_profile.json         # Dataset statistics
    research_sources.jsonl       # Strategy web-research log (working copy)
    research_summary.md          # Ranked research shortlist (working copy)
    kaggle_discovery.json        # Ranked Datasets/Models/Code/Discussions/Arena/Benchmarks metadata
    kaggle_discovery.md          # Oracle/Codex-readable Kaggle ecosystem shortlist
    research_storage.json        # Mapping to persisted knowledge paths
    method_scout_queries.json    # Competition-specific method discovery queries
    source_registry.json         # Attributed research/notebook/repo/discussion sources and planned retrieval queries
    method_registry.json         # Ranked/blocked method candidates for portfolio planning
    validation_registry.json     # Split redesign candidates and validation priority
    validation_lab_report.json   # Validation profile evidence and active split calibration
    win_contract.json            # Top1 score, validation, method, and submission done-definition contract
    private_robustness_report.json # Candidate public-overfit, correlation, and baseline-regression risks
    top1_exhaustion_report.json  # Remaining legal/executable top1 work and exhaustion status
    reference_reproduction_report.json # Mandatory reference baseline gate and attribution
    experiment_graph.json        # Candidate DAG and portfolio execution state
    campaign_outcomes.jsonl      # Method/profile/category outcome journal for self-improvement
    sample_submission.<suffix>   # Required submission format, preserving published suffix when possible
    sample_submission_head.csv      # Text preview of sample submission
    top1_public.json             # Leaderboard leader snapshot
    rules_url.txt                # Competition rules URL
    rules.md                     # Rules markdown (fetched or from --rules-file)
    rules.html                   # Rules HTML (if provided)
    overview.md                  # Competition overview (if available)
    data.md                      # Data description (if available)
    submission_format.md         # Submission format (if available)
    knowledge_hints.txt          # Similar competitions + hints
    agent/
      brief_for_strategy.md      # Codex brief
      strategy_plan.md           # GPT strategy section
      codex_instructions.md      # GPT implementation instructions
      strategy_transcript.txt    # Raw GPT stage output
  kernel/
    kernel.py                    # Authoritative kernel entrypoint (all compute modes)
    *.py                         # Optional competition-specific helper modules imported by kernel.py
  prompts/
    codex_plan_and_implement.md   # Initial plan + implementation prompt
    codex_improve.md             # Improvement iteration prompt
  kernels/
    <run-id>/                    # Kaggle kernel workspace
  runs/<run-id>/
    run.json                     # Run configuration and status
    run_state.json               # Submit stage state (attempted/ok/last fingerprint)
    submit_attempts.jsonl        # Submit attempts (success/failure/retry/abort/skip)
    iter-<k>/
      metrics.json               # Offline evaluation results
      diagnostics.md             # Agent-readable performance analysis
      submission.<suffix>        # Tabular predictions for this iteration, preserving selected output suffix
      submission_manifest.json   # Canonical submission artifact manifest for single-file/bundle outputs
      agent/
        prompt.md                # Codex input
        codex_last_message.txt   # Codex output summary
  submissions/
    ledger.jsonl                 # Deduplication log (append-only)
```

Knowledge Base lives in:
- `knowledge/kb.sqlite`
- `knowledge/taxonomy.yml`
- `knowledge/playbooks/*.md`
- `knowledge/skills/*.md`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent)

## Documentation

For detailed guides, see:
- [docs/autopilot.md](docs/autopilot.md) - Autopilot usage, configuration, troubleshooting

Submission artifacts are no longer assumed to be `submission.csv` only. Tabular submissions preserve supported output suffixes such as CSV, TSV/TAB/PSV/TXT, JSON/JSONL/JSONLINES/NDJSON, YAML/YML, Parquet/PARQ/PQ, Avro, HDF5/HDF, Feather/FTR/Arrow IPC, Stata, XML, Excel/XLSM/ODS, Pickle, and gzip/bzip2/xz/zstd-compressed tabular files. For non-tabular competitions, autopilot can carry a `submission_manifest.json` that points to a single-file artifact, a bundle, or a multi-file zip/tar/tar.zst staging directory; prebuilt 7z/RAR submission archives are also detected when competition context requires them. Directory array submissions such as `.zarr`, `.ome.zarr`, and `.n5` stores are zipped with their nested contents before submit. Shapefile submissions are treated as multi-file artifacts: `submission.shp` is zipped with the matching required `.dbf` sidecar and optional `.shx`/`.prj`/`.cpg`/`.qix`/`.sbn`/`.sbx`/`.shp.aux.xml` family files before submit. Georeferenced raster image submissions such as `.tif`, `.png`, or `.jpg` are zipped with adjacent world-file/GDAL sidecars such as `.tfw`, `.pgw`, `.jgw`, `.wld`, `.prj`, `.aux.xml`, and `.ovr` when those files are present; plain images without sidecars remain single-file submissions. GDAL VRT `.vrt` submissions are zipped with safe relative `SourceFilename`/`SourceDataset` raster sources before submit, preserving nested VRT layouts and rejecting missing or unsafe source paths. ENVI `.hdr` raster submissions are zipped with safe same-stem or `data file =` sidecars such as `.dat`, `.bil`, `.bsq`, and `.bip` before submit; ENVI headers are excluded from Analyze/NIfTI `.hdr`/`.img` pair detection. MapInfo TAB submissions are zipped with same-stem `.dat`, `.map`, and `.id` sidecars plus optional `.ind` files before submit; plain tab-delimited `.tab` files remain tabular unless MapInfo sidecars or context are present. MapInfo Interchange `.mif` submissions preserve and zip the matching same-stem `.mid` sidecar when present, and missing `.mid` sidecars are rejected when the submission format explicitly requests a MIF/MID pair. KML `.kml` submissions are zipped with safe relative local `<href>` sidecars such as icon and overlay images before submit, including percent-encoded local paths, while inline `data:` and same-document fragment references remain embedded. Analyze/NIfTI pair `.hdr`/`.img` submissions are zipped with the matching same-stem pair file before submit, and missing pair files are rejected before the Kaggle CLI call. MetaImage `.mhd` submissions are zipped with safe relative `ElementDataFile` sidecars before submit, and missing or unsafe sidecar paths are rejected before the Kaggle CLI call. Detached NRRD `.nhdr` submissions are zipped with safe relative `data file:` sidecars, including `LIST` members, before submit. LAS/LAZ `.las`/`.laz` submissions are zipped with adjacent projection/index sidecars such as `.prj`, `.wkt`, `.lax`, `.lasx`, and `.aux.xml` when those files are present; plain LAS/LAZ files without sidecars remain single-file submissions. PLY `.ply` submissions are zipped with safe relative `TextureFile` sidecars before submit. COLLADA `.dae` submissions are zipped with safe relative external image URI sidecars before submit. X3D `.x3d` submissions are zipped with safe relative `url` sidecars before submit. USD ASCII `.usd`/`.usda` submissions are zipped with safe relative `@asset@` sidecars before submit. OBJ submissions are zipped with safe relative `mtllib` material libraries and texture maps before submit, and missing or unsafe material/texture paths are rejected before the Kaggle CLI call. glTF `.gltf` submissions are zipped with safe relative external buffer/image URI sidecars, including percent-encoded local URI paths, before submit, while inline `data:` URIs remain embedded. Sharded model index submissions such as `model.safetensors.index.json` and `pytorch_model.bin.index.json` are zipped with their referenced `weight_map` shards before submit, and missing shard files are rejected before the Kaggle CLI call. Hugging Face/PEFT-style model weight submissions such as `adapter_model.safetensors` are zipped with adjacent config/tokenizer sidecars when present. When a tabular baseline can only emit a fallback for a requested non-tabular artifact such as `answers.nii.gz`, it writes `answers.tabular.csv` plus `requested_output_path` metadata instead of disguising CSV bytes as the requested artifact. Common manifest aliases such as `artifact_path`/`artifactPath`, `bundle_dir`/`stagingDir`, `folderPath`, list or dict `files`/`filePaths`/`entries`, member globs such as `bundle/*.tif`, directory members such as `bundle/`, and member path keys like `sourcePath`/`localPath` are accepted. Top-level submission and staging path values can also be path objects such as `{ "path": "predictions.zarr" }` or `{ "sourcePath": "answers.nii.gz" }`, and metadata-only nested objects are skipped when another path alias is available. Bundle members can also specify zip/tar internal paths via dict keys or `targetPath`/`archivePath`; glob members can target an internal directory such as `targetPath: masks/`, and the same source file may be included under distinct internal paths. Directory bundles preserve empty subdirectories when building zip/tar/tar.zst artifacts. Duplicate zip/tar internal member names are rejected, existing zip/tar/7z/RAR archives are opened before submit or output selection, and unsafe paths, duplicate members, symlinks, encrypted zip members, password-protected RAR members, or empty archives are rejected. Generic tar submissions are accepted as valid non-empty archives; legacy code-archive contracts such as top-level `deck.csv`, `main.py`, and `cg/` are enforced only when competition context explicitly names them. Artifact class values like `multi_file_zip`/`multiFileZip` and one-level `submission`/`bundle` payloads are accepted. Expanded bundle members are deduplicated while preserving order. When `artifact_class` is omitted, bundle/staging/member fields are used to infer bundle handling, including member-only manifests without a staging directory. Invalid/non-object manifests are skipped; if only nested manifests are found, the newest recursive match is preferred.

Single-file code submissions such as `submission.py`, `submission.ipynb`, `submission.r`, and `submission.jl` are treated as code artifacts rather than tabular/text fallbacks.
Context-explicit pickle/HDF5 model submissions such as `submission.pkl`, `submission.pkl.zst`, and `submission.hdf5` are treated as model artifacts, while ordinary pickle/HDF table submissions remain tabular.

Hugging Face/PEFT, MLflow, TensorFlow SavedModel, and TensorFlow checkpoint directory submissions are recognized by config/tokenizer plus weight files, `MLmodel` plus a model payload, `saved_model.pb`/`saved_model.pbtxt`, or `checkpoint` plus matching `.index`/`.data-*-of-*` marker files and packaged as deterministic zip artifacts before submit.
For asset-based competitions, fallback data synthesis can resolve common image, audio, video, medical-imaging, array, and point-cloud file references into `asset_path` columns when labels and sample submissions identify files by stem or filename.
- [docs/knowledge.md](docs/knowledge.md) - Knowledge Base system for cross-competition learning
- [docs/taxonomy.md](docs/taxonomy.md) - Tag taxonomy for competition similarity
- [docs/architecture.md](docs/architecture.md) - Control flow and safety gates (developer-focused)

## Testing

```bash
# Fast default: excludes broad orchestration/runner suites and competition artifacts.
uv run pytest -q

# Full mocked suite, including slow orchestration tests.
uv run pytest -q -m "not competition_artifact"

# Only the slow suite.
uv run pytest -q -m "slow and not competition_artifact"
```

## Example Workflow

```bash
# 1. Run autopilot (will bootstrap, plan, iterate, and submit when target met)
uv run kagglebot --force autopilot titanic --compute kaggle_gpu

# 2. If needed, edit plan.json to adjust target_score or evaluation strategy
nano artifacts/titanic/plan.json

# 3. Re-run autopilot with adjusted target
uv run kagglebot --force autopilot titanic --compute kaggle_gpu

# 4. Resume a crashed run (same competition)
uv run kagglebot --force autopilot titanic --compute kaggle_gpu --resume-run-id <run-id>
# or resume the latest run automatically
uv run kagglebot --force autopilot titanic --compute kaggle_gpu --resume-latest

# 5. Check Knowledge Base for learnings
uv run kagglebot knowledge show titanic

# 6. Find similar competitions
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```

## Notes

- **Non-interactive**: No prompts for input. All decisions via CLI flags or `plan.json`.
- **Crash recovery**: use `--resume-run-id <run-id>` (from `artifacts/<slug>/runs/<run-id>/`) or `--resume-latest` to continue a prior run. Interrupted Oracle-to-Codex improvement, kernel-fix, autofix, and repository self-improvement workflows are checkpointed and completed before watch can launch another kernel. Local-kernel training checkpoints are moved out of the disposable staging directory before it is recreated and are restored only when the fully staged kernel and plan fingerprints match exactly; incompatible checkpoints remain preserved for manual recovery.
- **Repair concurrency and source reload**: Codex sessions are serialized per repository. Repair snapshots always include `src/`; a source edit reloads the process and resumes the same run, while a repeated reload request for the identical source generation fails closed. Watch restarts append `resumed` rather than another `started` lifecycle event, and lifecycle notification delivery is deduplicated by compute, event type, competition, and run ID.
- **Reference artifact provenance**: staged Kaggle datasets retain the authoritative `dataset-metadata.json` response alongside downloaded files. Kernels can therefore validate the actual owner, slug, and license evidence instead of inferring licensing from filenames or an unrelated attached dataset.
- **Submit resume behavior**: resume can continue submitting new iteration outputs in the same run; duplicate submission SHA is skipped unless forced, the first valid leaderboard checkpoint is submitted when submit is enabled even if it is produced after iteration 1, and daily/rolling 24h submission limits are honored when rules expose them.
