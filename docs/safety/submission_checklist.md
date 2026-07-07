# Submission Safety Checklist

Use this checklist before real Kaggle submissions.

## 1. Rules Acceptance

- [ ] Competition rules accepted manually in browser.
- [ ] Rules URL is reachable: `https://www.kaggle.com/competitions/<slug>/rules`.

Tool behavior:
- `kagglebot` checks rules before submit.
- If not accepted, it exits with actionable guidance.

## 2. Authentication

- [ ] Kaggle CLI works: `uv run kaggle competitions list`.
- [ ] Credentials are configured via `~/.kaggle/kaggle.json` or env vars.
- [ ] Credentials are not committed into repository.

## 3. File Validation

- [ ] Submission columns match required format from the published sample submission file, such as `sample_submission.csv`, `.tsv`, `.jsonl`, `.parquet`, `.avro`, `.feather`, or compressed tabular variants like `.csv.zst` (or `submission_format.md` / `overview.md` when sample is placeholder/header-only).
- [ ] Row count matches the sample submission, or the runtime hidden/full test IDs for code/notebook competitions.
- [ ] ID alignment is correct when ID column exists.
- [ ] No NaN/inf in prediction columns.
- [ ] Non-CSV submissions use a supported format: TSV/TXT, JSON/JSONL/NDJSON, Parquet, Avro, Feather/Arrow IPC, Stata, XML, Excel, Pickle, SQLite DB, compressed tabular file, zip/tar bundle, prebuilt 7z/RAR archive, or supported code archive.
- [ ] Archive submissions are valid non-empty archives with no unsafe paths, duplicate member names, links/symlinks, encrypted zip members, or password-protected 7z/RAR members.
- [ ] Input archive extraction rejects password-protected 7z/RAR members before writing files.
- [ ] Output discovery ignores password-protected 7z/RAR candidates and unreadable 7z candidates instead of letting them shadow valid fallback submissions.
- [ ] Output discovery ignores corrupt compressed tabular candidates such as `.csv.gz` and `.csv.zst`, and invalid JSONL/NDJSON candidates, instead of treating unreadable bytes as payloads.
- [ ] Output discovery ignores symlink-only submission/prediction directories and manifest single-file directories instead of treating symlinks as usable payload files.
- [ ] Shapefile submissions include required sidecars; `submission.shp` is packaged with matching `.dbf` and optional `.shx`/`.prj`/`.cpg`/`.qix`/`.sbn`/`.sbx`/`.shp.aux.xml` files as a deterministic zip.
- [ ] Georeferenced raster image submissions preserve adjacent world-file/GDAL sidecars such as `.tfw`, `.pgw`, `.jgw`, `.wld`, `.prj`, `.aux.xml`, and `.ovr` when present; plain images without sidecars remain single-file submissions.
- [ ] GDAL VRT `.vrt` submissions include safe relative `SourceFilename`/`SourceDataset` raster source files; missing or unsafe VRT source paths are rejected before submit.
- [ ] ENVI `.hdr` raster submissions include safe same-stem or `data file =` sidecars such as `.dat`, `.bil`, `.bsq`, or `.bip`; ENVI headers are not misclassified as Analyze `.hdr/.img` pairs.
- [ ] MapInfo TAB submissions include same-stem `.dat`, `.map`, and `.id` sidecars; plain tab-delimited `.tab` files are not treated as MapInfo without sidecars or MapInfo context.
- [ ] MapInfo Interchange `.mif` submissions include the same-stem `.mid` sidecar when the published format asks for a MIF/MID pair.
- [ ] Analyze/NIfTI pair submissions include the matching same-stem `.hdr` or `.img` pair file; missing pair files are rejected before submit.
- [ ] MetaImage `.mhd` submissions include referenced `ElementDataFile` sidecars such as `raw/volume.raw`; unsafe or missing sidecar paths are rejected before submit.
- [ ] Detached NRRD `.nhdr` submissions include referenced `data file:` sidecars, including `LIST` members; unsafe or missing sidecar paths are rejected before submit.
- [ ] LAS/LAZ point-cloud submissions preserve adjacent projection/index sidecars such as `.prj`, `.wkt`, `.lax`, `.lasx`, and `.aux.xml` when present; plain LAS/LAZ files without sidecars remain single-file submissions.
- [ ] OBJ submissions include referenced `mtllib` material files and texture maps; unsafe or missing material/texture paths are rejected before submit.
- [ ] glTF `.gltf` submissions include external buffer/image URI sidecars, including percent-encoded local URI paths; unsafe or missing URI paths are rejected before submit.
- [ ] USD ASCII `.usd`/`.usda` submissions include referenced local `@asset@` sidecars; unsafe or missing asset paths are rejected before submit.
- [ ] Sharded model index submissions include every `weight_map` shard referenced by `*.safetensors.index.json` or `*.bin.index.json`.
- [ ] Classic ML model submissions such as CatBoost `.cbm`, XGBoost `.ubj`/`.xgb`/`.bst`, PMML `.pmml`, CoreML `.mlmodel`/`.mlpackage`/`.mlmodelc`, skops `.skops`, context-explicit pickle model `.pkl`/`.pickle`, and HDF5 model `.hdf`/`.hdf5` files are treated as model artifacts, not tabular/text fallbacks.
- [ ] Single-file code submissions such as `submission.py`, `submission.ipynb`, `submission.r`, and `submission.jl` are treated as code artifacts, not tabular/text fallbacks.
- [ ] Hugging Face/PEFT and MLflow model submissions include required config/tokenizer or `MLmodel` metadata when present; model directories with metadata plus weights/payloads are packaged as deterministic zip artifacts before submit.
- [ ] TensorFlow SavedModel directory submissions include `saved_model.pb` or `saved_model.pbtxt` and are packaged as deterministic zip artifacts before submit.
- [ ] TensorFlow checkpoint directory or index submissions such as `model.ckpt.index` include matching `.data-*-of-*` shards and optional `.meta`/`checkpoint` sidecars before submit.
- [ ] Plain `submission/`, `predictions/`, `masks/`, and similar final-output directories for multi-file submissions are packaged as deterministic zip artifacts only when they contain usable prediction files.
- [ ] Local directory submissions reject symlink members before building zip/tar archives.
- [ ] Submission duplicate hashes do not follow symlinks inside files or directory artifacts.
- [ ] Tabular fallbacks for requested non-tabular artifacts use a `*.tabular.csv` file plus `requested_output_path` metadata, not a disguised non-tabular payload.
- [ ] Extra tar code-archive members such as top-level `deck.csv`, `main.py`, and `cg/` are required only when the competition context explicitly says so.

Tool behavior:
- invalid files are rejected before CLI submit call.
- conservative column/order/id-suffix autofix can rewrite supported tabular formats without changing the intended file type.

## 4. Duplicate and Rate Limits

- [ ] Submission hash is not duplicate unless intentionally overridden.
- [ ] Local submit cooldown/rate checks pass.

Tool behavior:
- duplicate detection uses local ledger and deterministic hashes for files or directory submission artifacts
- retries are bounded and repeated fingerprints are aborted

## 5. Recommended Execution

Autopilot (default submit behavior):

```bash
uv run kagglebot autopilot <competition-url-or-slug> --compute local_gpu
```

Direct submit command:

```bash
uv run kagglebot submit <competition-url-or-slug> -f <submission-artifact> -m "message" --force
```

## 6. Security Constraints

- [ ] No secrets in prompts, kernel code, or logs.
- [ ] No automated rules acceptance.
- [ ] No external data usage unless competition rules explicitly allow it.
