# VersePulse Frontier

VersePulse Frontier is a wearable-first Scripture experience built for the
Kaggle **Scripture in New Frontiers** challenge. It detects meaningful workout
transitions from current and past biometric observations, retrieves a compatible
Scripture reference, and keeps authoritative Scripture separate from bounded,
safety-checked encouragement.

![VersePulse Frontier architecture](architecture.png)

## What is included

- `kernel.py` — the complete deterministic implementation used for the recorded run.
- `public_notebook.ipynb` — an explanatory notebook for the validation and safety evidence.
- `demo/` — a self-contained browser replay with no external CDN.
- `metrics.json` and `fold_metrics.csv` — grouped out-of-fold technical evidence.
- `api_contract_report.json` and `safety_eval.json` — API-boundary and safety checks.
- `cover.png` and `evaluation_dashboard.png` — submission media.

No API credentials, Kaggle credentials, private data, or organizer data are
stored in this repository.

## Reproduce

Use Python 3.11+ and the repository's `uv` environment:

```bash
uv sync
KAGGLEBOT_DATA_DIR=/path/to/organizer-files \
KAGGLEBOT_OUTPUT_DIR=/tmp/versepulse-output \
uv run python showcase/scripture-in-new-frontiers/kernel.py
```

The organizer directory is expected to contain the challenge-provided biometric
CSV, verse-movement mapping CSV, sample notebook, and sample submission. The
implementation discovers those files by schema rather than requiring credentials
or absolute paths.

The recorded run used Leave-One-Session-Out validation across three deterministic
seeds. Its selected technical fallback achieved grouped macro-F1 `0.6354`;
retrieval achieved Recall@3 `0.5833` and MRR@3 `0.3958`. These are offline
technical proxies, not an official judge score.

## API and safety boundary

In live mode, YouVersion is the authority for canonical Scripture and Gloo is
limited to schema-constrained encouragement. Secrets are read only from runtime
environment variables. Replay output is explicitly labeled and is never presented
as live API evidence.

Open `demo/index.html` locally to inspect the product replay and its auditable
decision trail.
