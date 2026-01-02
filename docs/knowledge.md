# Knowledge Base

The knowledge base stores cross-competition learnings to help new runs.
It is persisted in `knowledge/kb.sqlite` and governed by `knowledge/taxonomy.yml`.

## Schema (minimum)

- `competitions(slug, url, metric, task_type, created_at, last_seen_at)`
- `tags(tag)`
- `competition_tags(slug, tag)`
- `runs(run_id, slug, started_at, compute, goal_metric, goal_score, direction)`
- `iterations(run_id, iter, score_source, offline_value, offline_std, top1_public_score, met_target, git_commit, created_at)`
- `improvements(run_id, iter, summary, delta_offline, created_at)`

## Tagging

Tags are generated deterministically from dataset profiles:
modality (`tabular/text/image/timeseries`), task (`regression/binary/multiclass`),
size buckets, missingness, and high-cardinality categories.

## CLI

```bash
uv run kagglebot knowledge show <slug>
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```
