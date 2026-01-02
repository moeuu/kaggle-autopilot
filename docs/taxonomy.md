# Taxonomy

The taxonomy defines allowed tags and aliases. It lives at:

```
knowledge/taxonomy.yml
```

This file is JSON-compatible YAML. Example:

```json
{
  "tags": ["tabular", "text", "image", "timeseries"],
  "aliases": {"binary_classification": "binary"}
}
```

To extend:
1. Add new tags to `tags`.
2. Add aliases if needed.
3. Re-run `kagglebot bootstrap` to apply new tags.
