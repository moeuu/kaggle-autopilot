# Ranked Candidate Pipelines

## 1. Space–Time Conversion Ledger — 41 raw tactical attributes

**Leak-free features/encodings:** No encodings, learned transforms, global bins, fitted thresholds, normalization, or outcome fields. Fixed event filters generate counts, distinct counts, duration totals, distance totals, positive displacement totals, and absolute displacement totals. Families cover possession transport, passing-option supply/selection/reception, off-ball space creation, phase conversion, opponent-mapped defensive blocks, and pressure suppression.

**Model and concrete hyperparameters:** `deterministic_rule_based_feature_generator`; 41 output columns; `issued_from_different_phase` defines phase starts; `lead_to_different_phase` defines phase exits; high-intensity bands are `hsr|sprinting`; penetrative runs are `behind|run_ahead_of_the_ball`; width-stretch runs are `pulling_wide|pulling_half_space|overlap|underlap`; coordinates are not flipped; event chunks are 200,000 rows.

**Expected runtime/memory:** 8–20 minutes on CPU, less than 4 GB host RAM, negligible GPU usage.

**Leakage/rule risk:** Low after the field denylist. Remaining interpretation risk comes from using source phase, line-break, run, and pressing annotations; document them as provided categorical event labels and never aggregate source model scores.

**Fallback:** Geometry-only 27-feature pipeline if annotation fields are missing or fail the strict rule audit.

## 2. Mandatory reference-aligned 39 raw reproduction

**Leak-free features/encodings:** Thirty-nine counts, duration totals, distance totals, zone-entry counts, line-break counts, run counts, phase-start counts, and pressure counts. No ratio, mean, xThreat/xPass field, score target, or attack-direction flip.

**Model and concrete hyperparameters:** `deterministic_reference_feature_generator`; 39 columns; 200,000-row chunks; exact phase-start de-duplication; expected 20 output rows; mandatory reference ranking signal recorded as 39.0 but never used as the validation score.

**Expected runtime/memory:** 3–10 minutes, less than 3 GB RAM.

**Leakage/rule risk:** Very low. Main risk is assuming that every formula in the public notebook remains compatible with the latest raw-aggregate wording; reimplement concepts from the frozen specification instead of copying unsafe code.

**Fallback:** Use as a sanity benchmark only and emit a `reference_reproduction_report.json` documenting every deliberate deviation.

## 3. Geometry-only raw safety fallback — 27 features

**Leak-free features/encodings:** Event counts, duration totals, distance totals, coordinate displacement totals, zone-entry counts, targeted/received counts, pressure closure distance, and goal-side recovery. No phase annotations, line-break annotations, run subtypes, source value fields, or learned transforms.

**Model and concrete hyperparameters:** `deterministic_geometry_feature_generator`; 27 columns; canonical left-to-right coordinates; 200,000-row chunks; expected 20 rows.

**Expected runtime/memory:** 2–6 minutes, less than 2 GB RAM.

**Leakage/rule risk:** Minimal, but the approach has the greatest concept and novelty risk because generic event volume can dominate.

**Fallback:** This is itself the terminal safe fallback. It may be promoted only when richer candidates fail a hard contract check; it must still produce the full dictionary, validation, provenance, and write-up package.
