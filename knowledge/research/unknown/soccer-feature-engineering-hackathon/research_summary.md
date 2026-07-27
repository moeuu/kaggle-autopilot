# Ranked candidate pipelines

## 1. Relational Opportunity–Realization–Response Graph 50

**Leak-free features/encodings:** No encoder, target, fitted bin, normalization, external data, or outcome field. Fixed joins connect possession → targeted option → linked off-ball run and defensive engagement → affected line-breaking option. The 50 outputs are counts, distinct counts, positive distance sums, and raw distance totals. Fixed thresholds are `n_passing_options>=3`, `n_passing_options_ahead>=3`, `n_passing_options_line_break>=2`, `n_off_ball_runs>=2`, pressing-chain length `>=4`, build-up length `>=4` possessions, high line `>30m`, and cover distance `>5m`. Public research supports multiple-option progression and structural pass–defence interaction, but no learned graph model is used. ([arXiv][2])

**Model + key hyperparameters:** `deterministic_relational_feature_generator`; exactly 50 features; canonical provider orientation; float64; 150,000-row chunks; four verified relationship maps; hard redundancy threshold `0.95`; minimum active matches `5`; 5-fold match-group contract CV, three seeds, two repeats.

**Runtime/memory:** 30–60 minutes for all builds and evidence on CPU, under about 6 GB RAM; GPU memory negligible.

**Leakage/rule risk:** No target leakage exists, but provider line-break/run/engagement categories have model provenance. Deny all continuous model fields, document annotation dependencies, and require the geometry/linkage safety ablation.

**Fallback:** Promote `linked_event_motif_graph_50` if any relation, source, activity, or correlation gate fails.

## 2. Linked Event Motif Graph 50

**Leak-free features/encodings:** Existing raw one-touch, quick-pass, give-and-go, run-linked line-break, line-push, phase, pressing-chain, movement, and reception counts/totals; no learned transform or forbidden model score.

**Model + key hyperparameters:** `deterministic_rule_based_feature_generator`; 50 features; long pressing chain `>=4`; HSR bands `hsr|sprinting`; no coordinate flip; 150,000-row chunks; `|Spearman|<0.95` hard gate.

**Runtime/memory:** 20–45 minutes, under 5 GB RAM.

**Leakage/rule risk:** Low to moderate annotation-interpretation risk; lower implementation risk because the attached kernel already contains this candidate.

**Fallback:** Mandatory reference 39.

## 3. Mandatory Reference-Aligned 39

**Leak-free features/encodings:** Counts and totals for event volume, progression, line-break options, runs, phase starts, and pressure. Explicitly remove ratios, percentages, averages, xThreat/xPass fields, outcomes, and any second coordinate flip.

**Model + key hyperparameters:** `deterministic_reference_feature_generator`; exactly 39 executable columns; 150,000-row chunks; Code-tab `39.0` retained only as provenance metadata.

**Runtime/memory:** 8–20 minutes, under 4 GB RAM.

**Leakage/rule risk:** Minimal target leakage; moderate originality risk and some redundancy risk.

**Fallback:** Reimplement concepts from the official field specification rather than copying public code.

## 4. Geometry-Only Raw Safety 30

**Leak-free features/encodings:** Event counts, duration/distance totals, forward/backward/lateral displacement, entries, targeted/received volumes, pressure closure, simultaneous engagement, and goal-side recovery. No phase, line-break, or run-subtype annotation.

**Model + key hyperparameters:** `deterministic_geometry_feature_generator`; 30 features; float64; 150,000-row chunks; exact 20-row output.

**Runtime/memory:** 5–12 minutes, under 3 GB RAM.

**Leakage/rule risk:** Lowest rule risk, highest concept and generic-volume risk.

**Fallback:** Terminal emergency route only. Official SkillCorner documentation supports the event-family and raw physical interpretation; it also makes clear why model-probability fields must stay out. ([GitHub][4])
