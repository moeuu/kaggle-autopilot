# Ranked shortlist

## 1. Contrastive-assessed sparse progressive portfolio

**Leak-free features/encodings:** eight focused roots; explicit positive triggers and near-miss exclusions; root-level safety and exact-contract clauses; one-level references; fold-fit word 1–2 gram and character 3–5 gram TF-IDF; frozen description/full-body MiniLM embeddings; cross-encoder rerank margin; collision, activation-count, and compositional-query diagnostics; actual BenchFlow invocation traces from development only; same-task redacted success/failure contrasts; rejected-patch memory. Public tasks with answer/target leakage are quarantined, and final-holdout trajectories never enter authoring or calibration.

**Models and concrete settings:** deterministic compiler; `all-MiniLM-L6-v2`; `ms-marco-MiniLM-L-6-v2`; per-skill balanced logistic regression (`C=1.0`, `max_iter=1000`); optional local Qwen3-8B-AWQ proposer (`temperature=0.2`, `top_p=0.9`, `max_new_tokens=1536`, batch 1); maximum 8 skills, 3 activations, 8 description variants per skill, 2 patches per iteration, 12 changed lines, 8% token-change cap, 3 iterations. Promotion uses official matched paired lift and a hard safety gate.

**Runtime/memory:** 10–24 hours including paired agent evaluation; 8–11.5 GB VRAM while the AWQ proposer is loaded, otherwise roughly 2–4 GB. **Leakage risk:** trajectory overfit, contaminated public packages, or task wording copied into patches. **Fallback:** deterministic failure-category patches, incumbent rollback, lexical-only routing, smaller batches/chunks/candidate breadth, but never disable paired validation or safety.

## 2. Scripted progressive static portfolio

**Leak-free features/encodings:** the same eight hand-authored roots, deterministic artifact scripts, synthetic difficult near-misses, grouped routing folds, frozen blind routing set, and safe public metadata used only for diagnostics. **Models/settings:** deterministic compiler plus the same MiniLM/reranker/logistic screen; 8 roots, 3-skill cap, 3 folds, 3 cheap seeds, 64 description candidates, 12-task paired development screen, 20-task final holdout. **Runtime/memory:** 2–6 hours before paired evaluation; 2–4 GB VRAM. **Leakage risk:** proxy routing may improve while paired lift falls. **Fallback:** original descriptions with a fixed 0.60 word/0.40 character TF-IDF screen and actual BenchFlow selection only.

## 3. Flat exact-contract ablation

**Leak-free features/encodings:** six self-contained roots with no references, semantic-parity matrix, the same safety/contract clauses, static lint, and grouped paired evaluation. **Models/settings:** deterministic compiler and lexical activation screen; 6 skills, 2-skill routing cap for the proxy, 4,200-token/380-line root ceiling, no references, up to 2 scripts per skill. **Runtime/memory:** under 1 hour excluding paired runs; CPU or under 2 GB VRAM. **Leakage risk:** context bloat and instruction conflict rather than data leakage. **Fallback:** retain the progressive portfolio unless the flat candidate improves exact-output/long-artifact slices without any safety, timeout, or domain regression.

## 4. Mandatory broad reference control

**Leak-free features/encodings:** one independently authored generic plan–execute–verify skill, frontmatter/archive/safety checks, and a non-procedural length-matched placebo. **Models/settings:** deterministic generator only; 1 skill, 2,500-token/220-line root ceiling, no references/scripts, 4-task paired sanity screen. **Runtime/memory:** minutes, CPU-only. **Leakage risk:** accidental copying from the notebook or a placebo becoming procedural. **Fallback:** never auto-promote; keep only as required provenance, reference reproduction, and writeup control.

The first pipeline is the serious run because it converts public paired failures into bounded, replay-tested edits while preserving a frozen holdout. The second is the mandatory rollback. Routing metrics are admission diagnostics, not the competition score; no candidate is promoted without real matched verifier evidence when the official runner is available. The research basis is SkillsBench’s focused-skill result, SkillOpt’s held-out textual optimizer, SkillCAT’s patch replay, SkillRouter’s body-aware shortlist, ClawsBench’s independent safety measurement, and skill-shadowing evidence. ([arXiv][1])
