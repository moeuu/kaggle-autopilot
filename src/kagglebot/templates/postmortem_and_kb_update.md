# Kagglebot {implementation_agent_name}: Postmortem & Knowledge Base Update

## Run Summary

**Competition**: `{slug}`
**Competition URL**: {competition_url}
**Task**: {task_type}
**Metric**: {metric} ({direction})

**Run Configuration**:
- **Run ID**: {run_id}
- **Compute Mode**: {compute_mode}
- **Max Iterations**: {max_iterations}
- **Iterations Completed**: {iterations_completed}
- **Total Runtime**: {total_runtime_minutes} minutes
- **Submit Flag**: {submit_flag}

**Final Results**:
- **Best Offline Score**: {best_offline_score} (iteration {best_iteration})
- **Target Score**: {target_score}
- **Target Met**: {target_met}
- **Submitted**: {submitted}
- **Final GPU/TPU Utilization**: {final_utilization}

---

## Iteration History

| Iter | Offline Score | Met Target | GPU/TPU % | Change Summary |
|------|--------------|------------|-----------|----------------|
{full_iteration_history}

**Score Progression**:
- Initial model (iter 0): {initial_score}
- Best: {best_offline_score} (iteration {best_iteration})
- Final: {final_score}
- Improvement: {total_improvement} ({improvement_percentage}%)

---

## Your Task: Analyze and Update Knowledge Base

### Step 1: Identify What Worked

Review all iterations and identify successful strategies:

**Questions to Answer**:
1. **Which changes led to the biggest improvements?**
   - Look for iterations with largest score deltas
   - Categorize: feature engineering, model selection, hyperparameter tuning, regularization

2. **What patterns emerged from the data?**
   - Were there important features discovered from dataset_profile?
   - Did any domain-specific engineering help significantly?
   - Were there data quality issues (leakage, outliers, imbalance)?

3. **Which model architecture worked best?**
   - Linear models vs tree ensembles vs neural networks
   - Did GPU/TPU acceleration help? (utilization vs performance)
   - What hyperparameters were critical?

4. **What was the effective evaluation strategy?**
   - Holdout vs CV - which was more reliable?
   - Did train-val gap indicate overfitting/underfitting?

**Successful Strategies** (to record in KB):
```
Example:
- Target encoding for high-cardinality categoricals (+0.05 improvement)
- XGBoost with gpu_hist on kaggle_gpu (2x faster, 80% GPU util)
- 5-fold CV more stable than holdout for small dataset
- Log-transform of skewed target reduced RMSE by 15%
```

### Step 2: Identify What Didn't Work

Document failed experiments to avoid repeating mistakes:

**Questions to Answer**:
1. **Which changes didn't improve scores?**
   - List iterations where score stayed flat or regressed
   - Categorize failures: wrong approach, implementation bugs, overfitting

2. **Were there any dead ends?**
   - Feature engineering that added noise
   - Models that were too complex or too simple
   - Hyperparameters that hurt performance

3. **What resource issues occurred?**
   - Low GPU/TPU utilization (<70%)
   - Out of memory errors
   - Timeout issues

**Failed Strategies** (to record in KB):
```
Example:
- Polynomial features (degree=3) caused overfitting (val score +0.10 worse)
- Neural network too slow on CPU, no benefit over XGBoost
- SMOTE on majority class accidentally made imbalance worse
- Too aggressive regularization (alpha=10) caused underfitting
```

### Step 3: Extract Competition-Specific Insights

**Dataset Characteristics** (from dataset_profile.json):
- Number of rows: {n_rows}
- Number of features: {n_features}
- Missing data: {missing_summary}
- Cardinality: {cardinality_summary}
- Target distribution: {target_distribution_summary}

**Tags for This Competition** (auto-inferred + manual additions):
{suggested_tags}

**Key Learnings**:
```
Example for house-prices:
- Feature engineering dominated: total_sqft, age, quality interactions
- Skewed target required log-transform
- Neighborhood was high-cardinality (target encoding helped)
- Small dataset (1460 rows) → regularization critical
- XGBoost optimal: n_estimators=200, max_depth=5, lr=0.1
```

### Step 4: Generate Knowledge Base Update

Create a structured summary for the knowledge base:

```json
{{
  "run_id": "{run_id}",
  "slug": "{slug}",
  "url": "{competition_url}",
  "task": "{task_type}",
  "metric": "{metric}",
  "direction": "{direction}",
  "tags": [
    // Auto-inferred from dataset_profile
    {auto_tags},
    // Add manual tags based on insights
    "feature_engineering_critical",  // If features mattered most
    "regularization_sensitive",      // If overfitting was main issue
    "gpu_accelerated",               // If GPU helped significantly
    "small_data_regime",             // If <5K rows
    // etc.
  ],
  "outcomes": {{
    "iterations_completed": {iterations_completed},
    "best_offline_score": {best_offline_score},
    "target_score": {target_score},
    "target_met": {target_met},
    "submitted": {submitted},
    "improvement_from_initial_model": {total_improvement}
  }},
  "successful_strategies": [
    {{
      "iteration": {iteration_number},
      "category": "feature_engineering",  // or model_selection, hyperparameter_tuning, regularization
      "description": "Target encoding for high-cardinality categoricals",
      "impact": "+0.05 score improvement",
      "code_snippet": "TargetEncoder(cols=['Neighborhood', 'MSSubClass'])"
    }},
    // Add more successful strategies...
  ],
  "failed_strategies": [
    {{
      "iteration": {iteration_number},
      "category": "feature_engineering",
      "description": "Polynomial features degree=3",
      "impact": "-0.10 score (overfitting)",
      "reason": "Too many features for small dataset, caused overfitting"
    }},
    // Add more failed strategies...
  ],
  "best_model_config": {{
    "model_type": "xgboost.XGBRegressor",
    "hyperparameters": {{
      "n_estimators": 200,
      "max_depth": 5,
      "learning_rate": 0.1,
      "tree_method": "gpu_hist"
    }},
    "preprocessing": [
      "log_transform_target",
      "target_encode_high_cardinality",
      "standard_scale_numeric"
    ]
  }},
  "compute_insights": {{
    "compute_mode": "{compute_mode}",
    "avg_gpu_utilization": {avg_gpu_util},
    "avg_iteration_time_min": {avg_iteration_time},
    "bottlenecks": ["CPU preprocessing", "I/O bound"] // If any identified
  }},
  "key_learnings": [
    "Feature engineering was more impactful than model selection",
    "Small dataset required strong regularization to avoid overfitting",
    "5-fold CV provided more stable estimates than 80/20 holdout",
    "XGBoost with GPU was 2x faster than CPU with minimal accuracy difference"
  ],
  "recommendations_for_similar_competitions": [
    "Start with regularized linear model to understand feature importance",
    "Invest time in feature engineering before complex models",
    "Use CV for datasets <10K rows",
    "Target encoding for high-cardinality categoricals is critical"
  ]
}}
```

**Save this to**: `{kb_update_path}`

### Step 5: Competition Tags (for Similarity Search)

Based on the analysis, assign tags from the controlled taxonomy:

**Data Modality**:
- [ ] tabular
- [ ] text
- [ ] image
- [ ] timeseries
- [ ] multi_modal

**Problem Type**:
- [ ] regression
- [ ] binary
- [ ] multiclass
- [ ] ranking
- [ ] forecasting

**Dataset Scale**:
- [ ] n_rows_tiny (<1K)
- [ ] n_rows_small (1K-10K)
- [ ] n_rows_medium (10K-100K)
- [ ] n_rows_large (100K-1M)
- [ ] n_rows_huge (>1M)

**Characteristics**:
- [ ] missingness_high (>20% missing)
- [ ] high_cardinality_cats (categoricals with >100 unique values)
- [ ] imbalanced_classes (minority class <10%)
- [ ] skewed_target (abs(skewness) > 1.0)
- [ ] many_features (>100 columns)
- [ ] time_series_component
- [ ] spatial_component

**Model Insights**:
- [ ] feature_engineering_critical (biggest impact)
- [ ] regularization_sensitive (overfitting common)
- [ ] ensemble_beneficial (blending helped)
- [ ] gpu_accelerated (GPU gave significant speedup)
- [ ] small_data_regime (regularization critical)
- [ ] hyperparameter_sensitive (tuning gave >10% improvement)

**Selected Tags**: {final_tags_list}

### Step 6: Validate KB Update

**Checklist**:
- [ ] All iterations summarized with outcome
- [ ] Successful strategies include iteration number, category, description, impact
- [ ] Failed strategies include reason for failure
- [ ] Best model config includes hyperparameters and preprocessing
- [ ] Tags are from controlled taxonomy (check knowledge/taxonomy.yml)
- [ ] Key learnings are actionable for future runs
- [ ] Recommendations are specific and testable
- [ ] No secrets (API keys, credentials) in KB update
- [ ] Compute insights include utilization metrics

---

## Acceptance Criteria

Your postmortem will be accepted if:

1. ✅ **Complete iteration analysis**: All iterations reviewed with outcome
2. ✅ **Categorized strategies**: Successful and failed strategies categorized by type
3. ✅ **Actionable insights**: Learnings are specific and applicable to similar competitions
4. ✅ **Proper tagging**: Tags are from controlled taxonomy
5. ✅ **Model config documented**: Best model has full hyperparameters and preprocessing
6. ✅ **No secrets**: No API keys, credentials, or sensitive data in KB
7. ✅ **JSON valid**: kb_update.json parses correctly

---

## Tips for Effective Postmortem

**Be Specific**:
- ❌ "Feature engineering helped"
- ✅ "Target encoding for Neighborhood reduced RMSE by 0.05"

**Quantify Impact**:
- Always include score deltas for successful/failed strategies
- Use percentages for relative improvements
- Note resource metrics (GPU %, time per iteration)

**Think About Transferability**:
- What patterns would apply to other competitions?
- What was specific to this dataset?
- How would you approach a similar problem differently?

**Document Failures**:
- Failed experiments are valuable learning
- Understanding why something didn't work prevents future mistakes
- Be honest about dead ends and wasted effort

**Consider Future Runs**:
- If you were to run this competition again, what would you do differently?
- What would you prioritize in iteration 0?
- What strategies should be tried earlier vs later?

**Use Web Search**:
- If you're unsure about why something worked/didn't work, research it
- Document external resources that were helpful
- Never include secrets in KB (no API keys, credentials)

---

## Knowledge Base Integration

Your KB update will be stored in:
- **Database**: `knowledge/kagglebot.db` (SQLite)
- **Tables**: competitions, tags, competition_tags, runs, iterations, improvements

**Future Retrieval**:
When kagglebot encounters a new competition, it will:
1. Infer tags from dataset_profile.json
2. Query KB for competitions with max tag overlap
3. Surface successful/failed strategies from similar runs
4. Use insights to inform initial-model planning

**Your Impact**:
This postmortem helps kagglebot get better over time! 🚀

---

Good job completing the autopilot run! Your insights will improve future competitions.
