# Process Mining Anomaly Detection at Attribute Level

Code developed for attribute-level anomaly detection in business process event logs.

The repository contains two independent pipelines:

* **`unsupervised_AD/`** — Isolation Forest-based detection of **timestamp** and **activity** anomalies.
* **`supervised_AD/`** — supervised detection of **timestamp, activity, and resource** anomalies using Random Forest and XGBoost, including feature evaluation, robustness testing, plots and SHAP analysis.

## Dataset setup

Extract `dataset.zip` into a `dataset/` directory at the repository root:

```bash
mkdir -p dataset
unzip dataset.zip -d dataset
```

The resulting structure should be:

```text
dataset/
├── real_life_logs/
├── comuzzi_logs/
└── artificial_logs/
```

The supervised pipeline uses the real-life logs, while the unsupervised experiments use all three dataset groups.

## Unsupervised pipeline

Run the scripts from the repository root.

```bash
# 1. Generate injected datasets
python unsupervised_AD/01_inject_datasets.py

# 2. Train Isolation Forest models and compute anomaly scores/thresholds
python unsupervised_AD/02_train_score_threshold.py --target time
python unsupervised_AD/02_train_score_threshold.py --target act

# 3. Evaluate on the test partitions
python unsupervised_AD/03_evaluate_test_scores.py --target time
python unsupervised_AD/03_evaluate_test_scores.py --target act

# 4. Generate plots
python unsupervised_AD/04_plot.py --target time
python unsupervised_AD/04_plot.py --target act
```

Use `--force` with `02_train_score_threshold.py` to overwrite existing outputs.

## Supervised pipeline

```bash
# 1. Generate training and test datasets
python supervised_AD/01_injection.py

# 2. Optional: reproduce the feature-ranking stage
python supervised_AD/02_feature_eval.py --target time
python supervised_AD/02_feature_eval.py --target act
python supervised_AD/02_feature_eval.py --target res

# 3. Train the final models using the provided feature configurations
python supervised_AD/03_training_model.py --target time
python supervised_AD/03_training_model.py --target act
python supervised_AD/03_training_model.py --target res

# 4. Evaluate model robustness
python supervised_AD/04_model_robustness_test.py --target time
python supervised_AD/04_model_robustness_test.py --target act
python supervised_AD/04_model_robustness_test.py --target res

# 5. Generate plots
python supervised_AD/05_plot.py --target time
python supervised_AD/05_plot.py --target act
python supervised_AD/05_plot.py --target res
```

The final selected feature sets are stored in:

```text
features_config_time.json
features_config_act.json
features_config_res.json
```

### SHAP analysis

The SHAP analysis is optional and reproduces the off-target false-positive analysis:

```bash
python supervised_AD/06_shap_offtarget.py --target time
python supervised_AD/06_shap_offtarget.py --target act
python supervised_AD/06_shap_offtarget.py --target res
```

By default, SHAP analysis uses the **15% Seen anomaly scenario**.
