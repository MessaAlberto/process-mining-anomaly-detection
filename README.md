# Process Mining Anomaly Detection Framework

## Overview
This framework injects synthetic anomalies (activity and timestamp disruptions) into event logs and evaluates multiple machine learning models at different aggregation levels for anomaly detection in process mining.

## Pipeline

1. **Anomaly Injection** (`injection.py`)
   - Injects ~30% activity anomalies (changing activity labels)
   - Injects ~30% temporal anomalies (modifying event timestamps)

2. **Feature Engineering** (`feature_engineering.py`)
   - Computes temporal features: `delta_t`, `cum_t`, `trace_pos`
   - Computes circular time features: `hour_sin/cos`, `day_sin/cos`
   - Computes event density features

3. **Anomaly Scoring** (`run_01_scoring.py`)
   - Applies models (Isolation Forest, LOF, DBSCAN) at 3 aggregation levels:
     - L3_Global: dataset-wide
     - L2_Activity: per-activity groups
     - L1_Micro: per-activity-resource groups
   - Generates anomaly scores across multiple feature sets

4. **Evaluation** (`run_02_evaluation.py`)
   - Ground truth: temporal anomalies only (`TimeLabel`)
   - Computes precision, recall, F1-score using multiple thresholds (IQR-based)
   - Averages metrics across 10 independent runs

5. **Visualization** (`run_03_plot.py`)
   - Bar plots comparing best-performing models per dataset

6. **Comparison** (`run_04_AE_comparison.py`)
   - Benchmark against AE baseline

7. **Threshold Analysis** (`run_05_if_th_comparison.py`)
   - Threshold sensitivity analysis for Isolation Forest

## Key Insight
**Note:** Activity anomalies are injected but not used for evaluation. Detection of activity-only anomalies are counted as false positives (ground truth is temporal-only).