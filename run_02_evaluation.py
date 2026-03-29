import pandas as pd
import numpy as np
from config import SCORED_DIR, MODEL_COMP_DIR, DATASETS, NUM_RUNS, IQR_THRESHOLDS, LEVELS

def evaluate_scores():
    """
    Evaluate the anomaly scores from the scoring phase, compute metrics for different thresholds, and save results.
    """
    all_results = []

    for file_name in DATASETS:
        dataset_base = file_name.replace('.csv', '')
        print(f"Evaluating dataset: {dataset_base}")

        for run in range(1, NUM_RUNS + 1):
            file_path = SCORED_DIR / f"{dataset_base}_run_{run}.parquet"
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path)
            y_true = df['is_true_anomaly'].values

            score_columns = [c for c in df.columns if c.startswith('score_')]

            for col in score_columns:
                level = None
                for l in LEVELS:
                    if f"_{l}_" in col:
                        level = l
                        break

                if not level:
                    continue

                parts = col.split(f"_{level}_")
                model_name = parts[0].replace('score_', '')
                feat_name = parts[1]

                scores = df[col].values

                valid_mask = scores != 0.0
                if not valid_mask.any():
                    continue

                if model_name == 'DBSCAN':
                    y_pred = (scores == 1.0) & valid_mask
                    all_results.append(_calc_metrics_fast(dataset_base, run, model_name, level, feat_name, 'N/A', y_true, y_pred))
                else:
                    valid_scores = scores[valid_mask]

                    q1, q3 = np.percentile(valid_scores, [25, 75])
                    iqr = q3 - q1

                    # Compute metrics for each IQR threshold
                    for th in IQR_THRESHOLDS:
                        limit = q3 + th * iqr
                        y_pred = (scores > limit) & valid_mask
                        all_results.append(_calc_metrics_fast(dataset_base, run, model_name, level, feat_name, th, y_true, y_pred))

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Average metrics across runs for each dataset, model, level, feature set, and threshold
        group_cols = ['dataset', 'model', 'level', 'feature_set', 'threshold_value']
        avg_df = results_df.groupby(group_cols).agg({
            'precision_normal': 'mean',
            'recall_normal': 'mean',
            'f1_score_normal': 'mean',
            'precision_anomalous': 'mean',
            'recall_anomalous': 'mean',
            'f1_score_anomalous': 'mean'
        }).reset_index()

        avg_df.to_csv(MODEL_COMP_DIR / 'avg_metrics.csv', index=False)
        print(f"Evaluation completed. Results saved to {MODEL_COMP_DIR / 'avg_metrics.csv'}")

def _calc_metrics_fast(ds, run, model, level, feat, th_val, y_true, y_pred):
    tp = np.sum(y_true & y_pred)          # True Positives (Anomalies correctly identified as anomalies)
    fp = np.sum((~y_true) & y_pred)       # False Positives (Normal data incorrectly identified as anomalies)
    fn = np.sum(y_true & (~y_pred))       # False Negatives (Anomalies not identified)
    tn = np.sum((~y_true) & (~y_pred))    # True Negatives (Normal data correctly identified as normal)

    # Anomalous class metrics
    prec_anom = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_anom = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_anom = (2 * prec_anom * rec_anom) / (prec_anom + rec_anom) if (prec_anom + rec_anom) > 0 else 0.0

    # Normal class metrics
    prec_norm = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_norm = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_norm = (2 * prec_norm * rec_norm) / (prec_norm + rec_norm) if (prec_norm + rec_norm) > 0 else 0.0

    return {
        'dataset': ds,
        'run': run,
        'model': model,
        'level': level,
        'feature_set': feat,
        'threshold_value': th_val,
        
        'precision_normal': prec_norm,
        'recall_normal': rec_norm,
        'f1_score_normal': f1_norm,
        
        'precision_anomalous': prec_anom,
        'recall_anomalous': rec_anom,
        'f1_score_anomalous': f1_anom
    }

if __name__ == "__main__":
    evaluate_scores()