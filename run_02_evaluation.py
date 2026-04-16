import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from config import SCORED_DIR, MODEL_COMP_DIR, DATASETS, NUM_RUNS, LEVELS, DATASET_SCHEMAS, TEST_FEATURE_SETS

warnings.filterwarnings('ignore')


def evaluate_scores():
    gmm_results = []

    for file_name in DATASETS:
        dataset_base = file_name.replace('.csv', '')
        print(f"Evaluating dataset: {dataset_base}")

        schema = DATASET_SCHEMAS[file_name]
        act_col = schema['act']
        res_col = schema['res']

        for run in range(1, NUM_RUNS + 1):
            file_path = SCORED_DIR / f"{dataset_base}_run_{run}.parquet"
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path)
            y_true = df['is_true_anomaly'].values
            split_vals = df['split'].values
            score_columns = [c for c in df.columns if c.startswith('score_')]

            for level in LEVELS:
                for model_name in ['IsolationForest', 'LOF', 'DBSCAN']:

                    if level == 'L3_Global':
                        groups = {'global': np.arange(len(df))}
                    elif level == 'L2_Activity':
                        groups = df.groupby(act_col).indices
                    elif level == 'L1_Micro':
                        if res_col is None:
                            continue
                        groups = df.groupby([act_col, res_col]).indices
                    else:
                        continue

                    level_cols = [c for c in score_columns if f"_{model_name}_{level}_" in c]
                    if not level_cols:
                        continue

                    global_test_cms = defaultdict(lambda: np.zeros(4))
                    gmm_thresholds = defaultdict(list)
                    gmm_separations = defaultdict(list)

                    for group_id, idx in groups.items():
                        idx = np.array(idx)

                        idx_val = idx[split_vals[idx] == 'val']
                        idx_test = idx[split_vals[idx] == 'test']

                        if len(idx_val) == 0 or len(idx_test) == 0:
                            continue

                        y_true_test = y_true[idx_test]

                        for col in level_cols:
                            feat_name = col.split(f"_{level}_")[1]
                            scores_val = df[col].values[idx_val]
                            scores_test = df[col].values[idx_test]

                            v_mask_val = scores_val != 0.0
                            v_mask_test = scores_test != 0.0

                            valid_scores_val = scores_val[v_mask_val]
                            limit = 0.0

                            if model_name == 'DBSCAN':
                                # For DBSCAN, use the default threshold of 1.0 for anomalies
                                y_pred_test = np.zeros_like(y_true_test)
                                if v_mask_test.any():
                                    y_pred_test[v_mask_test] = (scores_test[v_mask_test] == 1.0)
                                cm_test = _calc_cm(y_true_test, y_pred_test)
                                global_test_cms[feat_name] += cm_test
                            else:
                                # For Isolation Forest and LOF, fit a GMM (2 components) on the validation scores to find an optimal threshold
                                if len(valid_scores_val) > 5:
                                    X_val = valid_scores_val.reshape(-1, 1)
                                    gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=42)

                                    try:
                                        gmm.fit(X_val)
                                        anom_idx = np.argmax(gmm.means_[:, 0])

                                        means = gmm.means_.flatten()
                                        covars = gmm.covariances_.flatten()
                                        std_0, std_1 = np.sqrt(covars[0]), np.sqrt(covars[1])
                                        sep_score = abs(means[0] - means[1]) / (std_0 + std_1)
                                        gmm_separations[feat_name].append(sep_score)

                                        min_s, max_s = X_val.min(), X_val.max()
                                        grid = np.linspace(min_s, max_s, 1000).reshape(-1, 1)
                                        probs = gmm.predict_proba(grid)

                                        anom_probs = probs[:, anom_idx]
                                        cross_indices = np.where(anom_probs >= 0.5)[0]

                                        if len(cross_indices) > 0:
                                            limit = grid[cross_indices[0], 0]
                                        else:
                                            limit = X_val.mean()
                                    except Exception:
                                        limit = np.median(valid_scores_val)
                                        gmm_separations[feat_name].append(0.0)
                                else:
                                    if len(valid_scores_val) > 0:
                                        limit = np.median(valid_scores_val)
                                    gmm_separations[feat_name].append(0.0)

                                gmm_thresholds[feat_name].append(limit)

                                y_pred_test = np.zeros_like(y_true_test)
                                y_pred_test[v_mask_test] = scores_test[v_mask_test] > limit
                                cm_test = _calc_cm(y_true_test, y_pred_test)

                                global_test_cms[feat_name] += cm_test

                    for feat_name in TEST_FEATURE_SETS.keys():
                        if model_name == 'DBSCAN':
                            best_th = np.nan
                            avg_sep = 0.0
                        else:
                            best_th = np.mean(gmm_thresholds[feat_name]) if gmm_thresholds[feat_name] else 0.0
                            avg_sep = np.mean(gmm_separations[feat_name]) if gmm_separations[feat_name] else 0.0

                        cm_test = global_test_cms.get(feat_name, np.zeros(4))
                        gmm_results.append(_build_metrics_dict(dataset_base, run, model_name,
                                           level, feat_name, best_th, cm_test, avg_sep))

    _aggregate_and_save(gmm_results, 'gmm_unsupervised')
    print("GMM evaluation completed on TEST Set.")


def _calc_cm(y_true, y_pred):
    # Calculate confusion matrix components: TP, FP, FN, TN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([tp, fp, fn, tn])


def _build_metrics_dict(ds, run, model, level, feat, th_val, cm, sep_score):
    tp, fp, fn, tn = cm

    prec_anom = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_anom = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_anom = (2 * prec_anom * rec_anom) / (prec_anom + rec_anom) if (prec_anom + rec_anom) > 0 else 0.0

    prec_norm = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_norm = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_norm = (2 * prec_norm * rec_norm) / (prec_norm + rec_norm) if (prec_norm + rec_norm) > 0 else 0.0

    tot_events = tp + fp + fn + tn
    predicted_anomaly_rate = ((tp + fp) / tot_events * 100) if tot_events > 0 else 0.0

    return {
        'dataset': ds, 'run': run, 'model': model, 'level': level,
        'feature_set': feat, 'threshold_value': th_val,
        'separation_score': sep_score,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'predicted_anomaly_rate': predicted_anomaly_rate,
        'precision_anomalous': prec_anom, 'recall_anomalous': rec_anom, 'f1_score_anomalous': f1_anom,
        'precision_normal': prec_norm, 'recall_normal': rec_norm, 'f1_score_normal': f1_norm
    }


def _aggregate_and_save(results_list, prefix):
    if not results_list:
        return
    df = pd.DataFrame(results_list)

    group_cols = ['dataset', 'model', 'level', 'feature_set']

    agg_df = df.groupby(group_cols).agg({
        'threshold_value': ['mean'],
        'separation_score': ['mean'],
        'predicted_anomaly_rate': ['mean', 'std'],
        'precision_anomalous': ['mean', 'std'],
        'recall_anomalous': ['mean', 'std'],
        'f1_score_anomalous': ['mean', 'std'],
        'precision_normal': ['mean', 'std'],
        'recall_normal': ['mean', 'std'],
        'f1_score_normal': ['mean', 'std'],
    }).reset_index()

    agg_df.columns = ['_'.join(c).strip('_') for c in agg_df.columns]

    if 'separation_score_mean' in agg_df.columns:
        agg_df.rename(columns={'separation_score_mean': 'separation_score'}, inplace=True)

    agg_df.to_csv(MODEL_COMP_DIR / f'{prefix}.csv', index=False)


if __name__ == "__main__":
    evaluate_scores()
