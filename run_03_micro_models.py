import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from config import POISONED_DIR, RESULTS_DIR, ML_MODELS, THRESHOLDS, DATASET_SCHEMAS, TEST_FEATURE_SETS
from core_pipeline import feature_engineering

MIN_SAMPLES = 30
MAX_DBSCAN_ROWS = 50000


def get_geometric_eps(X, k):
    if len(X) < k + 1:
        return 0.5

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)

    n_points = len(distances)
    if n_points < 3:
        return 0.5

    all_coords = np.vstack((range(n_points), distances)).T
    first_point = all_coords[0]
    last_point = all_coords[-1]

    line_vec = last_point - first_point
    line_norm = np.sqrt(np.sum(line_vec**2))
    if line_norm == 0:
        return 0.5

    line_vec_norm = line_vec / line_norm

    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel

    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    elbow_index = np.argmax(dist_to_line)

    best_eps = distances[elbow_index]
    max_eps = max(0.01, float(best_eps))
    return max_eps


def evaluate_micro_models():
    all_results = []

    for file_name, schema in DATASET_SCHEMAS.items():
        poisoned_name = file_name.replace('.csv', '_poisoned.csv')
        path = POISONED_DIR / poisoned_name
        if not path.exists():
            print(f"File {path} does not exist. Skipping.")
            continue

        print(f"\nAnalyzing {poisoned_name}...")

        case_col = schema['case']
        act_col = schema['act']
        time_col = schema['time']
        res_col = schema['res']

        df = pd.read_csv(path)
        df[time_col] = pd.to_datetime(df[time_col], utc=True, format='mixed')
        df = feature_engineering(df, case_col=case_col, act_col=act_col, time_col=time_col)
        y_true = df['is_true_anomaly']

        for model_name, model_info in ML_MODELS.items():
            for feat_set_name, X_cols in TEST_FEATURE_SETS.items():

                print(f"  -> Training cascade for: {model_name} with features {feat_set_name} ({len(X_cols)} features)")
                base_model = model_info['class'](**model_info['kwargs'])

                k_val = base_model.get_params().get('min_samples', 5) if model_name == 'DBSCAN' else None

                scores_L3 = np.zeros(len(df))
                X_global = df[X_cols].fillna(0).astype(np.float32)

                if model_name == 'DBSCAN':
                    if len(X_global) <= MAX_DBSCAN_ROWS:
                        eps_val = get_geometric_eps(X_global.values, k_val)
                        m_global = clone(base_model).set_params(eps=eps_val)
                        preds = m_global.fit_predict(X_global)
                        scores_L3 = (preds == -1).astype(int)
                else:
                    global_m = clone(base_model).fit(X_global.values if model_name == 'LOF' else X_global)
                    scores_L3 = -global_m.score_samples(X_global.values if model_name == 'LOF' else X_global)

                scores_L2 = np.copy(scores_L3)

                for act, group in df.groupby(act_col):
                    if len(group) >= MIN_SAMPLES:
                        X_act = group[X_cols].fillna(0).astype(np.float32)
                        if model_name == 'DBSCAN':
                            eps_val = get_geometric_eps(X_act.values, k_val)
                            m_act = clone(base_model).set_params(eps=eps_val)
                            preds = m_act.fit_predict(X_act)
                            scores_L2[group.index] = (preds == -1).astype(int)
                        else:
                            m_act = clone(base_model).fit(X_act.values if model_name == 'LOF' else X_act)
                            scores_L2[group.index] = - \
                                m_act.score_samples(X_act.values if model_name == 'LOF' else X_act)

                scores_L1 = np.copy(scores_L2)

                levels = {
                    'L3_Global': scores_L3,
                    'L2_Activity': scores_L2
                }

                if res_col is not None and res_col in df.columns:
                    for (act, res), group in df.groupby([act_col, res_col]):
                        if len(group) >= MIN_SAMPLES:
                            X_micro = group[X_cols].fillna(0).astype(np.float32)
                            if model_name == 'DBSCAN':
                                eps_val = get_geometric_eps(X_micro.values, k_val)
                                m_micro = clone(base_model).set_params(eps=eps_val)
                                preds = m_micro.fit_predict(X_micro)
                                scores_L1[group.index] = (preds == -1).astype(int)
                            else:
                                m_micro = clone(base_model).fit(X_micro.values if model_name == 'LOF' else X_micro)
                                scores_L1[group.index] = - \
                                    m_micro.score_samples(X_micro.values if model_name == 'LOF' else X_micro)
                    
                    levels['L1_Micro'] = scores_L1
                else:
                    print(f"    -> No resource column found for {file_name}. Skipping micro-level modeling.")

                for level_name, final_scores in levels.items():
                    if model_name == 'DBSCAN':
                        y_pred = final_scores == 1
                        all_results.append({
                            'dataset': poisoned_name,
                            'model': model_name,
                            'level': level_name,
                            'feature_set': feat_set_name,
                            'threshold': 'N/A',
                            # Metrics for Anomalies (pos_label=True)
                            'precision_anomalous': precision_score(y_true, y_pred, pos_label=True, zero_division=0),
                            'recall_anomalous': recall_score(y_true, y_pred, pos_label=True, zero_division=0),
                            'f1_score_anomalous': f1_score(y_true, y_pred, pos_label=True, zero_division=0),
                            # Metrics for Normal data (pos_label=False)
                            'precision_normal': precision_score(y_true, y_pred, pos_label=False, zero_division=0),
                            'recall_normal': recall_score(y_true, y_pred, pos_label=False, zero_division=0),
                            'f1_score_normal': f1_score(y_true, y_pred, pos_label=False, zero_division=0)
                        })
                    else:
                        for th in THRESHOLDS:
                            q1, q3 = np.percentile(final_scores, [25, 75])
                            limit = q3 + th * (q3 - q1)
                            y_pred = final_scores > limit

                            all_results.append({
                                'dataset': poisoned_name,
                                'model': model_name,
                                'level': level_name,
                                'feature_set': feat_set_name,
                                'threshold': th,
                                # Metrics for Anomalies (pos_label=True)
                                'precision_anomalous': precision_score(y_true, y_pred, pos_label=True, zero_division=0),
                                'recall_anomalous': recall_score(y_true, y_pred, pos_label=True, zero_division=0),
                                'f1_score_anomalous': f1_score(y_true, y_pred, pos_label=True, zero_division=0),
                                # Metrics for Normal data (pos_label=False)
                                'precision_normal': precision_score(y_true, y_pred, pos_label=False, zero_division=0),
                                'recall_normal': recall_score(y_true, y_pred, pos_label=False, zero_division=0),
                                'f1_score_normal': f1_score(y_true, y_pred, pos_label=False, zero_division=0)
                            })

    pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'model_evaluation.csv', index=False)
    print("\nSaved evaluation_results/model_evaluation.csv")


if __name__ == "__main__":
    evaluate_micro_models()