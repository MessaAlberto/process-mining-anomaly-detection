import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from config import *
from injection import inject_anomalies
from feature_engineering import compute_features

pd.options.mode.chained_assignment = None

def get_geometric_eps(X, k):
    """
    Compute a geometric-based epsilon for DBSCAN using the k-distance graph method.
    """
    if len(X) < k + 1:
        return 0.5
    neighbors = NearestNeighbors(n_neighbors=k)
    distances, _ = neighbors.fit(X).kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    n_points = len(distances)
    if n_points < 3:
        return 0.5

    all_coords = np.vstack((range(n_points), distances)).T
    first_point, last_point = all_coords[0], all_coords[-1]
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

    return max(0.01, float(distances[np.argmax(dist_to_line)]))

def process_single_run(file_name, schema, run):
    """
    Process a single dataset and run: inject anomalies, compute features, apply models, and save scores.
    """
    np.random.seed(run)
    input_path = RAW_DIR / file_name
    case_col, act_col, time_col, res_col = schema['case'], schema['act'], schema['time'], schema['res']

    raw_df = pd.read_csv(input_path)
    raw_df[time_col] = pd.to_datetime(raw_df[time_col], utc=True, format='mixed')

    # Inject anomalies
    df = inject_anomalies(raw_df, case_col, act_col, time_col, anomaly_rate=ANOMALY_RATE)
    # Considering only time-based anomalies for evaluation
    df['is_true_anomaly'] = (df['TimeLabel'] == 1).astype(int)

    cols_to_export = [case_col, act_col, time_col, 'is_true_anomaly']
    if res_col and res_col in df.columns:
        cols_to_export.append(res_col)

    # Save the poisoned dataset for reference
    poisoned_file = POISONED_DIR / f"{file_name.replace('.csv', '')}_poisoned_run_{run}.csv"
    df[cols_to_export].to_csv(poisoned_file, index=False)

    # Compute features and apply models
    df = compute_features(df, case_col=case_col, act_col=act_col, time_col=time_col)

    # Compute group indices for each level
    n_rows = len(df)
    groups = {
        'L3_Global': {'all': np.arange(n_rows)},
        'L2_Activity': df.groupby(act_col).indices,
        'L1_Micro': df.groupby([act_col, res_col]).indices if res_col in df.columns else {}
    }

    # Loop through feature sets, models, and levels to compute anomaly scores
    for feat_name, X_cols in TEST_FEATURE_SETS.items():
        X_full = df[X_cols].fillna(0).values.astype(np.float32)

        # Loop through models
        for model_name, model_info in ML_MODELS.items():
            base_model = model_info['class'](**model_info['kwargs'])
            k_val = base_model.get_params().get('min_samples', 5) if model_name == 'DBSCAN' else None

            # Loop through levels
            for level in LEVELS:
                print(f"[{file_name} - RUN {run}] Processing -> Feature: {feat_name} | Model: {model_name} | Level: {level}", flush=True)
                
                col_name = f"score_{model_name}_{level}_{feat_name}"
                df[col_name] = 0.0

                level_groups = groups.get(level, {})
                if not level_groups:
                    continue

                # Avoid DBSCAN on large datasets at global level (memory allocation issues)
                if model_name == 'DBSCAN' and level == 'L3_Global' and n_rows > MAX_DBSCAN_ROWS:
                    continue

                # Loop through each group in the current level
                for idx in level_groups.values():
                    if len(idx) < MIN_SAMPLES and level != 'L3_Global':
                        continue

                    X_subset = X_full[idx]
                    if len(X_subset) == 0:
                        continue

                    local_scaler = MinMaxScaler()
                    X_subset = local_scaler.fit_transform(X_subset)
                    X_subset += np.random.normal(0, 1e-6, X_subset.shape)

                    if model_name == 'DBSCAN':
                        eps_val = get_geometric_eps(X_subset, k_val)
                        m = clone(base_model).set_params(eps=eps_val)
                        preds = m.fit_predict(X_subset)
                        df.loc[df.index[idx], col_name] = (preds == -1).astype(int)
                    elif model_name == 'LOF':
                        m = clone(base_model)
                        m.fit_predict(X_subset)
                        df.loc[df.index[idx], col_name] = -m.negative_outlier_factor_
                    else:
                        m = clone(base_model).fit(X_subset)
                        df.loc[df.index[idx], col_name] = -m.score_samples(X_subset)

    output_file = SCORED_DIR / f"{file_name.replace('.csv', '')}_run_{run}.parquet"
    cols_to_save = ['is_true_anomaly'] + [c for c in df.columns if c.startswith('score_')]
    df[cols_to_save].to_parquet(output_file, index=False)

    gc.collect()

    return f"[{file_name}] - RUN {run}/{NUM_RUNS} completed."

def generate_and_score():
    """
    Process parallel runs for all datasets, injecting anomalies, computing features, applying models, and saving scores.
    """
    tasks = []
    for file_name in DATASETS:
        input_path = RAW_DIR / file_name
        if not input_path.exists():
            print(f"Skipping {file_name}: not found.")
            continue

        schema = DATASET_SCHEMAS[file_name]
        for run in range(1, NUM_RUNS + 1):
            tasks.append((file_name, schema, run))

    max_workers = 2
    print(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_run, t[0], t[1], t[2]) for t in tasks]

        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"Error occurred in a worker process: {e}")

if __name__ == "__main__":
    generate_and_score()