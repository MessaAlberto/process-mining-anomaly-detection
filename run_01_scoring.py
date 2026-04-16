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

from config import *
from injection import inject_anomalies
from feature_engineering import compute_features, compute_contextual_features

pd.options.mode.chained_assignment = None

def get_geometric_eps(X, k):
    if len(X) < k + 1: return 0.5
    neighbors = NearestNeighbors(n_neighbors=k)
    distances, _ = neighbors.fit(X).kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    n_points = len(distances)
    if n_points < 3: return 0.5

    all_coords = np.vstack((range(n_points), distances)).T
    first_point, last_point = all_coords[0], all_coords[-1]
    line_vec = last_point - first_point
    line_norm = np.sqrt(np.sum(line_vec**2))
    if line_norm == 0: return 0.5

    line_vec_norm = line_vec / line_norm
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))

    return max(0.01, float(distances[np.argmax(dist_to_line)]))

def process_single_run(file_name, schema, run):
    np.random.seed(run)
    case_col, act_col, time_col, res_col = schema['case'], schema['act'], schema['time'], schema['res']

    poisoned_file = POISONED_DIR / f"{file_name.replace('.csv', '')}_poisoned_run_{run}.csv"

    if poisoned_file.exists():
        print(f"[{file_name}] Poisoned file found! Loading existing RUN {run}...")
        df = pd.read_csv(poisoned_file)
        df[time_col] = pd.to_datetime(df[time_col], utc=True, format='mixed')
    else:
        print(f"[{file_name}] Creating NEW poisoned dataset for RUN {run}...")
        input_path = RAW_DIR / file_name
        raw_df = pd.read_csv(input_path)
        raw_df[time_col] = pd.to_datetime(raw_df[time_col], utc=True, format='mixed')
        
        # Sort chronologically before splitting
        raw_df = raw_df.sort_values(time_col).reset_index(drop=True)

        if file_name in ['small_log.csv', 'large_log.csv']:
            # Chronological split by Case ID to maintain temporal integrity for synthetic logs
            case_starts = raw_df.groupby(case_col)[time_col].min().sort_values()
            ordered_cases = case_starts.index.tolist()
            
            n_cases = len(ordered_cases)
            train_idx = int(n_cases * TRAIN_PCT)
            val_idx = int(n_cases * (TRAIN_PCT + VAL_PCT))
            
            train_set = set(ordered_cases[:train_idx])
            val_set = set(ordered_cases[train_idx:val_idx])
            
            raw_df['split'] = 'test'
            raw_df.loc[raw_df[case_col].isin(train_set), 'split'] = 'train'
            raw_df.loc[raw_df[case_col].isin(val_set), 'split'] = 'val'
        else:
            # Traditional row-based chronological split for real datasets
            n_rows = len(raw_df)
            train_end = int(n_rows * TRAIN_PCT)
            val_end = int(n_rows * (TRAIN_PCT + VAL_PCT))
            
            raw_df['split'] = 'train'
            raw_df.loc[train_end:val_end-1, 'split'] = 'val'
            raw_df.loc[val_end:, 'split'] = 'test'

        df = inject_anomalies(raw_df, case_col, act_col, time_col, anomaly_rate=ANOMALY_RATE, random_state=run)
        # Considering only timestamp anomalies for the "true" label
        df['is_true_anomaly'] = (df['TimeLabel'] == 1).astype(int)

        cols_to_export = [case_col, act_col, time_col, 'is_true_anomaly', 'split']
        if res_col and res_col in df.columns:
            cols_to_export.append(res_col)
        
        df[cols_to_export].to_csv(poisoned_file, index=False)

    df = compute_features(df, case_col=case_col, act_col=act_col, time_col=time_col).fillna(0)
    df = compute_contextual_features(df, case_col=case_col, act_col=act_col, res_col=res_col)

    groups = {
        'L3_Global': {'all': np.arange(len(df))},
        'L2_Activity': df.groupby(act_col).indices,
        'L1_Micro': df.groupby([act_col, res_col]).indices if res_col in df.columns else {}
    }

    feature_cache = {
        feat_name: df[X_cols].values.astype(np.float32)
        for feat_name, X_cols in TEST_FEATURE_SETS.items()
    }

    for feat_name in TEST_FEATURE_SETS.keys():
        X_full = feature_cache[feat_name]

        for model_name, model_info in ML_MODELS.items():
            base_model = model_info['class'](**model_info['kwargs'])
            k_val = base_model.get_params().get('min_samples', 5) if model_name == 'DBSCAN' else None

            for level in LEVELS:
                scores = np.zeros(len(df))
                print(f"[{file_name} - RUN {run}] Processing -> Feature: {feat_name} | Model: {model_name} | Level: {level}", flush=True)

                level_groups = groups.get(level, {})
                if not level_groups:
                    continue

                # Avoid running DBSCAN on large dataset (OOM risk)
                if model_name == 'DBSCAN' and level == 'L3_Global' and len(df) > MAX_DBSCAN_ROWS:
                    continue

                for idx in level_groups.values():
                    train_mask = df['split'].values[idx] == 'train'
                    idx_train = idx[train_mask]

                    if len(idx_train) < MIN_SAMPLES and level != 'L3_Global':
                        continue

                    X_train = X_full[idx_train]
                    X_all = X_full[idx]

                    min_val = X_train.min(axis=0)
                    ptp_val = np.ptp(X_train, axis=0) + 1e-8

                    X_train_scaled = (X_train - min_val) / ptp_val
                    X_train_scaled += np.random.normal(0, 1e-6, X_train_scaled.shape)

                    X_all_scaled = (X_all - min_val) / ptp_val
                    X_all_scaled += np.random.normal(0, 1e-6, X_all_scaled.shape)

                    if model_name == 'DBSCAN':
                        eps_val = get_geometric_eps(X_all_scaled, k_val)
                        m = clone(base_model).set_params(eps=eps_val)
                        preds = m.fit_predict(X_all_scaled)
                        scores[idx] = (preds == -1).astype(int)
                    else:
                        m = clone(base_model).fit(X_train_scaled)
                        scores[idx] = -m.score_samples(X_all_scaled)

                col_name = f"score_{model_name}_{level}_{feat_name}"
                df[col_name] = scores

    output_file = SCORED_DIR / f"{file_name.replace('.csv', '')}_run_{run}.parquet"
    
    cols_to_save = ['is_true_anomaly', act_col, 'split']
    if res_col and res_col in df.columns:
        cols_to_save.append(res_col)
    cols_to_save += [c for c in df.columns if c.startswith('score_')]
    
    df[cols_to_save].to_parquet(output_file, index=False)
    gc.collect()

    return f"[{file_name}] - RUN {run}/{NUM_RUNS} completed."

def generate_and_score():
    tasks = []
    for file_name in DATASETS:
        input_path = RAW_DIR / file_name
        if not input_path.exists():
            print(f"File {file_name} not found in {RAW_DIR}. Skipping.")
            continue

        schema = DATASET_SCHEMAS[file_name]
        for run in range(START_RUN, END_RUN + 1):
            tasks.append((file_name, schema, run))

    max_workers = 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_run, t[0], t[1], t[2]) for t in tasks]
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    generate_and_score()