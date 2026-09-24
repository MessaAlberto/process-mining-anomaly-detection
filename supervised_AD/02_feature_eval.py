import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import TargetEncoder

from config import (
    INJ_DIR,
    DATASET_SCHEMAS,
    FEATURE_EVAL_TRAIN_PCT,
    SPLIT_SEED,
    args as config_args,
    split_case_ids,
)
from feature_engineering_time import compute_features as compute_time_features, compute_resource_features as compute_time_resource_features, get_feature_columns as get_time_feature_columns
from feature_engineering_act import compute_features as compute_act_features, compute_resource_features as compute_act_resource_features, get_feature_columns as get_act_feature_columns
from feature_engineering_res import compute_features as compute_res_features, compute_resource_features as compute_res_resource_features, get_feature_columns as get_res_feature_columns


TARGET = config_args.target
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)
FEATURE_MODULES = {
    'time': {
        'compute_base': compute_time_features,
        'compute_resource': compute_time_resource_features,
        'feature_columns': get_time_feature_columns,
        'label': 'TimeLabel'
    },
    'act': {
        'compute_base': compute_act_features,
        'compute_resource': compute_act_resource_features,
        'feature_columns': get_act_feature_columns,
        'label': 'ActLabel'
    },
    'res': {
        'compute_base': compute_res_features,
        'compute_resource': compute_res_resource_features,
        'feature_columns': get_res_feature_columns,
        'label': 'ResLabel'
    }
}


def get_base_schema_name(file_name):
    for schema_name in DATASET_SCHEMAS:
        base_without_ext = schema_name.replace('.csv', '')
        if file_name.startswith(base_without_ext):
            return schema_name
    return None


def remove_correlated_features(df, features, threshold=0.95):
    corr_matrix = df[features].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    selected_features = [f for f in features if f not in to_drop]
    return selected_features, to_drop


def standardize_feature_name(feature_name, schema):
    """Map dataset-specific raw attribute names to common ranking names."""
    if feature_name == schema['act']:
        return '[STD_Activity]'

    resource_col = schema.get('res')
    if resource_col and feature_name == resource_col:
        return '[STD_Resource]'

    return feature_name


def print_ranking(global_importances, title):
    """Print the global ranking across all compatible datasets.

    Each compatible dataset contributes exactly one score per feature:
    - the measured permutation importance if the feature survives correlation filtering;
    - 0.0 if the feature is removed by correlation filtering.

    Features that are not available in a dataset are not included for that dataset.
    """
    print("\n" + "=" * 80)
    print(f"*** {title} ***")
    print("=" * 80)

    summary = []
    for feat, scores in global_importances.items():
        if not scores:
            continue

        avg_imp = float(np.mean(scores))
        positive_hits = sum(1 for score in scores if score > 0.001)
        compatible_datasets = len(scores)
        stability_rate = (positive_hits / compatible_datasets) * 100

        summary.append(
            (
                feat,
                avg_imp,
                stability_rate,
                positive_hits,
                compatible_datasets,
            )
        )

    summary.sort(key=lambda row: row[1], reverse=True)

    print(
        f"{'Feature':<40} | "
        f"{'Avg Importance':>14} | "
        f"{'Stability':>10} | "
        f"{'Positive/Compatible':>19}"
    )
    print("-" * 94)

    for feat, avg_imp, stability_rate, positive_hits, compatible_datasets in summary:
        print(
            f"{feat:<40} | "
            f"{avg_imp:>14.4f} | "
            f"{stability_rate:>9.1f}% | "
            f"{positive_hits:>8}/{compatible_datasets:<10}"
        )


def normalize_categorical_columns(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('__MISSING__').astype(str)
    return df


def compute_target_features(df, schema, train_stats=None):
    module_cfg = FEATURE_MODULES[TARGET]

    base_kwargs = {
        'case_col': schema['case'],
        'act_col': schema['act'],
        'time_col': schema['time'],
        'res_col': schema.get('res')
    }

    resource_kwargs = {
        'case_col': schema['case'],
        'act_col': schema['act'],
        'time_col': schema['time'],
        'res_col': schema.get('res')
    }

    if train_stats is None:
        df, base_stats = module_cfg['compute_base'](df, **base_kwargs)
        train_stats = {'base_stats': base_stats}
    else:
        df = module_cfg['compute_base'](df, train_stats=train_stats['base_stats'], **base_kwargs)

    if schema.get('res') and schema['res'] in df.columns:
        if 'res_stats' not in train_stats:
            df, res_stats = module_cfg['compute_resource'](df, **resource_kwargs)
            train_stats['res_stats'] = res_stats
        else:
            df = module_cfg['compute_resource'](df, train_stats=train_stats['res_stats'], **resource_kwargs)

    return df, train_stats


def analyze_dataset(base_name, file_name):
    schema = DATASET_SCHEMAS[base_name]

    if TARGET == 'res' and not schema.get('res'):
        return base_name, {}, {}

    module_cfg = FEATURE_MODULES[TARGET]

    df = pd.read_csv(INJ_DIR / file_name)
    df[schema['time']] = pd.to_datetime(df[schema['time']], utc=True, format='mixed')

    target_col = module_cfg['label']
    if target_col not in df.columns:
        return base_name, {}, {}

    fit_cases, eval_cases = split_case_ids(
        df,
        schema['case'],
        train_size=FEATURE_EVAL_TRAIN_PCT,
        random_state=SPLIT_SEED,
    )
    fit_raw = df[df[schema['case']].isin(fit_cases)].copy()
    eval_raw = df[df[schema['case']].isin(eval_cases)].copy()

    train_df, train_stats = compute_target_features(fit_raw, schema)
    test_df, _ = compute_target_features(eval_raw, schema, train_stats=train_stats)

    feature_cols = module_cfg['feature_columns'](
        case_col=schema['case'],
        act_col=schema['act'],
        time_col=schema['time'],
        res_col=schema.get('res'),
    )
    num_cols = [c for c in feature_cols['num_cols'] if c in train_df.columns]
    cat_cols = [c for c in feature_cols['cat_cols'] if c in train_df.columns]

    train_df = normalize_categorical_columns(train_df, cat_cols)
    test_df = normalize_categorical_columns(test_df, cat_cols)

    encoder = TargetEncoder(target_type='binary', random_state=42)

    train_encoded_cats = pd.DataFrame(
        encoder.fit_transform(train_df[cat_cols], train_df[target_col]),
        columns=cat_cols,
        index=train_df.index,
    ) if cat_cols else pd.DataFrame(index=train_df.index)

    test_encoded_cats = pd.DataFrame(
        encoder.transform(test_df[cat_cols]),
        columns=cat_cols,
        index=test_df.index,
    ) if cat_cols else pd.DataFrame(index=test_df.index)

    X_train = pd.concat([train_df[num_cols], train_encoded_cats], axis=1)
    X_test = pd.concat([test_df[num_cols], test_encoded_cats], axis=1)

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # Store the complete feature set available in this dataset before
    # correlation filtering. These are the features compatible with this dataset.
    candidate_features = list(X_train.columns)

    final_features, dropped_features = remove_correlated_features(
        X_train,
        candidate_features,
    )
    X_train = X_train[final_features]
    X_test = X_test[final_features]

    scale_pos_w = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))

    model_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=1,
        n_estimators=50,
    )
    model_rf.fit(X_train, y_train)
    result_rf = permutation_importance(
        model_rf,
        X_test,
        y_test,
        scoring='f1',
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    model_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_w,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
    )
    model_xgb.fit(X_train, y_train)
    result_xgb = permutation_importance(
        model_xgb,
        X_test,
        y_test,
        scoring='f1',
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    # Actual importance values exist only for features that survive the
    # correlation filter.
    retained_rf = {
        feat: float(importance)
        for feat, importance in zip(
            final_features,
            result_rf.importances_mean,
        )
    }
    retained_xgb = {
        feat: float(importance)
        for feat, importance in zip(
            final_features,
            result_xgb.importances_mean,
        )
    }

    global_importances_rf = {}
    global_importances_xgb = {}

    # Every compatible dataset contributes one value for every candidate feature.
    # A feature removed because of correlation contributes 0.0, while an
    # unavailable feature never enters candidate_features and is therefore
    # excluded from the denominator.
    for feat in candidate_features:
        standard_name = standardize_feature_name(feat, schema)

        importance_rf = retained_rf.get(feat, 0.0)
        importance_xgb = retained_xgb.get(feat, 0.0)

        global_importances_rf.setdefault(standard_name, []).append(importance_rf)
        global_importances_xgb.setdefault(standard_name, []).append(importance_xgb)

    if dropped_features:
        dropped_names = [
            standardize_feature_name(feat, schema)
            for feat in dropped_features
        ]
        print(
            f"[{TARGET.upper()}] {base_name} - "
            f"correlation drops counted as zero: {dropped_names}"
        )

    return base_name, global_importances_rf, global_importances_xgb


def merge_importance_dicts(target_dict, source_dict):
    for feature_name, scores in source_dict.items():
        target_dict.setdefault(feature_name, []).extend(scores)


def evaluate_features_globally():
    if TARGET not in FEATURE_MODULES:
        raise ValueError(f"Unsupported target: {TARGET}")

    all_files = [f for f in os.listdir(INJ_DIR) if f.endswith('.csv')]

    files_to_tune = {}
    for f in all_files:
        base = get_base_schema_name(f)
        if base and base not in files_to_tune:
            files_to_tune[base] = f

    global_importances_rf = {}
    global_importances_xgb = {}
    dataset_count = len(files_to_tune)

    print(f"Starting analysis on {dataset_count} unique datasets for TARGET: {TARGET.upper()}...\n")

    tasks = list(files_to_tune.items())
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(analyze_dataset, base_name, file_name): (base_name, file_name)
            for base_name, file_name in tasks
        }

        for future in as_completed(future_map):
            base_name, _ = future_map[future]
            print(f"Completed: {base_name}")
            result_base, local_rf, local_xgb = future.result()
            if result_base is None:
                continue
            merge_importance_dicts(global_importances_rf, local_rf)
            merge_importance_dicts(global_importances_xgb, local_xgb)

    print_ranking(global_importances_rf, f"GLOBAL FEATURE RANKING ({TARGET.upper()}) - RANDOM FOREST")
    print_ranking(global_importances_xgb, f"GLOBAL FEATURE RANKING ({TARGET.upper()}) - XGBOOST")


if __name__ == "__main__":
    evaluate_features_globally()
