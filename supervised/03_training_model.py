import json
import os
import joblib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import (
    INJ_DIR,
    BASE_DIR,
    RATE_STR,
    DATASET_SCHEMAS,
    TRAIN_PCT,
    SPLIT_SEED,
    get_models,
    FEATURES_CONFIG_PATH,
    args as config_args,
)
from feature_engineering_time import compute_features as compute_time_features, compute_resource_features as compute_time_resource_features, get_feature_columns as get_time_feature_columns
from feature_engineering_act import compute_features as compute_act_features, compute_resource_features as compute_act_resource_features, get_feature_columns as get_act_feature_columns
from feature_engineering_res import compute_features as compute_res_features, compute_resource_features as compute_res_resource_features, get_feature_columns as get_res_feature_columns


TARGET = config_args.target
LABEL_BY_TARGET = {
    'time': 'TimeLabel',
    'act': 'ActLabel',
    'res': 'ResLabel'
}

FEATURE_MODULES = {
    'time': {
        'compute_base': compute_time_features,
        'compute_resource': compute_time_resource_features,
        'feature_columns': get_time_feature_columns
    },
    'act': {
        'compute_base': compute_act_features,
        'compute_resource': compute_act_resource_features,
        'feature_columns': get_act_feature_columns
    },
    'res': {
        'compute_base': compute_res_features,
        'compute_resource': compute_res_resource_features,
        'feature_columns': get_res_feature_columns
    }
}


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


def normalize_categorical_columns(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('__MISSING__').astype(str)
    return df


def load_feature_config():
    if not FEATURES_CONFIG_PATH.exists():
        return None

    with open(FEATURES_CONFIG_PATH, 'r', encoding='utf-8') as file_handle:
        return json.load(file_handle)


def resolve_feature_columns(schema, df_columns):
    feature_config = load_feature_config()
    
    # Use the JSON configuration for every target when available.
    if feature_config:
        num_cols = list(feature_config.get('num_cols', []))
        cat_cols = [schema[key] for key in feature_config.get('cat_schema_keys', []) if key in schema]
        cat_cols.extend(feature_config.get('extra_cat_cols', []))

        if schema.get('res') and schema['res'] in df_columns:
            num_cols.extend(feature_config.get('res_num_cols', []))

            if feature_config.get('include_raw_resource', False):
                cat_cols.append(schema['res'])

            cat_cols.extend(feature_config.get('res_extra_cat_cols', []))

        num_cols = [c for c in num_cols if c in df_columns]
        cat_cols = [c for c in cat_cols if c in df_columns]
        return num_cols, list(dict.fromkeys(cat_cols))

    # Fall back to the feature module when the JSON file is missing.
    feature_cols = FEATURE_MODULES[TARGET]['feature_columns'](case_col=schema['case'], act_col=schema['act'], time_col=schema['time'], res_col=schema.get('res'))
    num_cols = [c for c in feature_cols['num_cols'] if c in df_columns]
    cat_cols = [c for c in feature_cols['cat_cols'] if c in df_columns]
    return num_cols, cat_cols

def get_base_schema_name(file_name):
    for schema_name in DATASET_SCHEMAS:
        base_without_ext = schema_name.replace('.csv', '')
        if file_name.startswith(base_without_ext):
            return schema_name
    return None

def train_and_save_run(file_name, models_dir):
    print(f"[{file_name}] Training and saving models...")
    base_name = get_base_schema_name(file_name)
    if not base_name:
        return

    schema = DATASET_SCHEMAS[base_name]
    
    if TARGET == 'res' and not schema.get('res'):
        return

    df = pd.read_csv(INJ_DIR / file_name)
    df[schema['time']] = pd.to_datetime(df[schema['time']], utc=True, format='mixed')

    # run_01 already contains only the 60% training partition.
    train_df = df.copy()

    # Fit all distribution-based feature statistics on the training partition only.
    train_df, train_stats = compute_target_features(train_df, schema)

    num_cols, cat_cols = resolve_feature_columns(schema, train_df.columns)

    print(f"\n[{file_name}]")
    print("NUM COLS:", num_cols)
    print("CAT COLS:", cat_cols)
    print("N num:", len(num_cols), "N cat:", len(cat_cols))

    train_df = normalize_categorical_columns(train_df, cat_cols)

    label_col = LABEL_BY_TARGET[TARGET]
    if label_col not in train_df.columns:
        raise ValueError(f"Missing label column {label_col} in {file_name} for target={TARGET}.")

    X_train, y_train_event = train_df[num_cols + cat_cols], train_df[label_col]

    cat_transformer = TargetEncoder(target_type='binary', random_state=42)

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())]), num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    scale_pos_w = len(y_train_event[y_train_event == 0]) / max(1, len(y_train_event[y_train_event == 1]))
    models = get_models(scale_pos_w)

    run_id = file_name.replace('.csv', '')

    for model_name, clf in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train_event)

        # Save the fitted pipeline, training statistics and feature columns.
        model_artifact = {
            'pipeline': pipeline,
            'train_stats': train_stats,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'label_col': label_col,
            'target': TARGET,
            'feature_config_path': str(FEATURES_CONFIG_PATH) if FEATURES_CONFIG_PATH.exists() else None,
            'train_partition_fraction': TRAIN_PCT,
            'split_seed': SPLIT_SEED,
        }

        model_filename = models_dir / f"{run_id}__{model_name}.joblib"
        joblib.dump(model_artifact, model_filename)
        print(f"Saved: {model_filename.name}")

if __name__ == "__main__":
    models_dir = BASE_DIR / 'saved_models' / TARGET / RATE_STR
    models_dir.mkdir(parents=True, exist_ok=True)

    all_files = [f for f in os.listdir(INJ_DIR) if f.endswith('.csv')]

    print(f"Training and saving models for {len(all_files)} runs...")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train_and_save_run, f, models_dir): f for f in all_files}
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            completed += 1
            pct = (completed / total) * 100

            try:
                future.result()
                print(f"[PROGRESS] {completed}/{total} completed ({pct:.1f}%)")
            except Exception as e:
                print(f"[PROGRESS] {completed}/{total} completed ({pct:.1f}%)")
                print(f"Error: {e}")

    print("All models successfully trained and saved!")