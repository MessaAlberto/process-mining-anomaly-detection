import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from config import (
    BASE_DIR,
    DATASET_SCHEMAS,
    RATE_STR,
    TEST_LOGS_DIR,
    args as config_args,
)
from feature_engineering_act import (
    compute_features as compute_act_features,
    compute_resource_features as compute_act_resource_features,
)
from feature_engineering_res import (
    compute_features as compute_res_features,
    compute_resource_features as compute_res_resource_features,
)
from feature_engineering_time import (
    compute_features as compute_time_features,
    compute_resource_features as compute_time_resource_features,
)

TARGET = config_args.target
LABEL_BY_TARGET = {
    'time': 'TimeLabel',
    'act': 'ActLabel',
    'res': 'ResLabel',
}


UNSEEN_TIME_ANOMALIES = (
    'Micro-Delay',
    'Macro-Delay',
    'TS-Round-Min',
    'TS-Round-Hour',
    'TS-Round-Day',
    'Trace-Level-Storm',
)
UNSEEN_TIME_SCENARIO_PREFIX = 'Unseen_Time'


def unseen_time_scenario_names():
    return tuple(
        f'{UNSEEN_TIME_SCENARIO_PREFIX}_{anomaly_type}'
        for anomaly_type in UNSEEN_TIME_ANOMALIES
    )

FEATURE_MODULES = {
    'time': {
        'compute_base': compute_time_features,
        'compute_resource': compute_time_resource_features,
    },
    'act': {
        'compute_base': compute_act_features,
        'compute_resource': compute_act_resource_features,
    },
    'res': {
        'compute_base': compute_res_features,
        'compute_resource': compute_res_resource_features,
    },
}


def compute_target_features(df, schema, train_stats=None):
    module_cfg = FEATURE_MODULES[TARGET]
    kwargs = {
        'case_col': schema['case'],
        'act_col': schema['act'],
        'time_col': schema['time'],
        'res_col': schema.get('res'),
    }

    if train_stats is None:
        df, base_stats = module_cfg['compute_base'](df, **kwargs)
        train_stats = {'base_stats': base_stats}
    else:
        df = module_cfg['compute_base'](
            df,
            train_stats=train_stats['base_stats'],
            **kwargs,
        )

    if schema.get('res') and schema['res'] in df.columns:
        if 'res_stats' not in train_stats:
            df, res_stats = module_cfg['compute_resource'](df, **kwargs)
            train_stats['res_stats'] = res_stats
        else:
            df = module_cfg['compute_resource'](
                df,
                train_stats=train_stats['res_stats'],
                **kwargs,
            )

    return df, train_stats


def normalize_categorical_columns(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna('__MISSING__').astype(str)
    return df


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'PRAUC': (
            average_precision_score(y_true, y_prob)
            if y_true.nunique() > 1
            else 0.0
        ),
    }


def scenario_names_for_target():
    if TARGET == 'time':
        return ('Seen',) + unseen_time_scenario_names()
    return ('Seen',)


def discover_test_files():
    test_files = []
    for scenario_name in scenario_names_for_target():
        scenario_dir = TEST_LOGS_DIR / scenario_name
        if not scenario_dir.exists():
            print(f'[WARN] Missing test directory: {scenario_dir}')
            continue
        test_files.extend(sorted(scenario_dir.glob('*/*_test.csv')))
    return test_files


def build_off_target_mask(df):
    target_label = LABEL_BY_TARGET[TARGET]
    off_target_labels = [
        label
        for perspective, label in LABEL_BY_TARGET.items()
        if perspective != TARGET
    ]
    return (df[target_label] == 0) & (df[off_target_labels].max(axis=1) == 1)


def process_test_scenario(test_file_path, models_dir, all_model_files):
    rate_str = test_file_path.parent.name
    scenario_name = test_file_path.parent.parent.name
    file_name = test_file_path.name

    base_name = file_name.replace('_test.csv', '') + '.csv'
    if base_name not in DATASET_SCHEMAS:
        return []

    schema = DATASET_SCHEMAS[base_name]
    if TARGET == 'res' and not schema.get('res'):
        return []
    if scenario_name.startswith(f'{UNSEEN_TIME_SCENARIO_PREFIX}_') and TARGET != 'time':
        return []

    print(f'Evaluating {file_name} | scenario={scenario_name} | rate={rate_str}')

    raw_df = pd.read_csv(test_file_path, low_memory=False)
    raw_df[schema['time']] = pd.to_datetime(
        raw_df[schema['time']], utc=True, errors='coerce', format='mixed'
    )

    label_col = LABEL_BY_TARGET[TARGET]
    dataset_prefix = f'{base_name.replace(".csv", "")}_run'
    dataset_model_files = [
        file_name
        for file_name in all_model_files
        if file_name.startswith(dataset_prefix)
    ]

    run_groups = {}
    for model_file in dataset_model_files:
        run_id = model_file.split('__')[0]
        run_groups.setdefault(run_id, []).append(model_file)

    results = []
    for run_id, model_files_for_run in sorted(run_groups.items()):
        model_files_for_run = sorted(model_files_for_run)
        first_model_file = model_files_for_run[0]
        first_artifact = joblib.load(models_dir / first_model_file)

        train_stats = first_artifact['train_stats']
        num_cols = list(first_artifact['num_cols'])
        cat_cols = list(first_artifact['cat_cols'])

        df, _ = compute_target_features(
            raw_df.copy(),
            schema,
            train_stats=train_stats,
        )
        df = normalize_categorical_columns(df, cat_cols)

        required_cols = num_cols + cat_cols
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f'{base_name} {run_id}: missing features {missing}')

        X_test = df[required_cols]
        y_test = df[label_col].fillna(0).astype(int)
        off_target_mask = build_off_target_mask(df)
        total_off_target = int(off_target_mask.sum())

        for model_file in model_files_for_run:
            artifact = (
                first_artifact
                if model_file == first_model_file
                else joblib.load(models_dir / model_file)
            )
            pipeline = artifact['pipeline']
            model_name = model_file.rsplit('__', 1)[1].replace('.joblib', '')

            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, predictions, probabilities)

            false_alarm_rate = 0.0
            if total_off_target > 0:
                false_alarm_rate = float(predictions[off_target_mask].sum() / total_off_target)

            results.append(
                {
                    'Base_Dataset': base_name,
                    'Run_ID': run_id,
                    'Model': model_name,
                    'Scenario': scenario_name,
                    'Test_Rate': rate_str.replace('_', '.'),
                    'F1': metrics['F1'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'PRAUC': metrics['PRAUC'],
                    'Off_Target_False_Alarm_Rate': false_alarm_rate,
                    'Target_Anomaly_Count': int(y_test.sum()),
                    'Off_Target_Anomaly_Count': total_off_target,
                }
            )

    return results


def main():
    models_dir = BASE_DIR / 'saved_models' / TARGET / RATE_STR
    result_dir = BASE_DIR / 'results' / TARGET / RATE_STR / 'advanced_evaluation'
    result_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        raise FileNotFoundError(models_dir)

    test_files = discover_test_files()
    if not test_files:
        raise FileNotFoundError(f'No compatible test files found under {TEST_LOGS_DIR}.')

    all_model_files = [
        file_name
        for file_name in os.listdir(models_dir)
        if file_name.endswith('.joblib')
    ]
    if not all_model_files:
        raise FileNotFoundError(f'No saved models found under {models_dir}.')

    all_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                process_test_scenario,
                path,
                models_dir,
                all_model_files,
            ): path
            for path in test_files
        }

        total = len(futures)
        for completed, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as exc:
                print(f'[ERROR] {path}: {exc}')
            print(f'[PROGRESS] {completed}/{total} ({completed / total:.1%})')

    if not all_results:
        raise RuntimeError('No evaluation rows were generated.')

    result_df = pd.DataFrame(all_results)
    detailed_path = result_dir / 'advanced_metrics_detailed.csv'
    result_df.to_csv(detailed_path, index=False)

    metric_cols = [
        'F1',
        'Precision',
        'Recall',
        'PRAUC',
        'Off_Target_False_Alarm_Rate',
    ]
    aggregated_df = (
        result_df.groupby(
            ['Base_Dataset', 'Model', 'Scenario', 'Test_Rate']
        )[metric_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    aggregated_df.columns = [
        '_'.join(column).strip('_')
        if isinstance(column, tuple)
        else column
        for column in aggregated_df.columns
    ]
    aggregated_path = result_dir / 'advanced_metrics_aggregated.csv'
    aggregated_df.to_csv(aggregated_path, index=False)

    mean_columns = [column for column in aggregated_df.columns if column.endswith('_mean')]
    global_df = (
        aggregated_df.groupby(['Model', 'Scenario', 'Test_Rate'], as_index=False)[mean_columns]
        .mean()
    )
    global_path = result_dir / 'advanced_metrics_global_mean.csv'
    global_df.to_csv(global_path, index=False)

    print(f'Evaluation completed for target={TARGET}.')
    print(f'Saved: {detailed_path}')
    print(f'Saved: {aggregated_path}')
    print(f'Saved: {global_path}')


if __name__ == '__main__':
    main()
