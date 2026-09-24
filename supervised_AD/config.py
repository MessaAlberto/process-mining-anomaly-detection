import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, choices=['time', 'act', 'res'], default='time')
args, _ = parser.parse_known_args()

TIME_ANOMALY_RATE = 0.05
ACT_ANOMALY_RATE = 0.05
RES_ANOMALY_RATE = 0.05
TEST_ANOMALY_RATES = (0.01, 0.05, 0.15, 0.30)

TRAIN_PCT = 0.60
TEST_PCT = 1.0 - TRAIN_PCT
FEATURE_EVAL_TRAIN_PCT = 0.80
SPLIT_SEED = 42
NUM_ITERS = 5

TIME_RATE_STR = f"{TIME_ANOMALY_RATE:.2f}".replace('.', '_')
ACT_RATE_STR = f"{ACT_ANOMALY_RATE:.2f}".replace('.', '_')
RES_RATE_STR = f"{RES_ANOMALY_RATE:.2f}".replace('.', '_')
TRAIN_RATE_STR = f"{TRAIN_PCT:.2f}".replace('.', '_')

RATE_STR = (
    f"train_{TRAIN_RATE_STR}_time_{TIME_RATE_STR}_"
    f"act_{ACT_RATE_STR}_res_{RES_RATE_STR}"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_DATASET_DIR = PROJECT_ROOT / 'dataset'
REAL_LIFE_LOG_DIR = INPUT_DATASET_DIR / 'real_life_logs'
COMUZZI_LOG_DIR = INPUT_DATASET_DIR / 'comuzzi_logs'
ARTIFICIAL_LOG_DIR = INPUT_DATASET_DIR / 'artificial_logs'

DATA_DIR = BASE_DIR / 'dataset'
INJECTED_LOGS_ROOT = DATA_DIR / 'injected_logs'
INJ_DIR = INJECTED_LOGS_ROOT / RATE_STR
TEST_LOGS_ROOT = DATA_DIR / 'test_logs'
TEST_LOGS_DIR = TEST_LOGS_ROOT / RATE_STR

RESULT_DIR = BASE_DIR / 'results' / args.target / RATE_STR
FEATURES_CONFIG_PATH = BASE_DIR / (
    'features_config_time.json' if args.target == 'time' else f'features_config_{args.target}.json'
)

for directory in [DATA_DIR, INJ_DIR, TEST_LOGS_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CASE_COL = 'case_id'
ACT_COL = 'activity'
TIME_COL = 'timestamp'
RES_COL = 'resource'

STANDARD_COLUMNS = [CASE_COL, ACT_COL, TIME_COL]


def split_case_ids(df, case_col, train_size=TRAIN_PCT, random_state=SPLIT_SEED):
    case_ids = sorted(df[case_col].dropna().unique().tolist(), key=str)
    if len(case_ids) < 2:
        raise ValueError(f"At least two cases are required to split '{case_col}'.")

    return train_test_split(
        case_ids,
        train_size=train_size,
        random_state=random_state,
    )

RAW_DATASET_PATHS = {
    # Comuzzi logs
    'bpi_2012.csv': COMUZZI_LOG_DIR / 'bpi_2012.csv',
    'bpi_2013.csv': COMUZZI_LOG_DIR / 'bpi_2013.csv',
    'small_log.csv': COMUZZI_LOG_DIR / 'small_log.csv',
    'large_log.csv': COMUZZI_LOG_DIR / 'large_log.csv',

    # Artificial logs
    'art_log_1.csv': ARTIFICIAL_LOG_DIR / 'art_log_1.csv',
    'art_log_2.csv': ARTIFICIAL_LOG_DIR / 'art_log_2.csv',

    # Real-life logs
    'BPIC12.csv': REAL_LIFE_LOG_DIR / 'BPIC12.csv',
    'BPIC13_C.csv': REAL_LIFE_LOG_DIR / 'BPIC13_C.csv',
    'BPIC13_I.csv': REAL_LIFE_LOG_DIR / 'BPIC13_I.csv',
    'BPIC13_O.csv': REAL_LIFE_LOG_DIR / 'BPIC13_O.csv',
    'BPIC20_D.csv': REAL_LIFE_LOG_DIR / 'BPIC20_D.csv',
    'BPIC20_I.csv': REAL_LIFE_LOG_DIR / 'BPIC20_I.csv',
    'BPIC20_PE.csv': REAL_LIFE_LOG_DIR / 'BPIC20_PE.csv',
    'BPIC20_PR.csv': REAL_LIFE_LOG_DIR / 'BPIC20_PR.csv',
    'BPIC20_R.csv': REAL_LIFE_LOG_DIR / 'BPIC20_R.csv',
}


def infer_raw_schema_from_columns(columns, file_name='dataset'):
    columns = set(columns)

    if {'case_id', 'activity', 'timestamp'}.issubset(columns):
        return {
            'case': 'case_id',
            'act': 'activity',
            'time': 'timestamp',
            'res': 'resource' if 'resource' in columns else None,
        }

    if {'Case ID', 'Activity', 'Complete Timestamp'}.issubset(columns):
        return {
            'case': 'Case ID',
            'act': 'Activity',
            'time': 'Complete Timestamp',
            'res': 'Resource' if 'Resource' in columns else None,
        }

    if {'case:concept:name', 'concept:name', 'time:timestamp'}.issubset(columns):
        return {
            'case': 'case:concept:name',
            'act': 'concept:name',
            'time': 'time:timestamp',
            'res': 'org:resource' if 'org:resource' in columns else ('org:role' if 'org:role' in columns else None),
        }

    raise ValueError(f"{file_name}: cannot infer schema. Available columns: {list(columns)}")


def infer_raw_schema(path: Path, file_name: str) -> dict:
    raw_head = pd.read_csv(path, nrows=1, low_memory=False)
    return infer_raw_schema_from_columns(raw_head.columns, file_name)


def standard_schema_for_raw_schema(raw_schema: dict) -> dict:
    return {
        'case': CASE_COL,
        'act': ACT_COL,
        'time': TIME_COL,
        'res': RES_COL if raw_schema.get('res') else None,
    }


def load_and_standardize_raw_log(path: Path, file_name: str | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw_schema = infer_raw_schema_from_columns(raw.columns, file_name or path.name)

    use_cols = [raw_schema['case'], raw_schema['act'], raw_schema['time']]
    if raw_schema.get('res') and raw_schema['res'] in raw.columns:
        use_cols.append(raw_schema['res'])

    df = raw[use_cols].copy()
    rename_map = {
        raw_schema['case']: CASE_COL,
        raw_schema['act']: ACT_COL,
        raw_schema['time']: TIME_COL,
    }
    if raw_schema.get('res') and raw_schema['res'] in raw.columns:
        rename_map[raw_schema['res']] = RES_COL

    df = df.rename(columns=rename_map)
    df[CASE_COL] = df[CASE_COL].astype('string')
    df[ACT_COL] = df[ACT_COL].astype('string').fillna('__MISSING_ACTIVITY__')
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors='coerce', format='mixed')

    if RES_COL in df.columns:
        df[RES_COL] = df[RES_COL].astype('string').fillna('__MISSING_RESOURCE__')

    before = len(df)
    df = df.dropna(subset=[CASE_COL, TIME_COL]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] {file_name or path.name}: dropped {dropped} rows with missing case_id or timestamp")

    return df.reset_index(drop=True)


def _build_dataset_schemas() -> dict:
    schemas = {}
    for dataset_name, path in RAW_DATASET_PATHS.items():
        if not path.exists():
            continue
        raw_schema = infer_raw_schema(path, dataset_name)
        schemas[dataset_name] = standard_schema_for_raw_schema(raw_schema)
    return schemas


DATASET_SCHEMAS = _build_dataset_schemas()
DATASETS = list(DATASET_SCHEMAS.keys())


def iter_raw_datasets():
    for dataset_name in DATASETS:
        path = RAW_DATASET_PATHS.get(dataset_name)
        if path is None:
            continue
        if not path.exists():
            print(f"[SKIP] {dataset_name}: file not found at {path}")
            continue
        yield dataset_name, path


def get_models(scale_pos_w=1.0):
    return {
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_w,
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        ),
    }
