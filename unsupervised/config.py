from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
REAL_LIFE_LOG_DIR = DATASET_DIR / "real_life_logs"
COMUZZI_LOG_DIR = DATASET_DIR / "comuzzi_logs"
ARTIFICIAL_LOG_DIR = DATASET_DIR / "artificial_logs"

UNSUPERVISED_DIR = PROJECT_ROOT / "unsupervised_AD"
DATA_DIR = UNSUPERVISED_DIR / "data"
RESULT_DIR = UNSUPERVISED_DIR / "results"

for directory in [DATA_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CASE_COL = "case_id"
ACT_COL = "activity"
TIME_COL = "timestamp"
RES_COL = "resource"

REAL_LIFE_DATASETS = {
    "BPIC12.csv": REAL_LIFE_LOG_DIR / "BPIC12.csv",
    "BPIC13_C.csv": REAL_LIFE_LOG_DIR / "BPIC13_C.csv",
    "BPIC13_I.csv": REAL_LIFE_LOG_DIR / "BPIC13_I.csv",
    "BPIC13_O.csv": REAL_LIFE_LOG_DIR / "BPIC13_O.csv",
    "BPIC20_D.csv": REAL_LIFE_LOG_DIR / "BPIC20_D.csv",
    "BPIC20_I.csv": REAL_LIFE_LOG_DIR / "BPIC20_I.csv",
    "BPIC20_PE.csv": REAL_LIFE_LOG_DIR / "BPIC20_PE.csv",
    "BPIC20_PR.csv": REAL_LIFE_LOG_DIR / "BPIC20_PR.csv",
    "BPIC20_R.csv": REAL_LIFE_LOG_DIR / "BPIC20_R.csv",
}

COMUZZI_DATASETS = {
    "bpi_2012.csv": COMUZZI_LOG_DIR / "bpi_2012.csv",
    "bpi_2013.csv": COMUZZI_LOG_DIR / "bpi_2013.csv",
    "small_log.csv": COMUZZI_LOG_DIR / "small_log.csv",
    "large_log.csv": COMUZZI_LOG_DIR / "large_log.csv",
}

ARTIFICIAL_DATASETS = {
    "art_log_1.csv": ARTIFICIAL_LOG_DIR / "art_log_1.csv",
    "art_log_2.csv": ARTIFICIAL_LOG_DIR / "art_log_2.csv",
}

DATASETS = {
    **REAL_LIFE_DATASETS,
    **COMUZZI_DATASETS,
    **ARTIFICIAL_DATASETS,
}

ACTIVE_DATASETS = list(DATASETS.keys())

NUM_RUNS = 10
ANOMALY_RATE = 0.15
RANDOM_SEED = 42

TRAIN_PCT = 0.60
VAL_PCT = 0.20
TEST_PCT = 0.20

MODEL_NAME = "IsolationForest"
MODEL_KWARGS = {
    "n_estimators": 200,
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}

MIN_GROUP_SIZE = 30
GMM_MIN_VALID_SCORES = 10
POSITION_MIN_STD = 0.05

LEVEL_GROUP_COLS = {
    "L3_Global": [],
    "L2_Activity": [ACT_COL],
    "L1_ActivityResource": [ACT_COL, RES_COL],
}

TIME_LEVELS = ["L3_Global", "L2_Activity", "L1_ActivityResource"]
ACT_LEVELS = ["L3_Global", "L2_Activity", "L1_ActivityResource"]

TIME_FEATURE_SETS = {
    "delta_t": ["delta_t"],
    "delta_z_micro": ["delta_t", "z_score_micro"],
    "cum_z_cum_t": ["cum_t", "z_score_cum_t"],
    "pure_cyclical": ["hour_sin", "hour_cos", "day_sin", "day_cos"],
    "delta_cyclical": ["delta_t", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    "omni_fusion_cyclical": [
        "delta_t",
        "cum_t",
        "z_score_micro",
        "z_score_transition",
        "z_score_cum_t",
        "resource_workload",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ],
}

ACT_FEATURE_SETS = {
    # Resource-free baselines
    "position_3_bins": [
        "position_surprise_3",
    ],
    "trigram": [
        "trigram_surprise",
    ],

    # Best resource-free model
    "position_trigram": [
        "position_surprise_3",
        "trigram_surprise",
    ],

    # Resource-based baseline
    "activity_given_resource": [
        "activity_given_resource_surprise",
    ],

    # Best resource-based model
    "activity_resource_trigram": [
        "activity_given_resource_surprise",
        "trigram_surprise",
    ],

    # Extended resource-based model
    "compact_best": [
        "activity_given_resource_surprise",
        "position_surprise_3",
        "trigram_surprise",
        "activity_surprise",
    ],
}

TARGET_ALIASES = {
    "time": "time",
    "timestamp": "time",
    "act": "act",
    "activity": "act",
}

TARGET_LABELS = {
    "time": "TimeLabel",
    "act": "ActivityLabel",
}


def normalize_target(target: str) -> str:
    key = str(target).strip().lower()
    if key not in TARGET_ALIASES:
        raise ValueError("target must be one of: time, act")
    return TARGET_ALIASES[key]


def get_label_col(target: str) -> str:
    return TARGET_LABELS[normalize_target(target)]


def get_feature_sets(target: str) -> dict[str, list[str]]:
    normalized = normalize_target(target)
    source = TIME_FEATURE_SETS if normalized == "time" else ACT_FEATURE_SETS
    return {name: list(columns) for name, columns in source.items()}


def get_levels(target: str) -> list[str]:
    normalized = normalize_target(target)
    return list(TIME_LEVELS if normalized == "time" else ACT_LEVELS)


def get_target_paths(target: str) -> dict[str, Path]:
    normalized = normalize_target(target)
    target_data_dir = DATA_DIR / normalized
    paths = {
        "data": target_data_dir,
        "injected": DATA_DIR / "injected_logs",
        "scored": target_data_dir / "scored_logs",
        "thresholds": target_data_dir / "saved_thresholds",
        "results": RESULT_DIR / normalized,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
