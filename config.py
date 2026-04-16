from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RAW_DIR = DATA_DIR / 'raw_data'
POISONED_DIR = DATA_DIR / 'poisoned_data'
SCORED_DIR = DATA_DIR / 'scored_data'
MODEL_COMP_DIR = RESULT_DIR / 'model_comparison_results'
AE_COMP_DIR = RESULT_DIR / 'ae_comparison_results'
DISTRIB_DIR = RESULT_DIR / 'distribution_analysis'

for d in [DATA_DIR, RESULT_DIR, RAW_DIR, POISONED_DIR, SCORED_DIR, MODEL_COMP_DIR, AE_COMP_DIR, DISTRIB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_RUNS = 10
ANOMALY_RATE = 0.1
START_RUN = 1
END_RUN = 10

TRAIN_PCT = 0.6
VAL_PCT = 0.2
TEST_PCT = 0.2

ML_MODELS = {
    'IsolationForest': {
        'class': IsolationForest,
        'kwargs': {'n_estimators': 200, 'contamination': 'auto', 'random_state': 42}
    },
    'LOF': {
        'class': LocalOutlierFactor,
        'kwargs': {'novelty': True, 'contamination': 'auto'}
    },
    'DBSCAN': {
        'class': DBSCAN,
        'kwargs': {
            'eps': 0.8,
            'min_samples': 5,
            'algorithm': 'auto',
            'n_jobs': 1
        }
    }
}

TEST_FEATURE_SETS = {
    'delta_t': ['delta_t'],
    'delta_z_micro': ['delta_t', 'z_score_micro'],
    'cum_z_cum_t': ['cum_t', 'z_score_cum_t'],

    'pure_cyclical': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
    'delta_cyclical': ['delta_t', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
    'omni_fusion_cyclical': ['delta_t', 'cum_t', 'z_score_micro', 'z_score_transition', 'z_score_cum_t', 'resource_workload', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
}

IQR_THRESHOLDS = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

LEVELS = ['L3_Global', 'L2_Activity', 'L1_Micro']
MIN_SAMPLES = 30
MAX_DBSCAN_ROWS = 30000

DATASET_SCHEMAS = {
    'bpi_2012.csv': {'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': 'Resource'},
    'bpi_2013.csv': {'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': 'Resource'},
    'BPI_Challenge_2012.csv': {'case': 'case:concept:name', 'act': 'concept:name', 'time': 'time:timestamp', 'res': 'org:resource'},
    'RequestForPayment.csv': {'case': 'case:concept:name', 'act': 'concept:name', 'time': 'time:timestamp', 'res': 'org:role'},
    # 'RequestForPayment.csv': {'case': 'case:concept:name', 'act': 'concept:name', 'time': 'time:timestamp', 'res': 'org:resource'},
    # No resources in this dataset
    'small_log.csv': {'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None},
    # No resources in this dataset
    'large_log.csv': {'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None}
}

DATASETS = list(DATASET_SCHEMAS.keys())
