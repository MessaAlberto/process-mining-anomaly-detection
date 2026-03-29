from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw_data'
POISONED_DIR = DATA_DIR / 'poisoned_data'
SCORED_DIR = DATA_DIR / 'scored_data'
MODEL_COMP_DIR = DATA_DIR / 'model_comparison_results'
AE_COMP_DIR = DATA_DIR / 'ae_comparison_results'
IF_THRESHOLD_COMP_DIR = DATA_DIR / 'if_threshold_comparison_results'

for d in [RAW_DIR, POISONED_DIR, SCORED_DIR, MODEL_COMP_DIR, AE_COMP_DIR, IF_THRESHOLD_COMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_RUNS = 10
ANOMALY_RATE = 0.3

ML_MODELS = {
    'IsolationForest': {
        'class': IsolationForest,
        'kwargs': {'contamination': 'auto', 'random_state': 42}
    },
    'LOF': {
        'class': LocalOutlierFactor,
        'kwargs': {'novelty': False, 'contamination': 'auto'}
    },
    'DBSCAN': {
        'class': DBSCAN,
        'kwargs': {
            'eps': 0.8,
            'min_samples': 5,
            'algorithm': 'auto',   # or 'ball_tree'
            'n_jobs': 1
        }
    }
}

TEST_FEATURE_SETS = {
    'minimal': ['delta_t'],
    'position_based': ['delta_t', 'trace_pos'],
    'velocity_based': ['delta_t', 'event_density'],
    'process_context': ['delta_t', 'cum_t', 'trace_pos'],
    'time_based_circular': ['delta_t', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
}

IQR_THRESHOLDS = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

LEVELS = ['L3_Global', 'L2_Activity', 'L1_Micro']
MIN_SAMPLES = 30
MAX_DBSCAN_ROWS = 30000

DATASET_SCHEMAS = {
    'bpi_2012.csv': {
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': 'Resource'
    },
    'bpi_2013.csv': {
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': 'Resource'
    },
    'BPI_Challenge_2012.csv': {
        'case': 'case:concept:name', 'act': 'concept:name', 'time': 'time:timestamp', 'res': 'org:resource'
    },
    'RequestForPayment.csv': {
        'case': 'case:concept:name', 'act': 'concept:name', 'time': 'time:timestamp', 'res': 'org:resource'
    },
    'small_log.csv': {
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None  # No resources in this dataset
    },
    'large_log.csv': {
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None  # No resources in this dataset
    }
}

DATASETS = list(DATASET_SCHEMAS.keys())