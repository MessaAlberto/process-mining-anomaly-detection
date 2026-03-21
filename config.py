from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw_data'
POISONED_DIR = DATA_DIR / 'poisoned_data'
K_DIST_PLOTS_DIR = DATA_DIR / 'k_dist_plots'
RESULTS_DIR = DATA_DIR / 'evaluation_results'
COMPARISON_DIR = DATA_DIR / 'comparison_results'

for d in [RAW_DIR, POISONED_DIR, K_DIST_PLOTS_DIR, RESULTS_DIR, COMPARISON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ML_MODELS = {
    'IsolationForest': {
        'class': IsolationForest,
        'kwargs': {'contamination': 'auto', 'random_state': 42}
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
            'algorithm': 'auto',   # or 'ball_tree'
            'n_jobs': -1
        }
    }
}

TEST_FEATURE_SETS = {
    'minimal': ['delta_t_scaled'],
    'position_based': ['delta_t_scaled', 'cum_t_scaled', 'trace_pos_scaled'],
    'time_based_circular': ['delta_t_scaled', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
    'time_based_linear': ['delta_t_scaled', 'hour_linear', 'day_linear'],
    'time_based': ['delta_t_scaled', 'hour_linear', 'day_linear', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
    'full': ['delta_t_scaled', 'cum_t_scaled', 'trace_pos_scaled', 'hour_linear', 'day_linear', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
}

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]

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
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None  # Non ha risorse!
    },
    'large_log.csv': {
        'case': 'Case ID', 'act': 'Activity', 'time': 'Complete Timestamp', 'res': None  # Non ha risorse!
    }
}

DATASETS = list(DATASET_SCHEMAS.keys())