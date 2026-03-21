import pandas as pd
import numpy as np
import pm4py
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(df, case_col='Case ID', time_col='Complete Timestamp'):
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[time_col])
    return df.sort_values(by=[case_col, time_col]).copy()


def feature_engineering(df, case_col='Case ID', act_col='Activity', time_col='Complete Timestamp'):
    df = df.copy()

    # Sort by case and time
    df = df.sort_values(by=[case_col, time_col])

    # 1. Feature base
    df['delta_t'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)
    df['cum_t'] = df.groupby(case_col)['delta_t'].cumsum()

    df['trace_len'] = df.groupby(case_col)[act_col].transform('count')
    df['event_idx'] = df.groupby(case_col).cumcount() + 1
    df['trace_pos'] = df['event_idx'] / df['trace_len']

    # 2. Circular time features
    hours = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

    days = df[time_col].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * days / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * days / 7.0)

    # 3. Linear time features
    df['hour_linear'] = hours / 24.0
    df['day_linear'] = days / 6.0

    # 4. Scaling
    scaler = MinMaxScaler()
    df[['delta_t_scaled', 'cum_t_scaled', 'trace_pos_scaled']] = scaler.fit_transform(
        df[['delta_t', 'cum_t', 'trace_pos']]
    )

    return df


def detect_heuristic_errors(df, case_col='Case ID', time_col='Complete Timestamp'):
    time_diff = df.groupby(case_col)[time_col].diff().dt.total_seconds()
    df['is_logical_error'] = (time_diff <= 0).fillna(False)
    return df


def detect_conformance_errors(df, case_col='Case ID', act_col='Activity', time_col='Complete Timestamp'):
    df_pm = pm4py.format_dataframe(df, case_id=case_col, activity_key=act_col, timestamp_key=time_col)
    net, im, fm = pm4py.discover_petri_net_inductive(df_pm, noise_threshold=0.2)

    replayed_traces = pm4py.conformance_diagnostics_token_based_replay(df_pm, net, im, fm)
    log = pm4py.convert_to_event_log(df_pm)

    trace_flags = {
        trace.attributes['concept:name']: not replayed_traces[i]['trace_is_fit']
        for i, trace in enumerate(log)
    }
    df['is_out_of_sequence'] = df[case_col].map(trace_flags)
    return df


def get_top_shap_features(model, X, outliers_mask):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[outliers_mask])

    feature_names = X.columns
    return [feature_names[np.argmax(np.abs(vals))] for vals in shap_values]


def apply_surgical_masking(df, mask_col, time_col='Complete Timestamp'):
    df.loc[df[mask_col], time_col] = pd.NaT
    return df


def generate_visualizations(df, base_name, output_dir):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df['anomaly_score'], bins=50, kde=True)
    plt.title(f'Anomaly Score Distribution - {base_name}')
    plt.savefig(output_dir / f'{base_name}_score_dist.png')
    plt.close()

    if 'trace_pos' in df.columns and 'delta_t' in df.columns and 'is_outlier_custom' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='trace_pos', y='delta_t', hue='is_outlier_custom', data=df, alpha=0.6)
        plt.title(f'Anomalies Scatter - {base_name}')
        plt.savefig(output_dir / f'{base_name}_scatter.png')
        plt.close()
