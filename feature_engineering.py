import numpy as np
import pandas as pd

def compute_features(df, case_col='Case ID', act_col='Activity', time_col='Complete Timestamp'):
    df = df.copy()

    df['delta_t'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)
    df['cum_t'] = df.groupby(case_col)['delta_t'].cumsum()

    df['trace_len'] = df.groupby(case_col)[act_col].transform('count')
    df['event_idx'] = df.groupby(case_col).cumcount() + 1
    df['trace_pos'] = df['event_idx'] / df['trace_len']

    df['act_rework_count'] = df.groupby([case_col, act_col]).cumcount()

    df['rolling_delta_t_mean'] = df.groupby(case_col)['delta_t'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0)
    df['momentum_ratio'] = df['delta_t'] / (df['rolling_delta_t_mean'] + 1e-6)

    hours = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

    days = df[time_col].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * days / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * days / 7.0)

    df['hour_linear'] = hours / 24.0
    df['day_linear'] = days / 6.0

    df['event_density'] = df['event_idx'] / (df['cum_t'] + 1.0)

    return df

def compute_contextual_features(df, case_col='Case ID', act_col='Activity', res_col=None, time_col='Complete Timestamp'):
    df = df.copy()

    df['prev_act'] = df.groupby(case_col)[act_col].shift(1).fillna('START')
    df['transition'] = df['prev_act'] + "->" + df[act_col]

    train_mask = df['split'] == 'train'
    train_df = df[train_mask].copy()

    act_stats = train_df.groupby(act_col)['delta_t'].agg(['mean', 'std']).fillna(0)
    df['expected_act_duration'] = df[act_col].map(act_stats['mean']).fillna(0)
    act_std = df[act_col].map(act_stats['std']).fillna(0)
    df['z_score_act'] = (df['delta_t'] - df['expected_act_duration']) / (act_std + 1e-6)

    trans_stats = train_df.groupby('transition')['delta_t'].agg(['mean', 'std']).fillna(0)
    df['expected_trans_duration'] = df['transition'].map(trans_stats['mean']).fillna(0)
    trans_std = df['transition'].map(trans_stats['std']).fillna(0)
    df['z_score_transition'] = (df['delta_t'] - df['expected_trans_duration']) / (trans_std + 1e-6)

    df['day_of_week'] = df[time_col].dt.dayofweek.astype(str)
    train_df['day_of_week'] = train_df[time_col].dt.dayofweek.astype(str)

    train_df['act_day_key'] = train_df[act_col].astype(str) + "_" + train_df['day_of_week']
    df['act_day_key'] = df[act_col].astype(str) + "_" + df['day_of_week']

    act_day_stats = train_df.groupby('act_day_key')['delta_t'].agg(['mean', 'std']).fillna(0)
    df['expected_act_day_duration'] = df['act_day_key'].map(act_day_stats['mean']).fillna(0)
    act_day_std = df['act_day_key'].map(act_day_stats['std']).fillna(0)
    df['z_score_act_day'] = (df['delta_t'] - df['expected_act_day_duration']) / (act_day_std + 1e-6)

    if res_col and res_col in df.columns:
        res_stats = train_df.groupby(res_col)['delta_t'].agg(['mean', 'std']).fillna(0)
        df['expected_res_duration'] = df[res_col].map(res_stats['mean']).fillna(0)
        res_std = df[res_col].map(res_stats['std']).fillna(0)
        df['z_score_res'] = (df['delta_t'] - df['expected_res_duration']) / (res_std + 1e-6)

        train_df['micro_key'] = train_df[act_col].astype(str) + "_" + train_df[res_col].astype(str)
        df['micro_key'] = df[act_col].astype(str) + "_" + df[res_col].astype(str)
        micro_stats = train_df.groupby('micro_key')['delta_t'].agg(['mean', 'std']).fillna(0)
        df['expected_micro_duration'] = df['micro_key'].map(micro_stats['mean']).fillna(0)
        micro_std = df['micro_key'].map(micro_stats['std']).fillna(0)
        df['z_score_micro'] = (df['delta_t'] - df['expected_micro_duration']) / (micro_std + 1e-6)
        df = df.drop(columns=['micro_key'])

        train_df['res_day_key'] = train_df[res_col].astype(str) + "_" + train_df['day_of_week']
        df['res_day_key'] = df[res_col].astype(str) + "_" + df['day_of_week']
        res_day_stats = train_df.groupby('res_day_key')['delta_t'].agg(['mean', 'std']).fillna(0)
        df['expected_res_day_duration'] = df['res_day_key'].map(res_day_stats['mean']).fillna(0)
        res_day_std = df['res_day_key'].map(res_day_stats['std']).fillna(0)
        df['z_score_res_day'] = (df['delta_t'] - df['expected_res_day_duration']) / (res_day_std + 1e-6)

        train_df['act_res_day_key'] = train_df['act_day_key'] + "_" + train_df[res_col].astype(str)
        df['act_res_day_key'] = df['act_day_key'] + "_" + df[res_col].astype(str)
        act_res_day_stats = train_df.groupby('act_res_day_key')['delta_t'].agg(['mean', 'std']).fillna(0)
        df['expected_act_res_day_duration'] = df['act_res_day_key'].map(act_res_day_stats['mean']).fillna(0)
        act_res_day_std = df['act_res_day_key'].map(act_res_day_stats['std']).fillna(0)
        df['z_score_act_res_day'] = (df['delta_t'] - df['expected_act_res_day_duration']) / (act_res_day_std + 1e-6)

        df = df.drop(columns=['res_day_key', 'act_res_day_key'])
    else:
        df['expected_res_duration'] = 0.0
        df['z_score_res'] = 0.0
        df['expected_micro_duration'] = 0.0
        df['z_score_micro'] = 0.0
        df['expected_res_day_duration'] = 0.0
        df['z_score_res_day'] = 0.0
        df['expected_act_res_day_duration'] = 0.0
        df['z_score_act_res_day'] = 0.0

    cum_stats = train_df.groupby('event_idx')['cum_t'].agg(['mean', 'std']).fillna(0)
    df['expected_cum_t'] = df['event_idx'].map(cum_stats['mean']).fillna(0)
    cum_std = df['event_idx'].map(cum_stats['std']).fillna(0)
    df['z_score_cum_t'] = (df['cum_t'] - df['expected_cum_t']) / (cum_std + 1e-6)

    if res_col and res_col in df.columns and time_col in df.columns:
        df = df.sort_values(time_col)
        df.set_index(time_col, drop=False, inplace=True)
        df['resource_workload'] = df.groupby(res_col)[case_col].transform(lambda x: x.rolling('2h').count()).fillna(0)
        df.reset_index(drop=True, inplace=True)
    else:
        df['resource_workload'] = 0.0

    df = df.drop(columns=['prev_act', 'transition', 'day_of_week', 'act_day_key'])

    return df.fillna(0)