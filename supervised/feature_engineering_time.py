import numpy as np
import pandas as pd


FEATURE_COLUMN_GROUPS = {
    'numeric': [
        'delta_t', 'cum_t', 'trace_pos', 'act_rework_count', 'momentum_ratio',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'z_score_global', 'z_score_act',
        'act_delta_t_percentile', 'trace_mean_deviation'
    ],
    'categorical': ['act_transition'],
    'resource_numeric': ['z_score_act_res', 'res_act_freq', 'res_workload_day'],
    'resource_categorical': ['handover']
}


def get_feature_columns(case_col='Case ID', act_col='Activity', time_col='Complete Timestamp', res_col=None):
    num_cols = list(FEATURE_COLUMN_GROUPS['numeric'])
    cat_cols = [act_col] + list(FEATURE_COLUMN_GROUPS['categorical'])

    if res_col and res_col is not None:
        num_cols.extend(FEATURE_COLUMN_GROUPS['resource_numeric'])
        cat_cols.extend([res_col] + FEATURE_COLUMN_GROUPS['resource_categorical'])

    return {'num_cols': num_cols, 'cat_cols': cat_cols}

def compute_features(df, case_col='Case ID', act_col='Activity', time_col='Complete Timestamp', res_col=None, train_stats=None):
    df = df.copy()
    df = df.sort_values([case_col, time_col]).reset_index(drop=True)

    df['delta_t'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)
    df['cum_t'] = df.groupby(case_col)['delta_t'].cumsum()
    df['trace_len'] = df.groupby(case_col)[act_col].transform('count')

    df['event_idx'] = df.groupby(case_col).cumcount() + 1
    df['trace_pos'] = df['event_idx'] / df['trace_len']

    # Number of times this specific activity was previously executed in the case
    df['act_rework_count'] = df.groupby([case_col, act_col]).cumcount()

    # Mean duration of the previous 3 events in the case
    df['rolling_delta_t_mean'] = df.groupby(case_col)['delta_t'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0)

    # Ratio of current duration to the rolling mean duration
    df['momentum_ratio'] = df['delta_t'] / (df['rolling_delta_t_mean'] + 1e-6)

    df['trace_expanding_mean'] = df.groupby(case_col)['delta_t'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    )

    # Ratio of current duration to the historical trace mean
    df['trace_mean_deviation'] = df['delta_t'] / (df['trace_expanding_mean'] + 1e-6)
    df = df.drop(columns=['trace_expanding_mean'])

    # Cyclical encoding of the hour of the day
    hours = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

    # Cyclical encoding of the day of the week
    days = df[time_col].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * days / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * days / 7.0)

    df['prev_act'] = df.groupby(case_col)[act_col].shift(1).fillna('START')

    # String representing the transition from previous to current activity
    df['act_transition'] = df['prev_act'].astype(str) + " -> " + df[act_col].astype(str)

    if train_stats is None:
        stats = {}
        mask = df['delta_t'] > 0
        stats['global_m'] = df[mask]['delta_t'].mean()
        stats['global_s'] = df[mask]['delta_t'].std()
        stats['act_stats'] = df[mask].groupby(act_col)['delta_t'].agg(['mean', 'std']).fillna(0)
        stats['act_dist'] = df[mask].groupby(act_col)['delta_t'].apply(lambda x: np.sort(x.values)).to_dict()

        df['z_score_global'] = (df['delta_t'] - stats['global_m']) / (stats['global_s'] + 1e-6)
        
        df = df.join(stats['act_stats'], on=act_col)
        df['z_score_act'] = (df['delta_t'] - df['mean']) / (df['std'] + 1e-6)
        df = df.drop(columns=['mean', 'std'])

        df['act_delta_t_percentile'] = df.groupby(act_col)['delta_t'].transform(lambda x: x.rank(pct=True))

        return df, stats
    else:
        df['z_score_global'] = (df['delta_t'] - train_stats['global_m']) / (train_stats['global_s'] + 1e-6)
        
        df = df.join(train_stats['act_stats'], on=act_col)
        df['mean'] = df['mean'].fillna(train_stats['global_m'])
        df['std'] = df['std'].fillna(train_stats['global_s'])
        df['z_score_act'] = (df['delta_t'] - df['mean']) / (df['std'] + 1e-6)
        df = df.drop(columns=['mean', 'std'])

        pct_list = []
        for act, group in df.groupby(act_col):
            dist = train_stats['act_dist'].get(act)
            if dist is not None and len(dist) > 0:
                pct = np.searchsorted(dist, group['delta_t'].values) / len(dist)
                pct_list.append(pd.Series(pct, index=group.index))
            else:
                pct_list.append(pd.Series(0.5, index=group.index))
        
        if pct_list:
            df['act_delta_t_percentile'] = pd.concat(pct_list).sort_index()
        else:
            df['act_delta_t_percentile'] = 0.5

        return df

def compute_resource_features(df, case_col, act_col, time_col, res_col, train_stats=None):
    df = df.copy()

    if res_col not in df.columns or df[res_col].isnull().all():
        return (df, {}) if train_stats is None else df

    df['prev_res'] = df.groupby(case_col)[res_col].shift(1).fillna('START')

    # String representing the handover transition between resources
    df['handover'] = df['prev_res'].astype(str) + " -> " + df[res_col].astype(str)

    # Cumulative number of events executed by this resource on the current day
    df['date_only'] = df[time_col].dt.date
    df['res_workload_day'] = df.groupby([res_col, 'date_only']).cumcount()
    df = df.drop(columns=['date_only'])

    if train_stats is None:
        stats = {}
        mask = df['delta_t'] > 0
        stats['act_res_stats'] = df[mask].groupby([act_col, res_col])['delta_t'].agg(['mean', 'std']).fillna(0)
        stats['res_act_freq'] = df.groupby([res_col, act_col]).size().to_dict()

        df = df.join(stats['act_res_stats'], on=[act_col, res_col], rsuffix='_act_res')
        df['expected_act_res_duration'] = df['mean'].fillna(0)
        act_res_std = df['std'].fillna(1e-6)
        df['z_score_act_res'] = (df['delta_t'] - df['expected_act_res_duration']) / (act_res_std + 1e-6)
        df = df.drop(columns=['mean', 'std'])

        df['res_act_freq'] = df.set_index([res_col, act_col]).index.map(stats['res_act_freq']).fillna(0).values

        return df, stats
    else:
        df = df.join(train_stats['act_res_stats'], on=[act_col, res_col], rsuffix='_act_res')
        df['expected_act_res_duration'] = df['mean'].fillna(0)
        act_res_std = df['std'].fillna(1e-6)
        df['z_score_act_res'] = (df['delta_t'] - df['expected_act_res_duration']) / (act_res_std + 1e-6)
        df = df.drop(columns=['mean', 'std'])

        df['res_act_freq'] = df.set_index([res_col, act_col]).index.map(train_stats['res_act_freq']).fillna(0).values

        return df