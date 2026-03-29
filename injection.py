import numpy as np
import pandas as pd

def inject_anomalies(df, case_col, act_col, time_col, anomaly_rate=0.3):
    df = df.copy().sort_values([case_col, time_col]).reset_index(drop=True)

    df['Duration'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)

    act_stats = df[df['Duration'] > 0].groupby(act_col)['Duration'].agg(['mean', 'std']).fillna(0)
    act_list = df[act_col].unique()

    n_total = int(len(df) * anomaly_rate)
    n_act = n_total
    n_time = n_total

    df['ActivityLabel'] = 0
    df['TimeLabel'] = 0

    act_idx = np.random.choice(df.index, size=n_act, replace=False)
    df.loc[act_idx, 'ActivityLabel'] = 1

    for idx in act_idx:
        current_act = df.loc[idx, act_col]
        possible_acts = [a for a in act_list if a != current_act]
        df.loc[idx, act_col] = np.random.choice(possible_acts)

    first_events_idx = df.groupby(case_col).head(1).index
    valid_time_pool = np.setdiff1d(df.index, first_events_idx)

    time_idx = np.random.choice(valid_time_pool, size=n_time, replace=False)

    df.loc[time_idx, 'TimeLabel'] = 1
    df['time_shift'] = 0.0

    for idx in time_idx:
        act = df.loc[idx, act_col]
        mean_val = act_stats.loc[act, 'mean'] if act in act_stats.index else 0
        std_val = act_stats.loc[act, 'std'] if act in act_stats.index else 0

        old_duration = df.loc[idx, 'Duration']
        r = np.random.random_sample()
        new_duration = (mean_val + std_val) * (1.0 + r)

        shift = max(0, new_duration - old_duration)
        df.loc[idx, 'time_shift'] += shift

    df[time_col] += pd.to_timedelta(df['time_shift'], unit='s')

    return df.drop(columns=['Duration', 'time_shift'])