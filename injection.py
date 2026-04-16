import numpy as np
import pandas as pd


def inject_anomalies(df, case_col, act_col, time_col, anomaly_rate=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    df = df.copy().sort_values([case_col, time_col]).reset_index(drop=True)

    df['Duration'] = (df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0))

    act_stats = (df[df['Duration'] > 0].groupby(act_col)['Duration'].agg(['mean', 'std']).fillna(0))
    act_list = df[act_col].unique()
    n_total = int(len(df) * anomaly_rate)

    df['ActivityLabel'] = 0
    df['TimeLabel'] = 0

    # Activity mutation
    act_idx = np.random.choice(df.index, size=n_total, replace=False)
    df.loc[act_idx, 'ActivityLabel'] = 1

    current_vals = df.loc[act_idx, act_col].values
    new_vals = np.random.choice(act_list, size=n_total)

    # ensure different activity
    mask_same = new_vals == current_vals
    while mask_same.any():
        new_vals[mask_same] = np.random.choice(act_list, size=mask_same.sum())
        mask_same = new_vals == current_vals

    df.loc[act_idx, act_col] = new_vals

    # Timestamp mutation
    first_idx = df.groupby(case_col).head(1).index
    valid_mask = ~df.index.isin(first_idx)
    valid_idx = df.index[valid_mask]

    time_idx = np.random.choice(valid_idx, size=n_total, replace=False)
    df.loc[time_idx, 'TimeLabel'] = 1

    means = df[act_col].map(act_stats['mean']).fillna(0)
    stds = df[act_col].map(act_stats['std']).fillna(0)

    r = np.random.random(size=len(df))
    new_duration = (means + stds) * (1.0 + r)

    shift = np.maximum(0, new_duration - df['Duration'])

    df['time_shift'] = 0.0
    df.loc[time_idx, 'time_shift'] = shift.loc[time_idx]

    df[time_col] = df[time_col] + pd.to_timedelta(df['time_shift'], unit='s')

    return df.drop(columns=['Duration', 'time_shift'])
