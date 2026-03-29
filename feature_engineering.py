import numpy as np

def compute_features(df, case_col='Case ID', act_col='Activity', time_col='Complete Timestamp'):
    df = df.copy()

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

    # 4. Event density
    df['event_density'] = df['event_idx'] / (df['cum_t'] + 1.0)

    return df