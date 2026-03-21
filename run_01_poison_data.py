import pandas as pd
import numpy as np
from config import RAW_DIR, POISONED_DIR, DATASET_SCHEMAS
from core_pipeline import preprocess_data


def create_golden_log_and_poison(file_name, anomaly_rate=0.15):
    input_path = RAW_DIR / file_name
    if not input_path.exists():
        print(f"File {input_path} does not exist. Skipping.")
        return

    schema = DATASET_SCHEMAS.get(file_name)
    if not schema:
        print(f"No schema found for {file_name}. Skipping.")
        return

    case_col = schema['case']
    act_col = schema['act']
    time_col = schema['time']

    df = pd.read_csv(input_path)
    df = preprocess_data(df, case_col=case_col, time_col=time_col)

    df['delta_t'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)

    valid_dt = df[df['delta_t'] > 0]
    act_mean = valid_dt.groupby(act_col)['delta_t'].mean().fillna(0)
    act_std = valid_dt.groupby(act_col)['delta_t'].std().fillna(0)

    df['is_true_anomaly'] = False
    df['anomaly_type'] = 'none'

    n_total = len(df)
    n_target = int(n_total * anomaly_rate)

    valid_time_indices = df[df.groupby(case_col).cumcount() > 0].index
    time_n_anomalies = min(len(valid_time_indices), n_target)
    time_anomaly_indices = np.random.choice(valid_time_indices, time_n_anomalies, replace=False)

    remaining_indices = df.index.difference(time_anomaly_indices)
    act_n_anomalies = min(len(remaining_indices), n_target)
    act_anomaly_indices = np.random.choice(remaining_indices, act_n_anomalies, replace=False)

    df.loc[time_anomaly_indices, 'is_true_anomaly'] = True
    df.loc[time_anomaly_indices, 'anomaly_type'] = 'time'

    df.loc[act_anomaly_indices, 'is_true_anomaly'] = True
    df.loc[act_anomaly_indices, 'anomaly_type'] = 'activity'

    df['time_shift_seconds'] = 0.0
    time_anomalies = df.loc[time_anomaly_indices].copy()

    means = time_anomalies[act_col].map(act_mean).fillna(0)
    stds = time_anomalies[act_col].map(act_std).fillna(0)

    r_values = np.random.uniform(0.0, 1.0, size=len(time_anomalies))
    new_dt = (means + stds) * (1 + r_values)

    shift_diff = np.maximum(0, new_dt - time_anomalies['delta_t'])
    df.loc[time_anomaly_indices, 'time_shift_seconds'] = shift_diff

    df['cumulative_shift'] = df.groupby(case_col)['time_shift_seconds'].cumsum()
    df[time_col] = df[time_col] + pd.to_timedelta(df['cumulative_shift'], unit='s')

    unique_acts = df[act_col].unique()

    def get_different_activity(current_act):
        choices = [a for a in unique_acts if a != current_act]
        return np.random.choice(choices) if choices else current_act

    df.loc[act_anomaly_indices, act_col] = df.loc[act_anomaly_indices, act_col].apply(get_different_activity)

    df = df.drop(columns=['time_shift_seconds', 'cumulative_shift'])

    poisoned_name = file_name.replace('.csv', '_poisoned.csv')
    df.to_csv(POISONED_DIR / poisoned_name, index=False)
    print(f"Saved {poisoned_name}")


if __name__ == "__main__":
    for ds in DATASET_SCHEMAS.keys():
        create_golden_log_and_poison(ds)
