import zlib

import numpy as np
import pandas as pd

from config import (
    ACT_ANOMALY_RATE,
    DATASET_SCHEMAS,
    INJ_DIR,
    NUM_ITERS,
    RES_ANOMALY_RATE,
    SPLIT_SEED,
    TEST_ANOMALY_RATES,
    TEST_LOGS_DIR,
    TIME_ANOMALY_RATE,
    TRAIN_PCT,
    iter_raw_datasets,
    load_and_standardize_raw_log,
    split_case_ids,
)

TIME_ANOMALIES = ('Delay', 'Batch', 'Rush')
ATTR_ANOMALIES = ('Swap', 'Mutate', 'Repeat')
UNSEEN_TIME_ANOMALIES = (
    'Micro-Delay',
    'Macro-Delay',
    'TS-Round-Min',
    'TS-Round-Hour',
    'TS-Round-Day',
    'Trace-Level-Storm',
)

TRAIN_SCENARIO = 'Train_Balanced'
SEEN_SCENARIO = 'Seen'
UNSEEN_TIME_SCENARIO_PREFIX = 'Unseen_Time'
OVERWRITE_EXISTING = False


def _is_swap_label(label):
    if isinstance(label, tuple):
        return label[-1] == 'Swap'
    return label == 'Swap'


def _redistribute_lost_counts(counts, lost):
    if lost <= 0:
        return counts

    receivers = [label for label in counts if not _is_swap_label(label)]
    if not receivers:
        raise ValueError('At least one non-swap subtype is required.')

    for index in range(lost):
        counts[receivers[index % len(receivers)]] += 1

    return counts


def get_exact_counts(n_total, labels):
    labels = list(labels)
    if n_total <= 0 or not labels:
        return {}

    base_count = n_total // len(labels)
    counts = {label: base_count for label in labels}

    remainder = n_total - base_count * len(labels)
    for label in labels[:remainder]:
        counts[label] += 1

    lost = 0
    for label in labels:
        if _is_swap_label(label) and counts[label] % 2 == 1:
            counts[label] -= 1
            lost += 1

    counts = _redistribute_lost_counts(counts, lost)
    if sum(counts.values()) != n_total:
        raise RuntimeError(f'Count allocation mismatch: {sum(counts.values())}/{n_total}.')

    return counts


def prepare_clean_metadata(df):
    df = df.copy()
    df['TimeLabel'] = 0
    df['ActLabel'] = 0
    df['ResLabel'] = 0
    df['AnomalyScenario'] = 'None'
    df['AnomalyPerspective'] = 'None'
    df['AnomalySubtype'] = 'None'
    return df


def get_first_indices(df, case_col):
    return set(df.groupby(case_col, dropna=False, sort=False).head(1).index.tolist())


def get_time_candidate_indices(df, case_col, time_col):
    grouped = df.groupby(case_col, dropna=False, sort=False)
    positions = grouped.cumcount()
    previous_times = grouped[time_col].shift(1)
    mask = (
        positions.gt(0)
        & df['AnomalyPerspective'].eq('None')
        & df[time_col].notna()
        & previous_times.notna()
    )
    return df.index[mask].to_numpy(dtype=int)


def get_mutate_candidate_indices(df, target_col, allowed_values):
    unique_values = pd.Series(allowed_values).dropna().unique()
    if len(unique_values) < 2:
        return np.array([], dtype=int)

    mask = df['AnomalyPerspective'].eq('None') & df[target_col].notna()
    return df.index[mask].to_numpy(dtype=int)


def get_repeat_candidates(df, case_col, target_col):
    values = df[target_col].copy()
    candidates = []

    free_indices = df.index[
        df['AnomalyPerspective'].eq('None') & values.notna()
    ]
    for idx in free_indices:
        idx = int(idx)
        current_value = values.at[idx]
        sources = []

        for distance in (1, 2):
            source_idx = idx - distance
            if source_idx < 0:
                continue
            if df.at[source_idx, case_col] != df.at[idx, case_col]:
                continue

            source_value = values.at[source_idx]
            if pd.isna(source_value) or source_value == current_value:
                continue
            sources.append(source_idx)

        if sources:
            candidates.append((idx, sources))

    return candidates, values


def get_disjoint_swap_pairs(df, case_col, target_col, rng):
    pairs = []

    for _, group in df.groupby(case_col, dropna=False, sort=False):
        indices = group.index.to_list()
        if len(indices) < 2:
            continue

        edge_positions = list(range(len(indices) - 1))
        if rng.random() < 0.5:
            edge_positions.reverse()

        used = set()
        for pos in edge_positions:
            left_idx = int(indices[pos])
            right_idx = int(indices[pos + 1])

            if left_idx in used or right_idx in used:
                continue
            if df.at[left_idx, 'AnomalyPerspective'] != 'None':
                continue
            if df.at[right_idx, 'AnomalyPerspective'] != 'None':
                continue

            left_value = df.at[left_idx, target_col]
            right_value = df.at[right_idx, target_col]
            if pd.isna(left_value) or pd.isna(right_value):
                continue
            if left_value == right_value:
                continue

            pairs.append((left_idx, right_idx))
            used.update((left_idx, right_idx))

    rng.shuffle(pairs)
    return pairs


def build_injection_reference(df, schema):
    reference_df = df.copy()
    case_col = schema['case']
    act_col = schema['act']
    time_col = schema['time']
    res_col = schema.get('res')

    reference_df[time_col] = pd.to_datetime(
        reference_df[time_col], utc=True, errors='coerce', format='mixed'
    )
    reference_df = reference_df.sort_values([case_col, time_col]).reset_index(drop=True)
    reference_df['Duration'] = (
        reference_df.groupby(case_col, dropna=False)[time_col]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    positive = reference_df[reference_df['Duration'] > 0]
    global_m = positive['Duration'].mean()
    global_s = positive['Duration'].std()
    global_max = positive['Duration'].max()

    act_stats = (
        positive.groupby(act_col, dropna=False)['Duration']
        .agg(['mean', 'std', 'max'])
        .fillna(0)
    )

    return {
        'time_stats': {
            'act_stats': act_stats,
            'global_m': 3600.0 if pd.isna(global_m) else float(global_m),
            'global_s': 600.0 if pd.isna(global_s) or global_s == 0 else float(global_s),
            'global_max': 7200.0 if pd.isna(global_max) or global_max <= 0 else float(global_max),
        },
        'act_values': reference_df[act_col].dropna().unique(),
        'res_values': (
            reference_df[res_col].dropna().unique()
            if res_col and res_col in reference_df.columns
            else np.array([], dtype=object)
        ),
    }


def inject_time_anomalies(
    df,
    schema,
    target_counts,
    random_state,
    scenario_name,
    injection_reference,
):
    rng = np.random.default_rng(random_state)
    df = df.copy()

    case_col = schema['case']
    act_col = schema['act']
    time_col = schema['time']
    stats = injection_reference['time_stats']
    act_stats = stats['act_stats']
    global_m = stats['global_m']
    global_s = stats['global_s']

    for anomaly_type, target_count in target_counts.items():
        candidates = get_time_candidate_indices(df, case_col, time_col)
        rng.shuffle(candidates)

        if target_count > len(candidates):
            raise RuntimeError(
                f'Time {anomaly_type}: requested {target_count} events, '
                f'but only {len(candidates)} non-first free events are available.'
            )

        injected = 0
        for idx in candidates:
            idx = int(idx)
            activity = df.at[idx, act_col]
            mean_duration = (
                float(act_stats.at[activity, 'mean'])
                if activity in act_stats.index
                else global_m
            )
            std_duration = (
                float(act_stats.at[activity, 'std'])
                if activity in act_stats.index
                else global_s
            )
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                mean_duration = global_m
            if not np.isfinite(std_duration) or std_duration <= 0:
                std_duration = global_s

            prev_time = df.at[idx - 1, time_col]
            current_time = df.at[idx, time_col]

            if anomaly_type == 'Delay':
                duration = (mean_duration + std_duration) * (1 + rng.random())
            elif anomaly_type == 'Batch':
                duration = float(rng.integers(1, 5))
            elif anomaly_type == 'Rush':
                duration = mean_duration - std_duration * (1 + rng.random())
                duration = max(15.0, duration, mean_duration * 0.10)
            else:
                raise ValueError(f'Unsupported training time anomaly: {anomaly_type}')

            new_timestamp = prev_time + pd.to_timedelta(duration, unit='s')
            if pd.isna(new_timestamp) or new_timestamp == current_time:
                continue

            df.at[idx, time_col] = new_timestamp
            df.at[idx, 'TimeLabel'] = 1
            df.at[idx, 'AnomalyScenario'] = scenario_name
            df.at[idx, 'AnomalyPerspective'] = 'Time'
            df.at[idx, 'AnomalySubtype'] = anomaly_type
            injected += 1

            if injected == target_count:
                break

        if injected != target_count:
            raise RuntimeError(
                f'Time {anomaly_type}: injected {injected}/{target_count} events.'
            )

    return df


def inject_attribute_anomalies(
    df,
    schema,
    perspective,
    target_counts,
    random_state,
    scenario_name,
    allowed_values,
):
    rng = np.random.default_rng(random_state)
    df = df.copy()

    case_col = schema['case']
    if perspective == 'act':
        target_col = schema['act']
        label_col = 'ActLabel'
        perspective_label = 'Act'
    elif perspective == 'res':
        target_col = schema.get('res')
        label_col = 'ResLabel'
        perspective_label = 'Res'
    else:
        raise ValueError(f'Unsupported attribute perspective: {perspective}')

    if target_col is None or target_col not in df.columns:
        raise ValueError(f'Missing column for perspective={perspective}.')

    unique_values = pd.Series(allowed_values).dropna().unique()
    injection_order = ('Repeat', 'Swap', 'Mutate')

    for anomaly_type in injection_order:
        target_count = int(target_counts.get(anomaly_type, 0))
        if target_count <= 0:
            continue

        if anomaly_type == 'Repeat':
            candidates, source_values = get_repeat_candidates(
                df, case_col, target_col
            )
            rng.shuffle(candidates)

            if target_count > len(candidates):
                raise RuntimeError(
                    f'{perspective.capitalize()} Repeat: requested {target_count} '
                    f'events, but only {len(candidates)} events have usable '
                    f'preceding values.'
                )

            for idx, sources in candidates[:target_count]:
                source_idx = int(rng.choice(sources))
                df.at[idx, target_col] = source_values.at[source_idx]
                df.at[idx, label_col] = 1
                df.at[idx, 'AnomalyScenario'] = scenario_name
                df.at[idx, 'AnomalyPerspective'] = perspective_label
                df.at[idx, 'AnomalySubtype'] = anomaly_type

        elif anomaly_type == 'Swap':
            if target_count % 2 != 0:
                raise ValueError('Swap must receive an even event count.')

            required_pairs = target_count // 2
            pairs = get_disjoint_swap_pairs(df, case_col, target_col, rng)
            if required_pairs > len(pairs):
                raise RuntimeError(
                    f'{perspective.capitalize()} Swap: requested {required_pairs} pairs '
                    f'({target_count} events), but only {len(pairs)} disjoint pairs '
                    f'are available.'
                )

            for left_idx, right_idx in pairs[:required_pairs]:
                left_value = df.at[left_idx, target_col]
                df.at[left_idx, target_col] = df.at[right_idx, target_col]
                df.at[right_idx, target_col] = left_value

                for event_idx in (left_idx, right_idx):
                    df.at[event_idx, label_col] = 1
                    df.at[event_idx, 'AnomalyScenario'] = scenario_name
                    df.at[event_idx, 'AnomalyPerspective'] = perspective_label
                    df.at[event_idx, 'AnomalySubtype'] = anomaly_type

        elif anomaly_type == 'Mutate':
            candidates = get_mutate_candidate_indices(df, target_col, unique_values)
            rng.shuffle(candidates)

            if target_count > len(candidates):
                raise RuntimeError(
                    f'{perspective.capitalize()} Mutate: requested {target_count} '
                    f'events, but only {len(candidates)} free events are available.'
                )

            injected = 0
            for idx in candidates:
                idx = int(idx)
                current_value = df.at[idx, target_col]
                choices = unique_values[unique_values != current_value]
                if len(choices) == 0:
                    continue

                df.at[idx, target_col] = rng.choice(choices)
                df.at[idx, label_col] = 1
                df.at[idx, 'AnomalyScenario'] = scenario_name
                df.at[idx, 'AnomalyPerspective'] = perspective_label
                df.at[idx, 'AnomalySubtype'] = anomaly_type
                injected += 1

                if injected == target_count:
                    break

            if injected != target_count:
                raise RuntimeError(
                    f'{perspective.capitalize()} Mutate: injected '
                    f'{injected}/{target_count} events.'
                )

        else:
            raise ValueError(f'Unsupported attribute anomaly: {anomaly_type}')

    unexpected = set(target_counts) - set(injection_order)
    if unexpected:
        raise ValueError(f'Unsupported attribute anomalies: {sorted(unexpected)}')

    return df


def validate_disjoint_injection(df, expected_counts, context, case_col):
    label_cols = ['TimeLabel', 'ActLabel', 'ResLabel']
    labels = df[label_cols].fillna(0).astype(int)

    actual_counts = {
        'time': int(labels['TimeLabel'].sum()),
        'act': int(labels['ActLabel'].sum()),
        'res': int(labels['ResLabel'].sum()),
    }
    anomaly_mask = labels.sum(axis=1) > 0
    overlap_count = int((labels.sum(axis=1) > 1).sum())
    union_count = int(anomaly_mask.sum())
    expected_union = sum(expected_counts.values())

    if actual_counts != expected_counts:
        raise RuntimeError(
            f'{context}: perspective counts {actual_counts}, expected {expected_counts}.'
        )
    if overlap_count != 0:
        raise RuntimeError(f'{context}: found {overlap_count} overlapping anomalous events.')
    if union_count != expected_union:
        raise RuntimeError(
            f'{context}: anomalous union {union_count}, expected {expected_union}.'
        )

    first_indices = get_first_indices(df, case_col)
    first_mask = df.index.isin(first_indices)
    subtype = df['AnomalySubtype'].astype(str)

    invalid_first_time = int((first_mask & labels['TimeLabel'].eq(1)).sum())
    invalid_first_repeat = int((first_mask & subtype.eq('Repeat')).sum())
    if invalid_first_time or invalid_first_repeat:
        raise RuntimeError(
            f'{context}: invalid first-event injections: '
            f'time={invalid_first_time}, repeat={invalid_first_repeat}.'
        )

    perspective = df['AnomalyPerspective'].astype(str)
    scenario = df['AnomalyScenario'].astype(str)
    normal_metadata = (perspective == 'None') & (subtype == 'None') & (scenario == 'None')

    invalid_normal = int(((~anomaly_mask) & (~normal_metadata)).sum())
    invalid_anomaly = int((anomaly_mask & normal_metadata).sum())
    if invalid_normal or invalid_anomaly:
        raise RuntimeError(
            f'{context}: metadata mismatch for {invalid_normal} normal and '
            f'{invalid_anomaly} anomalous events.'
        )

    expected_perspective = pd.Series('None', index=df.index, dtype=object)
    expected_perspective.loc[labels['TimeLabel'].eq(1)] = 'Time'
    expected_perspective.loc[labels['ActLabel'].eq(1)] = 'Act'
    expected_perspective.loc[labels['ResLabel'].eq(1)] = 'Res'
    perspective_mismatch = int((perspective != expected_perspective).sum())
    if perspective_mismatch:
        raise RuntimeError(
            f'{context}: found {perspective_mismatch} perspective metadata mismatches.'
        )

    for perspective_name in ('Act', 'Res'):
        swap_indices = df.index[
            perspective.eq(perspective_name) & subtype.eq('Swap')
        ].tolist()
        if len(swap_indices) % 2 != 0:
            raise RuntimeError(
                f'{context}: {perspective_name} Swap has an odd event count.'
            )

        swap_set = set(swap_indices)
        for idx in swap_indices:
            has_partner = False
            for partner_idx in (idx - 1, idx + 1):
                if partner_idx not in swap_set:
                    continue
                if partner_idx < 0 or partner_idx >= len(df):
                    continue
                if df.at[idx, case_col] == df.at[partner_idx, case_col]:
                    has_partner = True
                    break
            if not has_partner:
                raise RuntimeError(
                    f'{context}: swap event {idx} has no adjacent partner.'
                )

    return actual_counts, union_count


def inject_balanced_training_anomalies(df, schema, random_state):
    df = prepare_clean_metadata(df)
    df[schema['time']] = pd.to_datetime(
        df[schema['time']], utc=True, errors='coerce', format='mixed'
    )
    df = df.sort_values([schema['case'], schema['time']]).reset_index(drop=True)

    expected_counts = {
        'time': int(round(len(df) * TIME_ANOMALY_RATE)),
        'act': int(round(len(df) * ACT_ANOMALY_RATE)),
        'res': int(round(len(df) * RES_ANOMALY_RATE)),
    }

    reference = build_injection_reference(df, schema)

    # Constrained attribute anomalies are injected before temporal anomalies.
    df = inject_attribute_anomalies(
        df,
        schema,
        'res',
        get_exact_counts(expected_counts['res'], ATTR_ANOMALIES),
        random_state + 303,
        TRAIN_SCENARIO,
        reference['res_values'],
    )
    df = inject_attribute_anomalies(
        df,
        schema,
        'act',
        get_exact_counts(expected_counts['act'], ATTR_ANOMALIES),
        random_state + 202,
        TRAIN_SCENARIO,
        reference['act_values'],
    )
    df = inject_time_anomalies(
        df,
        schema,
        get_exact_counts(expected_counts['time'], TIME_ANOMALIES),
        random_state + 101,
        TRAIN_SCENARIO,
        reference,
    )

    validate_disjoint_injection(df, expected_counts, TRAIN_SCENARIO, schema['case'])
    return df


def inject_seen_scenario(test_df, schema, total_rate, reference, random_state):
    df = prepare_clean_metadata(test_df)
    df = df.sort_values([schema['case'], schema['time']]).reset_index(drop=True)

    total_anomalies = int(round(len(df) * total_rate))
    perspective_counts = get_exact_counts(total_anomalies, ('time', 'act', 'res'))

    df = inject_attribute_anomalies(
        df,
        schema,
        'res',
        get_exact_counts(perspective_counts['res'], ATTR_ANOMALIES),
        random_state + 303,
        SEEN_SCENARIO,
        reference['res_values'],
    )
    df = inject_attribute_anomalies(
        df,
        schema,
        'act',
        get_exact_counts(perspective_counts['act'], ATTR_ANOMALIES),
        random_state + 202,
        SEEN_SCENARIO,
        reference['act_values'],
    )
    df = inject_time_anomalies(
        df,
        schema,
        get_exact_counts(perspective_counts['time'], TIME_ANOMALIES),
        random_state + 101,
        SEEN_SCENARIO,
        reference,
    )

    expected = {
        'time': perspective_counts['time'],
        'act': perspective_counts['act'],
        'res': perspective_counts['res'],
    }
    validate_disjoint_injection(
        df,
        expected,
        f'{SEEN_SCENARIO}_{total_rate:.2%}',
        schema['case'],
    )
    return df


def unseen_time_scenario_name(anomaly_type):
    return f'{UNSEEN_TIME_SCENARIO_PREFIX}_{anomaly_type}'


def inject_unseen_time_scenario(
    test_df,
    schema,
    total_rate,
    reference,
    random_state,
    anomaly_type,
):
    if anomaly_type not in UNSEEN_TIME_ANOMALIES:
        raise ValueError(f'Unsupported unseen time anomaly: {anomaly_type}')

    rng = np.random.default_rng(random_state)
    df = prepare_clean_metadata(test_df)
    df = df.sort_values([schema['case'], schema['time']]).reset_index(drop=True)

    target_total = int(round(len(df) * total_rate))
    scenario_name = unseen_time_scenario_name(anomaly_type)

    act_col = schema['act']
    time_col = schema['time']
    case_col = schema['case']
    stats = reference['time_stats']
    act_stats = stats['act_stats']
    global_m = stats['global_m']
    global_s = stats['global_s']
    global_max = stats['global_max']

    candidates = get_time_candidate_indices(df, case_col, time_col)
    if target_total > len(candidates):
        raise RuntimeError(
            f'{anomaly_type}: requested {target_total} events, but only '
            f'{len(candidates)} non-first events are available.'
        )

    def activity_stat(activity, column, fallback):
        if activity in act_stats.index:
            value = float(act_stats.at[activity, column])
            if np.isfinite(value) and value > 0:
                return value
        return fallback

    injected = 0

    if anomaly_type == 'Trace-Level-Storm':
        candidate_set = set(int(idx) for idx in candidates)
        cases = np.array(df[case_col].dropna().unique(), dtype=object)
        rng.shuffle(cases)

        for case_id in cases:
            if injected >= target_total:
                break

            case_candidates = np.array(
                [
                    int(idx)
                    for idx in df.index[df[case_col] == case_id]
                    if int(idx) in candidate_set
                ],
                dtype=int,
            )
            if len(case_candidates) == 0:
                continue

            storm_size = min(
                max(1, int(len(case_candidates) * 0.80)),
                len(case_candidates),
                target_total - injected,
            )
            selected = rng.choice(case_candidates, size=storm_size, replace=False)

            for idx in selected:
                idx = int(idx)
                activity = df.at[idx, act_col]
                mean_duration = activity_stat(activity, 'mean', global_m)
                std_duration = activity_stat(activity, 'std', global_s)
                prev_time = df.at[idx - 1, time_col]
                current_time = df.at[idx, time_col]

                duration = (mean_duration + std_duration) * (1 + rng.random())
                new_timestamp = prev_time + pd.to_timedelta(duration, unit='s')
                if pd.isna(new_timestamp):
                    continue
                if new_timestamp == current_time:
                    new_timestamp += pd.to_timedelta(1, unit='s')

                df.at[idx, time_col] = new_timestamp
                df.at[idx, 'TimeLabel'] = 1
                df.at[idx, 'AnomalyScenario'] = scenario_name
                df.at[idx, 'AnomalyPerspective'] = 'Time'
                df.at[idx, 'AnomalySubtype'] = anomaly_type
                injected += 1

                if injected >= target_total:
                    break

    else:
        rng.shuffle(candidates)

        for idx in candidates:
            if injected >= target_total:
                break

            idx = int(idx)
            activity = df.at[idx, act_col]
            activity_max = activity_stat(activity, 'max', global_max)

            prev_time = df.at[idx - 1, time_col]
            current_time = df.at[idx, time_col]
            current_duration = (current_time - prev_time).total_seconds()

            if anomaly_type == 'Micro-Delay':
                duration = max(1.0, activity_max * 0.20)
                if np.isfinite(current_duration) and abs(duration - current_duration) < 1e-9:
                    duration += max(1.0, activity_max * 0.05)
                new_timestamp = prev_time + pd.to_timedelta(duration, unit='s')

            elif anomaly_type == 'Macro-Delay':
                duration = max(1.0, activity_max)
                if np.isfinite(current_duration) and abs(duration - current_duration) < 1e-9:
                    duration += max(1.0, activity_max * 0.10)
                new_timestamp = prev_time + pd.to_timedelta(duration, unit='s')

            elif anomaly_type == 'TS-Round-Min':
                new_timestamp = current_time.floor('min')

            elif anomaly_type == 'TS-Round-Hour':
                new_timestamp = current_time.floor('h')

            elif anomaly_type == 'TS-Round-Day':
                new_timestamp = current_time.floor('d')

            else:
                raise ValueError(f'Unsupported unseen time anomaly: {anomaly_type}')

            if pd.isna(new_timestamp) or new_timestamp == current_time:
                continue

            df.at[idx, time_col] = new_timestamp
            df.at[idx, 'TimeLabel'] = 1
            df.at[idx, 'AnomalyScenario'] = scenario_name
            df.at[idx, 'AnomalyPerspective'] = 'Time'
            df.at[idx, 'AnomalySubtype'] = anomaly_type
            injected += 1

    if injected != target_total:
        raise RuntimeError(
            f'{anomaly_type}: injected {injected}/{target_total} unseen events. '
            'The requested rate cannot be reached with the available valid events.'
        )

    validate_disjoint_injection(
        df,
        {'time': target_total, 'act': 0, 'res': 0},
        f'{scenario_name}_{total_rate:.2%}',
        case_col,
    )
    return df

def stable_seed(*parts):
    token = '|'.join(str(part) for part in parts).encode('utf-8')
    return int((SPLIT_SEED + zlib.crc32(token)) % (2**32 - 1))


def rate_dir_name(rate):
    return f'{rate:.2f}'.replace('.', '_')


def split_clean_partitions(raw_df, schema):
    train_cases, test_cases = split_case_ids(
        raw_df,
        schema['case'],
        train_size=TRAIN_PCT,
        random_state=SPLIT_SEED,
    )

    train_df = raw_df[raw_df[schema['case']].isin(train_cases)].copy()
    test_df = raw_df[raw_df[schema['case']].isin(test_cases)].copy()

    train_df = train_df.sort_values([schema['case'], schema['time']]).reset_index(drop=True)
    test_df = test_df.sort_values([schema['case'], schema['time']]).reset_index(drop=True)

    train_case_set = set(train_df[schema['case']].astype(str).unique())
    test_case_set = set(test_df[schema['case']].astype(str).unique())
    if train_case_set & test_case_set:
        raise RuntimeError('Train/test case leakage detected.')

    return train_df, test_df


def print_summary(df, dataset_name, scenario_name, detail):
    labels = df[['TimeLabel', 'ActLabel', 'ResLabel']].astype(int)
    total = int((labels.sum(axis=1) > 0).sum())
    print(
        f'[{dataset_name} | {scenario_name} | {detail}] '
        f'Total={total}/{len(df)} ({total / max(1, len(df)):.4f}) | '
        f'Time={int(labels.TimeLabel.sum())}, '
        f'Act={int(labels.ActLabel.sum())}, '
        f'Res={int(labels.ResLabel.sum())}'
    )


def save_training_run(df, dataset_name, run_idx):
    output_path = INJ_DIR / f'{dataset_name.replace(".csv", "")}_run{run_idx}.csv'
    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f'[SKIP] Existing file: {output_path}')
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return True


def save_test_scenario(df, dataset_name, scenario_name, rate):
    output_dir = TEST_LOGS_DIR / scenario_name / rate_dir_name(rate)
    output_path = output_dir / f'{dataset_name.replace(".csv", "")}_test.csv'

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f'[SKIP] Existing file: {output_path}')
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return True


def generate_dataset_outputs(file_name, raw_df, schema):
    if not schema.get('res') or schema['res'] not in raw_df.columns:
        print(f'[SKIP] {file_name}: unified train/test generation requires resources.')
        return 0

    clean_train_df, clean_test_df = split_clean_partitions(raw_df, schema)
    reference = build_injection_reference(clean_train_df, schema)

    print(
        f'[{file_name}] Clean split | '
        f'train={clean_train_df[schema["case"]].nunique()} cases, '
        f'test={clean_test_df[schema["case"]].nunique()} cases | '
        f'train events={len(clean_train_df)}, test events={len(clean_test_df)}'
    )

    saved_count = 0
    for run_idx in range(1, NUM_ITERS + 1):
        train_run = inject_balanced_training_anomalies(
            clean_train_df,
            schema,
            stable_seed(file_name, 'train', run_idx),
        )
        print_summary(train_run, file_name, TRAIN_SCENARIO, f'run={run_idx}')
        saved_count += int(save_training_run(train_run, file_name, run_idx))

    for rate in TEST_ANOMALY_RATES:
        seen_df = inject_seen_scenario(
            clean_test_df,
            schema,
            rate,
            reference,
            stable_seed(file_name, SEEN_SCENARIO, rate),
        )
        print_summary(seen_df, file_name, SEEN_SCENARIO, f'rate={rate:.2%}')
        saved_count += int(save_test_scenario(seen_df, file_name, SEEN_SCENARIO, rate))

        for anomaly_type in UNSEEN_TIME_ANOMALIES:
            scenario_name = unseen_time_scenario_name(anomaly_type)
            unseen_df = inject_unseen_time_scenario(
                clean_test_df,
                schema,
                rate,
                reference,
                stable_seed(file_name, scenario_name, rate),
                anomaly_type,
            )
            print_summary(unseen_df, file_name, scenario_name, f'rate={rate:.2%}')
            saved_count += int(
                save_test_scenario(unseen_df, file_name, scenario_name, rate)
            )

    return saved_count


def generate_train_and_test_datasets():
    processed_datasets = 0
    saved_files = 0

    for file_name, raw_path in iter_raw_datasets():
        schema = DATASET_SCHEMAS.get(file_name)
        if not schema:
            continue

        raw_df = load_and_standardize_raw_log(raw_path, file_name)
        generated = generate_dataset_outputs(file_name, raw_df, schema)
        if generated > 0 or not OVERWRITE_EXISTING:
            processed_datasets += 1
        saved_files += generated

    if processed_datasets == 0 and saved_files == 0:
        raise RuntimeError('run_01 produced no training or test logs.')

    print(
        f'Generation completed: datasets={processed_datasets}, '
        f'files written={saved_files}.'
    )


if __name__ == '__main__':
    generate_train_and_test_datasets()
