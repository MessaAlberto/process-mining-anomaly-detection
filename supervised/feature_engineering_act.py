import numpy as np
import pandas as pd


EPS = 1e-9
MISSING_TOKEN = "__MISSING__"
UNKNOWN_RANK_PENALTY = 1.0


FEATURE_COLUMN_GROUPS = {
    "numeric": [
        "delta_t",
        "event_idx",
        "trace_pos",

        "act_rework_count",
        "repeat_burst_len",
        "self_loop_flag",
        "act_repeat_last_5",
        "local_repeat_ratio_5",
        "window_activity_stability",

        "trans_in_prob",
        "trans_out_prob",
        "triplet_prob",
        "window3_rarity",
        "swap_local_gain",
        "act_context_prob",

        "act_given_res_prob",
        "res_given_act_prob",
        "act_res_rank_norm",
        "act_given_prev_res_curr_res_prob",
        "act_given_curr_res_next_res_prob",
        "act_given_res_pos_bin_prob",

        "act_repeat_given_res_pos_bin_prob",
        "act_repeat_given_prev_act_prob",
        "act_loop_resource_context_prob",
        "act_trace_ratio",

        "res_transition_out_prob",
        "res_workload_day_norm",
        "z_score_act_res",
    ],
    "categorical": [
        "act",
        "trans_in",
        "trans_out",
        "triplet",
    ],
    "resource_categorical": [
        "res",
        "act_res_pair",
        "res_transition",
        "handover",
    ],
}


def get_feature_columns(case_col="Case ID", act_col="Activity", time_col="Complete Timestamp", res_col=None):
    num_cols = list(FEATURE_COLUMN_GROUPS["numeric"])
    cat_cols = [
        act_col if col == "act" else col
        for col in FEATURE_COLUMN_GROUPS["categorical"]
    ]

    if res_col:
        cat_cols.extend([
            res_col if col == "res" else col
            for col in FEATURE_COLUMN_GROUPS["resource_categorical"]
        ])

    return {"num_cols": num_cols, "cat_cols": cat_cols}


def _normalize_column(df, col):
    df[col] = df[col].where(df[col].notna(), MISSING_TOKEN)
    df[col] = df[col].astype(str).str.strip()
    missing_values = {"", "nan", "NaN", "None", "none", "NULL", "null"}
    df.loc[df[col].isin(missing_values), col] = MISSING_TOKEN
    return df


def _safe_surprise(values):
    return -np.log(np.asarray(values, dtype=float) + EPS)


def _entropy(values):
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log2(counts + EPS)).sum())


def _as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def _conditional_prob_map(df, numerator_cols, denominator_cols):
    numerator = df.groupby(numerator_cols, dropna=False).size()
    denominator = df.groupby(denominator_cols, dropna=False).size()
    result = {}

    for key, value in numerator.items():
        key_tuple = _as_tuple(key)
        den_key = key_tuple[:len(denominator_cols)]
        den_key = den_key[0] if len(den_key) == 1 else den_key
        result[key] = float(value / max(denominator.get(den_key, 0), 1))

    return result


def _event_rate_map(df, condition_cols, event_col):
    return df.groupby(condition_cols, dropna=False)[event_col].mean().to_dict()


def _normalized_rank_map(df, group_col, item_col):
    rank_map = {}
    unknown_rank_map = {}

    for group_value, group in df.groupby(group_col, dropna=False):
        ranked_items = group[item_col].value_counts().index.tolist()
        n_items = max(1, len(ranked_items))
        unknown_rank_map[group_value] = 1.0 + UNKNOWN_RANK_PENALTY / n_items

        for rank, item in enumerate(ranked_items, 1):
            rank_map[(group_value, item)] = rank / n_items

    return rank_map, unknown_rank_map


def _add_local_repeat_features(df, case_col, act_col):
    repeat_last_5 = np.zeros(len(df), dtype=float)
    stability = np.zeros(len(df), dtype=float)

    for _, idx in df.groupby(case_col, dropna=False).groups.items():
        idx = list(idx)
        values = df.loc[idx, act_col].tolist()

        for i, value in enumerate(values):
            window = values[max(0, i - 4):i + 1]
            repeat_last_5[idx[i]] = window[:-1].count(value) if len(window) > 1 else 0
            stability[idx[i]] = len(set(window)) / max(1, len(window))

    df["act_repeat_last_5"] = repeat_last_5
    df["local_repeat_ratio_5"] = repeat_last_5 / 5.0
    df["window_activity_stability"] = stability
    return df


def _add_trace_activity_ratio(df, case_col, act_col):
    trace_len = df.groupby(case_col, dropna=False)[act_col].transform("count")
    df["act_trace_freq"] = df.groupby([case_col, act_col], dropna=False)[act_col].transform("count")
    df["act_trace_ratio"] = (df["act_trace_freq"] / trace_len.replace(0, np.nan)).fillna(0)
    df = df.drop(columns=["act_trace_freq"], errors="ignore")
    return df


def _add_resource_placeholders(df):
    defaults = {
        "act_given_res_prob": 0,
        "res_given_act_prob": 0,
        "act_res_rank_norm": 0,
        "act_given_prev_res_curr_res_prob": 0,
        "act_given_curr_res_next_res_prob": 0,
        "act_given_res_pos_bin_prob": 0,
        "act_repeat_given_res_pos_bin_prob": 0,
        "act_loop_resource_context_prob": 0,
        "res_transition_out_prob": 0,
        "res_workload_day_norm": 0,
        "z_score_act_res": 0,
    }
    for col, value in defaults.items():
        df[col] = value
    return df


def _build_activity_stats(df, case_col, act_col):
    stats = {}

    prev_counts = df.groupby("prev_act", dropna=False).size()
    curr_counts = df.groupby(act_col, dropna=False).size()
    pair_context_counts = df.groupby(["prev_act", "next_act"], dropna=False).size()

    pair_in = df.groupby(["prev_act", act_col], dropna=False).size()
    pair_out = df.groupby([act_col, "next_act"], dropna=False).size()
    triplets = df.groupby(["prev_act", act_col, "next_act"], dropna=False).size()

    stats["trans_in_prob"] = {
        k: float(v / max(prev_counts.get(k[0], 0), 1))
        for k, v in pair_in.items()
    }
    stats["trans_out_prob"] = {
        k: float(v / max(curr_counts.get(k[0], 0), 1))
        for k, v in pair_out.items()
    }
    stats["triplet_prob"] = {
        k: float(v / max(pair_context_counts.get((k[0], k[2]), 0), 1))
        for k, v in triplets.items()
    }

    context_counts = df.groupby(["prev_act", "next_act", act_col], dropna=False).size()
    stats["act_context_prob"] = {
        k: float(v / max(pair_context_counts.get((k[0], k[1]), 0), 1))
        for k, v in context_counts.items()
    }

    tmp = df.copy()
    tmp["act_repeat_event"] = (tmp[act_col] == tmp["prev_act"]).astype(int)
    stats["act_repeat_given_prev_act_prob"] = _event_rate_map(
        tmp, ["prev_act"], "act_repeat_event"
    )

    return stats


def _apply_activity_stats(df, stats, act_col):
    df["trans_in_prob"] = [
        stats.get("trans_in_prob", {}).get((p, a), 0)
        for p, a in zip(df["prev_act"], df[act_col])
    ]
    df["trans_out_prob"] = [
        stats.get("trans_out_prob", {}).get((a, n), 0)
        for a, n in zip(df[act_col], df["next_act"])
    ]
    df["triplet_prob"] = [
        stats.get("triplet_prob", {}).get((p, a, n), 0)
        for p, a, n in zip(df["prev_act"], df[act_col], df["next_act"])
    ]

    df["window3_rarity"] = _safe_surprise(df["triplet_prob"])

    reverse_in = [
        stats.get("trans_in_prob", {}).get((a, p), 0)
        for p, a in zip(df["prev_act"], df[act_col])
    ]
    prev_to_next = [
        stats.get("trans_in_prob", {}).get((p, n), 0)
        for p, n in zip(df["prev_act"], df["next_act"])
    ]
    df["swap_local_gain"] = (
        np.asarray(reverse_in)
        + np.asarray(prev_to_next)
        - df["trans_in_prob"].to_numpy()
        - df["trans_out_prob"].to_numpy()
    )

    df["act_context_prob"] = [
        stats.get("act_context_prob", {}).get((p, n, a), 0)
        for p, n, a in zip(df["prev_act"], df["next_act"], df[act_col])
    ]
    df["act_repeat_given_prev_act_prob"] = [
        stats.get("act_repeat_given_prev_act_prob", {}).get(p, 0)
        for p in df["prev_act"]
    ]

    return df


def _build_resource_context_stats(df, act_col, res_col):
    stats = {}

    act_counts = df.groupby(act_col, dropna=False).size()
    res_counts = df.groupby(res_col, dropna=False).size()
    act_res_counts = df.groupby([act_col, res_col], dropna=False).size()

    stats["act_given_res_prob"] = {
        k: float(v / max(res_counts.get(k[1], 0), 1))
        for k, v in act_res_counts.items()
    }
    stats["res_given_act_prob"] = {
        k: float(v / max(act_counts.get(k[0], 0), 1))
        for k, v in act_res_counts.items()
    }

    rank_map, unknown_rank_map = _normalized_rank_map(df, res_col, act_col)
    stats["act_res_rank_norm"] = rank_map
    stats["res_unknown_norm_rank"] = unknown_rank_map

    stats["act_given_prev_res_curr_res_prob"] = _conditional_prob_map(
        df, ["prev_res", res_col, act_col], ["prev_res", res_col]
    )
    stats["act_given_curr_res_next_res_prob"] = _conditional_prob_map(
        df, [res_col, "next_res", act_col], [res_col, "next_res"]
    )
    stats["act_given_res_pos_bin_prob"] = _conditional_prob_map(
        df, [res_col, "pos_bin", act_col], [res_col, "pos_bin"]
    )

    tmp = df.copy()
    tmp["act_repeat_event"] = (tmp[act_col] == tmp["prev_act"]).astype(int)
    stats["act_repeat_given_res_pos_bin_prob"] = _event_rate_map(
        tmp, [res_col, "pos_bin"], "act_repeat_event"
    )
    stats["act_loop_resource_context_prob"] = _event_rate_map(
        tmp, ["prev_res", res_col, "next_res"], "act_repeat_event"
    )

    curr_res_counts = df.groupby(res_col, dropna=False).size()
    res_pair_out = df.groupby([res_col, "next_res"], dropna=False).size()
    stats["res_transition_out_prob"] = {
        k: float(v / max(curr_res_counts.get(k[0], 0), 1))
        for k, v in res_pair_out.items()
    }

    return stats


def _apply_resource_context_stats(df, stats, act_col, res_col):
    df["act_given_res_prob"] = [
        stats.get("act_given_res_prob", {}).get((a, r), 0)
        for a, r in zip(df[act_col], df[res_col])
    ]
    df["res_given_act_prob"] = [
        stats.get("res_given_act_prob", {}).get((a, r), 0)
        for a, r in zip(df[act_col], df[res_col])
    ]
    df["act_res_rank_norm"] = [
        stats.get("act_res_rank_norm", {}).get(
            (r, a),
            stats.get("res_unknown_norm_rank", {}).get(r, 1.0)
        )
        for r, a in zip(df[res_col], df[act_col])
    ]

    df["act_given_prev_res_curr_res_prob"] = [
        stats.get("act_given_prev_res_curr_res_prob", {}).get((pr, r, a), 0)
        for pr, r, a in zip(df["prev_res"], df[res_col], df[act_col])
    ]
    df["act_given_curr_res_next_res_prob"] = [
        stats.get("act_given_curr_res_next_res_prob", {}).get((r, nr, a), 0)
        for r, nr, a in zip(df[res_col], df["next_res"], df[act_col])
    ]
    df["act_given_res_pos_bin_prob"] = [
        stats.get("act_given_res_pos_bin_prob", {}).get((r, pb, a), 0)
        for r, pb, a in zip(df[res_col], df["pos_bin"], df[act_col])
    ]

    df["act_repeat_given_res_pos_bin_prob"] = [
        stats.get("act_repeat_given_res_pos_bin_prob", {}).get((r, pb), 0)
        for r, pb in zip(df[res_col], df["pos_bin"])
    ]
    df["act_loop_resource_context_prob"] = [
        stats.get("act_loop_resource_context_prob", {}).get((pr, r, nr), 0)
        for pr, r, nr in zip(df["prev_res"], df[res_col], df["next_res"])
    ]
    df["res_transition_out_prob"] = [
        stats.get("res_transition_out_prob", {}).get((r, nr), 0)
        for r, nr in zip(df[res_col], df["next_res"])
    ]

    return df


def _add_time_resource_features(df, stats, act_col, time_col, res_col, has_time):
    if not has_time or res_col is None or res_col not in df.columns:
        df["z_score_act_res"] = 0
        df["res_workload_day_norm"] = 0
        return df

    act_res_time = stats.get("act_res_time", pd.DataFrame())
    df = df.join(act_res_time, on=[act_col, res_col], rsuffix="_act_res")

    if "mean" in df.columns:
        mean = df["mean"].fillna(stats.get("global_duration_mean", 0.0))
    else:
        mean = stats.get("global_duration_mean", 0.0)

    if "std" in df.columns:
        std = df["std"].replace(0, np.nan).fillna(stats.get("global_duration_std", 1.0))
    else:
        std = stats.get("global_duration_std", 1.0)

    df["z_score_act_res"] = (df["delta_t"] - mean) / (std + EPS)
    df = df.drop(columns=["mean", "std"], errors="ignore")

    res_load_stats = stats.get("res_load_stats", pd.DataFrame())
    df = df.join(res_load_stats, on=res_col, rsuffix="_load")

    if "mean" in df.columns:
        load_mean = df["mean"].fillna(stats.get("global_load_mean", 0.0))
    else:
        load_mean = stats.get("global_load_mean", 0.0)

    if "std" in df.columns:
        load_std = df["std"].replace(0, np.nan).fillna(stats.get("global_load_std", 1.0))
    else:
        load_std = stats.get("global_load_std", 1.0)

    df["res_workload_day_norm"] = df["res_workload_day"] / (load_mean + load_std + 1.0)
    df = df.drop(columns=["mean", "std"], errors="ignore")
    return df


def _build_time_resource_stats(df, act_col, time_col, res_col, has_time):
    stats = {}
    if not has_time or res_col is None or res_col not in df.columns:
        stats["act_res_time"] = pd.DataFrame()
        stats["global_duration_mean"] = 0.0
        stats["global_duration_std"] = 1.0
        stats["res_load_stats"] = pd.DataFrame()
        stats["global_load_mean"] = 0.0
        stats["global_load_std"] = 1.0
        return stats

    mask = df["delta_t"] > 0
    stats["act_res_time"] = df[mask].groupby([act_col, res_col])["delta_t"].agg(["mean", "std"]).fillna(0)
    stats["global_duration_mean"] = float(df.loc[mask, "delta_t"].mean()) if mask.any() else 0.0
    global_std = df.loc[mask, "delta_t"].std() if mask.any() else 1.0
    stats["global_duration_std"] = float(global_std) if pd.notna(global_std) and global_std != 0 else 1.0

    stats["res_load_stats"] = df.groupby(res_col)["res_workload_day"].agg(["mean", "std"]).fillna(0)
    stats["global_load_mean"] = float(df["res_workload_day"].mean())
    global_load_std = df["res_workload_day"].std()
    stats["global_load_std"] = float(global_load_std) if pd.notna(global_load_std) and global_load_std != 0 else 1.0
    return stats


def compute_features(
    df,
    case_col="Case ID",
    act_col="Activity",
    time_col="Complete Timestamp",
    res_col=None,
    train_stats=None
):
    df = df.copy()

    if act_col not in df.columns:
        raise ValueError(f"Activity column '{act_col}' not found in DataFrame.")

    df = _normalize_column(df, act_col)
    has_resource = res_col is not None and res_col in df.columns
    if has_resource:
        df = _normalize_column(df, res_col)

    has_time = time_col in df.columns
    if has_time:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, format="mixed")
        df = df.sort_values([case_col, time_col]).reset_index(drop=True)
        df["delta_t"] = df.groupby(case_col, dropna=False)[time_col].diff().dt.total_seconds().fillna(0)

        if has_resource:
            df["date_only"] = df[time_col].dt.date
            df["res_workload_day"] = df.groupby([res_col, "date_only"], dropna=False).cumcount()
            df = df.drop(columns=["date_only"], errors="ignore")
        else:
            df["res_workload_day"] = 0
    else:
        df = df.sort_values(case_col).reset_index(drop=True)
        df["delta_t"] = 0
        df["res_workload_day"] = 0

    grouped = df.groupby(case_col, dropna=False)
    df["event_idx"] = grouped.cumcount() + 1
    df["trace_len"] = grouped[act_col].transform("count")
    df["trace_pos"] = (df["event_idx"] / df["trace_len"].replace(0, np.nan)).fillna(0)

    df["pos_bin"] = pd.cut(
        df["trace_pos"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    df["prev_act"] = grouped[act_col].shift(1).fillna("START")
    df["next_act"] = grouped[act_col].shift(-1).fillna("END")

    df["trans_in"] = df["prev_act"].astype(str) + " -> " + df[act_col].astype(str)
    df["trans_out"] = df[act_col].astype(str) + " -> " + df["next_act"].astype(str)
    df["triplet"] = df["prev_act"].astype(str) + " -> " + df[act_col].astype(str) + " -> " + df["next_act"].astype(str)

    df["act_rework_count"] = df.groupby([case_col, act_col], dropna=False).cumcount()
    df["self_loop_flag"] = (df[act_col] == df["prev_act"]).astype(int)

    prev_same = grouped[act_col].shift(1)
    run_id = (df[act_col] != prev_same).groupby(df[case_col]).cumsum()
    df["repeat_burst_len"] = df.groupby([case_col, run_id], dropna=False)[act_col].transform("size")

    df = _add_local_repeat_features(df, case_col, act_col)
    df = _add_trace_activity_ratio(df, case_col, act_col)

    if has_resource:
        df["prev_res"] = grouped[res_col].shift(1).fillna("START")
        df["next_res"] = grouped[res_col].shift(-1).fillna("END")
        df["handover"] = df["prev_res"].astype(str) + " -> " + df[res_col].astype(str)
        df["res_transition"] = df[res_col].astype(str) + " -> " + df["next_res"].astype(str)
        df["act_res_pair"] = df[act_col].astype(str) + "_" + df[res_col].astype(str)
    else:
        df = _add_resource_placeholders(df)

    if train_stats is None:
        stats = {
            "activity_stats": _build_activity_stats(df, case_col, act_col),
            "has_resource": has_resource,
        }
        if has_resource:
            stats["resource_context_stats"] = _build_resource_context_stats(df, act_col, res_col)
            stats["time_resource_stats"] = _build_time_resource_stats(df, act_col, time_col, res_col, has_time)
    else:
        stats = train_stats

    df = _apply_activity_stats(df, stats.get("activity_stats", {}), act_col)

    if has_resource:
        df = _apply_resource_context_stats(df, stats.get("resource_context_stats", {}), act_col, res_col)
        df = _add_time_resource_features(df, stats.get("time_resource_stats", {}), act_col, time_col, res_col, has_time)
    else:
        df = _add_resource_placeholders(df)

    helper_cols = [
        "trace_len",
        "pos_bin",
        "prev_res",
        "next_res",
        "res_workload_day",
    ]
    df = df.drop(columns=helper_cols, errors="ignore")

    if train_stats is None:
        return df, stats
    return df


def compute_resource_features(df, case_col, act_col, time_col, res_col, train_stats=None):
    if train_stats is None:
        return df, {}
    return df