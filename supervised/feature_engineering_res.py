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

        "res_given_act_prob",
        "act_given_res_prob",
        "res_act_rank_norm",
        "res_given_prev_act_curr_act_prob",
        "res_given_curr_act_next_act_prob",
        "res_given_act_pos_bin_prob",

        "res_transition_in_prob",
        "res_transition_out_prob",
        "res_transition_out_surprise",
        "res_triplet_prob",
        "res_window3_rarity",
        "res_swap_local_gain",

        "self_handover_flag",
        "res_repeat_burst_len",
        "res_repeat_last_5",
        "local_res_repeat_ratio_5",
        "window_res_stability",
        "res_repeat_given_act_pos_bin_prob",
        "res_repeat_given_prev_res_prob",
        "res_loop_context_prob",
        "window_3_res_entropy",
        "res_trace_ratio",

        "res_idle_time",
        "z_score_act_res",
        "res_workload_day_norm",
    ],
    "categorical": [
        "act",
        "handover",
        "res_transition",
        "res_act_pair",
        "act_transition",
    ],
    "resource_categorical": []
}


def get_feature_columns(case_col="Case ID", act_col="Activity", time_col="Complete Timestamp", res_col="Resource"):
    num_cols = list(FEATURE_COLUMN_GROUPS["numeric"])
    cat_cols = [act_col if col == "act" else col for col in FEATURE_COLUMN_GROUPS["categorical"]]
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


def _add_local_repeat_features(df, case_col, res_col):
    repeat_last_5 = np.zeros(len(df), dtype=float)
    stability = np.zeros(len(df), dtype=float)

    for _, idx in df.groupby(case_col).groups.items():
        idx = list(idx)
        values = df.loc[idx, res_col].tolist()

        for i, value in enumerate(values):
            window = values[max(0, i - 4):i + 1]
            repeat_last_5[idx[i]] = window[:-1].count(value) if len(window) > 1 else 0
            stability[idx[i]] = len(set(window)) / max(1, len(window))

    df["res_repeat_last_5"] = repeat_last_5
    df["local_res_repeat_ratio_5"] = repeat_last_5 / 5.0
    df["window_res_stability"] = stability
    return df


def _add_trace_resource_ratio(df, case_col, res_col):
    trace_len = df.groupby(case_col)[res_col].transform("count")
    df["res_trace_freq"] = df.groupby([case_col, res_col])[res_col].transform("count")
    df["res_trace_ratio"] = (df["res_trace_freq"] / trace_len.replace(0, np.nan)).fillna(0)
    df = df.drop(columns=["res_trace_freq"], errors="ignore")
    return df


def _build_train_stats(df, case_col, act_col, time_col, res_col):
    stats = {}

    act_counts = df.groupby(act_col, dropna=False).size()
    res_counts = df.groupby(res_col, dropna=False).size()
    act_res_counts = df.groupby([act_col, res_col], dropna=False).size()

    stats["res_given_act_prob"] = {
        k: float(v / max(act_counts.get(k[0], 0), 1))
        for k, v in act_res_counts.items()
    }
    stats["act_given_res_prob"] = {
        k: float(v / max(res_counts.get(k[1], 0), 1))
        for k, v in act_res_counts.items()
    }

    rank_map, unknown_rank_map = _normalized_rank_map(df, act_col, res_col)
    stats["res_act_rank_norm"] = rank_map
    stats["act_unknown_norm_rank"] = unknown_rank_map

    stats["res_given_prev_act_curr_act_prob"] = _conditional_prob_map(
        df, ["prev_act", act_col, res_col], ["prev_act", act_col]
    )
    stats["res_given_curr_act_next_act_prob"] = _conditional_prob_map(
        df, [act_col, "next_act", res_col], [act_col, "next_act"]
    )
    stats["res_given_act_pos_bin_prob"] = _conditional_prob_map(
        df, [act_col, "pos_bin", res_col], [act_col, "pos_bin"]
    )

    prev_counts = df.groupby("prev_res", dropna=False).size()
    curr_counts = df.groupby(res_col, dropna=False).size()
    pair_context_counts = df.groupby(["prev_res", "next_res"], dropna=False).size()

    pair_in = df.groupby(["prev_res", res_col], dropna=False).size()
    pair_out = df.groupby([res_col, "next_res"], dropna=False).size()
    triplets = df.groupby(["prev_res", res_col, "next_res"], dropna=False).size()

    stats["res_transition_in_prob"] = {
        k: float(v / max(prev_counts.get(k[0], 0), 1))
        for k, v in pair_in.items()
    }
    stats["res_transition_out_prob"] = {
        k: float(v / max(curr_counts.get(k[0], 0), 1))
        for k, v in pair_out.items()
    }
    stats["res_triplet_prob"] = {
        k: float(v / max(pair_context_counts.get((k[0], k[2]), 0), 1))
        for k, v in triplets.items()
    }

    tmp = df.copy()
    tmp["res_repeat_event"] = (tmp[res_col] == tmp["prev_res"]).astype(int)
    stats["res_repeat_given_act_pos_bin_prob"] = _event_rate_map(
        tmp, [act_col, "pos_bin"], "res_repeat_event"
    )
    stats["res_repeat_given_prev_res_prob"] = _event_rate_map(
        tmp, ["prev_res"], "res_repeat_event"
    )
    stats["res_loop_context_prob"] = _event_rate_map(
        tmp, ["prev_act", act_col, "next_act"], "res_repeat_event"
    )

    stats["has_time"] = time_col in df.columns
    if stats["has_time"]:
        mask = df["delta_t"] > 0
        stats["act_res_time"] = df[mask].groupby([act_col, res_col])["delta_t"].agg(["mean", "std"]).fillna(0)
        stats["global_duration_mean"] = float(df.loc[mask, "delta_t"].mean()) if mask.any() else 0.0
        global_std = df.loc[mask, "delta_t"].std() if mask.any() else 1.0
        stats["global_duration_std"] = float(global_std) if pd.notna(global_std) and global_std != 0 else 1.0

        stats["res_load_stats"] = df.groupby(res_col)["res_workload_day"].agg(["mean", "std"]).fillna(0)
        stats["global_load_mean"] = float(df["res_workload_day"].mean())
        global_load_std = df["res_workload_day"].std()
        stats["global_load_std"] = float(global_load_std) if pd.notna(global_load_std) and global_load_std != 0 else 1.0
    else:
        stats["act_res_time"] = pd.DataFrame()
        stats["global_duration_mean"] = 0.0
        stats["global_duration_std"] = 1.0
        stats["res_load_stats"] = pd.DataFrame()
        stats["global_load_mean"] = 0.0
        stats["global_load_std"] = 1.0

    return stats


def _apply_train_stats(df, stats, act_col, res_col):
    df["res_given_act_prob"] = [
        stats.get("res_given_act_prob", {}).get((a, r), 0)
        for a, r in zip(df[act_col], df[res_col])
    ]
    df["act_given_res_prob"] = [
        stats.get("act_given_res_prob", {}).get((a, r), 0)
        for a, r in zip(df[act_col], df[res_col])
    ]
    df["res_act_rank_norm"] = [
        stats.get("res_act_rank_norm", {}).get(
            (a, r),
            stats.get("act_unknown_norm_rank", {}).get(a, 1.0)
        )
        for a, r in zip(df[act_col], df[res_col])
    ]

    df["res_given_prev_act_curr_act_prob"] = [
        stats.get("res_given_prev_act_curr_act_prob", {}).get((pa, a, r), 0)
        for pa, a, r in zip(df["prev_act"], df[act_col], df[res_col])
    ]
    df["res_given_curr_act_next_act_prob"] = [
        stats.get("res_given_curr_act_next_act_prob", {}).get((a, na, r), 0)
        for a, na, r in zip(df[act_col], df["next_act"], df[res_col])
    ]
    df["res_given_act_pos_bin_prob"] = [
        stats.get("res_given_act_pos_bin_prob", {}).get((a, pb, r), 0)
        for a, pb, r in zip(df[act_col], df["pos_bin"], df[res_col])
    ]

    df["res_transition_in_prob"] = [
        stats.get("res_transition_in_prob", {}).get((p, r), 0)
        for p, r in zip(df["prev_res"], df[res_col])
    ]
    df["res_transition_out_prob"] = [
        stats.get("res_transition_out_prob", {}).get((r, n), 0)
        for r, n in zip(df[res_col], df["next_res"])
    ]
    df["res_triplet_prob"] = [
        stats.get("res_triplet_prob", {}).get((p, r, n), 0)
        for p, r, n in zip(df["prev_res"], df[res_col], df["next_res"])
    ]

    df["res_transition_out_surprise"] = _safe_surprise(df["res_transition_out_prob"])
    df["res_window3_rarity"] = _safe_surprise(df["res_triplet_prob"])

    reverse_in = [
        stats.get("res_transition_in_prob", {}).get((r, p), 0)
        for p, r in zip(df["prev_res"], df[res_col])
    ]
    prev_to_next = [
        stats.get("res_transition_in_prob", {}).get((p, n), 0)
        for p, n in zip(df["prev_res"], df["next_res"])
    ]
    df["res_swap_local_gain"] = (
        np.asarray(reverse_in)
        + np.asarray(prev_to_next)
        - df["res_transition_in_prob"].to_numpy()
        - df["res_transition_out_prob"].to_numpy()
    )

    df["res_repeat_given_act_pos_bin_prob"] = [
        stats.get("res_repeat_given_act_pos_bin_prob", {}).get((a, pb), 0)
        for a, pb in zip(df[act_col], df["pos_bin"])
    ]
    df["res_repeat_given_prev_res_prob"] = [
        stats.get("res_repeat_given_prev_res_prob", {}).get(p, 0)
        for p in df["prev_res"]
    ]
    df["res_loop_context_prob"] = [
        stats.get("res_loop_context_prob", {}).get((pa, a, na), 0)
        for pa, a, na in zip(df["prev_act"], df[act_col], df["next_act"])
    ]

    return df


def _add_time_features(df, stats, act_col, time_col, res_col, has_time):
    if has_time:
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
    else:
        df["z_score_act_res"] = 0

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


def compute_features(
    df,
    case_col="Case ID",
    act_col="Activity",
    time_col="Complete Timestamp",
    res_col=None,
    train_stats=None
):
    df = df.copy()

    if res_col is None:
        raise ValueError("res_col must be provided for resource feature computation.")

    if res_col not in df.columns:
        raise ValueError(f"Resource column '{res_col}' not found in DataFrame.")

    df = _normalize_column(df, act_col)
    df = _normalize_column(df, res_col)

    has_time = time_col in df.columns
    if has_time:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, format="mixed")
        df = df.sort_values(time_col).reset_index(drop=True)
        df["res_idle_time"] = df.groupby(res_col)[time_col].diff().dt.total_seconds().fillna(0)

        df = df.sort_values([case_col, time_col]).reset_index(drop=True)
        df["delta_t"] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)
        df["date_only"] = df[time_col].dt.date
        df["res_workload_day"] = df.groupby([res_col, "date_only"]).cumcount()
        df = df.drop(columns=["date_only"], errors="ignore")
    else:
        df = df.sort_values(case_col).reset_index(drop=True)
        df["delta_t"] = 0
        df["res_idle_time"] = 0
        df["res_workload_day"] = 0

    grouped = df.groupby(case_col, dropna=False)
    df["event_idx"] = grouped.cumcount() + 1
    df["trace_len"] = grouped[res_col].transform("count")
    df["trace_pos"] = (df["event_idx"] / df["trace_len"].replace(0, np.nan)).fillna(0)

    df["pos_bin"] = pd.cut(
        df["trace_pos"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    df["prev_res"] = grouped[res_col].shift(1).fillna("START")
    df["next_res"] = grouped[res_col].shift(-1).fillna("END")
    df["prev_act"] = grouped[act_col].shift(1).fillna("START")
    df["next_act"] = grouped[act_col].shift(-1).fillna("END")

    df["handover"] = df["prev_res"].astype(str) + " -> " + df[res_col].astype(str)
    df["res_transition"] = df[res_col].astype(str) + " -> " + df["next_res"].astype(str)
    df["res_act_pair"] = df[res_col].astype(str) + "_" + df[act_col].astype(str)
    df["act_transition"] = df["prev_act"].astype(str) + " -> " + df[act_col].astype(str)

    df["self_handover_flag"] = (df[res_col] == df["prev_res"]).astype(int)
    previous_same = grouped[res_col].shift(1)
    run_id = (df[res_col] != previous_same).groupby(df[case_col]).cumsum()
    df["res_repeat_burst_len"] = df.groupby([case_col, run_id], dropna=False)[res_col].transform("size")

    df = _add_local_repeat_features(df, case_col, res_col)
    df = _add_trace_resource_ratio(df, case_col, res_col)

    if train_stats is None:
        stats = _build_train_stats(df, case_col, act_col, time_col, res_col)
    else:
        stats = train_stats

    df = _apply_train_stats(df, stats, act_col, res_col)

    df["window_3_res_entropy"] = [
        _entropy([p, c, n])
        for p, c, n in zip(df["prev_res"], df[res_col], df["next_res"])
    ]

    df = _add_time_features(df, stats, act_col, time_col, res_col, has_time)

    helper_cols = [
        "trace_len",
        "pos_bin",
        "prev_act",
        "next_act",
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
