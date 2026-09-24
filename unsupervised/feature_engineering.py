import numpy as np
import pandas as pd

import config as cfg


EPS = 1e-6


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.index.name = None
    return out


def _surprise(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    return -np.log(np.clip(values, EPS, None))


def _conditional_probability(
    out: pd.DataFrame,
    train: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
) -> np.ndarray:
    joint_cols = condition_cols + [outcome_col]
    joint_counts = train.groupby(joint_cols, dropna=False, observed=True).size()
    condition_counts = train.groupby(condition_cols, dropna=False, observed=True).size()

    joint_keys = pd.MultiIndex.from_frame(out[joint_cols])
    joint = joint_counts.reindex(joint_keys, fill_value=0).to_numpy(dtype=float)

    if len(condition_cols) == 1:
        condition = out[condition_cols[0]].map(condition_counts).fillna(0.0).to_numpy(dtype=float)
    else:
        condition_keys = pd.MultiIndex.from_frame(out[condition_cols])
        condition = condition_counts.reindex(condition_keys, fill_value=0).to_numpy(dtype=float)

    return np.divide(joint, condition, out=np.zeros_like(joint), where=condition > 0)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_index(df)

    if "original_order" not in out.columns:
        out["original_order"] = np.arange(len(out))

    out = out.sort_values(
        [cfg.CASE_COL, cfg.TIME_COL, "original_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    out["delta_t"] = out.groupby(cfg.CASE_COL)[cfg.TIME_COL].diff().dt.total_seconds().fillna(0.0)
    out["delta_t"] = out["delta_t"].clip(lower=0.0)
    out["cum_t"] = out.groupby(cfg.CASE_COL)["delta_t"].cumsum()

    out["trace_len"] = out.groupby(cfg.CASE_COL)[cfg.CASE_COL].transform("size")
    out["event_idx"] = out.groupby(cfg.CASE_COL).cumcount()
    denominator = (out["trace_len"] - 1).clip(lower=1)
    out["trace_pos"] = out["event_idx"] / denominator

    hours = out[cfg.TIME_COL].dt.hour + out[cfg.TIME_COL].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    weekdays = out[cfg.TIME_COL].dt.dayofweek
    out["day_sin"] = np.sin(2 * np.pi * weekdays / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * weekdays / 7.0)

    out["prev_activity"] = out.groupby(cfg.CASE_COL)[cfg.ACT_COL].shift(1).fillna("START")
    out["next_activity"] = out.groupby(cfg.CASE_COL)[cfg.ACT_COL].shift(-1).fillna("END")
    out["prev2_activity"] = out.groupby(cfg.CASE_COL)[cfg.ACT_COL].shift(2).fillna("START2")
    out["transition"] = out["prev_activity"].astype(str) + "->" + out[cfg.ACT_COL].astype(str)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_index(df)
    train = out[out["split"].astype(str) == "train"].copy()

    def add_zscore(group_cols: list[str], output_col: str) -> None:
        if train.empty:
            out[output_col] = 0.0
            return

        stats = train.groupby(group_cols, dropna=False, observed=True)["delta_t"].agg(["mean", "std"])
        global_mean = float(train["delta_t"].mean())

        if len(group_cols) == 1:
            mean = out[group_cols[0]].map(stats["mean"])
            std = out[group_cols[0]].map(stats["std"])
        else:
            keys = pd.MultiIndex.from_frame(out[group_cols])
            mean = pd.Series(stats["mean"].reindex(keys).to_numpy(), index=out.index)
            std = pd.Series(stats["std"].reindex(keys).to_numpy(), index=out.index)

        mean = mean.fillna(global_mean)
        std = std.fillna(0.0)
        out[output_col] = (out["delta_t"] - mean) / (std + EPS)

    add_zscore([cfg.ACT_COL], "z_score_activity")
    add_zscore(["transition"], "z_score_transition")
    add_zscore([cfg.ACT_COL, cfg.RES_COL], "z_score_activity_resource")
    out["z_score_micro"] = out["z_score_activity_resource"]

    if train.empty:
        out["z_score_cum_t"] = 0.0
    else:
        cum_mean = float(train["cum_t"].mean())
        cum_std = float(train["cum_t"].std())
        out["z_score_cum_t"] = (out["cum_t"] - cum_mean) / (cum_std + EPS)

    out["resource_workload_2h"] = 0.0
    window_ns = pd.Timedelta(hours=2).value
    sorted_out = out.sort_values([cfg.RES_COL, cfg.TIME_COL, "original_order"], kind="mergesort")

    for _, group in sorted_out.groupby(cfg.RES_COL, sort=False, observed=True):
        times = group[cfg.TIME_COL].astype("int64").to_numpy()
        left_bounds = np.searchsorted(times, times - window_ns, side="left")
        counts = np.arange(len(group)) - left_bounds + 1
        out.loc[group.index, "resource_workload_2h"] = counts.astype(float)

    out["resource_workload"] = out["resource_workload_2h"]
    return out


def _add_position_surprise(out: pd.DataFrame, train: pd.DataFrame, n_bins: int) -> None:
    column = f"_position_bin_{n_bins}"
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out[column] = pd.cut(
        out["trace_pos"],
        bins=edges,
        labels=False,
        include_lowest=True,
    ).fillna(0).astype(int)
    train = out[out["split"].astype(str) == "train"]
    probabilities = _conditional_probability(out, train, [cfg.ACT_COL], column)
    out[f"position_surprise_{n_bins}"] = _surprise(probabilities)
    out.drop(columns=[column], inplace=True)


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_index(df)
    train = out[out["split"].astype(str) == "train"].copy()

    if train.empty:
        raise ValueError("The train split is empty.")

    activity_probabilities = train[cfg.ACT_COL].value_counts(normalize=True, dropna=False)
    out["activity_probability"] = out[cfg.ACT_COL].map(activity_probabilities).fillna(0.0)
    out["activity_surprise"] = _surprise(out["activity_probability"])

    p_current_given_previous = _conditional_probability(
        out,
        train,
        ["prev_activity"],
        cfg.ACT_COL,
    )
    p_next_given_current = _conditional_probability(
        out,
        train,
        [cfg.ACT_COL],
        "next_activity",
    )

    out["transition_in_surprise"] = _surprise(p_current_given_previous)
    out["transition_out_surprise"] = _surprise(p_next_given_current)
    out["transition_surprise"] = out["transition_in_surprise"] + out["transition_out_surprise"]
    out["transition_asymmetry"] = np.abs(
        out["transition_in_surprise"] - out["transition_out_surprise"]
    )

    trigram_probability = _conditional_probability(
        out,
        train,
        ["prev2_activity", "prev_activity"],
        cfg.ACT_COL,
    )
    out["trigram_surprise"] = _surprise(trigram_probability)

    for n_bins in [3, 5, 10]:
        _add_position_surprise(out, train, n_bins)

    position_stats = train.groupby(cfg.ACT_COL, dropna=False, observed=True)["trace_pos"].agg(["mean", "std"])
    global_std = float(train["trace_pos"].std())
    fallback_std = max(global_std, cfg.POSITION_MIN_STD)
    mean_position = out[cfg.ACT_COL].map(position_stats["mean"]).fillna(float(train["trace_pos"].mean()))
    std_position = out[cfg.ACT_COL].map(position_stats["std"]).fillna(fallback_std).clip(lower=cfg.POSITION_MIN_STD)
    out["position_deviation"] = np.abs(out["trace_pos"] - mean_position) / std_position

    real_resource = (
        cfg.RES_COL in out.columns
        and out[cfg.RES_COL].astype(str).ne("<no_resource>").any()
        and out[cfg.RES_COL].nunique(dropna=False) > 1
    )

    if real_resource:
        p_activity_given_resource = _conditional_probability(
            out,
            train,
            [cfg.RES_COL],
            cfg.ACT_COL,
        )
        p_resource_given_activity = _conditional_probability(
            out,
            train,
            [cfg.ACT_COL],
            cfg.RES_COL,
        )
        out["activity_given_resource_surprise"] = _surprise(p_activity_given_resource)
        out["resource_given_activity_surprise"] = _surprise(p_resource_given_activity)
    else:
        out["activity_given_resource_surprise"] = 0.0
        out["resource_given_activity_surprise"] = 0.0

    return out


def compute_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    normalized_target = cfg.normalize_target(target)
    out = add_base_features(df)

    if "split" not in out.columns:
        raise ValueError("Missing required 'split' column before feature engineering.")

    if normalized_target == "time":
        out = add_time_features(out)
    else:
        out = add_activity_features(out)

    out = out.sort_values(
        [cfg.CASE_COL, cfg.TIME_COL, "original_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
