import numpy as np
import pandas as pd

from config import ACT_COL, CASE_COL, TIME_COL


def _inject_comuzzi_activity(
    original: pd.DataFrame,
    output: pd.DataFrame,
    n_anomalies: int,
    rng: np.random.Generator,
) -> None:
    activities = original[ACT_COL].dropna().unique()
    if n_anomalies <= 0 or len(activities) <= 1:
        return

    n_anomalies = min(n_anomalies, len(original))
    indices = rng.choice(original.index.to_numpy(), size=n_anomalies, replace=False)

    current = original.loc[indices, ACT_COL].to_numpy()
    replacements = np.empty(n_anomalies, dtype=object)

    for position, activity in enumerate(current):
        candidates = activities[activities != activity]
        replacements[position] = rng.choice(candidates)

    output.loc[indices, ACT_COL] = replacements
    output.loc[indices, "ActivityLabel"] = 1


def _inject_comuzzi_time(
    original: pd.DataFrame,
    output: pd.DataFrame,
    n_anomalies: int,
    rng: np.random.Generator,
) -> None:
    original_duration = (
        original.groupby(CASE_COL, sort=False)[TIME_COL]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    activity_stats = (
        original.assign(_duration=original_duration)
        .groupby(ACT_COL, dropna=False, observed=True)["_duration"]
        .agg(["mean", "std"])
    )
    activity_stats["std"] = activity_stats["std"].fillna(0.0)

    # The first event is excluded because it has no preceding duration.
    first_indices = original.groupby(CASE_COL, sort=False).head(1).index
    eligible_indices = original.index[~original.index.isin(first_indices)].to_numpy()

    if n_anomalies <= 0 or len(eligible_indices) == 0:
        return
    if n_anomalies > len(eligible_indices):
        raise ValueError(
            f"Requested {n_anomalies} temporal anomalies but only "
            f"{len(eligible_indices)} non-initial events are available."
        )

    indices = rng.choice(eligible_indices, size=n_anomalies, replace=False)
    anomalous_duration = original_duration.copy()

    means = original.loc[indices, ACT_COL].map(activity_stats["mean"]).to_numpy(dtype=float)
    stds = original.loc[indices, ACT_COL].map(activity_stats["std"]).to_numpy(dtype=float)
    extreme_duration = (1.0 + rng.random(n_anomalies)) * (means + stds)

    anomalous_duration.loc[indices] = extreme_duration
    output.loc[indices, "TimeLabel"] = 1

    # Rebuild timestamps from the previous original timestamp and current duration.
    rebuilt_timestamp = original[TIME_COL].copy()
    for _, group in original.groupby(CASE_COL, sort=False, observed=True):
        group_indices = group.index.to_numpy()
        if len(group_indices) <= 1:
            continue

        previous_original = original.loc[group_indices[:-1], TIME_COL].to_numpy()
        current_duration = anomalous_duration.loc[group_indices[1:]].to_numpy(dtype=float)
        rebuilt_timestamp.loc[group_indices[1:]] = (
            pd.to_datetime(previous_original)
            + pd.to_timedelta(current_duration, unit="s")
        )

    output[TIME_COL] = rebuilt_timestamp


def inject_anomalies(
    df: pd.DataFrame,
    anomaly_rate: float = 0.15,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Inject Comuzzi-style activity and temporal anomalies into the same log."""
    rng = np.random.default_rng(random_state)

    original = df.copy()
    original = original.sort_values(
        [CASE_COL, TIME_COL, "original_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    output = original.copy()
    output["ActivityLabel"] = 0
    output["TimeLabel"] = 0

    n_activity = int(len(output) * anomaly_rate)
    n_time = int(len(output) * anomaly_rate)

    # Activity and timestamp anomalies are sampled independently, so the same event may contain both.
    _inject_comuzzi_activity(original, output, n_activity, rng)
    _inject_comuzzi_time(original, output, n_time, rng)

    actual_activity = int(output["ActivityLabel"].sum())
    actual_time = int(output["TimeLabel"].sum())
    if actual_activity != n_activity or actual_time != n_time:
        raise RuntimeError(
            "Unexpected injected anomaly counts: "
            f"ActivityLabel={actual_activity}/{n_activity}, "
            f"TimeLabel={actual_time}/{n_time}"
        )

    return output


