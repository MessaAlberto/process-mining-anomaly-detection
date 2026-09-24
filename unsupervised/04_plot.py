import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg


METRICS = [
    ("f1_anomalous", "F1 anomalies"),
    ("precision_anomalous", "Precision anomalies"),
    ("recall_anomalous", "Recall anomalies"),
    ("predicted_anomaly_rate", "Predicted anomaly rate (%)"),
    ("f1_normal", "F1 normal"),
    ("precision_normal", "Precision normal"),
    ("recall_normal", "Recall normal"),
]

LEVEL_LABELS = {
    "L3_Global": "L3 Global",
    "L2_Activity": "L2 Activity",
    "L1_ActivityResource": "L1 Activity-Resource",
}

TARGET_LABELS = {
    "time": "Timestamp anomalies",
    "act": "Activity anomalies",
}

EXCLUDED_FROM_GLOBAL_SUMMARY = {
    "art_log_1",
    "art_log_2",
    "small_log",
    "large_log",
    "bpi_2012",
    "bpi_2013",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot target-specific Isolation Forest results.")
    parser.add_argument("--target", required=True, choices=["time", "act"])
    return parser.parse_args()


def normalize_dataset_name(value) -> str:
    name = str(value).strip()
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return name.lower()


def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def add_normal_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["precision_anomalous"] = out["precision"]
    out["recall_anomalous"] = out["recall"]
    out["f1_anomalous"] = out["f1"]

    tn = out["tn"].astype(float)
    fp = out["fp"].astype(float)
    fn = out["fn"].astype(float)

    out["precision_normal"] = safe_div(tn, tn + fn)
    out["recall_normal"] = safe_div(tn, tn + fp)
    out["f1_normal"] = safe_div(
        2 * out["precision_normal"] * out["recall_normal"],
        out["precision_normal"] + out["recall_normal"],
    )
    return out


def aggregate_results(
    df: pd.DataFrame,
    levels: list[str],
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    group_cols = ["dataset", "model", "level", "feature_set"]
    metric_cols = [metric for metric, _ in METRICS]

    agg_df = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.columns = [
        "_".join(column).strip("_")
        for column in agg_df.columns.to_flat_index()
    ]

    level_order = [level for level in levels if level in agg_df["level"].unique()]
    feature_order = [
        name for name in feature_sets
        if name in agg_df["feature_set"].unique()
    ]

    agg_df["level"] = pd.Categorical(
        agg_df["level"],
        categories=level_order,
        ordered=True,
    )
    agg_df["feature_set"] = pd.Categorical(
        agg_df["feature_set"],
        categories=feature_order,
        ordered=True,
    )

    return agg_df.sort_values(["dataset", "level", "feature_set"])


def load_and_prepare_results(
    target: str,
    by_run_path,
    levels: list[str],
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not by_run_path.exists():
        raise FileNotFoundError(f"Missing file: {by_run_path}")

    df = pd.read_csv(by_run_path)
    required_cols = [
        "dataset",
        "model",
        "level",
        "feature_set",
        "precision",
        "recall",
        "f1",
        "predicted_anomaly_rate",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    if "target" in df.columns:
        df = df[df["target"].astype(str) == target].copy()

    df = df[df["model"].astype(str) == cfg.MODEL_NAME].copy()
    df = add_normal_class_metrics(df)

    all_agg_df = aggregate_results(df, levels, feature_sets)

    real_df = df[
        ~df["dataset"].map(normalize_dataset_name).isin(EXCLUDED_FROM_GLOBAL_SUMMARY)
    ].copy()
    removed = len(df) - len(real_df)
    excluded = ", ".join(sorted(EXCLUDED_FROM_GLOBAL_SUMMARY))
    print(f"[INFO] Global summary excludes {removed} rows from: {excluded}")

    real_agg_df = aggregate_results(real_df, levels, feature_sets)
    return all_agg_df, real_agg_df


def plot_global_summary(df: pd.DataFrame, target: str, plot_dir) -> None:
    out_dir = plot_dir / "real_log_global_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for metric, metric_label in METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in df.columns:
            continue

        summary = (
            df.groupby(["level", "feature_set"], observed=True)[mean_col]
            .mean()
            .reset_index()
        )
        if summary.empty:
            continue

        summary["level_label"] = summary["level"].astype(str).map(
            lambda value: LEVEL_LABELS.get(value, value)
        )

        grid = sns.catplot(
            data=summary,
            x="feature_set",
            y=mean_col,
            col="level_label",
            hue="feature_set",
            legend=False,
            kind="bar",
            palette="viridis",
            height=5,
            aspect=1.2,
            sharey=True,
        )

        grid.set_axis_labels("Feature Set", metric_label)
        grid.set_titles("{col_name}", fontweight="bold", pad=2)

        for axis in grid.axes.flat:
            if metric == "predicted_anomaly_rate":
                axis.set_ylim(0, 100)
            else:
                axis.set_ylim(0, 1.05)
                axis.set_yticks(np.arange(0, 1.1, 0.1))

            axis.set_xticks(axis.get_xticks())
            axis.set_xticklabels(axis.get_xticklabels(), rotation=35, ha="right")

            for patch in axis.patches:
                height = patch.get_height()
                if pd.notnull(height) and height > 0:
                    label = f"{height:.1f}" if metric == "predicted_anomaly_rate" else f"{height:.2f}"
                    axis.annotate(
                        label,
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        xytext=(0, 4),
                        textcoords="offset points",
                        fontweight="bold",
                    )

        plt.subplots_adjust(top=0.88)
        grid.fig.suptitle(
            f"{TARGET_LABELS[target]} - real logs global mean - {metric_label}",
            fontsize=15,
            fontweight="bold",
        )

        out_path = out_dir / f"real_logs_global_{metric}.png"
        grid.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(grid.fig)
        print(f"[SAVED] {out_path}")


def plot_dataset_metric_grid(df: pd.DataFrame, target: str, plot_dir) -> None:
    out_dir = plot_dir / "dataset_metric_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    datasets = sorted(df["dataset"].dropna().astype(str).unique())
    metric_labels = dict(METRICS)
    value_vars = [
        f"{metric}_mean"
        for metric, _ in METRICS
        if f"{metric}_mean" in df.columns
    ]

    for dataset in datasets:
        dataset_df = df[df["dataset"].astype(str) == dataset].copy()
        if dataset_df.empty:
            continue

        melt_df = dataset_df.melt(
            id_vars=["level", "feature_set"],
            value_vars=value_vars,
            var_name="metric_key",
            value_name="value",
        )
        melt_df["metric_key"] = melt_df["metric_key"].str.replace("_mean", "", regex=False)
        melt_df["metric_name"] = melt_df["metric_key"].map(metric_labels)
        melt_df["metric_name"] = pd.Categorical(
            melt_df["metric_name"],
            categories=list(metric_labels.values()),
            ordered=True,
        )
        melt_df["level_label"] = melt_df["level"].astype(str).map(
            lambda value: LEVEL_LABELS.get(value, value)
        )

        grid = sns.catplot(
            data=melt_df,
            x="feature_set",
            y="value",
            row="metric_name",
            col="level_label",
            hue="feature_set",
            legend=False,
            kind="bar",
            palette="viridis",
            height=3.5,
            aspect=1.4,
            sharey=False,
            sharex=False,
        )

        grid.set_axis_labels("", "")
        grid.set_titles("{col_name} | {row_name}", fontweight="bold", pad=2)

        for (row_value, _), axis in grid.axes_dict.items():
            metric_key = next(key for key, label in metric_labels.items() if label == row_value)

            if metric_key == "predicted_anomaly_rate":
                axis.set_ylim(0, 100)
            else:
                axis.set_ylim(0, 1.05)
                axis.set_yticks(np.arange(0, 1.1, 0.2))

            if axis.get_subplotspec().is_last_row():
                axis.set_xticks(axis.get_xticks())
                axis.set_xticklabels(axis.get_xticklabels(), rotation=35, ha="right")
                axis.set_xlabel("Feature Set")
            else:
                axis.set_xticklabels([])

            for patch in axis.patches:
                height = patch.get_height()
                if pd.notnull(height) and height > 0:
                    label = f"{height:.1f}" if metric_key == "predicted_anomaly_rate" else f"{height:.2f}"
                    axis.annotate(
                        label,
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        xytext=(0, 4),
                        textcoords="offset points",
                        fontweight="bold",
                    )

        plt.subplots_adjust(top=0.92, hspace=0.3)
        grid.fig.suptitle(
            f"{dataset} - {TARGET_LABELS[target]} - Isolation Forest evaluation",
            fontsize=16,
            fontweight="bold",
        )

        out_path = out_dir / f"{dataset}_metrics_grid.png"
        grid.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(grid.fig)
        print(f"[SAVED] {out_path}")



def load_timing_totals(target: str, result_dir) -> pd.DataFrame | None:
    totals_path = result_dir / "training_times_level_totals_by_run.csv"

    if not totals_path.exists():
        print(f"[WARN] Missing timing file: {totals_path}")
        return None

    totals_df = pd.read_csv(totals_path)

    required = [
        "target",
        "dataset",
        "run",
        "model",
        "level",
        "cpu_seconds",
    ]
    missing = [column for column in required if column not in totals_df.columns]
    if missing:
        raise ValueError(
            f"{totals_path.name}: missing required columns {missing}. "
            f"Available columns: {list(totals_df.columns)}"
        )

    totals_df = totals_df[
        (totals_df["target"].astype(str) == target)
        & (totals_df["model"].astype(str) == cfg.MODEL_NAME)
    ].copy()

    return totals_df


def _add_horizontal_labels(axis, logarithmic: bool = False) -> None:
    for container in axis.containers:
        labels = []
        for bar in container:
            value = bar.get_width()
            if not np.isfinite(value) or value <= 0:
                labels.append("")
            elif value >= 1000:
                labels.append(f"{value:,.0f}")
            elif value >= 100:
                labels.append(f"{value:.0f}")
            elif value >= 10:
                labels.append(f"{value:.1f}")
            else:
                labels.append(f"{value:.2f}")

        axis.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=8,
        )

    if logarithmic:
        axis.margins(x=0.18)
    else:
        axis.margins(x=0.12)


def plot_timing_level_totals_by_dataset(
    totals_df: pd.DataFrame,
    target: str,
    levels: list[str],
    plot_dir,
) -> None:
    if totals_df.empty:
        print("[WARN] No level-total timing rows to plot.")
        return

    out_dir = plot_dir / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        totals_df.groupby(["dataset", "level"], as_index=False)
        .agg(cpu_seconds_mean=("cpu_seconds", "mean"))
    )

    level_order = [
        level
        for level in levels
        if level in summary["level"].astype(str).unique()
    ]

    pivot = summary.pivot(
        index="dataset",
        columns="level",
        values="cpu_seconds_mean",
    )

    sort_level = (
        "L1_ActivityResource"
        if "L1_ActivityResource" in pivot.columns
        else level_order[-1]
    )
    pivot = pivot.sort_values(sort_level, ascending=True)
    pivot = pivot.reindex(columns=level_order)

    axis = pivot.plot(
        kind="barh",
        figsize=(12, max(7, len(pivot) * 0.52)),
        width=0.78,
    )

    positive_values = pivot.to_numpy(dtype=float)
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0)
    ]
    use_log_scale = (
        len(positive_values) > 1
        and positive_values.max() / positive_values.min() >= 20
    )

    if use_log_scale:
        axis.set_xscale("log")

    axis.set_xlabel(
        "Mean CPU time over runs (seconds, logarithmic scale)"
        if use_log_scale
        else "Mean CPU time over runs (seconds)"
    )
    axis.set_ylabel("Dataset")
    axis.set_title(
        f"{TARGET_LABELS[target]} - total CPU time by level and dataset"
    )
    axis.legend(
        [
            LEVEL_LABELS.get(level, level)
            for level in level_order
        ],
        title="Level",
        loc="best",
    )
    axis.grid(axis="x", alpha=0.3)
    axis.grid(axis="y", visible=False)

    _add_horizontal_labels(axis, logarithmic=use_log_scale)

    figure = axis.get_figure()
    figure.tight_layout()

    out_path = out_dir / "timing_level_totals_by_dataset.png"
    figure.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[SAVED] {out_path}")


def plot_timing_level_ratios_by_dataset(
    totals_df: pd.DataFrame,
    target: str,
    plot_dir,
) -> None:
    if totals_df.empty:
        print("[WARN] No timing rows available for level-ratio plot.")
        return

    out_dir = plot_dir / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_pivot = totals_df.pivot_table(
        index=["dataset", "run"],
        columns="level",
        values="cpu_seconds",
        aggfunc="sum",
    )

    if "L3_Global" not in run_pivot.columns:
        raise ValueError(
            "Cannot compute timing ratios because L3_Global is missing."
        )

    ratio_rows = []
    for level, ratio_label in [
        ("L2_Activity", "L2 / L3"),
        ("L1_ActivityResource", "L1 / L3"),
    ]:
        if level not in run_pivot.columns:
            continue

        valid = (
            run_pivot["L3_Global"].notna()
            & run_pivot[level].notna()
            & (run_pivot["L3_Global"] > 0)
        )

        level_ratios = (
            run_pivot.loc[valid, level]
            / run_pivot.loc[valid, "L3_Global"]
        )

        frame = level_ratios.rename("ratio").reset_index()
        frame["comparison"] = ratio_label
        ratio_rows.append(frame)

    if not ratio_rows:
        print("[WARN] L2 and L1 timing levels are missing.")
        return

    ratios = pd.concat(ratio_rows, ignore_index=True)

    summary = (
        ratios.groupby(["dataset", "comparison"], as_index=False)
        .agg(ratio_mean=("ratio", "mean"))
    )

    pivot = summary.pivot(
        index="dataset",
        columns="comparison",
        values="ratio_mean",
    )

    sort_column = "L1 / L3" if "L1 / L3" in pivot.columns else "L2 / L3"
    pivot = pivot.sort_values(sort_column, ascending=True)

    comparison_order = [
        value for value in ["L2 / L3", "L1 / L3"]
        if value in pivot.columns
    ]
    pivot = pivot.reindex(columns=comparison_order)

    axis = pivot.plot(
        kind="barh",
        figsize=(11, max(7, len(pivot) * 0.52)),
        width=0.72,
    )

    positive_values = pivot.to_numpy(dtype=float)
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0)
    ]
    use_log_scale = (
        len(positive_values) > 1
        and positive_values.max() / positive_values.min() >= 20
    )

    if use_log_scale:
        axis.set_xscale("log")

    axis.axvline(1.0, linewidth=1, linestyle="--")
    axis.set_xlabel(
        "Mean CPU-time ratio over runs (logarithmic scale)"
        if use_log_scale
        else "Mean CPU-time ratio over runs"
    )
    axis.set_ylabel("Dataset")
    axis.set_title(
        f"{TARGET_LABELS[target]} - L2/L3 and L1/L3 CPU-time ratios by dataset"
    )
    axis.legend(title="Comparison", loc="best")
    axis.grid(axis="x", alpha=0.3)
    axis.grid(axis="y", visible=False)

    _add_horizontal_labels(axis, logarithmic=use_log_scale)

    figure = axis.get_figure()
    figure.tight_layout()

    out_path = out_dir / "timing_level_ratios_by_dataset.png"
    figure.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[SAVED] {out_path}")


def plot_timing_results(
    target: str,
    result_dir,
    levels: list[str],
    plot_dir,
) -> None:
    totals_df = load_timing_totals(target, result_dir)
    if totals_df is None:
        return

    plot_timing_level_totals_by_dataset(
        totals_df,
        target,
        levels,
        plot_dir,
    )
    plot_timing_level_ratios_by_dataset(
        totals_df,
        target,
        plot_dir,
    )

def main() -> None:
    args = parse_args()
    target = cfg.normalize_target(args.target)
    target_paths = cfg.get_target_paths(target)
    result_dir = target_paths["results"]
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_run_path = result_dir / "test_metrics_by_run.csv"
    levels = cfg.get_levels(target)
    feature_sets = cfg.get_feature_sets(target)

    all_df, real_df = load_and_prepare_results(
        target,
        by_run_path,
        levels,
        feature_sets,
    )

    print(f"[INFO] Target: {target}")
    print(f"[INFO] All rows: {len(all_df)}")
    print(f"[INFO] All datasets: {all_df['dataset'].nunique()}")
    print(f"[INFO] Real-log global rows: {len(real_df)}")
    print(f"[INFO] Real-log global datasets: {real_df['dataset'].nunique()}")
    print(f"[INFO] Levels: {sorted(all_df['level'].dropna().astype(str).unique())}")
    print(f"[INFO] Feature sets: {sorted(all_df['feature_set'].dropna().astype(str).unique())}")

    plot_global_summary(real_df, target, plot_dir)
    plot_dataset_metric_grid(all_df, target, plot_dir)
    plot_timing_results(
        target,
        result_dir,
        levels,
        plot_dir,
    )

    print(f"\nDone. Plots saved in: {plot_dir}")


if __name__ == "__main__":
    main()
