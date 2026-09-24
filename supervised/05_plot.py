import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import BASE_DIR

LOCAL_RESULTS_ROOT = BASE_DIR / "results"

MODEL_ORDER = ["RandomForest", "XGBoost"]
MODEL_COLORS = {
    "RandomForest": "#4C72B0",
    "XGBoost": "#DD8452",
}

METRICS_CONFIG = {
    "F1": {
        "label": "Mean F1-Score",
        "title": "F1-Score Profile",
        "ylim": (0, 1.05),
    },
    "Precision": {
        "label": "Mean Precision",
        "title": "Precision Profile",
        "ylim": (0, 1.05),
    },
    "Recall": {
        "label": "Mean Recall",
        "title": "Recall Profile",
        "ylim": (0, 1.05),
    },
    "PRAUC": {
        "label": "Mean PR-AUC",
        "title": "PR-AUC Profile",
        "ylim": (0, 1.05),
    },
    "Off_Target_False_Alarm_Rate": {
        "label": "Mean Off-Target False Alarm Rate",
        "title": "Off-Target False Alarm Profile",
        "ylim": (0, 1.05),
    },
}


SCENARIO_REPLACEMENTS = {
    "Seen": "Seen",
    "Unseen_Time_Micro-Delay": "Micro-Delay",
    "Unseen_Time_Macro-Delay": "Macro-Delay",
    "Unseen_Time_TS-Round-Min": "Timestamp Rounding Minute",
    "Unseen_Time_TS-Round-Hour": "Timestamp Rounding Hour",
    "Unseen_Time_TS-Round-Day": "Timestamp Rounding Day",
    "Unseen_Time_Trace-Level-Storm": "Trace-Level Storm",
}


def safe_filename(value: str) -> str:
    value = str(value)
    replacements = {
        " ": "_",
        "(": "",
        ")": "",
        ":": "",
        "%": "",
        "+": "and",
        "/": "_",
        "\\": "_",
        "|": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def clean_model_name(value: str) -> str:
    value = str(value)

    if "RandomForest" in value:
        return "RandomForest"
    if "XGBoost" in value:
        return "XGBoost"

    return "Other"


def categorize_scenario(scenario: str) -> str:
    if scenario == "Seen":
        return "Seen"
    if scenario.startswith("Unseen_Time_"):
        return "Unseen Time"
    return "Other"

def normalize_test_rate(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.replace(",", ".").strip()

    return float(value)


def clean_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "Base_Dataset",
        "Model",
        "Scenario",
        "Test_Rate",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    df["Model_Clean"] = df["Model"].apply(clean_model_name)
    df = df[df["Model_Clean"].isin(MODEL_ORDER)].copy()

    df["Category"] = df["Scenario"].apply(categorize_scenario)
    df["Scenario_Clean"] = df["Scenario"].replace(SCENARIO_REPLACEMENTS)
    df["Test_Rate_Num"] = df["Test_Rate"].apply(normalize_test_rate)

    metric_ids = [
        metric_id
        for metric_id in METRICS_CONFIG
        if f"{metric_id}_mean" in df.columns
    ]

    if not metric_ids:
        raise ValueError(
            "No supported metric columns found. Expected columns like F1_mean, Precision_mean, Recall_mean."
        )

    group_cols = [
        "Base_Dataset",
        "Model_Clean",
        "Category",
        "Scenario_Clean",
        "Test_Rate_Num",
    ]

    def aggregate_group(group):
        values = {}

        for metric_id in metric_ids:
            mean_col = f"{metric_id}_mean"
            std_col = f"{metric_id}_std"

            values[mean_col] = group[mean_col].mean()

            if std_col in group.columns:
                if len(group) > 1:
                    values[std_col] = group[mean_col].std()
                else:
                    values[std_col] = group[std_col].iloc[0]
            else:
                values[std_col] = 0.0

        return pd.Series(values)

    clean_df = (
        df.groupby(group_cols, dropna=False)
        .apply(aggregate_group, include_groups=False)
        .reset_index()
    )

    for metric_id in metric_ids:
        std_col = f"{metric_id}_std"
        if std_col in clean_df.columns:
            clean_df[std_col] = clean_df[std_col].fillna(0.0)

    clean_df = clean_df.sort_values(
        ["Category", "Scenario_Clean", "Base_Dataset", "Model_Clean", "Test_Rate_Num"]
    )

    return clean_df


def plot_models_with_bands(data, metric_mean, metric_std, ylim, **kwargs):
    ax = plt.gca()

    for model in MODEL_ORDER:
        model_df = data[data["Model_Clean"] == model].sort_values("Test_Rate_Num")

        if model_df.empty:
            continue

        x = model_df["Test_Rate_Num"].to_numpy(dtype=float)
        y = model_df[metric_mean].to_numpy(dtype=float)
        std = model_df[metric_std].to_numpy(dtype=float)

        color = MODEL_COLORS[model]

        ax.plot(
            x,
            y,
            marker="o",
            color=color,
            linewidth=2.5,
            markersize=6,
            label=model,
        )

        if np.nansum(std) > 0:
            lower = np.clip(y - std, ylim[0], ylim[1])
            upper = np.clip(y + std, ylim[0], ylim[1])
            ax.fill_between(x, lower, upper, alpha=0.18, color=color)


def plot_all_metrics(df_agg: pd.DataFrame, output_dir: Path, target: str):
    sns.set_theme(style="whitegrid")

    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(df_agg["Scenario_Clean"].dropna().unique())

    for metric_id, cfg in METRICS_CONFIG.items():
        mean_col = f"{metric_id}_mean"
        std_col = f"{metric_id}_std"

        if mean_col not in df_agg.columns:
            continue

        metric_dir = output_dir / metric_id
        metric_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Generating plots for metric: {metric_id}")

        for scenario in scenarios:
            plot_df = df_agg[df_agg["Scenario_Clean"] == scenario].copy()

            if plot_df.empty:
                continue

            if plot_df["Category"].eq("Unseen Time").all():
                continue

            if metric_id == "Off_Target_False_Alarm_Rate":
                if plot_df[mean_col].fillna(0).sum() == 0:
                    continue

            datasets = sorted(plot_df["Base_Dataset"].dropna().astype(str).unique())

            if not datasets:
                continue

            col_wrap = 3 if len(datasets) > 4 else 2

            g = sns.FacetGrid(
                plot_df,
                col="Base_Dataset",
                col_wrap=col_wrap,
                height=4.0,
                aspect=1.35,
                sharex=True,
                sharey=True,
            )

            g.map_dataframe(
                plot_models_with_bands,
                metric_mean=mean_col,
                metric_std=std_col,
                ylim=cfg["ylim"],
            )

            g.set_axis_labels("Injection rate", cfg["label"])
            g.set(ylim=cfg["ylim"])
            g.set_titles(col_template="{col_name}", fontweight="bold")

            unique_rates = sorted(plot_df["Test_Rate_Num"].dropna().unique())

            for ax in g.axes.flat:
                ax.set_yticks(np.arange(cfg["ylim"][0], cfg["ylim"][1] + 0.001, 0.1))
                ax.grid(True, linestyle="--", alpha=0.55)

                if unique_rates:
                    ax.set_xticks(unique_rates)

                ax.tick_params(axis="x", rotation=45, labelbottom=True)
                ax.tick_params(axis="y", labelleft=True)

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=MODEL_COLORS[model],
                    lw=3,
                    marker="o",
                )
                for model in MODEL_ORDER
            ]

            g.fig.legend(
                handles,
                MODEL_ORDER,
                title="Algorithms",
                loc="center right",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=11,
                title_fontsize=12,
                frameon=True,
            )

            g.fig.subplots_adjust(
                top=0.90,
                right=0.91,
                hspace=0.32,
                wspace=0.12,
            )

            g.fig.suptitle(
                f"{target.upper()} - {cfg['title']}:\n{scenario}",
                fontsize=17,
                fontweight="bold",
            )

            out_file = metric_dir / f"comparison_{target}_{metric_id}_{safe_filename(scenario)}.png"

            g.savefig(out_file, bbox_inches="tight", dpi=300)
            plt.close(g.fig)

            print(f"[SAVED] {out_file}")



UNSEEN_TIME_SCENARIO_ORDER = [
    "Micro-Delay",
    "Macro-Delay",
    "Timestamp Rounding Minute",
    "Timestamp Rounding Hour",
    "Timestamp Rounding Day",
    "Trace-Level Storm",
]


def plot_unseen_timestamp_f1_summary(df_detailed: pd.DataFrame, output_dir: Path, target: str):
    if target != "time":
        return

    required_cols = {
        "Base_Dataset",
        "Run_ID",
        "Model",
        "Scenario",
        "Test_Rate",
        "F1",
    }
    missing = sorted(required_cols - set(df_detailed.columns))
    if missing:
        print(
            f"[WARN] Detailed unseen results are missing columns {missing}. "
            "Skipping unseen summary plot."
        )
        return

    unseen_df = df_detailed.copy()
    unseen_df = unseen_df[unseen_df["Base_Dataset"].isin(SELECTED_DATASETS)].copy()
    unseen_df["Model_Clean"] = unseen_df["Model"].apply(clean_model_name)
    unseen_df = unseen_df[unseen_df["Model_Clean"].isin(MODEL_ORDER)].copy()
    unseen_df["Category"] = unseen_df["Scenario"].apply(
        lambda x: categorize_scenario(x, target)
    )
    unseen_df["Scenario_Clean"] = unseen_df["Scenario"].replace(SCENARIO_REPLACEMENTS)
    unseen_df["Test_Rate_Num"] = unseen_df["Test_Rate"].apply(normalize_test_rate)
    unseen_df["F1"] = pd.to_numeric(unseen_df["F1"], errors="coerce")
    unseen_df = unseen_df[
        (unseen_df["Category"] == "Unseen Time")
        & unseen_df["F1"].notna()
    ].copy()

    if unseen_df.empty:
        print("[WARN] No unseen timestamp detailed results found. Skipping unseen summary plot.")
        return

    # Extract the training replicate number for run-level macro-averaging.
    unseen_df["Training_Run"] = pd.to_numeric(
        unseen_df["Run_ID"].astype(str).str.extract(r"_run(\d+)$", expand=False),
        errors="coerce",
    )
    invalid_run_ids = unseen_df["Training_Run"].isna()
    if invalid_run_ids.any():
        bad_ids = sorted(unseen_df.loc[invalid_run_ids, "Run_ID"].astype(str).unique())
        raise ValueError(
            "Could not extract training-run number from Run_ID values: "
            f"{bad_ids[:10]}"
        )
    unseen_df["Training_Run"] = unseen_df["Training_Run"].astype(int)

    summary_dir = output_dir / "unseen_timestamp_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-run macro-averages across datasets.
    dataset_run = (
        unseen_df
        .groupby(
            [
                "Base_Dataset",
                "Training_Run",
                "Model_Clean",
                "Scenario_Clean",
                "Test_Rate_Num",
            ],
            dropna=False,
        )["F1"]
        .mean()
        .reset_index(name="Dataset_Run_F1")
    )

    run_macro = (
        dataset_run
        .groupby(
            ["Training_Run", "Model_Clean", "Scenario_Clean", "Test_Rate_Num"],
            dropna=False,
        )
        .agg(
            Run_Macro_F1=("Dataset_Run_F1", "mean"),
            Num_Datasets=("Base_Dataset", "nunique"),
        )
        .reset_index()
    )

    # The plotted point is the mean of the five run-level macro-averages.
    # The shaded band is ±1 standard deviation across those five values.
    summary = (
        run_macro
        .groupby(
            ["Model_Clean", "Scenario_Clean", "Test_Rate_Num"],
            dropna=False,
        )
        .agg(
            Mean_F1=("Run_Macro_F1", "mean"),
            Std_Across_Runs=("Run_Macro_F1", "std"),
            Num_Runs=("Training_Run", "nunique"),
            Num_Datasets=("Num_Datasets", "min"),
        )
        .reset_index()
    )
    summary["Std_Across_Runs"] = summary["Std_Across_Runs"].fillna(0.0)

    summary_csv = summary_dir / "time_unseen_f1_macro_average.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] {summary_csv}")

    run_macro_csv = summary_dir / "time_unseen_f1_macro_average_by_run.csv"
    run_macro.to_csv(run_macro_csv, index=False)
    print(f"[SAVED] {run_macro_csv}")

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(UNSEEN_TIME_SCENARIO_ORDER))
    scenario_colors = dict(zip(UNSEEN_TIME_SCENARIO_ORDER, palette))
    scenario_markers = {
        "Micro-Delay": "o",
        "Macro-Delay": "s",
        "Timestamp Rounding Minute": "^",
        "Timestamp Rounding Hour": "D",
        "Timestamp Rounding Day": "v",
        "Trace-Level Storm": "P",
    }

    def spread_label_positions(items, min_gap=0.055, lower=0.035, upper=1.015):
        if not items:
            return {}

        ordered = sorted(items, key=lambda item: item[1])
        ys = [float(np.clip(y, lower, upper)) for _, y in ordered]

        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + min_gap)

        if ys[-1] > upper:
            shift = ys[-1] - upper
            ys = [y - shift for y in ys]

        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)

        if ys[0] < lower:
            shift = lower - ys[0]
            ys = [y + shift for y in ys]

        return {scenario: y for (scenario, _), y in zip(ordered, ys)}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        sharex=True,
        sharey=True,
    )

    plotted_any = False
    all_rates = sorted(summary["Test_Rate_Num"].dropna().unique())
    final_rate = max(all_rates) if all_rates else 0.30

    plot_right = final_rate + 0.008
    label_x = plot_right + 0.008

    for ax, model in zip(axes, MODEL_ORDER):
        model_df = summary[summary["Model_Clean"] == model].copy()
        endpoint_items = []
        endpoint_values = {}

        for scenario in UNSEEN_TIME_SCENARIO_ORDER:
            scenario_df = model_df[
                model_df["Scenario_Clean"] == scenario
            ].sort_values("Test_Rate_Num")

            if scenario_df.empty:
                continue

            plotted_any = True
            x = scenario_df["Test_Rate_Num"].to_numpy(dtype=float)
            y = scenario_df["Mean_F1"].to_numpy(dtype=float)
            std = scenario_df["Std_Across_Runs"].to_numpy(dtype=float)
            color = scenario_colors[scenario]

            ax.plot(
                x,
                y,
                marker=scenario_markers.get(scenario, "o"),
                linewidth=2.2,
                markersize=6,
                color=color,
            )

            lower = np.clip(y - std, 0.0, 1.05)
            upper = np.clip(y + std, 0.0, 1.05)
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

            end_idx = int(np.argmax(x))
            end_x = float(x[end_idx])
            end_y = float(y[end_idx])
            endpoint_items.append((scenario, end_y))
            endpoint_values[scenario] = (end_x, end_y)

        label_positions = spread_label_positions(endpoint_items)
        for scenario in UNSEEN_TIME_SCENARIO_ORDER:
            if scenario not in endpoint_values:
                continue

            end_x, end_y = endpoint_values[scenario]
            text_y = label_positions[scenario]
            color = scenario_colors[scenario]

            ax.plot(
                [end_x, plot_right + 0.006],
                [end_y, text_y],
                color=color,
                linewidth=0.9,
                alpha=0.75,
                clip_on=False,
            )
            ax.text(
                label_x,
                text_y,
                scenario,
                color=color,
                fontsize=9.5,
                va="center",
                ha="left",
                fontweight="medium",
                clip_on=False,
            )

        if all_rates:
            ax.set_xticks(all_rates)
            ax.set_xticklabels([f"{rate:.0%}" for rate in all_rates])

        ax.set_xlim(min(all_rates) - 0.01 if all_rates else 0.0, plot_right)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlabel("Injection rate")
        ax.set_title(model, fontweight="bold", fontsize=14)

    axes[0].set_ylabel("Mean F1-Score across datasets")

    fig.suptitle(
        "TIME - Generalization to Unseen Timestamp Anomalies",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    if not plotted_any:
        plt.close(fig)
        print("[WARN] No unseen timestamp series available to plot.")
        return

    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.11, top=0.89, wspace=0.35)

    out_file = summary_dir / "comparison_time_F1_Unseen_RF_XGB.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[SAVED] {out_file}")

def plot_category_summary(df_agg: pd.DataFrame, output_dir: Path, target: str):
    sns.set_theme(style="whitegrid")

    summary_dir = output_dir / "category_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    main_metrics = ["F1", "Precision", "Recall", "PRAUC"]

    for metric_id in main_metrics:
        mean_col = f"{metric_id}_mean"
        std_col = f"{metric_id}_std"

        if mean_col not in df_agg.columns:
            continue

        summary = (
            df_agg.groupby(["Category", "Model_Clean"], dropna=False)[[mean_col, std_col]]
            .mean()
            .reset_index()
        )

        categories = sorted(summary["Category"].dropna().astype(str).unique())
        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.2), 6))

        for i, model in enumerate(MODEL_ORDER):
            model_df = summary[summary["Model_Clean"] == model]

            values = []
            errors = []

            for category in categories:
                rows = model_df[model_df["Category"].astype(str) == category]

                if rows.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                else:
                    values.append(rows[mean_col].mean())
                    errors.append(rows[std_col].mean())

            offset = (i - 0.5) * width

            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=4,
                color=MODEL_COLORS[model],
                ecolor="black",
                error_kw={"elinewidth": 1.1, "capthick": 1.1},
                label=model,
            )

            for bar, value in zip(bars, values):
                if pd.notna(value) and value > 0:
                    ax.annotate(
                        f"{value:.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        xytext=(0, 4),
                        textcoords="offset points",
                    )

        ax.set_title(f"{target.upper()} - Mean {metric_id} by scenario category")
        ax.set_ylabel(f"Mean {metric_id}")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.55)
        ax.legend(title="Algorithms")

        fig.tight_layout()

        out_file = summary_dir / f"{target}_{metric_id}_category_summary.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"[SAVED] {out_file}")



def plot_offtarget_worst_cases(df_agg: pd.DataFrame, output_dir: Path, target: str, top_n: int = 25):

    sns.set_theme(style="whitegrid")

    mean_col = "Off_Target_False_Alarm_Rate_mean"

    if mean_col not in df_agg.columns:
        print("[WARN] Off_Target_False_Alarm_Rate_mean not found. Skipping off-target summary.")
        return

    summary_dir = output_dir / "offtarget_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    work_df = df_agg.copy()
    work_df[mean_col] = pd.to_numeric(work_df[mean_col], errors="coerce").fillna(0.0)

    work_df = work_df[work_df[mean_col] > 0].copy()

    if work_df.empty:
        print("[WARN] No non-zero off-target false alarm rates found. Skipping off-target summary.")
        return

    ranked = (
        work_df
        .groupby(["Scenario_Clean", "Category", "Model_Clean"], dropna=False)
        .agg(
            Mean_Off_Target_FAR=(mean_col, "mean"),
            Std_Off_Target_FAR=(mean_col, "std"),
            Max_Off_Target_FAR=(mean_col, "max"),
            Num_Points=(mean_col, "count"),
        )
        .reset_index()
    )

    ranked["Std_Off_Target_FAR"] = ranked["Std_Off_Target_FAR"].fillna(0.0)

    ranked = ranked.sort_values(
        ["Mean_Off_Target_FAR", "Max_Off_Target_FAR"],
        ascending=False,
    )

    ranked_path = summary_dir / f"{target}_offtarget_worst_cases_ranked.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"[SAVED] {ranked_path}")

    top_ranked = ranked.head(top_n).copy()
    top_ranked["Scenario_Model"] = (
        top_ranked["Scenario_Clean"].astype(str)
        + " | "
        + top_ranked["Model_Clean"].astype(str)
    )

    fig_height = max(7, 0.42 * len(top_ranked))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    sns.barplot(
        data=top_ranked,
        y="Scenario_Model",
        x="Mean_Off_Target_FAR",
        hue="Model_Clean",
        order=top_ranked["Scenario_Model"].tolist(),
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        dodge=False,
        ax=ax,
    )

    ax.set_title(f"{target.upper()} - Worst off-target false alarm scenarios")
    ax.set_xlabel("Mean off-target false alarm rate")
    ax.set_ylabel("Scenario | model")
    ax.set_xlim(0, min(1.05, max(0.05, top_ranked["Mean_Off_Target_FAR"].max() * 1.15)))
    ax.grid(axis="x", linestyle="--", alpha=0.55)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    ax.legend(title="Algorithm", loc="lower right")
    fig.tight_layout()

    out_file = summary_dir / f"{target}_worst_offtarget_false_alarm_cases.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[SAVED] {out_file}")

    heatmap_df = (
        ranked
        .pivot_table(
            index="Scenario_Clean",
            columns="Model_Clean",
            values="Mean_Off_Target_FAR",
            aggfunc="mean",
        )
        .fillna(0.0)
    )

    if not heatmap_df.empty:
        heatmap_df["__sort_key__"] = heatmap_df.max(axis=1)
        heatmap_df = heatmap_df.sort_values("__sort_key__", ascending=False).drop(columns="__sort_key__")
        heatmap_df = heatmap_df.head(top_n)

        fig_height = max(6, 0.38 * len(heatmap_df))
        fig, ax = plt.subplots(figsize=(8, fig_height))

        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=min(1.0, max(0.05, float(heatmap_df.max().max()))),
            cmap="Reds",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean off-target FAR"},
            ax=ax,
        )

        ax.set_title(f"{target.upper()} - Off-target false alarm heatmap")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Scenario")
        fig.tight_layout()

        heatmap_file = summary_dir / f"{target}_offtarget_false_alarm_heatmap.png"
        fig.savefig(heatmap_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[SAVED] {heatmap_file}")

def find_result_files(results_root: Path, target_filter=None, rate_filter=None):
    candidate_dirs = []

    if not results_root.exists():
        return candidate_dirs

    for target_dir in sorted(results_root.iterdir()):
        if not target_dir.is_dir():
            continue

        target = target_dir.name

        if target_filter and target != target_filter:
            continue

        for rate_dir in sorted(target_dir.iterdir()):
            if not rate_dir.is_dir():
                continue

            rate = rate_dir.name

            if rate_filter and rate != rate_filter:
                continue

            agg_path = rate_dir / "advanced_evaluation" / "advanced_metrics_aggregated.csv"

            if agg_path.exists():
                candidate_dirs.append((target, rate, agg_path))

    return candidate_dirs

SELECTED_DATASETS = [
    "BPIC12.csv",
    "BPIC13_C.csv",
    "BPIC13_I.csv",
    "BPIC13_O.csv",
    "BPIC20_D.csv",
    "BPIC20_I.csv",
    "BPIC20_PE.csv",
    "BPIC20_PR.csv",
    "BPIC20_R.csv",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["time", "act", "res"], default=None)
    parser.add_argument("--rate", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--no-category-summary", action="store_true")
    parser.add_argument("--no-offtarget-summary", action="store_true")
    parser.add_argument("--offtarget-top-n", type=int, default=25)
    args, _ = parser.parse_known_args()

    print("\n--- [SUPERVISED ADVANCED EVALUATION PLOTS] ---")

    results_root = (
        Path(args.results_root).expanduser().resolve()
        if args.results_root
        else LOCAL_RESULTS_ROOT
    )
    print(f"[INFO] Results root: {results_root}")

    candidate_dirs = find_result_files(
        results_root,
        target_filter=args.target,
        rate_filter=args.rate,
    )

    if not candidate_dirs:
        raise FileNotFoundError(
            f"No advanced_metrics_aggregated.csv found under {results_root}"
        )

    for target, rate, agg_path in candidate_dirs:
        print(f"\n[INFO] Processing target={target}, rate={rate}")
        print(f"[INFO] Source: {agg_path}")

        raw_df = pd.read_csv(agg_path)
        raw_df = raw_df[raw_df["Base_Dataset"].isin(SELECTED_DATASETS)].copy()
        clean_df = clean_data(raw_df, target=target)

        out_dir = agg_path.parent / "plots_png" / "algorithm_comparisons"
        plot_all_metrics(clean_df, out_dir, target)

        detailed_path = agg_path.parent / "advanced_metrics_detailed.csv"
        if target == "time":
            if detailed_path.exists():
                detailed_df = pd.read_csv(detailed_path)
                plot_unseen_timestamp_f1_summary(detailed_df, out_dir, target)
            else:
                print(
                    f"[WARN] Missing detailed results required for run-level unseen STD: "
                    f"{detailed_path}"
                )

        if not args.no_category_summary:
            plot_category_summary(clean_df, out_dir, target)

        if not args.no_offtarget_summary:
            plot_offtarget_worst_cases(clean_df, out_dir, target, top_n=args.offtarget_top_n)

        clean_out = out_dir / "cleaned_aggregated_metrics_for_plots.csv"
        clean_df.to_csv(clean_out, index=False)
        print(f"[SAVED] {clean_out}")
        print(f"[DONE] Plots saved in: {out_dir}")

    print("\nAll plots completed.")


if __name__ == "__main__":
    main()