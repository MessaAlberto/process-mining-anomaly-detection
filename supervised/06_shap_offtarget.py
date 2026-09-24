from __future__ import annotations

import argparse
import os
import re
import shutil
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config as cfg
from feature_engineering_time import (
    compute_features as compute_time_features,
    compute_resource_features as compute_time_resource_features,
)
from feature_engineering_act import (
    compute_features as compute_act_features,
    compute_resource_features as compute_act_resource_features,
)
from feature_engineering_res import (
    compute_features as compute_res_features,
    compute_resource_features as compute_res_resource_features,
)


LABEL_BY_TARGET = {
    "time": "TimeLabel",
    "act": "ActLabel",
    "res": "ResLabel",
}

OFF_TARGET_LABELS_BY_TARGET = {
    "time": ("ActLabel", "ResLabel"),
    "act": ("TimeLabel", "ResLabel"),
    "res": ("TimeLabel", "ActLabel"),
}

FEATURE_MODULES = {
    "time": {
        "compute_base": compute_time_features,
        "compute_resource": compute_time_resource_features,
    },
    "act": {
        "compute_base": compute_act_features,
        "compute_resource": compute_act_resource_features,
    },
    "res": {
        "compute_base": compute_res_features,
        "compute_resource": compute_res_resource_features,
    },
}

MODELS_TO_USE = ("RandomForest", "XGBoost")
POPULATIONS = ("off_target_fp", "target_tp")
TARGET = "time"
SCRIPT_VERSION = "seen_scenario_tp_vs_offtarget_fp_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RQ3 SHAP analysis: compare mean positive SHAP profiles "
            "between off-target false positives and target true positives."
        )
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["time", "act", "res"],
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.15,
        help="Seen-scenario total anomaly rate to analyse.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of leading features used only for console summaries "
            "and plot labels. The CSV always contains every feature."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help=(
            "Maximum sampled events per dataset/run/model/population. "
            "Use a larger value for a denser estimate at higher cost."
        ),
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Background events used by interventional Tree SHAP.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional development limit on test files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Reuse the existing rq3_feature_profiles.csv and regenerate "
            "only the PNG plots, without recalculating SHAP."
        ),
    )
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    return int(
        (base_seed + zlib.crc32(text.encode("utf-8")))
        % (2**32 - 1)
    )


def clean_model_name(model_file: str) -> str:
    name = model_file.rsplit("__", 1)[1].replace(".joblib", "")
    if "RandomForest" in name:
        return "RandomForest"
    if "XGBoost" in name or "XGB" in name:
        return "XGBoost"
    return name


def extract_run_label(run_id: str) -> str:
    match = re.search(r"_run(\d+)", run_id)
    return f"run{match.group(1)}" if match else run_id


def get_base_name_from_test_file(test_file: Path) -> str:
    return test_file.name.replace("_test.csv", ".csv")


def normalise_rate_name(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "_")


def real_life_dataset_names() -> list[str]:
    real_root = Path(cfg.REAL_LIFE_LOG_DIR).resolve()
    names = [
        name
        for name, path in cfg.RAW_DATASET_PATHS.items()
        if Path(path).resolve().parent == real_root
    ]
    if names:
        return sorted(names)

    return [
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


def compute_target_features(
    df: pd.DataFrame,
    schema: dict,
    train_stats: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    module = FEATURE_MODULES[TARGET]
    kwargs = {
        "case_col": schema["case"],
        "act_col": schema["act"],
        "time_col": schema["time"],
        "res_col": schema.get("res"),
    }

    if train_stats is None:
        df, base_stats = module["compute_base"](df, **kwargs)
        train_stats = {"base_stats": base_stats}
    else:
        df = module["compute_base"](
            df,
            train_stats=train_stats["base_stats"],
            **kwargs,
        )

    if schema.get("res") and schema["res"] in df.columns:
        if "res_stats" not in train_stats:
            df, res_stats = module["compute_resource"](df, **kwargs)
            train_stats["res_stats"] = res_stats
        else:
            df = module["compute_resource"](
                df,
                train_stats=train_stats["res_stats"],
                **kwargs,
            )

    return df, train_stats


def normalize_categorical_columns(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
) -> pd.DataFrame:
    result = df.copy()
    for column in cat_cols:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna("__MISSING__")
                .astype(str)
            )
    return result


def get_pipeline_parts(pipeline):
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Expected a sklearn Pipeline.")
    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline has no 'preprocessor' step.")
    if "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline has no 'classifier' step.")
    return (
        pipeline.named_steps["preprocessor"],
        pipeline.named_steps["classifier"],
    )


def to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def make_unique_names(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    output: list[str] = []

    for raw_name in names:
        name = str(raw_name)
        for prefix in ("num__", "cat__", "remainder__"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        counts[name] += 1
        output.append(
            name if counts[name] == 1
            else f"{name}__{counts[name]}"
        )

    return output


def get_transformed_feature_names(
    preprocessor,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    transformed_width: int,
) -> list[str]:
    try:
        names = make_unique_names(
            list(preprocessor.get_feature_names_out())
        )
        if len(names) == transformed_width:
            return names
    except Exception:
        pass

    fallback = make_unique_names([*num_cols, *cat_cols])
    if len(fallback) == transformed_width:
        return fallback

    return [f"feature_{index}" for index in range(transformed_width)]


def normalize_shap_value_matrix(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] > 1:
            values = values[:, :, 1]
        elif values.shape[0] > 1 and values.shape[1] != values.shape[2]:
            values = values[1]
        else:
            values = values[:, :, 0]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    return values.astype(float)


def compute_probability_shap_matrix(
    classifier,
    X_selected: np.ndarray,
    X_background: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values strictly on the probability scale.

    No raw-margin/log-odds fallback is allowed because mixing output scales
    would make the numerical profiles harder to compare across models.
    """
    try:
        explainer = shap.TreeExplainer(
            classifier,
            data=X_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        try:
            explanation = explainer(
                X_selected,
                check_additivity=False,
            )
            values = explanation.values
        except Exception:
            values = explainer.shap_values(
                X_selected,
                check_additivity=False,
            )
    except Exception as exc:
        raise RuntimeError(
            "Probability-scale Tree SHAP failed. The analysis stops instead "
            "of silently switching to a different SHAP output scale. "
            f"Original error: {exc}"
        ) from exc

    return normalize_shap_value_matrix(values)


def iter_test_files(
    test_root: Path,
    rate_dir_name: str,
    real_logs: set[str],
):
    rate_dir = test_root / "Seen" / rate_dir_name
    if not rate_dir.exists():
        print(f"[WARN] Missing directory: {rate_dir}")
        return

    for test_file in sorted(rate_dir.glob("*_test.csv")):
        if get_base_name_from_test_file(test_file) in real_logs:
            yield test_file


def group_model_files_by_run(
    all_model_files: Sequence[str],
    base_name: str,
) -> dict[str, list[str]]:
    prefix = f"{base_name.replace('.csv', '')}_run"
    grouped: dict[str, list[str]] = {}

    for model_file in all_model_files:
        if not model_file.startswith(prefix):
            continue

        model_name = clean_model_name(model_file)
        if model_name not in MODELS_TO_USE:
            continue

        run_id = model_file.split("__")[0]
        grouped.setdefault(run_id, []).append(model_file)

    return grouped


def sample_indices(
    indices: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum:
        return np.sort(indices)

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(indices, size=maximum, replace=False)
    )


def build_population_masks(
    df: pd.DataFrame,
    target_label: str,
    off_target_label: str,
    preds: np.ndarray,
) -> dict[str, np.ndarray]:
    target_values = (
        df[target_label]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    off_values = (
        df[off_target_label]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    return {
        "off_target_fp": (
            (target_values == 0)
            & (off_values == 1)
            & (preds == 1)
        ),
        "target_tp": (
            (target_values == 1)
            & (preds == 1)
        ),
    }


def process_model(
    *,
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    masks: dict[str, np.ndarray],
    preprocessor,
    classifier,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    base_name: str,
    scenario_name: str,
    off_target_label: str,
    run_id: str,
    model_name: str,
    args: argparse.Namespace,
) -> list[dict]:
    population_indices: dict[str, np.ndarray] = {}

    for population in POPULATIONS:
        indices = np.flatnonzero(masks[population])
        population_indices[population] = sample_indices(
            indices,
            args.max_events,
            stable_seed(
                args.seed,
                TARGET,
                base_name,
                scenario_name,
                run_id,
                model_name,
                population,
            ),
        )

    selected_parts = [
        indices
        for indices in population_indices.values()
        if len(indices)
    ]
    if not selected_parts:
        return []

    selected_indices = np.unique(np.concatenate(selected_parts))
    X_selected_trans = to_dense(
        preprocessor.transform(X_test.iloc[selected_indices])
    )

    feature_names = get_transformed_feature_names(
        preprocessor,
        num_cols,
        cat_cols,
        X_selected_trans.shape[1],
    )

    # The same deterministic background is used for RF and XGBoost within
    # the same dataset/run/scenario. Model name is intentionally excluded.
    background_indices = sample_indices(
        np.arange(len(X_test), dtype=int),
        min(args.background_size, len(X_test)),
        stable_seed(
            args.seed,
            TARGET,
            base_name,
            scenario_name,
            run_id,
            "shared_background",
        ),
    )
    X_background_trans = to_dense(
        preprocessor.transform(X_test.iloc[background_indices])
    )

    shap_matrix = compute_probability_shap_matrix(
        classifier,
        X_selected_trans,
        X_background_trans,
    )
    if shap_matrix.shape != X_selected_trans.shape:
        raise ValueError(
            "SHAP matrix does not match transformed feature matrix: "
            f"{shap_matrix.shape} vs {X_selected_trans.shape}"
        )

    local_position = {
        int(global_index): local_index
        for local_index, global_index in enumerate(selected_indices)
    }

    rows: list[dict] = []

    for population, indices in population_indices.items():
        if len(indices) == 0:
            continue

        local_indices = np.array(
            [local_position[int(index)] for index in indices],
            dtype=int,
        )
        population_shap = shap_matrix[local_indices]

        # Keep only the SHAP evidence that pushes toward class 1.
        positive_shap = np.clip(population_shap, 0.0, None)

        # Mean first, normalise later at the final aggregated profile.
        mean_positive_shap = positive_shap.mean(axis=0)

        for feature_index, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "Target": TARGET,
                    "Dataset": base_name,
                    "Scenario": scenario_name,
                    "OffTargetPerspective": (
                        off_target_label.replace("Label", "")
                    ),
                    "Run": extract_run_label(run_id),
                    "Model": model_name,
                    "Population": population,
                    "Feature": feature_name,
                    "Mean_Positive_SHAP": float(
                        mean_positive_shap[feature_index]
                    ),
                    "Sampled_Events": int(len(indices)),
                }
            )

    return rows


def process_test_file(
    *,
    scenario_name: str,
    off_target_label: str,
    test_file: Path,
    models_dir: Path,
    all_model_files: Sequence[str],
    args: argparse.Namespace,
) -> list[dict]:
    base_name = get_base_name_from_test_file(test_file)
    if base_name not in cfg.DATASET_SCHEMAS:
        print(f"[SKIP] Missing schema for {base_name}")
        return []

    schema = cfg.DATASET_SCHEMAS[base_name]
    if TARGET == "res" and not schema.get("res"):
        return []

    raw_df = pd.read_csv(test_file, low_memory=False)
    raw_df[schema["time"]] = pd.to_datetime(
        raw_df[schema["time"]],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    run_groups = group_model_files_by_run(
        all_model_files,
        base_name,
    )
    if not run_groups:
        print(f"[WARN] No models found for {base_name}")
        return []

    target_label = LABEL_BY_TARGET[TARGET]
    profile_rows: list[dict] = []

    for run_id, model_files in sorted(run_groups.items()):
        model_files = sorted(model_files)
        first_artifact = joblib.load(models_dir / model_files[0])

        train_stats = first_artifact["train_stats"]
        num_cols = list(first_artifact["num_cols"])
        cat_cols = list(first_artifact["cat_cols"])

        df, _ = compute_target_features(
            raw_df.copy(),
            schema,
            train_stats=train_stats,
        )
        df = normalize_categorical_columns(df, cat_cols)

        required = [*num_cols, *cat_cols]
        missing = [
            column
            for column in required
            if column not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{base_name} {run_id}: missing features {missing}"
            )

        X_test = df[required]

        for model_file in model_files:
            artifact = (
                first_artifact
                if model_file == model_files[0]
                else joblib.load(models_dir / model_file)
            )
            pipeline = artifact["pipeline"]
            model_name = clean_model_name(model_file)

            probs = pipeline.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            masks = build_population_masks(
                df,
                target_label,
                off_target_label,
                preds,
            )
            preprocessor, classifier = get_pipeline_parts(pipeline)

            profile_rows.extend(
                process_model(
                    df=df,
                    X_test=X_test,
                    masks=masks,
                    preprocessor=preprocessor,
                    classifier=classifier,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    base_name=base_name,
                    scenario_name=scenario_name,
                    off_target_label=off_target_label,
                    run_id=run_id,
                    model_name=model_name,
                    args=args,
                )
            )

    return profile_rows


def aggregate_profiles(
    run_profiles: pd.DataFrame,
    expected_datasets: Sequence[str],
) -> pd.DataFrame:
    """
    Aggregate with equal weight:
      events -> run mean (already computed)
      runs -> dataset mean
      datasets -> overall mean

    Missing feature/dataset combinations are filled with zero before the
    final dataset mean, so a feature absent from one model representation
    does not receive an artificially favourable denominator.
    """
    group_cols = [
        "Target",
        "Scenario",
        "OffTargetPerspective",
        "Model",
        "Population",
        "Feature",
    ]

    dataset_profiles = (
        run_profiles.groupby(
            [*group_cols, "Dataset"],
            as_index=False,
        )["Mean_Positive_SHAP"]
        .mean()
    )

    completed_parts: list[pd.DataFrame] = []

    higher_group_cols = [
        "Target",
        "Scenario",
        "OffTargetPerspective",
        "Model",
        "Population",
    ]

    for key, group in dataset_profiles.groupby(
        higher_group_cols,
        sort=False,
    ):
        features = sorted(group["Feature"].unique())
        index = pd.MultiIndex.from_product(
            [expected_datasets, features],
            names=["Dataset", "Feature"],
        )
        completed = (
            group.set_index(["Dataset", "Feature"])[
                "Mean_Positive_SHAP"
            ]
            .reindex(index, fill_value=0.0)
            .reset_index()
        )

        for column, value in zip(higher_group_cols, key):
            completed[column] = value

        completed_parts.append(completed)

    completed_dataset_profiles = pd.concat(
        completed_parts,
        ignore_index=True,
    )

    overall = (
        completed_dataset_profiles.groupby(
            group_cols,
            as_index=False,
        )["Mean_Positive_SHAP"]
        .mean()
    )

    totals = (
        overall.groupby(
            [
                "Target",
                "Scenario",
                "OffTargetPerspective",
                "Model",
                "Population",
            ]
        )["Mean_Positive_SHAP"]
        .transform("sum")
    )
    overall["Positive_SHAP_Share"] = np.divide(
        overall["Mean_Positive_SHAP"],
        totals,
        out=np.zeros(len(overall), dtype=float),
        where=totals.to_numpy() > 0,
    )

    overall["Rank"] = (
        overall.groupby(
            [
                "Target",
                "Scenario",
                "OffTargetPerspective",
                "Model",
                "Population",
            ]
        )["Positive_SHAP_Share"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return overall


def build_all_feature_table(
    overall_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per scenario, off-target perspective and feature."""
    base_keys = [
        "Target",
        "Scenario",
        "OffTargetPerspective",
        "Feature",
    ]

    parts: list[pd.DataFrame] = []

    for model_name, model_prefix in (
        ("RandomForest", "RF"),
        ("XGBoost", "XGB"),
    ):
        model_df = overall_profiles[
            overall_profiles["Model"] == model_name
        ]

        fp = model_df[
            model_df["Population"] == "off_target_fp"
        ][
            [
                *base_keys,
                "Mean_Positive_SHAP",
                "Positive_SHAP_Share",
                "Rank",
            ]
        ].rename(
            columns={
                "Mean_Positive_SHAP": (
                    f"{model_prefix}_OffTarget_FP_Mean_Positive_SHAP"
                ),
                "Positive_SHAP_Share": (
                    f"{model_prefix}_OffTarget_FP_Share"
                ),
                "Rank": f"{model_prefix}_OffTarget_FP_Rank",
            }
        )

        tp = model_df[
            model_df["Population"] == "target_tp"
        ][
            [
                *base_keys,
                "Mean_Positive_SHAP",
                "Positive_SHAP_Share",
                "Rank",
            ]
        ].rename(
            columns={
                "Mean_Positive_SHAP": (
                    f"{model_prefix}_Target_TP_Mean_Positive_SHAP"
                ),
                "Positive_SHAP_Share": (
                    f"{model_prefix}_Target_TP_Share"
                ),
                "Rank": f"{model_prefix}_Target_TP_Rank",
            }
        )

        model_wide = fp.merge(
            tp,
            on=base_keys,
            how="outer",
        )

        value_cols = [
            f"{model_prefix}_OffTarget_FP_Mean_Positive_SHAP",
            f"{model_prefix}_OffTarget_FP_Share",
            f"{model_prefix}_Target_TP_Mean_Positive_SHAP",
            f"{model_prefix}_Target_TP_Share",
        ]
        model_wide[value_cols] = model_wide[value_cols].fillna(0.0)

        for rank_col in (
            f"{model_prefix}_OffTarget_FP_Rank",
            f"{model_prefix}_Target_TP_Rank",
        ):
            model_wide[rank_col] = model_wide.groupby(
                ["Scenario", "OffTargetPerspective"],
                dropna=False,
            )[rank_col].transform(
                lambda values: values.fillna(len(values) + 1)
            )
            model_wide[rank_col] = model_wide[rank_col].astype(int)

        model_wide[f"{model_prefix}_Delta_Share"] = (
            model_wide[f"{model_prefix}_OffTarget_FP_Share"]
            - model_wide[f"{model_prefix}_Target_TP_Share"]
        )
        parts.append(model_wide)

    table = parts[0].merge(
        parts[1],
        on=base_keys,
        how="outer",
    )

    rank_cols = [column for column in table.columns if column.endswith("_Rank")]
    value_cols = [
        column
        for column in table.columns
        if column not in base_keys and column not in rank_cols
    ]
    table[value_cols] = table[value_cols].fillna(0.0)

    for rank_col in rank_cols:
        table[rank_col] = table.groupby(
            ["Scenario", "OffTargetPerspective"],
            dropna=False,
        )[rank_col].transform(
            lambda values: values.fillna(len(values) + 1)
        )
        table[rank_col] = table[rank_col].astype(int)

    for prefix in ("RF", "XGB"):
        table[f"{prefix}_Delta_Share"] = (
            table[f"{prefix}_OffTarget_FP_Share"]
            - table[f"{prefix}_Target_TP_Share"]
        )

    table["Top5_In_Any_Profile"] = (
        (table["RF_OffTarget_FP_Rank"] <= 5)
        | (table["RF_Target_TP_Rank"] <= 5)
        | (table["XGB_OffTarget_FP_Rank"] <= 5)
        | (table["XGB_Target_TP_Rank"] <= 5)
    )

    return (
        table.sort_values(
            [
                "Scenario",
                "OffTargetPerspective",
                "RF_OffTarget_FP_Rank",
                "XGB_OffTarget_FP_Rank",
                "Feature",
            ]
        )
        .reset_index(drop=True)
    )

def labelled_features_for_plot(
    scenario_table: pd.DataFrame,
    top_k: int,
) -> set[str]:
    mask = (
        (scenario_table["RF_OffTarget_FP_Rank"] <= top_k)
        | (scenario_table["RF_Target_TP_Rank"] <= top_k)
        | (scenario_table["XGB_OffTarget_FP_Rank"] <= top_k)
        | (scenario_table["XGB_Target_TP_Rank"] <= top_k)
    )
    return set(scenario_table.loc[mask, "Feature"].astype(str))


def make_model_agreement_plot(
    scenario_table: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> None:
    """
    Create one image with two scatter plots, one per model.

    In both subplots:
      x = Target TP positive SHAP share
      y = Off-target FP positive SHAP share

    Every point is a feature. Only features in the Top-k of at least one
    FP/TP profile are labelled to keep the figure readable.
    """
    plot_table = scenario_table.reset_index(drop=True).copy()
    if plot_table.empty:
        print(f"[WARN] No points for {output_path.name}")
        return

    required = {
        "Feature",
        "Scenario",
        "OffTargetPerspective",
        "RF_Target_TP_Share",
        "RF_OffTarget_FP_Share",
        "XGB_Target_TP_Share",
        "XGB_OffTarget_FP_Share",
        "RF_Target_TP_Rank",
        "RF_OffTarget_FP_Rank",
        "XGB_Target_TP_Rank",
        "XGB_OffTarget_FP_Rank",
    }
    missing = sorted(required - set(plot_table.columns))
    if missing:
        raise ValueError(
            f"Cannot create {output_path.name}: missing CSV columns {missing}"
        )

    scenario = str(plot_table["Scenario"].iloc[0])
    perspective = str(plot_table["OffTargetPerspective"].iloc[0])
    labels = labelled_features_for_plot(plot_table, top_k)

    model_specs = [
        ("RF", "Random Forest"),
        ("XGB", "XGBoost"),
    ]

    # Shared axes are essential: the two point clouds are directly comparable.
    max_value = 0.0
    for prefix, _ in model_specs:
        x = plot_table[f"{prefix}_Target_TP_Share"].to_numpy(dtype=float) * 100.0
        y = plot_table[f"{prefix}_OffTarget_FP_Share"].to_numpy(dtype=float) * 100.0
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            max_value = max(
                max_value,
                float(np.max(x[finite])),
                float(np.max(y[finite])),
            )

    max_value = max(max_value, 1.0)
    axis_limit = max_value + max(max_value * 0.10, 0.5)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.8),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.07},
    )

    for ax, (prefix, model_title) in zip(axes, model_specs):
        x = plot_table[f"{prefix}_Target_TP_Share"].to_numpy(dtype=float) * 100.0
        y = plot_table[f"{prefix}_OffTarget_FP_Share"].to_numpy(dtype=float) * 100.0
        finite = np.isfinite(x) & np.isfinite(y)
        local_table = plot_table.loc[finite].reset_index(drop=True)
        x = x[finite]
        y = y[finite]

        ax.scatter(x, y, s=42, alpha=0.75)
        ax.plot(
            [0.0, axis_limit],
            [0.0, axis_limit],
            linestyle="--",
            linewidth=1.0,
        )

        for _, row in local_table.iterrows():
            feature = str(row["Feature"])
            if feature not in labels:
                continue
            ax.annotate(
                feature,
                (
                    float(row[f"{prefix}_Target_TP_Share"]) * 100.0,
                    float(row[f"{prefix}_OffTarget_FP_Share"]) * 100.0,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlim(0.0, axis_limit)
        ax.set_ylim(0.0, axis_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Target TP positive SHAP share (%)")
        if ax is axes[0]:
            ax.set_ylabel("Off-target FP positive SHAP share (%)")
        else:
            # Shared y-axis: omit the repeated label to reduce the gap.
            ax.set_ylabel("")
        ax.set_title(model_title)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{TARGET} target — {scenario} ({perspective} off-target)",
        y=0.98,
    )
    # Compact horizontal spacing while retaining room for the shared title.
    fig.subplots_adjust(
        # left=,
        # right=0.985,
        # bottom=0.13,
        # top=0.86,
        # wspace=0.03,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots_from_feature_table(
    feature_table: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> list[Path]:
    required_columns = {
        "Scenario",
        "OffTargetPerspective",
        "Feature",
        "RF_Target_TP_Share",
        "RF_OffTarget_FP_Share",
        "XGB_Target_TP_Share",
        "XGB_OffTarget_FP_Share",
        "RF_Target_TP_Rank",
        "RF_OffTarget_FP_Rank",
        "XGB_Target_TP_Rank",
        "XGB_OffTarget_FP_Rank",
    }
    missing = sorted(required_columns - set(feature_table.columns))
    if missing:
        raise ValueError(
            "The existing rq3_feature_profiles.csv is incompatible. "
            f"Missing columns: {missing}"
        )

    plot_paths: list[Path] = []
    for (scenario, perspective), scenario_table in feature_table.groupby(
        ["Scenario", "OffTargetPerspective"],
        sort=False,
    ):
        scenario_suffix = str(scenario).lower()
        perspective_suffix = str(perspective).lower()
        plot_path = output_dir / (
            f"rq3_{TARGET}_{scenario_suffix}_{perspective_suffix}.png"
        )
        make_model_agreement_plot(
            scenario_table,
            plot_path,
            top_k,
        )
        plot_paths.append(plot_path)

    return plot_paths

def print_top_profiles(
    table: pd.DataFrame,
    top_k: int,
) -> None:
    for (scenario, perspective), group in table.groupby(
        ["Scenario", "OffTargetPerspective"],
        sort=False,
    ):
        print(f"\n[{scenario} | {perspective} off-target]")

        for prefix, label in (
            ("RF", "Random Forest"),
            ("XGB", "XGBoost"),
        ):
            print(f"\n  {label} — Off-target FP Top-{top_k}")
            columns = [
                "Feature",
                f"{prefix}_OffTarget_FP_Share",
                f"{prefix}_OffTarget_FP_Rank",
                f"{prefix}_Target_TP_Share",
                f"{prefix}_Target_TP_Rank",
                f"{prefix}_Delta_Share",
            ]
            available = group[
                group[f"{prefix}_OffTarget_FP_Rank"] <= top_k
            ]
            if available.empty:
                print("  No off-target false-positive profile was available.")
                continue

            top = (
                available.sort_values(
                    f"{prefix}_OffTarget_FP_Rank"
                )[columns]
                .head(top_k)
                .copy()
            )

            for share_col in (
                f"{prefix}_OffTarget_FP_Share",
                f"{prefix}_Target_TP_Share",
                f"{prefix}_Delta_Share",
            ):
                top[share_col] = (
                    top[share_col] * 100.0
                ).round(2)

            print(top.to_string(index=False))


def main() -> None:
    global TARGET

    args = parse_args()
    TARGET = args.target

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive.")
    if args.background_size <= 0:
        raise ValueError("--background-size must be positive.")

    rate_dir = normalise_rate_name(args.rate)
    real_logs = real_life_dataset_names()
    real_log_set = set(real_logs)

    test_root = Path(cfg.TEST_LOGS_DIR)
    models_dir = (
        Path(cfg.BASE_DIR)
        / "saved_models"
        / TARGET
        / cfg.RATE_STR
    )
    output_dir = (
        Path(cfg.BASE_DIR)
        / "results"
        / TARGET
        / cfg.RATE_STR
        / "explainability_rq3"
    )

    if args.plots_only:
        if args.force:
            raise ValueError(
                "Do not combine --plots-only with --force: plots-only must "
                "preserve and reuse the existing CSV."
            )

        csv_path = output_dir / "rq3_feature_profiles.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Existing feature-profile CSV not found: {csv_path}"
            )

        feature_table = pd.read_csv(csv_path)
        plot_paths = generate_plots_from_feature_table(
            feature_table,
            output_dir,
            args.top_k,
        )

        print(f"Script version: {SCRIPT_VERSION}")
        print("\n--- [PLOTS ONLY COMPLETED] ---")
        print(f"Read: {csv_path}")
        for plot_path in plot_paths:
            print(f" - {plot_path.name}")
        return

    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} already exists. Use --force."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_root.exists():
        raise FileNotFoundError(test_root)
    if not models_dir.exists():
        raise FileNotFoundError(models_dir)

    all_model_files = sorted(
        name
        for name in os.listdir(models_dir)
        if name.endswith(".joblib")
    )

    off_target_labels = OFF_TARGET_LABELS_BY_TARGET[TARGET]
    test_files = list(
        iter_test_files(
            test_root,
            rate_dir,
            real_log_set,
        )
    )
    if args.max_files is not None:
        test_files = test_files[: args.max_files]

    if not test_files:
        raise FileNotFoundError(
            "No Seen-scenario test files found."
        )

    work_items = [
        (test_file, off_target_label)
        for test_file in test_files
        for off_target_label in off_target_labels
    ]

    print(f"Script version: {SCRIPT_VERSION}")
    print("\n--- [RQ3 SHAP PROFILE ANALYSIS] ---")
    print(f"Target: {TARGET}")
    print(f"Test rate: {args.rate:.2%}")
    print(f"Real-life logs: {len(real_logs)}")
    print(f"Test files: {len(test_files)}")
    print(f"Off-target comparisons: {len(work_items)}")
    print("SHAP output scale: probability only")

    profile_rows: list[dict] = []

    for index, (test_file, off_target_label) in enumerate(
        work_items,
        1,
    ):
        perspective = off_target_label.replace("Label", "")
        print(
            f"[{index}/{len(work_items)}] "
            f"Seen | {perspective} off-target | {test_file.name}"
        )

        profile_rows.extend(
            process_test_file(
                scenario_name="Seen",
                off_target_label=off_target_label,
                test_file=test_file,
                models_dir=models_dir,
                all_model_files=all_model_files,
                args=args,
            )
        )

    if not profile_rows:
        raise RuntimeError("No SHAP profiles generated.")

    run_profiles = pd.DataFrame(profile_rows)
    processed_datasets = sorted(
        run_profiles["Dataset"].dropna().unique().tolist()
    )
    overall_profiles = aggregate_profiles(
        run_profiles,
        expected_datasets=processed_datasets,
    )
    feature_table = build_all_feature_table(
        overall_profiles
    )

    csv_path = output_dir / "rq3_feature_profiles.csv"
    feature_table.to_csv(csv_path, index=False)

    plot_paths = generate_plots_from_feature_table(
        feature_table,
        output_dir,
        args.top_k,
    )

    print_top_profiles(feature_table, args.top_k)

    print("\n--- [COMPLETED] ---")
    print(f"Saved in: {output_dir}")
    print(f" - {csv_path.name}")
    for plot_path in plot_paths:
        print(f" - {plot_path.name}")


if __name__ == "__main__":
    main()
