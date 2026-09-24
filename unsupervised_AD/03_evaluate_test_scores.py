import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import config as cfg
from utils import dataset_stem, group_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate target-specific test scores.")
    parser.add_argument("--target", required=True, choices=["time", "act"])
    return parser.parse_args()


def empty_metrics(y_true: np.ndarray, valid_count: int = 0) -> dict:
    return {
        "test_events": int(len(y_true)),
        "valid_test_scores": int(valid_count),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "prauc": 0.0,
        "rocauc": np.nan,
        "predicted_anomaly_rate": 0.0,
        "true_anomaly_rate": float(100 * y_true.mean()) if len(y_true) else 0.0,
        "tp": 0,
        "fp": 0,
        "fn": int((y_true == 1).sum()),
        "tn": int((y_true == 0).sum()),
    }


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray,
) -> dict:
    if valid.sum() == 0:
        return empty_metrics(y_true, 0)

    y = y_true[valid]
    scores_valid = scores[valid]
    predictions = y_pred[valid]

    tp = int(((y == 1) & (predictions == 1)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())

    return {
        "test_events": int(len(y_true)),
        "valid_test_scores": int(valid.sum()),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "prauc": float(average_precision_score(y, scores_valid)) if len(np.unique(y)) > 1 else 0.0,
        "rocauc": float(roc_auc_score(y, scores_valid)) if len(np.unique(y)) > 1 else np.nan,
        "predicted_anomaly_rate": float(100 * predictions.mean()) if len(predictions) else 0.0,
        "true_anomaly_rate": float(100 * y_true.mean()) if len(y_true) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_file(dataset_name: str, run: int, target: str) -> list[dict]:
    stem = dataset_stem(dataset_name)
    paths = cfg.get_target_paths(target)
    parquet_path = paths["scored"] / stem / f"run_{run}_scores.parquet"
    pickle_path = paths["scored"] / stem / f"run_{run}_scores.pkl"
    thresholds_path = paths["thresholds"] / stem / f"run_{run}_thresholds.csv"

    if not parquet_path.exists() and not pickle_path.exists():
        print(f"[SKIP] Missing scored file: {parquet_path} or {pickle_path}")
        return []
    if not thresholds_path.exists():
        print(f"[SKIP] Missing thresholds file: {thresholds_path}")
        return []

    print(f"[EVAL] target={target} | {dataset_name} | run={run}")

    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_pickle(pickle_path)
    thresholds = pd.read_csv(thresholds_path)
    if thresholds.empty:
        return []
    if "group_key" not in thresholds.columns:
        raise ValueError(f"{thresholds_path}: missing 'group_key'. Expected local thresholds.")

    test = df[df["split"].astype(str) == "test"].copy().reset_index(drop=True)
    label_col = cfg.get_label_col(target)
    if label_col not in test.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y_true = test[label_col].to_numpy(dtype=int)

    rows = []
    group_cols = ["model", "level", "feature_set", "score_col"]

    for keys, threshold_group in thresholds.groupby(group_cols, dropna=False):
        model, level, feature_set, score_col = keys
        if score_col not in test.columns:
            continue

        scores = test[score_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(scores) & (scores != 0.0)
        y_pred = np.zeros(len(test), dtype=int)
        test_groups = group_indices(test, level)

        applied_thresholds = 0
        finite_thresholds = []
        separations = []
        validation_sizes = []
        train_events = []

        for _, threshold_row in threshold_group.iterrows():
            group_key = str(threshold_row["group_key"])
            if group_key not in test_groups:
                continue

            threshold = (
                float(threshold_row["threshold"])
                if pd.notna(threshold_row["threshold"])
                else np.nan
            )
            if not np.isfinite(threshold):
                continue

            idx = test_groups[group_key]
            local_valid = valid[idx]
            if not local_valid.any():
                continue

            local_idx = idx[local_valid]
            y_pred[local_idx] = (scores[local_idx] > threshold).astype(int)

            applied_thresholds += 1
            finite_thresholds.append(threshold)
            separations.append(float(threshold_row.get("gmm_separation", 0.0)))
            validation_sizes.append(float(threshold_row.get("n_validation_scores", 0.0)))
            train_events.append(float(threshold_row.get("group_train_events", 0.0)))

        metrics = compute_metrics_from_predictions(y_true, scores, y_pred, valid)
        rows.append({
            "target": target,
            "dataset": stem,
            "run": run,
            "model": model,
            "level": level,
            "feature_set": feature_set,
            "score_col": score_col,
            "threshold": float(np.nanmean(finite_thresholds)) if finite_thresholds else np.nan,
            "threshold_method": "local_group_thresholds",
            "gmm_separation": float(np.nanmean(separations)) if separations else np.nan,
            "n_validation_scores": float(np.nansum(validation_sizes)) if validation_sizes else 0.0,
            "trained_groups": int(len(threshold_group)),
            "applied_thresholds": int(applied_thresholds),
            "train_events": float(np.nansum(train_events)) if train_events else 0.0,
            **metrics,
        })

    return rows


def main() -> None:
    args = parse_args()
    target = cfg.normalize_target(args.target)
    result_dir = cfg.get_target_paths(target)["results"]
    all_rows = []

    for dataset_name in cfg.ACTIVE_DATASETS:
        for run in range(1, cfg.NUM_RUNS + 1):
            all_rows.extend(evaluate_file(dataset_name, run, target))

    result = pd.DataFrame(all_rows)
    detail_path = result_dir / "test_metrics_by_run.csv"
    aggregate_path = result_dir / "test_metrics_mean_std.csv"
    result.to_csv(detail_path, index=False)

    if not result.empty:
        group_cols = ["target", "dataset", "model", "level", "feature_set"]
        metric_cols = [
            "precision",
            "recall",
            "f1",
            "prauc",
            "rocauc",
            "predicted_anomaly_rate",
            "true_anomaly_rate",
            "gmm_separation",
            "n_validation_scores",
            "trained_groups",
            "applied_thresholds",
            "train_events",
        ]
        present_metrics = [column for column in metric_cols if column in result.columns]
        aggregated = result.groupby(group_cols, as_index=False).agg(
            {column: ["mean", "std"] for column in present_metrics}
        )
        aggregated.columns = ["_".join(column).strip("_") for column in aggregated.columns]
        aggregated.to_csv(aggregate_path, index=False)

    print(f"Saved detailed metrics: {detail_path}")
    print(f"Saved aggregated metrics: {aggregate_path}")


if __name__ == "__main__":
    main()
