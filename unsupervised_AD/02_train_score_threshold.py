import argparse
import gc
import json
import os
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

import config as cfg
from feature_engineering import compute_features
from utils import dataset_stem, group_indices, validate_existing_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and score a target-specific Isolation Forest pipeline.")
    parser.add_argument("--target", required=True, choices=["time", "act"])
    parser.add_argument("--force", action="store_true", help="Overwrite completed outputs.")
    return parser.parse_args()


def _workers() -> int:
    raw = os.getenv("PIPELINE_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, min(2, (os.cpu_count() or 2) - 1))


def _score_train() -> bool:
    return os.getenv("SCORE_TRAIN", "0") == "1"


def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [cfg.CASE_COL, cfg.ACT_COL, cfg.RES_COL, "split"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        out[float_cols] = out[float_cols].astype("float32")
    return out


def load_injected_with_split(dataset_name: str, run: int, target: str) -> pd.DataFrame | None:
    stem = dataset_stem(dataset_name)
    injected_dir = cfg.get_target_paths(target)["injected"]
    input_path = injected_dir / f"{stem}_poisoned_run_{run}.csv"

    if not input_path.exists():
        print(f"[SKIP] Missing injected file: {input_path}", flush=True)
        return None

    df = pd.read_csv(input_path, low_memory=False)
    df[cfg.TIME_COL] = pd.to_datetime(df[cfg.TIME_COL], errors="coerce", format="mixed")
    validate_existing_split(df, input_path.name)
    return df


def output_paths(dataset_name: str, run: int, target: str):
    stem = dataset_stem(dataset_name)
    paths = cfg.get_target_paths(target)
    score_dir = paths["scored"] / stem
    threshold_dir = paths["thresholds"] / stem
    return (
        score_dir,
        threshold_dir,
        score_dir / f"run_{run}_scores.parquet",
        score_dir / f"run_{run}_scores.pkl",
        threshold_dir / f"run_{run}_thresholds.csv",
        threshold_dir / f"run_{run}_thresholds.json",
        threshold_dir / f"run_{run}_timings.csv",
        threshold_dir / f"run_{run}_complete.json",
    )


def already_completed(dataset_name: str, run: int, target: str, force: bool) -> bool:
    if force:
        return False

    (
        _,
        _,
        parquet_path,
        pickle_path,
        thresholds_path,
        _,
        timings_path,
        complete_path,
    ) = output_paths(dataset_name, run, target)
    if (
        not (parquet_path.exists() or pickle_path.exists())
        or not thresholds_path.exists()
        or not timings_path.exists()
        or not complete_path.exists()
    ):
        return False

    try:
        with open(complete_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        return metadata.get("target") == target and metadata.get("status") == "complete"
    except Exception:
        return False


def fit_gmm_threshold(scores: np.ndarray) -> dict:
    valid_scores = scores[np.isfinite(scores) & (scores != 0.0)]

    if len(valid_scores) == 0:
        return {
            "threshold": np.nan,
            "method": "empty_scores",
            "n_validation_scores": 0,
            "gmm_separation": 0.0,
        }

    if len(valid_scores) < cfg.GMM_MIN_VALID_SCORES or np.unique(valid_scores).size < 2:
        return {
            "threshold": float(np.median(valid_scores)),
            "method": "median_fallback",
            "n_validation_scores": int(len(valid_scores)),
            "gmm_separation": 0.0,
        }

    x = valid_scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="spherical", random_state=cfg.RANDOM_SEED)

    try:
        gmm.fit(x)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        anomaly_component = int(np.argmax(means))
        separation = float(abs(means[0] - means[1]) / max(stds[0] + stds[1], 1e-12))

        grid = np.linspace(float(valid_scores.min()), float(valid_scores.max()), 1000).reshape(-1, 1)
        probabilities = gmm.predict_proba(grid)[:, anomaly_component]
        candidates = np.where(probabilities >= 0.5)[0]

        if len(candidates) == 0:
            threshold = float(np.median(valid_scores))
            method = "gmm_no_crossing_median_fallback"
        else:
            threshold = float(grid[candidates[0], 0])
            method = "gmm_2_components"

        return {
            "threshold": threshold,
            "method": method,
            "n_validation_scores": int(len(valid_scores)),
            "gmm_separation": separation,
            "gmm_mean_normal": float(np.min(means)),
            "gmm_mean_anomaly": float(np.max(means)),
        }
    except Exception:
        return {
            "threshold": float(np.median(valid_scores)),
            "method": "gmm_exception_median_fallback",
            "n_validation_scores": int(len(valid_scores)),
            "gmm_separation": 0.0,
        }


def fit_and_score_group(
    df: pd.DataFrame,
    idx: np.ndarray,
    feature_cols: list[str],
    run: int,
    score_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    scores = np.zeros(len(idx), dtype=np.float32)

    split_values = df.loc[idx, "split"].astype(str).to_numpy()
    train_idx = idx[split_values == "train"]

    if len(train_idx) < cfg.MIN_GROUP_SIZE:
        return scores, 0, 0

    local_score_mask = score_mask[idx]
    score_idx = idx[local_score_mask]
    if len(score_idx) == 0:
        return scores, int(len(train_idx)), 0

    X_train = df.loc[train_idx, feature_cols].to_numpy(dtype=np.float32)
    X_score = df.loc[score_idx, feature_cols].to_numpy(dtype=np.float32)

    finite_train = np.isfinite(X_train).all(axis=1)
    if not finite_train.any():
        return scores, 0, 0

    X_train = X_train[finite_train]
    if np.all(np.nanstd(X_train, axis=0) == 0):
        return scores, 0, 0

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_score = scaler.transform(X_score)

    model_kwargs = dict(cfg.MODEL_KWARGS)
    model_kwargs["random_state"] = cfg.RANDOM_SEED + run
    model_kwargs["n_jobs"] = 1

    model = IsolationForest(**model_kwargs)
    model.fit(X_train)

    scored_values = -model.score_samples(X_score).astype(np.float32)
    local_positions = np.flatnonzero(local_score_mask)
    scores[local_positions] = scored_values

    return scores, int(len(X_train)), int(len(score_idx))


def train_score_threshold_file(
    dataset_name: str,
    run: int,
    target: str,
    force: bool,
) -> str:
    if already_completed(dataset_name, run, target, force):
        return f"[SKIP] target={target} | {dataset_name} | run={run}: already completed"

    print(f"[TRAIN+SCORE+THRESHOLD] target={target} | {dataset_name} | run={run}", flush=True)

    df = load_injected_with_split(dataset_name, run, target)
    if df is None:
        return f"[SKIP] target={target} | {dataset_name} | run={run}: missing injected log"

    feature_sets = cfg.get_feature_sets(target)
    levels = cfg.get_levels(target)

    df = compute_features(df, target=target)
    df = _optimize_memory(df)

    group_cache = {level_name: group_indices(df, level_name) for level_name in levels}
    split_values = df["split"].astype(str).to_numpy()
    score_mask = np.isin(split_values, ["train", "val", "test"] if _score_train() else ["val", "test"])
    val_mask = split_values == "val"

    threshold_rows = []
    timing_rows = []

    for feature_set_name, feature_cols in feature_sets.items():
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{dataset_name} | run={run}: missing features {missing}")

        for level_name in levels:
            print(f"[LEVEL] target={target} | {dataset_name} | run={run} | level={level_name} | feature_set={feature_set_name}", flush=True)
            cpu_start = time.process_time()
            full_scores = np.zeros(len(df), dtype=np.float32)
            score_col = f"score_{target}_{cfg.MODEL_NAME}_{level_name}_{feature_set_name}"
            candidate_groups = len(group_cache[level_name])
            trained_groups = 0

            for group_key, idx in group_cache[level_name].items():
                group_scores, n_train, n_score = fit_and_score_group(
                    df,
                    idx,
                    feature_cols,
                    run,
                    score_mask,
                )
                if n_train <= 0:
                    continue

                trained_groups += 1
                full_scores[idx] = group_scores
                threshold_info = fit_gmm_threshold(group_scores[val_mask[idx]])
                threshold_rows.append({
                    "target": target,
                    "dataset": dataset_stem(dataset_name),
                    "run": run,
                    "model": cfg.MODEL_NAME,
                    "level": level_name,
                    "feature_set": feature_set_name,
                    "feature_columns": "|".join(feature_cols),
                    "score_col": score_col,
                    "group_key": str(group_key),
                    "group_train_events": int(n_train),
                    "group_scored_events": int(n_score),
                    **threshold_info,
                })

            df[score_col] = full_scores
            del full_scores
            gc.collect()

            cpu_seconds = time.process_time() - cpu_start
            timing_rows.append({
                "target": target,
                "dataset": dataset_stem(dataset_name),
                "run": run,
                "model": cfg.MODEL_NAME,
                "level": level_name,
                "feature_set": feature_set_name,
                "groups": int(trained_groups),
                "cpu_seconds": float(cpu_seconds),
                "candidate_groups": int(candidate_groups),
                "skipped_groups": int(candidate_groups - trained_groups),
                "feature_columns": "|".join(feature_cols),
            })

    label_cols = [
        column
        for column in ["ActivityLabel", "TimeLabel", "ActLabel", "ResLabel", "is_true_anomaly"]
        if column in df.columns
    ]
    base_cols = [
        column
        for column in [cfg.CASE_COL, cfg.ACT_COL, cfg.RES_COL, cfg.TIME_COL, "split"]
        if column in df.columns
    ]
    score_cols = [column for column in df.columns if column.startswith("score_")]

    (
        score_dir,
        threshold_dir,
        scored_parquet_path,
        scored_pickle_path,
        thresholds_path,
        thresholds_json_path,
        timings_path,
        complete_path,
    ) = output_paths(dataset_name, run, target)
    score_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir.mkdir(parents=True, exist_ok=True)

    tmp_parquet_path = scored_parquet_path.with_suffix(scored_parquet_path.suffix + ".tmp")
    tmp_pickle_path = scored_pickle_path.with_suffix(scored_pickle_path.suffix + ".tmp")
    tmp_thresholds_path = thresholds_path.with_suffix(thresholds_path.suffix + ".tmp")
    tmp_json_path = thresholds_json_path.with_suffix(thresholds_json_path.suffix + ".tmp")
    tmp_timings_path = timings_path.with_suffix(timings_path.suffix + ".tmp")
    tmp_complete_path = complete_path.with_suffix(complete_path.suffix + ".tmp")

    scored_frame = df[base_cols + label_cols + score_cols]
    scored_format = "parquet"
    try:
        scored_frame.to_parquet(tmp_parquet_path, index=False)
    except ImportError:
        scored_format = "pickle"
        scored_frame.to_pickle(tmp_pickle_path)

    pd.DataFrame(threshold_rows).to_csv(tmp_thresholds_path, index=False)
    pd.DataFrame(timing_rows).to_csv(tmp_timings_path, index=False)

    with open(tmp_json_path, "w", encoding="utf-8") as file:
        json.dump(threshold_rows, file, indent=2)

    with open(tmp_complete_path, "w", encoding="utf-8") as file:
        json.dump({
            "status": "complete",
            "target": target,
            "dataset": dataset_stem(dataset_name),
            "run": run,
            "threshold_rows": len(threshold_rows),
            "timing_rows": len(timing_rows),
            "scored_format": scored_format,
        }, file, indent=2)

    if scored_format == "parquet":
        tmp_parquet_path.replace(scored_parquet_path)
        scored_pickle_path.unlink(missing_ok=True)
    else:
        tmp_pickle_path.replace(scored_pickle_path)
        scored_parquet_path.unlink(missing_ok=True)
    tmp_thresholds_path.replace(thresholds_path)
    tmp_json_path.replace(thresholds_json_path)
    tmp_timings_path.replace(timings_path)
    tmp_complete_path.replace(complete_path)

    return (f"[DONE] target={target} | {dataset_name} | run={run} | "
        f"threshold_rows={len(threshold_rows)} | timing_rows={len(timing_rows)}")


def collect_training_timings(target: str) -> None:
    result_dir = cfg.get_target_paths(target)["results"]
    frames = []

    for dataset_name in cfg.ACTIVE_DATASETS:
        stem = dataset_stem(dataset_name)
        threshold_dir = cfg.get_target_paths(target)["thresholds"] / stem

        for run in range(1, cfg.NUM_RUNS + 1):
            timing_path = threshold_dir / f"run_{run}_timings.csv"
            if not timing_path.exists():
                print(f"[WARN] Missing timing file: {timing_path}")
                continue

            frame = pd.read_csv(timing_path)
            if not frame.empty:
                frames.append(frame)

    if not frames:
        print("[WARN] No training timing files were found.")
        return

    detail = pd.concat(frames, ignore_index=True)
    level_order = {
        "L3_Global": 0,
        "L2_Activity": 1,
        "L1_ActivityResource": 2,
    }

    baseline_keys = ["target", "dataset", "run", "model", "feature_set"]
    baseline = (
        detail[detail["level"] == "L3_Global"][baseline_keys + ["cpu_seconds"]]
        .rename(columns={"cpu_seconds": "l3_cpu_seconds"})
    )
    detail = detail.merge(baseline, on=baseline_keys, how="left")
    detail["slowdown_vs_L3"] = np.where(
        detail["l3_cpu_seconds"] > 0,
        detail["cpu_seconds"] / detail["l3_cpu_seconds"],
        np.nan,
    )
    detail["_level_order"] = detail["level"].map(level_order).fillna(99)
    detail = detail.sort_values(
        ["dataset", "run", "_level_order", "feature_set"]
    ).drop(columns=["_level_order"])

    detail.to_csv(result_dir / "training_times_by_run.csv", index=False)

    feature_summary = (
        detail.groupby(
            ["target", "dataset", "model", "level", "feature_set"],
            as_index=False,
        )
        .agg(
            runs=("run", "nunique"),
            groups_mean=("groups", "mean"),
            groups_std=("groups", "std"),
            cpu_seconds_mean=("cpu_seconds", "mean"),
            cpu_seconds_std=("cpu_seconds", "std"),
            slowdown_vs_L3_mean=("slowdown_vs_L3", "mean"),
            slowdown_vs_L3_std=("slowdown_vs_L3", "std"),
        )
    )
    feature_summary.to_csv(
        result_dir / "training_times_mean_std.csv",
        index=False,
    )

    level_totals = (
        detail.groupby(
            ["target", "dataset", "run", "model", "level"],
            as_index=False,
        )
        .agg(
            feature_sets=("feature_set", "nunique"),
            groups=("groups", "sum"),
            candidate_groups=("candidate_groups", "sum"),
            skipped_groups=("skipped_groups", "sum"),
            cpu_seconds=("cpu_seconds", "sum"),
        )
    )

    baseline_keys = ["target", "dataset", "run", "model"]
    baseline = (
        level_totals[level_totals["level"] == "L3_Global"][
            baseline_keys + ["cpu_seconds"]
        ]
        .rename(columns={"cpu_seconds": "l3_total_cpu_seconds"})
    )
    level_totals = level_totals.merge(baseline, on=baseline_keys, how="left")
    level_totals["slowdown_vs_L3"] = np.where(
        level_totals["l3_total_cpu_seconds"] > 0,
        level_totals["cpu_seconds"] / level_totals["l3_total_cpu_seconds"],
        np.nan,
    )
    level_totals.to_csv(
        result_dir / "training_times_level_totals_by_run.csv",
        index=False,
    )

    totals_summary = (
        level_totals.groupby(
            ["target", "dataset", "model", "level"],
            as_index=False,
        )
        .agg(
            runs=("run", "nunique"),
            feature_sets_mean=("feature_sets", "mean"),
            groups_mean=("groups", "mean"),
            groups_std=("groups", "std"),
            cpu_seconds_mean=("cpu_seconds", "mean"),
            cpu_seconds_std=("cpu_seconds", "std"),
            slowdown_vs_L3_mean=("slowdown_vs_L3", "mean"),
            slowdown_vs_L3_std=("slowdown_vs_L3", "std"),
        )
    )
    totals_summary.to_csv(
        result_dir / "training_times_level_totals_mean_std.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    target = cfg.normalize_target(args.target)
    cfg.get_target_paths(target)

    tasks = [
        (dataset_name, run)
        for dataset_name in cfg.ACTIVE_DATASETS
        for run in range(1, cfg.NUM_RUNS + 1)
    ]
    workers = _workers()
    print(f"Running {len(tasks)} target={target} dataset/run tasks with {workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(train_score_threshold_file, dataset_name, run, target, args.force): (
                dataset_name,
                run,
            )
            for dataset_name, run in tasks
        }
        for future in as_completed(futures):
            dataset_name, run = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                print(f"[ERROR] target={target} | {dataset_name} | run={run}: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)

    collect_training_timings(target)


if __name__ == "__main__":
    main()
