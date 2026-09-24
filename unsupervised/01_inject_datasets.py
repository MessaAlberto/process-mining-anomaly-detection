import argparse

import pandas as pd

import config as cfg
from injection import inject_anomalies
from utils import (
    add_train_val_test_split,
    iter_dataset_paths,
    load_and_standardize_log,
    safe_name,
    validate_existing_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject Comuzzi-style activity and temporal anomalies into shared event logs."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing injected files.",
    )
    return parser.parse_args()


def add_split_to_existing_file(path, run: int) -> bool:
    df = pd.read_csv(path, low_memory=False)
    if "split" in df.columns:
        validate_existing_split(df, path.name)
        return False

    df[cfg.TIME_COL] = pd.to_datetime(
        df[cfg.TIME_COL],
        errors="coerce",
        format="mixed",
    )
    df = add_train_val_test_split(
        df,
        train_pct=cfg.TRAIN_PCT,
        val_pct=cfg.VAL_PCT,
        test_pct=cfg.TEST_PCT,
        random_state=cfg.RANDOM_SEED + run,
        case_col=cfg.CASE_COL,
    )
    df.to_csv(path, index=False)
    return True


def main() -> None:
    args = parse_args()
    injected_dir = cfg.DATA_DIR / "injected_logs"
    injected_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, path in iter_dataset_paths(cfg.DATASETS):
        stem = safe_name(dataset_name)
        source_df = None

        for run in range(1, cfg.NUM_RUNS + 1):
            output_path = injected_dir / f"{stem}_poisoned_run_{run}.csv"

            if output_path.exists() and not args.force:
                changed = add_split_to_existing_file(output_path, run)
                status = "added split" if changed else "already has split"
                print(f"[SKIP] {dataset_name} | run={run}: {status}")
                continue

            if source_df is None:
                source_df = load_and_standardize_log(path)

            print(f"[INJECT+SPLIT] {dataset_name} | run={run}")

            injected = inject_anomalies(
                source_df,
                anomaly_rate=cfg.ANOMALY_RATE,
                random_state=cfg.RANDOM_SEED + run,
            )
            injected = add_train_val_test_split(
                injected,
                train_pct=cfg.TRAIN_PCT,
                val_pct=cfg.VAL_PCT,
                test_pct=cfg.TEST_PCT,
                random_state=cfg.RANDOM_SEED + run,
                case_col=cfg.CASE_COL,
            )
            injected.to_csv(output_path, index=False)

            expected = int(len(injected) * cfg.ANOMALY_RATE)
            activity_count = int(injected["ActivityLabel"].sum())
            time_count = int(injected["TimeLabel"].sum())

            if activity_count != expected or time_count != expected:
                raise RuntimeError(
                    f"{dataset_name} | run={run}: unexpected anomaly counts "
                    f"ActivityLabel={activity_count}/{expected}, "
                    f"TimeLabel={time_count}/{expected}"
                )

            split_counts = injected["split"].value_counts().to_dict()
            print(
                f"Saved {output_path} | "
                f"events={len(injected):,}, "
                f"train={split_counts.get('train', 0):,}, "
                f"val={split_counts.get('val', 0):,}, "
                f"test={split_counts.get('test', 0):,}, "
                f"ActivityLabel={activity_count:,}, "
                f"TimeLabel={time_count:,}"
            )


if __name__ == "__main__":
    main()
