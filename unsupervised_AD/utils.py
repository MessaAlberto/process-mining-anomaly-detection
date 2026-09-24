import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ACT_COL, CASE_COL, RES_COL, TIME_COL


STANDARD_COLUMNS = [CASE_COL, ACT_COL, TIME_COL, RES_COL]
VALID_SPLITS = {"train", "val", "test"}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace(".csv", ""))


def dataset_stem(dataset_name: str) -> str:
    return safe_name(dataset_name)


def infer_schema(df: pd.DataFrame, file_name: str) -> dict:
    columns = set(df.columns)

    if {"case_id", "activity", "timestamp"}.issubset(columns):
        return {
            "case": "case_id",
            "act": "activity",
            "time": "timestamp",
            "res": "resource" if "resource" in columns else None,
        }

    if {"Case ID", "Activity", "Complete Timestamp"}.issubset(columns):
        return {
            "case": "Case ID",
            "act": "Activity",
            "time": "Complete Timestamp",
            "res": "Resource" if "Resource" in columns else None,
        }

    if {"case:concept:name", "concept:name", "time:timestamp"}.issubset(columns):
        return {
            "case": "case:concept:name",
            "act": "concept:name",
            "time": "time:timestamp",
            "res": "org:resource" if "org:resource" in columns else (
                "org:role" if "org:role" in columns else None
            ),
        }

    raise ValueError(f"{file_name}: cannot infer schema. Available columns: {list(df.columns)}")


def load_and_standardize_log(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    schema = infer_schema(raw, path.name)

    case_col = schema["case"]
    act_col = schema["act"]
    time_col = schema["time"]
    res_col = schema["res"]

    use_cols = [case_col, act_col, time_col]
    if res_col is not None and res_col in raw.columns:
        use_cols.append(res_col)

    df = raw[use_cols].copy()

    rename_map = {
        case_col: CASE_COL,
        act_col: ACT_COL,
        time_col: TIME_COL,
    }
    if res_col is not None and res_col in raw.columns:
        rename_map[res_col] = RES_COL

    df = df.rename(columns=rename_map)

    if RES_COL not in df.columns:
        df[RES_COL] = "<no_resource>"

    df[CASE_COL] = df[CASE_COL].astype("string")
    df[ACT_COL] = df[ACT_COL].astype("string").fillna("<missing_activity>")
    df[RES_COL] = df[RES_COL].astype("string").fillna("<missing_resource>")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", format="mixed")

    before = len(df)
    df = df.dropna(subset=[CASE_COL, TIME_COL]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] {path.name}: dropped {dropped} rows with missing case_id or timestamp")

    df["original_order"] = np.arange(len(df))
    return df[STANDARD_COLUMNS + ["original_order"]]


def add_train_val_test_split(
    df: pd.DataFrame,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    random_state: int,
    case_col: str = CASE_COL,
) -> pd.DataFrame:
    if not np.isclose(train_pct + val_pct + test_pct, 1.0):
        raise ValueError("TRAIN_PCT + VAL_PCT + TEST_PCT must be 1.0.")

    out = df.copy()
    case_ids = out[case_col].dropna().unique()

    if len(case_ids) < 3:
        raise ValueError("At least 3 cases are required for train/val/test split.")

    train_cases, tmp_cases = train_test_split(
        case_ids,
        train_size=train_pct,
        random_state=random_state,
        shuffle=True,
    )

    relative_val_pct = val_pct / (val_pct + test_pct)
    val_cases, test_cases = train_test_split(
        tmp_cases,
        train_size=relative_val_pct,
        random_state=random_state,
        shuffle=True,
    )

    train_cases = set(train_cases)
    val_cases = set(val_cases)
    test_cases = set(test_cases)

    out["split"] = "test"
    out.loc[out[case_col].isin(train_cases), "split"] = "train"
    out.loc[out[case_col].isin(val_cases), "split"] = "val"
    out.loc[out[case_col].isin(test_cases), "split"] = "test"

    return out


def validate_existing_split(df: pd.DataFrame, file_name: str = "dataset") -> None:
    if "split" not in df.columns:
        raise ValueError(f"{file_name}: missing required 'split' column.")

    values = set(df["split"].dropna().astype(str).unique())
    invalid = values - VALID_SPLITS
    if invalid:
        raise ValueError(f"{file_name}: invalid split values found: {sorted(invalid)}")

    missing = VALID_SPLITS - values
    if missing:
        raise ValueError(f"{file_name}: missing split partitions: {sorted(missing)}")


def get_level_group_cols(level_name: str) -> list[str]:
    import config as cfg

    return list(cfg.LEVEL_GROUP_COLS.get(level_name, []))


def group_indices(df: pd.DataFrame, level_name: str) -> dict[str, np.ndarray]:
    group_cols = get_level_group_cols(level_name)
    if not group_cols:
        return {"GLOBAL": df.index.to_numpy()}

    if any(col not in df.columns for col in group_cols):
        return {}

    groups = df.groupby(group_cols, dropna=False, observed=True).groups
    return {str(key): indices.to_numpy() for key, indices in groups.items()}


def iter_dataset_paths(dataset_dict: dict):
    for dataset_name, path in dataset_dict.items():
        path = Path(path)
        if not path.exists():
            print(f"[SKIP] {dataset_name}: file not found at {path}")
            continue
        yield dataset_name, path
