from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import mat73
except ImportError as exc:  # pragma: no cover
    mat73 = None
    MAT73_IMPORT_ERROR = exc
else:
    MAT73_IMPORT_ERROR = None


SUMMARY_COLUMN_MAP = {
    "QDischarge": "QDischarge",
    "QCharge": "QCharge",
    "IR": "IR",
    "Tmax": "Tmax",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "chargetime": "chargetime",
}

SIGNAL_COLUMNS = list(SUMMARY_COLUMN_MAP.keys())
DEFAULT_OUTPUT_DIR = Path("preprocessed_data")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "preprocessed_cycles_all.csv"
BATCH_ALIAS_MAP = {
    "2017-05-12": "batch1",
    "2018-02-20": "batch2",
    "2018-04-12": "batch3",
}
SIGNAL_BOUNDS = {
    "QDischarge": (0.0, 5.0),
    "QCharge": (0.0, 5.0),
    "IR": (0.0, 1.0),
    "Tmax": (-20.0, 100.0),
    "Tavg": (-20.0, 100.0),
    "Tmin": (-50.0, 80.0),
    "chargetime": (0.0, 500.0),
}


def batch_name_from_path(path: Path) -> str:
    suffix = "_batchdata_updated_struct_errorcorrect"
    stem = path.stem
    return stem.replace(suffix, "") if suffix in stem else stem


def scalarize(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def to_1d_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape == ():
        arr = arr.reshape(1)
    return arr.reshape(-1)


def coerce_float(value) -> float:
    value = scalarize(value)
    try:
        return float(value)
    except Exception:
        return np.nan


def load_mat_file(path: str | Path) -> dict:
    if mat73 is None:  # pragma: no cover
        raise ImportError(
            "mat73 is required to read the Stanford battery v7.3 MAT files."
        ) from MAT73_IMPORT_ERROR

    logging.disable(logging.ERROR)
    return mat73.loadmat(str(path))


def iter_batch_cells(batch) -> Iterable[dict]:
    if isinstance(batch, dict):
        keys = list(batch.keys())
        if not keys:
            return
        n_cells = len(batch[keys[0]])
        for idx in range(n_cells):
            yield {key: batch[key][idx] for key in keys}
        return

    if isinstance(batch, (list, tuple, np.ndarray)):
        for cell in batch:
            yield cell
        return

    raise TypeError(f"Unsupported batch container: {type(batch)}")


def extract_summary_rows(
    mat_path: str | Path,
    cycle_window: tuple[int, int] | None = None,
    drop_missing_cycle_life: bool = True,
) -> pd.DataFrame:
    mat_path = Path(mat_path)
    batch_name = batch_name_from_path(mat_path)
    batch = load_mat_file(mat_path)["batch"]

    if cycle_window is None:
        start_cycle, end_cycle = 1, float("inf")
    else:
        start_cycle, end_cycle = cycle_window
    rows: list[dict] = []

    for cell_idx, cell in enumerate(iter_batch_cells(batch)):
        summary = cell.get("summary")
        if not isinstance(summary, dict):
            continue

        cycle_life = coerce_float(cell.get("cycle_life"))
        if drop_missing_cycle_life and np.isnan(cycle_life):
            continue

        arrays = {}
        lengths = []
        for out_col, src_col in SUMMARY_COLUMN_MAP.items():
            if src_col not in summary:
                arrays[out_col] = np.array([], dtype=float)
                continue
            arr = to_1d_array(summary[src_col]).astype(float, copy=False)
            arrays[out_col] = arr
            lengths.append(len(arr))

        if not lengths:
            continue

        n_cycles = min(lengths)
        policy = cell.get("policy_readable") or cell.get("policy") or "unknown"
        cell_uid = f"{batch_name}_cell_{cell_idx:03d}"

        for cycle_idx in range(n_cycles):
            cycle = cycle_idx + 1
            if cycle < start_cycle or cycle > end_cycle:
                continue

            row = {
                "file_name": mat_path.name,
                "batch_name": batch_name,
                "cell_uid": cell_uid,
                "cell_local_id": cell_idx,
                "cycle": cycle,
                "cycle_life": cycle_life,
                "charging_policy": str(policy),
            }
            row.update(
                {
                    name: arrays[name][cycle_idx] if len(arrays[name]) > cycle_idx else np.nan
                    for name in SIGNAL_COLUMNS
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def remove_zero_signal_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    zero_mask = (df[SIGNAL_COLUMNS] == 0).all(axis=1)
    return df.loc[~zero_mask].copy()


def clip_to_physical_bounds(series: pd.Series, signal_name: str) -> pd.Series:
    lower, upper = SIGNAL_BOUNDS[signal_name]
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out[(out < lower) | (out > upper)] = np.nan
    return out


def hampel_filter(series: pd.Series, window: int = 5, n_sigma: float = 3.0) -> pd.Series:
    values = series.astype(float).copy()
    rolling_median = values.rolling(window=window, center=True, min_periods=1).median()
    abs_diff = (values - rolling_median).abs()
    mad = abs_diff.rolling(window=window, center=True, min_periods=1).median()
    threshold = 1.4826 * n_sigma * mad
    outlier_mask = abs_diff > threshold
    values[outlier_mask] = np.nan
    return values


def replace_with_neighbor_values(series: pd.Series) -> pd.Series:
    filled = series.astype(float).copy()

    # Replace missing or spike-marked points with neighboring cycle values.
    filled = filled.interpolate(method="linear", limit_direction="both")
    filled = filled.bfill().ffill()

    if filled.isna().all():
        return filled.fillna(0.0)

    return filled.fillna(filled.median())


def preprocess_signal(series: pd.Series, signal_name: str) -> tuple[pd.Series, pd.Series]:
    bounded = clip_to_physical_bounds(series, signal_name)
    despiked = hampel_filter(bounded)
    was_touched = bounded.isna() | despiked.isna()
    cleaned = replace_with_neighbor_values(despiked)
    return cleaned, was_touched


def preprocess_cell_cycles(
    cell_df: pd.DataFrame,
    cycle_window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    if cycle_window is None:
        start_cycle = int(cell_df["cycle"].min())
        end_cycle = int(cell_df["cycle"].max())
    else:
        start_cycle, end_cycle = cycle_window
    target_cycles = pd.Index(range(start_cycle, end_cycle + 1), name="cycle")

    meta_cols = ["file_name", "batch_name", "cell_uid", "cell_local_id", "cycle_life", "charging_policy"]
    meta = {col: cell_df[col].iloc[0] for col in meta_cols}

    cell_df = (
        cell_df.sort_values("cycle")
        .drop_duplicates(subset="cycle", keep="first")
        .set_index("cycle")
        .reindex(target_cycles)
    )

    for col, value in meta.items():
        cell_df[col] = value

    touch_columns = {}
    for signal_name in SIGNAL_COLUMNS:
        original = cell_df[signal_name]
        cleaned, touched = preprocess_signal(original, signal_name)
        cell_df[signal_name] = cleaned
        touch_columns[f"{signal_name}_was_imputed"] = touched.fillna(True).astype(bool)

    touch_df = pd.DataFrame(touch_columns, index=cell_df.index)
    cell_df["was_imputed"] = touch_df.any(axis=1)
    cell_df = cell_df.reset_index()

    ordered_columns = meta_cols[:4] + ["cycle"] + meta_cols[4:] + SIGNAL_COLUMNS + ["was_imputed"]
    return cell_df[ordered_columns]


def preprocess_cycle_frame(
    cycle_df: pd.DataFrame,
    cycle_window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    if cycle_df.empty:
        ordered_columns = [
            "file_name",
            "batch_name",
            "cell_uid",
            "cell_local_id",
            "cycle",
            "cycle_life",
            "charging_policy",
            *SIGNAL_COLUMNS,
            "was_imputed",
        ]
        return pd.DataFrame(columns=ordered_columns)

    cycle_df = remove_zero_signal_rows(cycle_df)
    groups = []
    for _, cell_df in cycle_df.groupby("cell_uid", sort=True):
        if cycle_window is not None and cell_df["cycle"].max() < cycle_window[1]:
            continue
        groups.append(preprocess_cell_cycles(cell_df, cycle_window=cycle_window))

    if not groups:
        return preprocess_cycle_frame(pd.DataFrame())

    return pd.concat(groups, ignore_index=True)


def load_preprocessed_cycles(
    data_dir: str | Path = "data",
    mat_files: Iterable[str | Path] | None = None,
    cycle_window: tuple[int, int] | None = None,
    drop_missing_cycle_life: bool = True,
) -> pd.DataFrame:
    if mat_files is None:
        data_dir = Path(data_dir)
        resolved_mat_files = sorted(data_dir.glob("*.mat"))
    else:
        resolved_mat_files = [Path(path) for path in mat_files]

    if not resolved_mat_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")

    raw_frames = [
        extract_summary_rows(
            mat_path,
            cycle_window=cycle_window,
            drop_missing_cycle_life=drop_missing_cycle_life,
        )
        for mat_path in resolved_mat_files
    ]
    raw_cycle_df = pd.concat(raw_frames, ignore_index=True)
    return preprocess_cycle_frame(raw_cycle_df, cycle_window=cycle_window)


def to_wide_cycle_frame(cycle_df: pd.DataFrame) -> pd.DataFrame:
    if cycle_df.empty:
        return pd.DataFrame()

    base = cycle_df[["cell_uid", "batch_name", "cell_local_id", "cycle_life", "charging_policy"]]
    base = base.drop_duplicates(subset="cell_uid").set_index("cell_uid")

    wide_parts = []
    for signal_name in SIGNAL_COLUMNS:
        pivot = cycle_df.pivot(index="cell_uid", columns="cycle", values=signal_name)
        pivot.columns = [f"{signal_name}_cycle_{cycle:03d}" for cycle in pivot.columns]
        wide_parts.append(pivot)

    imputed = cycle_df.pivot(index="cell_uid", columns="cycle", values="was_imputed")
    imputed.columns = [f"was_imputed_cycle_{cycle:03d}" for cycle in imputed.columns]
    wide_parts.append(imputed)

    return pd.concat([base, *wide_parts], axis=1).reset_index()


def export_preprocessed_cycles(
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    data_dir: str | Path = "data",
    cycle_window: tuple[int, int] | None = None,
    drop_missing_cycle_life: bool = True,
    wide: bool = False,
) -> pd.DataFrame:
    cycle_df = load_preprocessed_cycles(
        data_dir=data_dir,
        cycle_window=cycle_window,
        drop_missing_cycle_life=drop_missing_cycle_life,
    )
    output_df = to_wide_cycle_frame(cycle_df) if wide else cycle_df

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix and output_path.suffix != ".csv":
        raise ValueError("Only .csv export is supported.")

    if not output_path.suffix:
        output_path = output_path.with_suffix(".csv")

    output_df.to_csv(output_path, index=False)

    return output_df


def export_preprocessed_cycles_by_batch(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = "data",
    mat_files: Iterable[str | Path] | None = None,
    cycle_window: tuple[int, int] | None = None,
    drop_missing_cycle_life: bool = True,
    wide: bool = False,
) -> dict[str, Path]:
    cycle_df = load_preprocessed_cycles(
        data_dir=data_dir,
        mat_files=mat_files,
        cycle_window=cycle_window,
        drop_missing_cycle_life=drop_missing_cycle_life,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}
    for batch_name, batch_df in cycle_df.groupby("batch_name", sort=True):
        output_df = to_wide_cycle_frame(batch_df) if wide else batch_df
        batch_alias = BATCH_ALIAS_MAP.get(batch_name, batch_name)
        suffix = (
            f"{cycle_window[0]:03d}_{cycle_window[1]:03d}"
            if cycle_window is not None
            else "all"
        )
        output_path = output_dir / f"{batch_alias}_preprocessed_cycles_{suffix}.csv"
        output_df.to_csv(output_path, index=False)
        exported_paths[batch_name] = output_path

    return exported_paths


def export_preprocessed_cycles_for_inputs(
    mat_files: Iterable[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cycle_window: tuple[int, int] | None = None,
    drop_missing_cycle_life: bool = True,
    wide: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}
    for mat_file in mat_files:
        mat_path = Path(mat_file)
        cycle_df = extract_summary_rows(
            mat_path=mat_path,
            cycle_window=cycle_window,
            drop_missing_cycle_life=drop_missing_cycle_life,
        )
        cycle_df = preprocess_cycle_frame(cycle_df, cycle_window=cycle_window)
        output_df = to_wide_cycle_frame(cycle_df) if wide else cycle_df

        batch_name = batch_name_from_path(mat_path)
        batch_alias = BATCH_ALIAS_MAP.get(batch_name, batch_name)
        suffix = (
            f"{cycle_window[0]:03d}_{cycle_window[1]:03d}"
            if cycle_window is not None
            else "all"
        )
        output_path = output_dir / f"{batch_alias}_preprocessed_cycles_{suffix}.csv"
        output_df.to_csv(output_path, index=False)
        exported_paths[str(mat_path)] = output_path

    return exported_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the Stanford battery MAT files.")
    parser.add_argument(
        "mat_files",
        nargs="*",
        help="Optional MAT file paths. If omitted, all data/*.mat files are processed.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory that contains the .mat files.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="CSV export path. Defaults to preprocessed_data/preprocessed_cycles_all.csv.",
    )
    parser.add_argument("--start-cycle", type=int)
    parser.add_argument("--end-cycle", type=int)
    parser.add_argument(
        "--include-missing-targets",
        action="store_true",
        help="Keep cells whose cycle_life is missing.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Export one row per cell instead of one row per cycle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.start_cycle is None) ^ (args.end_cycle is None):
        raise ValueError("Both --start-cycle and --end-cycle must be provided together.")
    cycle_window = (
        (args.start_cycle, args.end_cycle)
        if args.start_cycle is not None and args.end_cycle is not None
        else None
    )
    if args.mat_files:
        exported = export_preprocessed_cycles_for_inputs(
            mat_files=args.mat_files,
            output_dir=Path(args.output).parent if Path(args.output).suffix else Path(args.output),
            cycle_window=cycle_window,
            drop_missing_cycle_life=not args.include_missing_targets,
            wide=args.wide,
        )
        for mat_file, output_path in exported.items():
            print(f"{mat_file} -> {output_path}")
    elif Path(args.output).suffix or str(args.output) != str(DEFAULT_OUTPUT_FILE):
        df = export_preprocessed_cycles(
            output_path=args.output,
            data_dir=args.data_dir,
            cycle_window=cycle_window,
            drop_missing_cycle_life=not args.include_missing_targets,
            wide=args.wide,
        )

        print(df.head().to_string(index=False))
        print(f"\nshape={df.shape}")
        print(f"output={Path(args.output) if Path(args.output).suffix else Path(args.output).with_suffix('.csv')}")
    else:
        exported = export_preprocessed_cycles_by_batch(
            output_dir=DEFAULT_OUTPUT_DIR,
            data_dir=args.data_dir,
            cycle_window=cycle_window,
            drop_missing_cycle_life=not args.include_missing_targets,
            wide=args.wide,
        )
        for batch_name, output_path in exported.items():
            print(f"{batch_name} -> {output_path}")


if __name__ == "__main__":
    main()
