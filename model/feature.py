from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_BATCH = "2017-05-12"
TEST2_BATCH = "2018-02-20"
TEST3_BATCH = "2018-04-12"
FILTER_MAX_CYCLE_LIFE = 1100

MAT_FILES = {
    TRAIN_BATCH: DATA_DIR / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    TEST2_BATCH: DATA_DIR / "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    TEST3_BATCH: DATA_DIR / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

FEATURE_COLUMNS = ["DeltaQ_var", "charge_time_avg", "temp_integral"]
EPS = 1e-12


@dataclass
class CellData:
    cycle_life: float
    cycles: list[dict[str, np.ndarray]]
    summary: dict[str, np.ndarray]


def deref_scalar(f: h5py.File, ref) -> float:
    arr = np.asarray(f[ref][()]).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def deref_vector(f: h5py.File, ref) -> np.ndarray:
    return np.asarray(f[ref][()]).reshape(-1).astype(float, copy=False)


def read_cycle_group(f: h5py.File, cycle_group_ref) -> list[dict[str, np.ndarray]]:
    cycle_group = f[cycle_group_ref]
    n_cycles = cycle_group["t"].shape[0]
    cycles: list[dict[str, np.ndarray]] = []
    for idx in range(n_cycles):
        cycles.append(
            {
                "t": deref_vector(f, cycle_group["t"][idx, 0]),
                "T": deref_vector(f, cycle_group["T"][idx, 0]),
                "Qdlin": deref_vector(f, cycle_group["Qdlin"][idx, 0]),
            }
        )
    return cycles


def read_summary_group(f: h5py.File, summary_ref) -> dict[str, np.ndarray]:
    summary = f[summary_ref]
    return {
        "QDischarge": np.asarray(summary["QDischarge"][()]).reshape(-1).astype(float, copy=False),
        "IR": np.asarray(summary["IR"][()]).reshape(-1).astype(float, copy=False),
        "chargetime": np.asarray(summary["chargetime"][()]).reshape(-1).astype(float, copy=False),
    }


def find_time_gap_cycle_indices(cycles: list[dict[str, np.ndarray]]) -> list[int]:
    bad_indices: list[int] = []
    for idx in range(1, len(cycles)):
        t = cycles[idx]["t"]
        if t.size < 2:
            continue
        dt = np.diff(t)
        finite_dt = dt[np.isfinite(dt)]
        if finite_dt.size == 0:
            continue
        mean_dt = float(np.mean(finite_dt))
        if mean_dt <= 0:
            continue
        if float(np.max(finite_dt)) > 5.0 * mean_dt:
            bad_indices.append(idx)
    return bad_indices


def remove_indices_1d(arr: np.ndarray, bad_indices: list[int]) -> np.ndarray:
    if not bad_indices:
        return arr
    return np.delete(arr, bad_indices)


def cleaned_cell_data(
    f: h5py.File,
    cycle_life_ref,
    cycles_ref,
    summary_ref,
) -> CellData | None:
    cycle_life = deref_scalar(f, cycle_life_ref)
    if not np.isfinite(cycle_life) or cycle_life <= 0:
        return None

    cycles = read_cycle_group(f, cycles_ref)
    summary = read_summary_group(f, summary_ref)

    bad_indices = find_time_gap_cycle_indices(cycles)
    if bad_indices:
        cycles = [cycle for idx, cycle in enumerate(cycles) if idx not in bad_indices]
        summary = {key: remove_indices_1d(value, bad_indices) for key, value in summary.items()}

    if len(cycles) < 100:
        return None
    if any(len(summary[key]) < 100 for key in ("QDischarge", "IR", "chargetime")):
        return None

    return CellData(cycle_life=cycle_life, cycles=cycles, summary=summary)


def compute_features(cell: CellData) -> dict[str, float] | None:
    qdlin_10 = cell.cycles[9]["Qdlin"]
    qdlin_100 = cell.cycles[99]["Qdlin"]
    if qdlin_10.size == 0 or qdlin_100.size == 0 or qdlin_10.size != qdlin_100.size:
        return None

    delta_q = qdlin_100 - qdlin_10
    delta_q_var = float(np.var(delta_q))

    charge_2_6 = cell.summary["chargetime"][1:6]
    if charge_2_6.size != 5:
        return None

    temp_integral = 0.0
    for cyc in cell.cycles[1:100]:
        t = cyc["t"]
        T = cyc["T"]
        if t.size < 2 or T.size < 2 or t.size != T.size:
            return None
        temp_integral += float(np.trapezoid(T, t))

    return {
        "DeltaQ_var": float(np.log10(abs(delta_q_var) + EPS)),
        "charge_time_avg": float(np.mean(charge_2_6)),
        "temp_integral": float(temp_integral),
    }


def extract_batch_features(batch_name: str, mat_path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        cycle_life_ds = batch["cycle_life"]
        cycles_ds = batch["cycles"]
        summary_ds = batch["summary"]

        for cell_index in range(cycle_life_ds.shape[0]):
            cell = cleaned_cell_data(
                f=f,
                cycle_life_ref=cycle_life_ds[cell_index, 0],
                cycles_ref=cycles_ds[cell_index, 0],
                summary_ref=summary_ds[cell_index, 0],
            )
            if cell is None or cell.cycle_life >= FILTER_MAX_CYCLE_LIFE:
                continue

            features = compute_features(cell)
            if features is None:
                continue

            row = {
                "cell_uid": f"{batch_name}_cell_{cell_index:03d}",
                "batch_name": batch_name,
                "cycle_life": float(cell.cycle_life),
            }
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows)


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = extract_batch_features(TRAIN_BATCH, MAT_FILES[TRAIN_BATCH])
    test2_df = extract_batch_features(TEST2_BATCH, MAT_FILES[TEST2_BATCH])
    test3_df = extract_batch_features(TEST3_BATCH, MAT_FILES[TEST3_BATCH])
    return train_df, test2_df, test3_df
