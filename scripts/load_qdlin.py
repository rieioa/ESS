"""
archive/*.mat에서 모든 셀의 Qdlin(cycle 10, cycle 100)을 추출해
preprocessed_data/qdlin.csv 로 저장

컬럼 구성:
  cell_uid, batch_name, cycle_life,
  qdlin_010_000 ~ qdlin_010_999,   (10번째 사이클, 1000포인트)
  qdlin_100_000 ~ qdlin_100_999    (100번째 사이클, 1000포인트)
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ARCHIVE_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_PATH = Path(__file__).parent.parent / 'preprocessed_data' / 'qdlin.csv'

BATCH_ALIAS = {  # preprocess.py의 BATCH_ALIAS_MAP과 동일
    '2017-05-12': 'batch1',
    '2018-02-20': 'batch2',
    '2018-04-12': 'batch3',
}

CYCLE_10  = 9    # 0-indexed
CYCLE_100 = 99

N_POINTS  = 1000


def to_fixed_grid(arr: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """가변 길이 배열을 n포인트 고정 그리드로 선형 보간"""
    if len(arr) == n:
        return arr.astype(float)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return interp1d(x_old, arr, bounds_error=False, fill_value='extrapolate')(x_new)


def extract_qdlin_from_mat(mat_path: Path) -> list[dict]:
    stem        = mat_path.stem.replace('_batchdata_updated_struct_errorcorrect', '')
    batch_name  = stem                      # preprocess.py와 동일: "2017-05-12"
    batch_alias = BATCH_ALIAS.get(stem, stem)  # 로그용
    records = []

    with h5py.File(mat_path, 'r') as f:
        cycles_ds = f['batch']['cycles']      # (n_cells, 1) refs
        cl_data   = f['batch']['cycle_life']  # (n_cells, 1) refs
        n_cells   = cycles_ds.shape[0]

        for i in range(n_cells):
            # cycle_life
            try:
                cl = float(f[cl_data[i, 0]][()].flat[0])
            except Exception:
                continue
            if np.isnan(cl):
                continue

            # Qdlin refs for this cell
            try:
                cell_group = f[cycles_ds[i, 0]]       # Group {Qdlin, V, I, ...}
                qdlin_refs = cell_group['Qdlin']       # (n_cycles, 1) refs
            except Exception:
                continue

            n_cycles = qdlin_refs.shape[0]
            if n_cycles <= CYCLE_100:
                logging.warning(f'{batch_alias}_c{i:03d}: 사이클 수 부족 ({n_cycles}), 스킵')
                continue

            try:
                arr10  = f[qdlin_refs[CYCLE_10,  0]][()].flatten()
                arr100 = f[qdlin_refs[CYCLE_100, 0]][()].flatten()
            except Exception:
                continue

            row = {
                'cell_uid'   : f'{batch_name}_cell_{i:03d}',
                'batch_name' : batch_name,
                'cycle_life' : int(cl),
            }
            for k, v in enumerate(to_fixed_grid(arr10)):
                row[f'qdlin_010_{k:03d}'] = v
            for k, v in enumerate(to_fixed_grid(arr100)):
                row[f'qdlin_100_{k:03d}'] = v

            records.append(row)
            print(f'  {row["cell_uid"]} done')

    return records


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(ARCHIVE_DIR.glob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(f'archive에 .mat 파일 없음: {ARCHIVE_DIR}')

    all_records = []
    for mat_path in mat_files:
        print(f'\n[{mat_path.name}] 처리 중...')
        all_records.extend(extract_qdlin_from_mat(mat_path))

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n저장 완료: {OUTPUT_PATH}  shape={df.shape}')


if __name__ == '__main__':
    main()
