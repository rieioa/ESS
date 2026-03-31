"""
_feature.py — 보조 cycle_life 계산

각 셀의 초기 QDischarge 대비 82% 아래로 처음 떨어지는 사이클을
cycle_life_2로 계산해 preprocessed_data/cycle_life_2.csv에 저장.

Batch 2는 EOL 기준이 다를 수 있으므로 별도 추적용.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import extract_summary_rows

ARCHIVE_DIR      = Path(__file__).parent.parent / 'data'
PREPROCESSED_DIR = Path(__file__).parent.parent / 'preprocessed_data'
FEATURES_PATH    = PREPROCESSED_DIR / 'features.csv'

BATCH2_MAT = ARCHIVE_DIR / '2018-02-20_batchdata_updated_struct_errorcorrect.mat'


def load_batch2_raw() -> pd.DataFrame:
    """Batch 2 raw summary만 추출 (스무딩 없음)"""
    return extract_summary_rows(BATCH2_MAT, drop_missing_cycle_life=False)


def compute_cycle_life_at_threshold(
    summary_df: pd.DataFrame,
    threshold: float = 0.88,
) -> pd.DataFrame:
    """
    각 셀에서 초기 QDischarge(cycle 2~6 중앙값)의 threshold 이하로
    처음 떨어지는 사이클 번호를 cycle_life_2로 반환.
    끝까지 안 떨어지면 NaN.
    """
    qd_col = 'QDischarge' if 'QDischarge' in summary_df.columns else 'QD'
    records = []

    for cell_uid, g in summary_df.groupby('cell_uid'):
        g          = g.sort_values('cycle')
        initial_qd = g[g['cycle'].between(2, 6)][qd_col].median()
        cutoff     = threshold

        below        = g[g[qd_col] < cutoff]
        cycle_life_2 = int(below['cycle'].iloc[0]) if len(below) > 0 else np.nan

        batch_name = g['batch_name'].iloc[0]
        records.append({
            'cell_uid'    : cell_uid,
            'batch_name'  : batch_name,
            'cycle_life_2': cycle_life_2,
        })

    return pd.DataFrame(records).set_index(['cell_uid', 'batch_name'])


def main() -> None:
    print('Batch 2 raw summary 로드 중...')
    summary_df = load_batch2_raw()

    print('cycle_life_2 계산 중...')
    batch2_result = compute_cycle_life_at_threshold(summary_df, threshold=0.88)

    # features.csv 로드 후 Batch 2 셀의 cycle_life를 cycle_life_2로 교체
    features = pd.read_csv(FEATURES_PATH)
    features = features.set_index(['cell_uid', 'batch_name'])

    batch2_result = batch2_result[~batch2_result['cycle_life_2'].isna()]
    features.loc[features.index.isin(batch2_result.index), 'cycle_life'] = \
        batch2_result['cycle_life_2']

    features.reset_index().to_csv(FEATURES_PATH, index=False)
    print(f'features.csv 업데이트 완료 (Batch 2 {len(batch2_result)}개 셀)')
    print(features.loc[features.index.get_level_values('batch_name') == '2018-02-20', 'cycle_life'])


if __name__ == '__main__':
    main()
