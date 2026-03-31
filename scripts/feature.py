"""
feature.py — 셀 단위 피처 추출 및 저장

입력:
  preprocessed_data/batch*_preprocessed_cycles_all.csv  (preprocess.py 출력)
  preprocessed_data/qdlin.csv                            (load_qdlin.py 출력)

출력:
  preprocessed_data/features.csv
    인덱스: cell_uid, batch_name
    피처:
      cycle_life, charging_policy,
      c1, c2,                         (policy에서 파싱)
      qdlin_010_min, qdlin_010_var,   (cycle 10 Qdlin 통계)
      qdlin_100_min, qdlin_100_var,   (cycle 100 Qdlin 통계)
      delta_q_min,   delta_q_var      (Qdlin_100 - Qdlin_010 통계)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PREPROCESSED_DIR = Path(__file__).parent.parent / 'preprocessed_data'
QDLIN_PATH       = PREPROCESSED_DIR / 'qdlin.csv'
OUTPUT_PATH      = PREPROCESSED_DIR / 'features.csv'

QDLIN_10_COLS  = [f'qdlin_010_{k:03d}' for k in range(1000)]
QDLIN_100_COLS = [f'qdlin_100_{k:03d}' for k in range(1000)]


def parse_policy(policy: str) -> tuple[float, float]:
    """
    '3.6C(80%)-3.6C' → (3.6, 3.6)
    '4C(80%)-2C'     → (4.0, 2.0)
    파싱 실패 시 (nan, nan)
    """
    nums = re.findall(r'(\d+(?:\.\d+)?)C', str(policy))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), np.nan
    return np.nan, np.nan


def load_summary(preprocessed_dir: Path) -> pd.DataFrame:
    """preprocessed_data 아래 summary CSV 로드 (배치별 or 통합 파일 자동 감지)"""
    # 배치별 파일 우선
    files = sorted(preprocessed_dir.glob('batch*_preprocessed_cycles_*.csv'))
    # 없으면 통합 파일
    if not files:
        files = sorted(preprocessed_dir.glob('preprocessed_cycles_*.csv'))
    if not files:
        raise FileNotFoundError(f'summary CSV 없음: {preprocessed_dir}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def extract_features(summary_df: pd.DataFrame, qdlin_df: pd.DataFrame) -> pd.DataFrame:
    # 셀 메타 (cell_uid당 1행)
    meta = (
        summary_df
        .drop_duplicates('cell_uid')
        .set_index('cell_uid')[['batch_name', 'cycle_life', 'charging_policy']]
    )

    # C1, C2 파싱
    parsed = meta['charging_policy'].apply(parse_policy)
    meta['c1'] = parsed.apply(lambda x: x[0])
    meta['c2'] = parsed.apply(lambda x: x[1])

    # Qdlin 통계
    q = qdlin_df.set_index('cell_uid')

    arr10  = q[QDLIN_10_COLS].values
    arr100 = q[QDLIN_100_COLS].values
    delta  = arr100 - arr10

    delta_var = delta.var(axis=1)

    qdlin_stats = pd.DataFrame({
        'qdlin_010_min'  : arr10.min(axis=1),
        'qdlin_010_var'  : arr10.var(axis=1),
        'qdlin_100_min'  : arr100.min(axis=1),
        'qdlin_100_var'  : arr100.var(axis=1),
        'delta_q_min'    : delta.min(axis=1),
        'delta_q_var'    : delta_var,
        'log_delta_q_var': np.log(delta_var + 1e-10),
    }, index=q.index)

    result = meta.join(qdlin_stats, how='inner')
    result.index.name = 'cell_uid'
    return result.reset_index().set_index(['cell_uid', 'batch_name'])


def main() -> None:
    print('summary 로드 중...')
    summary_df = load_summary(PREPROCESSED_DIR)

    print('qdlin 로드 중...')
    qdlin_df = pd.read_csv(QDLIN_PATH)

    print('피처 추출 중...')
    features = extract_features(summary_df, qdlin_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)
    print(f'저장 완료: {OUTPUT_PATH}  shape={features.shape}')
    print(features.head().to_string())


if __name__ == '__main__':
    main()
