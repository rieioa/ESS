from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_feature_engineering_calibration import build_expanded_features
from run_paper_like_elasticnet import MAT_FILES
from geonwook_model.model import FEATURE_COLUMNS


FILTER_MAX_CYCLE_LIFE = 1100
BATCH1 = "2017-05-12"
BATCH2 = "2018-02-20"
BATCH3 = "2018-04-12"
BATCH3_PATH = ROOT / "data" / "2018-04-12_batchdata_updated_struct_errorcorrect.mat"


def build_dataset(batch_name: str) -> pd.DataFrame:
    if batch_name == BATCH3:
        mat_path = BATCH3_PATH
    else:
        mat_path = MAT_FILES[batch_name]

    df = build_expanded_features(batch_name, mat_path)
    df = df[df["cycle_life"] < FILTER_MAX_CYCLE_LIFE].copy()
    keep_cols = ["cell_uid", "batch_name", "cycle_life", *FEATURE_COLUMNS]
    return df[keep_cols].dropna(subset=keep_cols[2:]).reset_index(drop=True)
