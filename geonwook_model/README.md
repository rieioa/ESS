# geonwook_model

`geonwook_model`은 배터리 수명 예측 실험을 정리한 패키지다. 이 문서는 특히 아래 결과를 다른 사람이 그대로 재현하는 방법을 적는다.

- feature: `DeltaQ_var | charge_time_avg | temp_integral`
- model: `Ridge(alpha=1.0)`
- target: `log(cycle_life)`로 학습, 예측 후 `exp()`로 복원
- expected result:
  - `test2_mape = 10.781592`
  - `test3_mape = 9.564545`

## 재현 대상

데이터 분할:

- train batch: `2017-05-12`
- test2 batch: `2018-02-20`
- test3 batch: `2018-04-12`

필터:

- `cycle_life < 1100`
- feature 3개와 `cycle_life`가 모두 존재하는 셀만 사용

재현 시 사용 row 수:

- train: `41`
- test2: `37`
- test3: `30`

## 사용 feature

아래 3개만 쓴다.

- `DeltaQ_var`
- `charge_time_avg`
- `temp_integral`

이 feature들은 `preprocessed_data/features.csv`가 아니라 raw `.mat` 파일에서 직접 계산한다.

구체적으로는 [experiment_feature_engineering_calibration.py](/Users/geonwook/workspace/ESS/scripts/experiment_feature_engineering_calibration.py)의 `build_expanded_features()` 경로를 사용한다.

## 모델 설정

모델은 아래 파이프라인이다.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42)),
])
```

학습 타깃은 `np.log(y_train)`이고, 예측은 `np.exp(pred)`로 원복한다.

## 전제조건

프로젝트 루트에서 실행한다고 가정한다.

필수 데이터 파일:

- `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-04-12_batchdata_updated_struct_errorcorrect.mat`

필수 Python 패키지는 [requirements.txt](/Users/geonwook/workspace/ESS/geonwook_model/requirements.txt)를 따른다.

중요:

- 이 코드는 raw `.mat` 파일이 있어야만 재현된다.
- 따라서 코드만 push하고 데이터가 없으면 다른 사람은 같은 결과를 재현할 수 없다.
- 다른 사람이 따라 하게 하려면 저장소에 위 파일이 이미 있어야 하거나, 별도 다운로드 위치를 함께 안내해야 한다.

설치:

```bash
python -m pip install -r geonwook_model/requirements.txt
```

가상환경 사용 시:

```bash
.venv/bin/pip install -r geonwook_model/requirements.txt
```

## 재현 명령

현재 `geonwook_model/eval.py`는 위 Ridge 3-feature 실험과 동일하지 않다. 정확히 같은 수치를 재현하려면 아래 명령을 프로젝트 루트에서 실행한다.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

root = Path('/Users/geonwook/workspace/ESS')
sys.path.insert(0, str(root / 'scripts'))

from experiment_feature_engineering_calibration import build_expanded_features
from run_paper_like_elasticnet import MAT_FILES, TRAIN_BATCH, TEST_BATCH, mape

TEST_BATCH3 = '2018-04-12'
MAT3 = root / 'data' / '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
FEATURES = ['DeltaQ_var', 'charge_time_avg', 'temp_integral']
FILTER_MAX_CYCLE_LIFE = 1100

train_df = build_expanded_features(TRAIN_BATCH, MAT_FILES[TRAIN_BATCH])
test2_df = build_expanded_features(TEST_BATCH, MAT_FILES[TEST_BATCH])
test3_df = build_expanded_features(TEST_BATCH3, MAT3)

train_df = train_df[train_df['cycle_life'] < FILTER_MAX_CYCLE_LIFE].dropna(subset=FEATURES + ['cycle_life']).copy()
test2_df = test2_df[test2_df['cycle_life'] < FILTER_MAX_CYCLE_LIFE].dropna(subset=FEATURES + ['cycle_life']).copy()
test3_df = test3_df[test3_df['cycle_life'] < FILTER_MAX_CYCLE_LIFE].dropna(subset=FEATURES + ['cycle_life']).copy()

X_train = train_df[FEATURES].to_numpy(dtype=float)
y_train = train_df['cycle_life'].to_numpy(dtype=float)
X_test2 = test2_df[FEATURES].to_numpy(dtype=float)
y_test2 = test2_df['cycle_life'].to_numpy(dtype=float)
X_test3 = test3_df[FEATURES].to_numpy(dtype=float)
y_test3 = test3_df['cycle_life'].to_numpy(dtype=float)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0, random_state=42)),
])
model.fit(X_train, np.log(y_train))
pred2 = np.exp(model.predict(X_test2))
pred3 = np.exp(model.predict(X_test3))

print('train_rows', len(train_df))
print('test2_rows', len(test2_df))
print('test3_rows', len(test3_df))
print('test2_mape', f'{mape(y_test2, pred2):.6f}')
print('test3_mape', f'{mape(y_test3, pred3):.6f}')
PY
```

정상 재현 시 출력은 아래와 같아야 한다.

```text
train_rows 41
test2_rows 37
test3_rows 30
test2_mape 10.781592
test3_mape 9.564545
```

`python -m geonwook_model.eval`도 같은 결과를 내도록 맞춰져 있다.

```bash
.venv/bin/python -m geonwook_model.eval
```

## 파일 관계

- [experiment_feature_engineering_calibration.py](/Users/geonwook/workspace/ESS/scripts/experiment_feature_engineering_calibration.py)
  - `DeltaQ_var`, `charge_time_avg`, `temp_integral` 생성
- [run_paper_like_elasticnet.py](/Users/geonwook/workspace/ESS/scripts/run_paper_like_elasticnet.py)
  - raw `.mat` 파싱 유틸과 `mape`
- [grid_search_result.log](/Users/geonwook/workspace/ESS/grid_search_result.log)
  - 동일 설정 기록 포함
- [final/data/grid_search_feature_models_summary.txt](/Users/geonwook/workspace/ESS/final/data/grid_search_feature_models_summary.txt)
  - 요약 결과 기록 포함

## 주의

- 이 결과는 현재 `geonwook_model/eval.py` 기본 설정을 실행한 결과가 아니다.
- 현재 `geonwook_model/eval.py`는 바로 그 Ridge 3-feature 실험을 실행하도록 맞춰져 있다.
- raw `.mat`를 직접 읽기 때문에 실행 시간이 짧지 않다.
- 모델 저장 경로에 권한이 없으면 `/tmp/geonwook_model/`로 fallback 저장될 수 있다.
