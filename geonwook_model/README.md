# geonwook_model

`geonwook_model`은 현재 저장소 기준으로 재현된 배터리 수명 예측 실험 패키지다.

목적:

- `2017-05-12` batch로 학습
- `2018-02-20`, `2018-04-12` batch로 평가
- raw `.mat` 파일에서 직접 feature 생성
- 아래 3개 feature만 사용
  - `DeltaQ_var`
  - `charge_time_avg`
  - `temp_integral`

모델:

- `StandardScaler -> Ridge(alpha=1.0)`
- 학습 타깃은 `log(cycle_life)`
- 예측 시 `exp()`로 원래 scale의 `cycle_life`로 복원

## 파일 구성

- [model.py](/Users/geonwook/workspace/ESS/geonwook_model/model.py)
  - 최종 모델 정의
- [feature.py](/Users/geonwook/workspace/ESS/geonwook_model/feature.py)
  - raw `.mat`에서 feature 생성
- [eval.py](/Users/geonwook/workspace/ESS/geonwook_model/eval.py)
  - train/test 평가 실행

## 전제조건

프로젝트 루트에서 실행한다고 가정한다.

필수 데이터 파일:

- `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-04-12_batchdata_updated_struct_errorcorrect.mat`

필수 Python 패키지:

- `numpy`
- `pandas`
- `scikit-learn`
- `h5py`
- `matplotlib`

권장 사항:

- 가상환경 사용
- `MPLCONFIGDIR`를 쓰기 가능한 디렉터리로 지정
  - 예: `/tmp/mpl`

## 설치 예시

이미 프로젝트 가상환경이 있으면 그 환경을 사용하면 된다.

예시:

```bash
python -m pip install numpy pandas scikit-learn h5py matplotlib
```

또는 가상환경 사용 시:

```bash
.venv/bin/pip install numpy pandas scikit-learn h5py matplotlib
```

## 실행 방법

프로젝트 루트에서 실행:

```bash
env MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m geonwook_model.eval
```

가상환경 이름이 다르면 같은 의미로 바꿔서 실행하면 된다.

예시:

```bash
env MPLCONFIGDIR=/tmp/mpl python -m geonwook_model.eval
```

## 내부 동작

### 1. 데이터 필터

현재 구현은 각 batch에서 아래 조건을 적용한다.

- `cycle_life < 1100`
- 필요한 raw signal이 존재하는 셀만 유지

### 2. feature 정의

#### `DeltaQ_var`

- cycle 10의 `Qdlin`과 cycle 100의 `Qdlin`을 읽는다.
- `DeltaQ = Qdlin_100 - Qdlin_10`
- `DeltaQ_var = log10(abs(var(DeltaQ)) + 1e-12)`

#### `charge_time_avg`

- cycle `2~6` 구간의 `chargetime` 평균

#### `temp_integral`

- cycle `2~100` 구간 각각에 대해 raw temperature 시계열 `T(t)`를 시간축 `t`에 대해 적분
- 각 cycle 적분값을 전부 합산

## 출력

`eval.py`는 아래 정보를 stdout으로 출력한다.

- train batch 이름과 row 수
- 사용 feature 목록
- `log_target=True/False`
- train MAPE
- batch2 MAPE
- batch3 MAPE

출력 예시:

```text
=== geonwook_model evaluation ===
train_batch=2017-05-12 rows=41
features=['DeltaQ_var', 'charge_time_avg', 'temp_integral']
log_target=True
train_mape=7.394056
test_batch=2018-02-20 rows=37 mape=10.781592
test_batch=2018-04-12 rows=30 mape=9.564545
```

## 해석 시 주의

- 이 모델은 논문 완전 재현용이 아니다.
- 현재 저장소의 로컬 데이터 구성과 필터 기준에서 잘 작동한 조합을 정리한 것이다.
- 특히 `charge_time_avg`, `temp_integral`은 실험 조건 차이를 반영할 수 있으므로, 다른 데이터셋에 그대로 일반화된다고 가정하면 안 된다.
- 논문 재현 목적이라면 batch 재구성과 target semantics를 별도로 확인해야 한다.

## 팀 공용 사용 팁

- 절대경로 대신 프로젝트 루트 기준 상대경로를 유지한다.
- 공용 문서에는 실행한 Python 환경과 패키지 버전을 함께 남긴다.
- 다른 사람이 같은 결과를 재현하려면:
  - 같은 `.mat` 파일
  - 같은 cutoff 조건
  - 같은 feature 3개
  - 같은 모델 하이퍼파라미터
  - 같은 `log(cycle_life)` 타깃 설정
    이 네 가지가 모두 같아야 한다.
