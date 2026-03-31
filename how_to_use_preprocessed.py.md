# `scripts/preprocess.py` Usage

## 목적

`scripts/preprocess.py`는 Stanford battery `.mat` 파일을 읽어서 전처리된 `.csv` 파일로 내보내는 스크립트다.

- 기본 동작: 각 입력 `.mat` 파일 전체 cycle 전처리
- 선택 동작: `--start-cycle`, `--end-cycle`로 특정 cycle 범위만 전처리
- 출력 위치: `preprocessed_data/`
- 출력 형식: `.csv`

입력 파일 기준 배치 이름은 아래처럼 매핑된다.

- `2017-05-12...mat` -> `batch1`
- `2018-02-20...mat` -> `batch2`
- `2018-04-12...mat` -> `batch3`

## 필요한 환경

`mat73`, `numpy`, `pandas`가 설치된 Python 환경이 필요하다.

현재 작업 기록 기준 실행 예시는 아래 가상환경을 사용했다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python
```

## 기본 실행

아무 입력 파일도 주지 않으면 `data/*.mat` 전체를 읽어서 배치별 CSV를 만든다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py
```

생성 예시:

- `preprocessed_data/batch1_preprocessed_cycles_all.csv`
- `preprocessed_data/batch2_preprocessed_cycles_all.csv`
- `preprocessed_data/batch3_preprocessed_cycles_all.csv`

## 특정 파일만 실행

원하는 `.mat` 파일만 인자로 넘기면 그 파일만 전처리한다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat
```

생성 예시:

- `preprocessed_data/batch1_preprocessed_cycles_all.csv`

여러 파일도 한 번에 넣을 수 있다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  data/2018-02-20_batchdata_updated_struct_errorcorrect.mat \
  data/2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

## 1~100 cycle만 전처리

특정 cycle 범위를 쓰려면 `--start-cycle`, `--end-cycle`를 같이 넘긴다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  data/2018-02-20_batchdata_updated_struct_errorcorrect.mat \
  data/2018-04-12_batchdata_updated_struct_errorcorrect.mat \
  --start-cycle 1 --end-cycle 100
```

생성 예시:

- `preprocessed_data/batch1_preprocessed_cycles_001_100.csv`
- `preprocessed_data/batch2_preprocessed_cycles_001_100.csv`
- `preprocessed_data/batch3_preprocessed_cycles_001_100.csv`

주의:

- `--start-cycle`만 주거나 `--end-cycle`만 주면 에러가 난다.
- 둘 다 같이 줘야 한다.

## wide 형식으로 저장

기본 출력은 long-form이다.

- long-form: 한 행이 한 셀의 한 cycle
- wide-form: 한 행이 한 셀

wide-form으로 저장하려면 `--wide`를 붙인다.

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  --start-cycle 1 --end-cycle 100 \
  --wide
```

## cycle_life 결측 셀 포함

기본적으로 `cycle_life`가 없는 셀은 제외한다.

결측 타깃 셀도 포함하려면:

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2018-02-20_batchdata_updated_struct_errorcorrect.mat \
  --include-missing-targets
```

## 출력 컬럼

기본 long-form CSV 컬럼은 아래와 같다.

- `file_name`
- `batch_name`
- `cell_uid`
- `cell_local_id`
- `cycle`
- `cycle_life`
- `charging_policy`
- `QDischarge`
- `QCharge`
- `IR`
- `Tmax`
- `Tavg`
- `Tmin`
- `chargetime`
- `was_imputed`

원본 데이터 컬럼명은 그대로 유지한다.

## 전처리에서 실제로 하는 일

- `cycle_life` 결측 셀은 기본 제외
- all-zero summary row 제거
- cycle을 지정 범위로 재정렬
- 비정상 물리값을 `NaN` 처리
- spike를 Hampel filter로 감지
- spike 또는 결측 지점은 주변 cycle 값으로 대체
- 대체 또는 보정이 발생한 행은 `was_imputed=True`

## 원본 데이터 변경 여부

원본 `.mat` 파일은 수정하지 않는다.

- 변경 대상: `preprocessed_data/*.csv`
- 원본 유지: `data/*.mat`

## 자주 쓰는 예시

전체 배치, 전체 cycle:

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py
```

전체 배치, 1~100 cycle:

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  --start-cycle 1 --end-cycle 100
```

batch1만, 1~100 cycle:

```bash
/Users/geonwook/workspace/skala_archive/ml_ai/data_mini_project_배기주/.venv/bin/python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  --start-cycle 1 --end-cycle 100
```
