# `scripts/preprocess.py` 사용 가이드

## 목적

`scripts/preprocess.py`는 Stanford battery `.mat` 파일을 읽어 전처리된 `.csv` 파일로 저장하는 스크립트다.

- 기본 동작: `data/*.mat` 전체를 읽어 배치별 CSV 생성
- 선택 동작: 특정 `.mat` 파일만 지정해서 처리
- 선택 동작: `--start-cycle`, `--end-cycle`로 원하는 cycle 범위만 처리
- 출력 형식: 기본은 long-form CSV, `--wide` 사용 시 wide-form CSV
- 기본 출력 위치: `preprocessed_data/`

## 실행 전 준비

프로젝트 루트에서 실행하는 것을 기준으로 한다.

예시:

```bash
cd /path/to/ESS
```

Python 환경에는 최소한 아래 패키지가 필요하다.

- `mat73`
- `numpy`
- `pandas`

가상환경을 쓰는 경우 예시는 아래 둘 중 하나로 통일해서 사용하면 된다.

```bash
python scripts/preprocess.py
```

또는

```bash
.venv/bin/python scripts/preprocess.py
```

## 입력 데이터 위치

기본적으로 스크립트는 `data/` 디렉터리 아래의 `.mat` 파일을 찾는다.

- 기본 입력 경로: `data/*.mat`
- 다른 디렉터리를 쓰려면 `--data-dir` 사용

예시:

```bash
python scripts/preprocess.py --data-dir /path/to/mat_files
```

## 기본 실행

아무 파일도 넘기지 않으면 `data/*.mat` 전체를 읽어서 배치별 CSV를 만든다.

```bash
python scripts/preprocess.py
```

생성 예시:

- `preprocessed_data/batch1_preprocessed_cycles_all.csv`
- `preprocessed_data/batch2_preprocessed_cycles_all.csv`
- `preprocessed_data/batch3_preprocessed_cycles_all.csv`

## 특정 파일만 실행

원하는 `.mat` 파일만 인자로 넘기면 해당 파일만 전처리한다.

```bash
python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat
```

여러 파일도 한 번에 넣을 수 있다.

```bash
python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  data/2018-02-20_batchdata_updated_struct_errorcorrect.mat
```

생성 파일명은 입력 파일명에서 배치 이름을 추론해 아래 형식으로 저장된다.

- `preprocessed_data/{batch_alias}_preprocessed_cycles_all.csv`

현재 매핑:

- `2017-05-12...mat` -> `batch1`
- `2018-02-20...mat` -> `batch2`
- `2018-04-12...mat` -> `batch3`

## 특정 cycle 범위만 전처리

특정 cycle 범위만 사용하려면 `--start-cycle`과 `--end-cycle`를 함께 넘긴다.

```bash
python scripts/preprocess.py \
  --start-cycle 1 \
  --end-cycle 100
```

생성 예시:

- `preprocessed_data/batch1_preprocessed_cycles_001_100.csv`
- `preprocessed_data/batch2_preprocessed_cycles_001_100.csv`
- `preprocessed_data/batch3_preprocessed_cycles_001_100.csv`

주의:

- `--start-cycle`만 주거나 `--end-cycle`만 주면 에러가 난다.
- 두 옵션은 반드시 같이 써야 한다.
- 지정한 범위의 마지막 cycle까지 데이터가 없는 셀은 결과에서 제외될 수 있다.

## wide 형식으로 저장

기본 출력은 long-form이다.

- long-form: 한 행이 한 셀의 한 cycle
- wide-form: 한 행이 한 셀

wide-form으로 저장하려면 `--wide`를 사용한다.

```bash
python scripts/preprocess.py \
  --start-cycle 1 \
  --end-cycle 100 \
  --wide
```

## `cycle_life` 결측 셀 포함

기본적으로 `cycle_life`가 없는 셀은 제외한다.

결측 타깃도 남기려면 `--include-missing-targets`를 사용한다.

```bash
python scripts/preprocess.py \
  data/2018-02-20_batchdata_updated_struct_errorcorrect.mat \
  --include-missing-targets
```

## 출력 경로 직접 지정

`--output`을 사용하면 출력 위치를 직접 지정할 수 있다.

```bash
python scripts/preprocess.py \
  --output artifacts/preprocessed_batch_all.csv
```

이 경우에는 배치별 개별 파일이 아니라, 처리 결과 전체를 하나의 CSV로 저장한다.

특징:

- `--output results/foo.csv`처럼 `.csv` 파일명을 주면 해당 파일로 저장
- `--output results/foo`처럼 확장자가 없으면 `results/foo.csv`로 저장
- 입력 파일을 직접 넘기는 경우에는 `--output` 경로 자체보다 그 상위 디렉터리가 사용되며, 실제 파일명은 배치별 규칙에 따라 생성된다

예시:

```bash
python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  --output custom_outputs/out.csv
```

실제 생성 예시:

- `custom_outputs/batch1_preprocessed_cycles_all.csv`

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

wide-form에서는 각 signal이 `*_cycle_001` 같은 컬럼으로 펼쳐진다.

예시:

- `QDischarge_cycle_001`
- `IR_cycle_001`
- `was_imputed_cycle_001`

## 전처리에서 실제로 하는 일

스크립트는 아래 순서로 데이터를 정리한다.

- 기본값 기준 `cycle_life` 결측 셀 제외
- summary 값이 모두 0인 row 제거
- cycle 기준 정렬 및 중복 cycle 제거
- 지정 범위에 맞춰 cycle index 재구성
- 물리적으로 비정상적인 값은 `NaN` 처리
- 일부 signal에 대해 Hampel filter로 spike 감지
- 결측치 또는 spike 구간은 인접 cycle 값을 기준으로 보간
- 보정이 발생한 row는 `was_imputed=True`

참고:

- `chargetime`은 spike 제거 없이 숫자형 변환만 수행한다.
- 신호값이 전부 비어 있는 경우에는 보간 과정에서 0으로 채워질 수 있다.

## 원본 데이터 변경 여부

원본 `.mat` 파일은 수정하지 않는다.

- 원본 유지: `data/*.mat`
- 생성 대상: `preprocessed_data/*.csv` 또는 `--output`으로 지정한 경로

## 자주 쓰는 명령

전체 배치, 전체 cycle:

```bash
python scripts/preprocess.py
```

전체 배치, 1~100 cycle:

```bash
python scripts/preprocess.py \
  --start-cycle 1 \
  --end-cycle 100
```

특정 배치만, 1~100 cycle:

```bash
python scripts/preprocess.py \
  data/2017-05-12_batchdata_updated_struct_errorcorrect.mat \
  --start-cycle 1 \
  --end-cycle 100
```

wide 형식으로 저장:

```bash
python scripts/preprocess.py \
  --start-cycle 1 \
  --end-cycle 100 \
  --wide
```

출력 파일 하나로 저장:

```bash
python scripts/preprocess.py \
  --start-cycle 1 \
  --end-cycle 100 \
  --output artifacts/preprocessed_cycles_001_100.csv
```

## 팀 공용 사용 시 권장 사항

- 프로젝트 루트에서 실행한다.
- 문서, 노트북, 실험 로그에는 절대경로 대신 상대경로를 기록한다.
- 결과 재현이 필요하면 사용한 옵션(`--start-cycle`, `--end-cycle`, `--wide`, `--include-missing-targets`, `--output`)을 함께 남긴다.
- 공용 산출물은 `preprocessed_data/` 또는 별도 산출물 디렉터리로 정리해 개인 로컬 경로 의존성을 없앤다.
