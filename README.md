# ESS 배터리 수명 예측 
초기 사이클(1~100회) 데이터만을 활용하여 리튬이온 배터리의 전체 수명(Cycle Life)을 조기에 예측하는 회귀 모델을 개발한다.
EDA를 통해 열화 패턴과 충전 조건이 수명에 미치는 영향을 분석하고, 이를 바탕으로 유의미한 피처를 선별하여 ESS 배터리 관리 및 교체 의사결정에 활용 가능한 예측 모델을 구축한다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측)

## 파일 구조
```
├── data/
│   ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat   # Batch1
│   ├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat   # Batch2
│   └── 2018-04-12_batchdata_updated_struct_errorcorrect.mat   # Batch3
├── EDA/
│   ├── ESS_EDA.ipynb          # 메인 EDA (Q1~Q5 전체)
│   ├── Batch2.ipynb
│   ├── Batch3.ipynb
│   └── eda_Q5.ipynb
├── preprocessed_data/
│   ├── batch1_preprocessed_cycles_001_100.csv
│   ├── batch2_preprocessed_cycles_001_100.csv
│   └── batch3_preprocessed_cycles_001_100.csv
├── scripts/
│   ├── preprocess.py
│   ├── load_qdlin.py
│   ├── feature.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── test.py
├── figures/
│   └── eda_day1/              # EDA 결과 시각화 이미지
├── requirements.txt
└── README.md
```

# Git 협업 가이드

## 초기 설정 (최초 1회)
```bash
git clone https://github.com/rieioa/ESS
cd ESS
git config --global push.autoSetupRemote true
```

> 히스토리 충돌 시: `git pull origin main --allow-unrelated-histories`

## 작업 흐름
```bash
# 1. main 최신화
git checkout main
git pull origin main

# 2. 브랜치 생성
git checkout -b branch_name

# 3. 작업 & 커밋
git add .
git commit -m "message"

# 4. push
git push

# 5. GitHub에서 PR 생성 → 리뷰 → Merge
```

## 다음 작업 시
```bash
git checkout main
git pull origin main
git checkout -b new_branch_name
```

## EDA
- **Cycle Life 분포**
  - Batch1: 평균 845 cycle, 장수명(≥1000) 21.7%, 단수명(<500) 0개
  - Batch2: 평균 566 cycle, 장수명 7.7%, 단수명 71.8% (고의로 단수명 셀 위주 구성, 실험 환경 상이)
  - Batch3: 평균 1060 cycle, 장수명(≥1000) 52.3%, 단수명(<500) 0개
  - 핵심 발견 : Batch별 EOL cutoff 기준이 다르므로(Batch1·3: 0.88Ah/82%SOH, Batch2: 0.825Ah/75%SOH) 배치 간 cycle_life 직접 비교는 불가

- **열화 곡선 분석**
  - 장수명 셀은 완만한 선형 감소, 단수명 셀은 초반부터 급격한 용량 감소
  - Knee point 존재 확인 — QD가 일정 구간까지 선형 감소 후 특정 시점에 가속하여 급격히 하락하는 패턴이 세 배치 공통으로 관찰됨
  - 핵심 발견 : 초기 사이클에서의 열화 속도 차이가 최종 수명을 결정하는 핵심 신호이며, Batch3에서 이상치 셀 3개는 초기부터 급격한 QD 감소를 보임

- **ΔQ(V) 곡선 분석**
  - 2.3~2.4V 구간에서 최대 감소, 단수명 셀일수록 2.4~2.6V 구간 감소 폭이 큼(-0.04~-0.06 수준)
  - 핵심 발견 : 초기 10~100 사이클의 ΔQ(V) 패턴만으로 수명 예측 가능. dQ_min(r=+0.827)과 log_dQ_var(r=-0.851)가 cycle life와 가장 강한 상관관계를 보이는 핵심 피처

- **충전 속도(C-rate)와 수명의 관계**
  - 낮은 C-rate 정책일수록 수명이 긴 경향이 세 배치 공통으로 나타남 (4C→8C 구간에서 약 40% 수명 단축)
  - 핵심 발견 : C-rate 절댓값보다 SOC가 높아지는 구간에서의 충전 속도가 수명에 더 큰 영향을 미치며, mean_chargetime이 배치를 막론하고 수명과 가장 안정적인 상관관계(r≈+0.58~0.64)를 보임

## Modeling 

### 피처 엔지니어링 전략
EDA 결과를 바탕으로 선택한 피처와 그 근거를 기술


### 모델 선택 및 근거
- 후보 모델 : ElasticNet, XGBoost
- 최종 모델 : ElasticNet
- 선택 이유 : 


## 성능 결과

<!-- | 구분 | | MAPE (%) | 비고 |
|---|---|---|---|
| Train (Batch 1 CV) | | | 비고 |
| Valid (Batch 1 Hold-out) | | | 비고 |
| Test (Batch 2) | | | 비고 |
| | Gap (Train-Valid) | | (+) : 과적합 의심 |
| | Gap (Valid-Test) | | (+) : 배치간 일반화 저하 의심 |
| | Gap (Target-Test) | | Target : 원논문 9.1% |
| Test (Batch 3) | | | 비고 |
| | Gap (Batch2-Batch3) | | Test 성능 간 비교 |
| | Gap (Target-Test) | | Batch 3 기준, 원논문 성능 비교 | -->


## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점
- 원인 가설 및 개선 방향


## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?


## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.


## 팀 구성
- 손수민 : EDA
- 유건욱 : EDA
- 조성현 : EDA
- 최종민 : EDA