# Git 협업 가이드

## 초기 설정 (최초 1회)
```bash
git clone https://github.com/rieioa/ESS
cd repo
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
- Cycle Life 분포
	- 분포 형태 및 장단수명 비율 요약
    Batch 1 : 
        - 평균인 845 cycle 부근에 많은 셀이 분포해있고, 그 외 영역에서는 상당히 고르게 분포
        - 500cycle 미만의 수명을 가지는 단수명 셀은 0개, 1000cycle 이상의 수명을 가지는 셀은 21.7%(10개)
    Batch 2 :
        - 평균인 566 cycle보다 낮은 구간에 30개 셀이 균일하게, 나머지 9개 셀은 비교적 긴 수명 영역에 듬성듬성 분포
        - 500cycle 미만의 수명을 가지는 단수명 셀은 71.8%(28개), 1000cycle 이상의 수명을 가지는 장수명 셀은 7.7%(3개)
    Batch 3 :
        - 평균인 1060 cycle 이하로 많은 셀이 분포해있고, 그 외 영역에서는 드물게 분포
        - 500cycle 미만의 수명을 가지는 단수명 셀은 0개, 1000cycle 이상의 수명을 가지는 셀은 52.3%(23개)
	- 핵심 발견 : 이상치 셀은 QD curve와 비교해보았을 때, Batch 3에서 3개의 배터리 셀은 초기부터 급격한 방전 용량의 감소를 보여 평균보다 낮은 cycle_life를 가지는 것을 확인. 

- 열화 곡선 분석
	- 장수명 vs 단수명 셀의 열화 속도 차이

	- Knee point 존재 여부 및 발생 시점
	- 핵심 발견 :

- ΔQ(V) 곡선 분석
	- Cycle 100 - Cycle 10 차이 곡선 형태
    약 2.3~2.4V 구간에서 가장 큰 변화가 발생하며 cycle life에 따라 곡선의 깊이가 점진적으로 달라지는 경향이 나타남.
	- 장단수명 셀 간 ΔQ 형태 비교
    세 배치 모두 약 2.4~2.6V 구간에서 ΔQ(V)의 최저값이 -0.01~-0.03 범위에 위치하며, 전반적으로 완만한 곡선 형태를 보임. 단수명 배터리는 동일 전압 구간에서 ΔQ(V)가 -0.04~-0.06 수준까지 크게 감소하며, 더 깊고 급격한 열화 패턴을 나타냄.
	- 핵심 발견 :
    특정 전압 구간(2.4~2.6V)에서의 ΔQ(V) 감소 정도는 cycle life와 강한 상관관계를 보임.

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
    세 배치 전체에서 C-rate가 낮은 정책이 높은 수명을 기록하는 경향이 공통적으로 나타남.
	- 핵심 발견 :
    C-rate와 cycle life 간의 관계는 선형적이지 않으며, 특정 임계값 이상의 C-rate는 수명을 급격히 단축시키는 것으로 보임.

- (추가 확인한 내용 작성) 