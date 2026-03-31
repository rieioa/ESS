# Git 협업 가이드

## 초기 설정 (최초 1회)
```bash
git clone https://github.com/rieioa/slack.git
cd slack
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
