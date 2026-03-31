# 실행 가이드

## 1. 레포지토리 클론

```bash
git clone https://github.com/rieioa/ESS
cd ESS
```

## 2. 데이터 준비

`./data` 안에 `.mat` raw file이 존재해야 합니다.

## 3. 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. 의존성 설치

```bash
pip install -r requirements.txt
```

## 5. 실행

```bash
python model/eval.py
```