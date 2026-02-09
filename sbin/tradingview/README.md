# TradingView Chart App

백테스트(Backtest) 결과를 시각화하고, 전략의 진입/청산 시점과 기술적 지표들을 웹 브라우저에서 인터랙티브하게 분석할 수 있는 애플리케이션입니다.

## 📌 기능 소개
- **Lightweight Charts** 기반의 고성능 캔들 차트 (줌, 스크롤, 크로스헤어 지원)
- **거래 내역 표시**: 차트 상에 진입/청산 마커(Long/Short/Close) 및 호버 시 상세 정보(가격, PnL, 진입 사유) 제공
- **보조지표 오버레이**: 백테스트 시 사용된 다양한 보조지표(MA, BB, RSI, MACD 등)를 차트에 함께 표시
- **사이드바 리스트**: 좌측 사이드바에서 거래별 PnL, 승률, MDD 등을 카드 형태로 한눈에 확인 가능

---

## 🚀 사전 준비 사항 (Prerequisites)

### 1. Python 환경 (Backend & Backtest)
- Python 3.8 이상
- 필요 패키지: `websockets`, `pandas`, `numpy` 등 (백테스트 의존성 포함)

### 2. Node.js 환경 (Frontend)
- Node.js LTS 버전 (16.x 이상 권장)
- npm (Node Package Manager)

---

## 🛠 실행 가이드 (How to Run)

전체 실행 과정은 **"데이터 생성 → 백엔드 실행 → 프론트엔드 실행"** 순서로 진행됩니다.

### 1단계: 백테스트 실행 및 데이터 생성 (Data Generation)

먼저 백테스트를 실행하여 거래 내역과 차트 데이터(OHLCV + 지표)를 JSON 파일로 추출해야 합니다.
반드시 `--export_trades` 옵션을 사용해야 시각화에 필요한 데이터가 생성됩니다.

# 프로젝트 루트 디렉토리에서 실행 (예: ~/Projects/quant)
``` bash
python sbin/backtest/backtest_runner.py \
    --buy_config sbin/buy_strategy/config/buy_config_test.json \
    --buy_config_idx 0 \
    --sell_config sbin/sell_strategy/config/sell_config_test.json \
    --sell_config_idx 0 \
    --ticker KRW-BTC \
    --interval day \
    --market coin \
    --parallel \
    --workers 8 \
    --output var/results_test.csv \
    --export_trades var/trades/
```

- **옵션 설명**:
    - `--ticker`: 분석할 종목 코드 (예: KRW-BTC, KRW-ETH)
    - `--interval`: 캔들 간격 (minute1, minute5, minute15, minute30, minute60, minute240, day)
    - `--export_trades`: [필수] 거래 내역 및 차트 데이터를 저장할 경로 (기본값: var/trades/)

실행이 완료되면 `var/trades/` 디렉토리에 `KRW-BTC_minute15_xxxx.json` 형태의 파일이 생성됩니다.

---

### 2단계: 백엔드 서버 실행 (Backend Server)

생성된 JSON 데이터를 웹소켓을 통해 프론트엔드로 전송하는 파이썬 서버를 실행합니다.

```bash
# 프로젝트 루트에서 실행
cd sbin/tradingview

# 서버 실행 (기본 포트: 8765)
python run_backend.py
```
> **성공 메시지**: `Serving on localhost:8765` 가 출력되면 정상 작동 중입니다.

---

### 3단계: 프론트엔드 앱 실행 (Frontend App)

React 기반의 웹 애플리케이션을 실행합니다. (최초 실행 시 의존성 설치 필요)

```bash
# 프로젝트 루트에서 실행
cd sbin/tradingview/chart-app

# 1. 의존성 설치 (최초 1회만 실행)
npm install

# 2. 개발 서버 실행
npm start
```
> **성공 메시지**: 브라우저가 자동으로 열리거나 `http://localhost:3000` 에서 실행됩니다.

---

## 💻 사용 방법 (Usage)

1. **브라우저 접속**: [http://localhost:3000](http://localhost:3000)
2. **거래 내역 로드**:
    - 우측 상단 **"Trades"** 버튼을 클릭하여 사이드바를 엽니다.
    - 백엔드에 맵핑된 거래 데이터 파일 목록이 표시됩니다.
    - 원하는 거래 내역 카드를 클릭하면 차트에 해당 데이터가 로드됩니다.
3. **차트 분석**:
    - **마우스 휠/드래그**: 줌 인/아웃 및 차트 이동
    - **캔들 호버**: 좌측 상단 패널에 OHLC 정보, 보조지표 값, **해당 캔들의 거래 내역(진입/청산)**이 표시됩니다.
    - **마커 확인**: 차트 상의 화살표(▲/▼)는 진입/청산 시점을 나타내며, 마우스 오버 시 상세 정보(가격, PnL)가 툴팁으로 표시됩니다.

---

## 📂 프로젝트 구조

```
sbin/tradingview/
├── run_backend.py          # 백엔드 웹소켓 서버
├── README.md               # (현재 파일) 실행 가이드
└── chart-app/              # React 프론트엔드 프로젝트
    ├── public/
    ├── src/
    │   ├── components/
    │   │   └── ChartComponent.jsx  # 핵심 차트 렌더링 로직 (Lightweight Charts)
    │   ├── App.js                  # 메인 앱 컴포넌트 (웹소켓 연결 및 상태 관리)
    │   └── ...
    └── package.json
```
