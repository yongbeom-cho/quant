# Backtest Module

통합 백테스트 엔진을 제공하는 모듈입니다.
모든 Buy/Sell 전략 조합을 테스트하고 결과를 분석할 수 있습니다.

## 구조

```
backtest/
├── __init__.py          # 모듈 초기화
├── engine.py            # UnifiedBacktestEngine
├── data_loader.py       # OHLCV 데이터 로더
└── backtest_runner.py   # CLI 실행기
```

## 핵심 기능

### 1. Cross-Combination Test
모든 Buy/Sell 전략 조합을 테스트합니다.

```
VBT Buy ─┬─ VBT Sell
         ├─ Enhanced Sell
         ├─ Ratio Sell
         └─ Timeseries Sell

Low BB Buy ─┬─ VBT Sell
            ├─ Enhanced Sell
            └─ ...
```

### 2. 멀티프로세스 병렬 처리
`--parallel` 옵션으로 CPU 병렬 처리를 지원합니다.

### 3. 결과 정렬 및 필터링
PnL, 승률, MDD, 샤프 비율 등으로 정렬 가능합니다.

## CLI 사용법

### 기본 사용

```bash
python sbin/backtest/backtest_runner.py \
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx 0 \
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx 0 \
    --market coin --interval minute60 --ticker KRW-BTC
```

### 병렬 처리

```bash
python sbin/backtest/backtest_runner.py \
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx all \
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx all \
    --parallel --workers 4
```

### 결과 CSV 저장

```bash
python sbin/backtest/backtest_runner.py \
    ... \
    --output results.csv
```

## CLI 옵션 목록

### Config 옵션

| 옵션 | 설명 |
|------|------|
| `--buy_config` | Buy 전략 설정 파일 경로 |
| `--buy_config_idx` | Buy config 인덱스 (0, 1, 2 또는 'all') |
| `--sell_config` | Sell 전략 설정 파일 경로 |
| `--sell_config_idx` | Sell config 인덱스 (0, 1, 2 또는 'all') |

### 데이터 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--market` | 시장 유형 | `coin` |
| `--interval` | 봉 간격 | `minute60` |
| `--ticker` | 종목 (쉼표 구분) | 전체 |
| `--start_date` | 시작 날짜 | - |
| `--end_date` | 종료 날짜 | - |

### 백테스트 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--commission_fee` | 수수료율 | 0.0005 (0.05%) |
| `--slippage_fee` | 슬리피지율 | 0.002 (0.2%) |
| `--parallel` | 병렬 처리 활성화 | False |
| `--workers` | 워커 수 | CPU 코어 수 |

### 결과 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--sort_by` | 정렬 기준 | `total_pnl` |
| `--top_n` | 출력 개수 | 10 |
| `--min_trades` | 최소 거래 수 | 1 |
| `--output` | CSV 저장 경로 | - |

## 프로그래밍 사용법

```python
from backtest.engine import UnifiedBacktestEngine
from backtest.data_loader import load_ohlcv_data
from buy_strategy.registry import get_buy_strategy
from sell_strategy.registry import get_sell_strategy

# 데이터 로드
data = load_ohlcv_data(
    db_path='var/data/coin_ohlcv_minute60.db',
    table_name='coin_ohlcv_minute60',
    tickers=['KRW-BTC']
)

# 전략 생성
buy_strat = get_buy_strategy('vbt_with_filters', {'k_long': 0.5})
sell_strat = get_sell_strategy('bailout_sell', {'stop_loss_ratio': 0.02})

# 백테스트 실행
engine = UnifiedBacktestEngine(commission_fee=0.0005, slippage_fee=0.002)
result = engine.run_single_backtest(data, buy_strat, sell_strat)

print(f"PnL: {result.total_pnl:.2%}")
print(f"Win Rate: {result.win_ratio:.2%}")
print(f"Trades: {result.trade_count}")
```

## 결과 분석

```python
# 여러 조합 테스트
results = engine.run_cross_combination_test(
    data=data,
    buy_strategies=buy_strategies,
    sell_strategies=sell_strategies
)

# 상위 결과 가져오기
top_results = engine.get_top_results(
    results,
    sort_by='total_pnl',  # 'win_ratio', 'mdd', 'sharpe_ratio'
    top_n=10,
    min_trades=5
)

# DataFrame 변환
df = engine.results_to_dataframe(results)
df.to_csv('analysis.csv', index=False)
```

## 데이터 로더

`data_loader.py`는 SQLite DB에서 OHLCV 데이터를 로드합니다.

```python
from backtest.data_loader import load_ohlcv_data, get_tickers

# 사용 가능한 종목 조회
tickers = get_tickers(db_path, table_name)

# 데이터 로드 (멀티스레드)
data = load_ohlcv_data(
    db_path='var/data/coin_ohlcv_minute60.db',
    table_name='coin_ohlcv_minute60',
    tickers=['KRW-BTC', 'KRW-ETH'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```
