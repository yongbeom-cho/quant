# Sell Strategy Module

청산(매도) 전략을 정의하는 모듈입니다.

## 구조

```
sell_strategy/
├── __init__.py                    # 모듈 초기화
├── base.py                        # BaseSellStrategy 추상 클래스
├── metrics.py                     # TradeRecord, PerformanceMetrics
├── registry.py                    # 전략 레지스트리
├── vbt_sell_strategy.py           # VBT 청산 전략
├── vbt_enhanced_sell_strategy.py  # VBT 고도화 청산 (ATR 트레일링)
├── simple_ratio_sell_strategy.py  # 단순 비율 청산
├── timeseries_sell_strategy.py    # 분할 청산 전략
└── config/
    └── sell_config.json           # 전략 설정 파일
```

## 사용 가능한 전략

| 전략 이름 | 클래스 | 설명 |
|----------|--------|------|
| `bailout_sell` | VBTSellStrategy | 손절/익절/타임아웃 |
| `bailout_sell_enhanced` | VBTEnhancedSellStrategy | ATR 트레일링 스탑 추가 |
| `simple_ratio_sell` | SimpleRatioSellStrategy | 고정 비율 손절/익절 |
| `timeseries_sell` | TimeseriesSellStrategy | 분할 청산 |

## 청산 조건 요약

### bailout_sell (VBTSellStrategy)
1. **손절**: `low <= entry_price * (1 - stop_loss_ratio)`
2. **익절**: `bailout_profit_days`일 후 시가가 진입가 상회
3. **타임아웃**: `bailout_no_profit_days`일 후에도 수익 없음

### bailout_sell_enhanced (VBTEnhancedSellStrategy)
위 조건 + **ATR 트레일링 스탑**: 수익권에서 ATR * mult 만큼 되돌리면 익절

### simple_ratio_sell (SimpleRatioSellStrategy)
- **손절**: `low_limit_ratio` (예: 0.9 = -10%)
- **익절**: `high_limit_ratio` (예: 1.1 = +10%)

### timeseries_sell (TimeseriesSellStrategy)
- **분할 청산**: 가격이 각 단계(uppers) 도달 시 일부 청산
- **손절**: `lower` 비율 이하로 하락 시 전량 청산

## 사용법

```python
from sell_strategy.registry import get_sell_strategy

# 전략 생성
config = {
    'stop_loss_ratio': 0.02,
    'bailout_profit_days': 1,
    'bailout_no_profit_days': 4
}
strategy = get_sell_strategy('bailout_sell', config)
```

## Buy/Sell 호환성

모든 Sell 전략은 Buy 전략에서 제공하는 지표가 없어도 자체적으로 기본값을 처리합니다.

| Buy 전략 | bailout_sell | bailout_enhanced | simple_ratio | timeseries |
|----------|--------------|------------------|--------------|------------|
| vbt_with_filters | ✅ | ✅ (fallback ATR) | ✅ | ✅ |
| vbt_enhanced | ✅ | ✅ (ATR 전달) | ✅ | ✅ |
| low_bb_dru | ✅ | ✅ (fallback ATR) | ✅ | ✅ |

> ⚠️ **참고**: `bailout_sell_enhanced`는 Buy 전략에서 ATR을 제공하지 않으면 `high - low`로 대체합니다.

## 새 전략 추가하기

### 1. 전략 클래스 생성

```python
# my_sell_strategy.py
from sell_strategy.base import BaseSellStrategy

class MySellStrategy(BaseSellStrategy):
    def should_exit(self, position, current_bar, current_idx):
        # 청산 로직
        # Returns: (should_exit, reason, exit_price)
        return True, 'my_reason', current_bar['close']
```

### 2. 레지스트리에 등록

```python
# registry.py
from .my_sell_strategy import MySellStrategy

return {
    'my_sell': MySellStrategy,
    ...
}
```

## Performance Metrics

`PerformanceMetrics` 클래스는 백테스트 결과를 집계합니다:

| 지표 | 설명 |
|------|------|
| `total_pnl` | 총 수익률 |
| `cumulative_return` | 누적 수익률 |
| `trade_count` | 거래 횟수 |
| `win_ratio` | 승률 |
| `mdd` | 최대 낙폭 (1.0 = 낙폭 없음) |
| `sharpe_ratio` | 샤프 비율 |
