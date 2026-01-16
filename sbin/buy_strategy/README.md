# Buy Strategy Module

매수(진입) 전략을 정의하는 모듈입니다.

## 구조

```
buy_strategy/
├── __init__.py              # 모듈 초기화
├── base.py                  # BaseBuyStrategy 추상 클래스
├── position.py              # PositionInfo 데이터클래스
├── registry.py              # 전략 레지스트리
├── vbt_strategy.py          # VBT 변동성 돌파 전략
├── vbt_enhanced_strategy.py # VBT 고도화 전략 (ADX, ATR 추가)
├── low_bb_dru_strategy.py   # 볼린저 밴드 반등 전략
└── config/
    └── buy_config.json      # 전략 설정 파일
```

## 사용 가능한 전략

| 전략 이름 | 클래스 | 설명 |
|----------|--------|------|
| `vbt_with_filters` | VBTBuyStrategy | 변동성 돌파 (EMA, RSI, 거래량 필터) |
| `vbt_with_filters_enhanced` | VBTEnhancedBuyStrategy | VBT + ADX/ATR 추가 |
| `low_bb_dru` | LowBBDRUBuyStrategy | 볼린저 밴드 하단 반등 |

## 사용법

### 전략 인스턴스 생성

```python
from buy_strategy.registry import get_buy_strategy

# 전략 생성
config = {
    'k_long': 0.5,
    'k_short': 0.5,
    'ema_period': 15,
    'rsi_period': 14
}
strategy = get_buy_strategy('vbt_with_filters', config)
```

### 파라미터 조합 생성

```python
from buy_strategy.registry import get_all_buy_param_combinations

# _list suffix가 붙은 파라미터는 자동으로 조합 생성
config = {
    'k_long_list': [0.4, 0.5, 0.6],
    'ema_period_list': [15, 20]
}
combinations = get_all_buy_param_combinations('vbt_with_filters', config)
# 결과: 6개 (3 * 2) 파라미터 조합
```

## 새 전략 추가하기

### 1. 전략 클래스 생성

```python
# my_strategy.py
from buy_strategy.base import BaseBuyStrategy

class MyBuyStrategy(BaseBuyStrategy):
    def calculate_signals(self, df, cached_data=None):
        # 매수 신호 계산 로직
        return {
            'direction': direction_array,  # 1: Long, -1: Short, 0: None
            'target_long': target_prices,
            'target_short': target_prices
        }
    
    def create_position(self, df, idx, signal_type, signals, cash, total_asset, ticker):
        # PositionInfo 생성
        pass
```

### 2. 레지스트리에 등록

```python
# registry.py
from .my_strategy import MyBuyStrategy

return {
    'my_strategy': MyBuyStrategy,
    'my_strategy_quick': MyBuyStrategy,  # quick 버전 (같은 클래스)
    ...
}
```

### 3. Config 파일에 추가

```json
{
    "index": 6,
    "strategy_name": "my_strategy",
    "description": "내 전략 설명",
    "buy_signal_config": {
        "param1_list": [1, 2, 3],
        "param2": "fixed_value"
    }
}
```

## Config 파일 구조

```json
{
    "index": 0,                    // 인덱스 (CLI에서 참조용)
    "strategy_name": "전략이름",    // 레지스트리 키와 일치해야 함
    "description": "설명",
    "buy_signal_config": {
        "param_list": [1, 2, 3],   // _list: 조합 생성
        "param_fixed": 10          // 고정값
    }
}
```
