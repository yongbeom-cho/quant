"""
Low BB DU 2 (Bollinger Band Down Up V2) 매수 전략

볼린저 밴드 하단에서 반등할 때 진입하는 전략 V2입니다.
기존 sbin/strategy/strategy.py의 low_bb_du_2 함수를 리팩토링한 구현입니다.

=============================================================================
low_bb_du와의 차이점:
- 7개 조건 사용 (더 세밀한 필터)
- cond3_idx 파라미터로 조건 3 변형 선택 가능 (0-8)
- 올바른 방향의 골든크로스 사용
=============================================================================

전략 조건 (7가지 조건 모두 만족시 매수)
=============================================================================
1. cond1: close/open 범위 내 (co_ratio_range)
2. cond2: 현재 종가가 BB 하단 위
3. cond3: 스토캐스틱 과매도 + 골든크로스 변형 (cond3_idx에 따라)
4. cond4: 직전 N봉 이내에 BB 하단 아래 음봉 존재
5. cond5: close / 직전 N봉 최저 close < cwlc_ratio_lower
6. cond6: low / open < lo_ratio_lower (아래꼬리)
7. cond7: 현재 종가가 BB 하위 N% 이내

=============================================================================

이식전 청산 전략: simple_ratio_sell
- low_limit_ratio: [0.95, 0.975, 0.9]
- high_limit_ratio: [1.026, 1.055, 1.11, 1.165, 1.22]
=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. LowBBDU2 strategy requires talib.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class LowBBDU2Strategy(BaseBuyStrategy):
    """
    볼린저 밴드 하단 반등 매수 전략 V2 (Low BB DU 2)
    
    7가지 세밀한 조건을 사용하며, cond3_idx로 조건 3의 변형을 선택할 수 있습니다.
    올바른 방향의 스토캐스틱 골든크로스를 사용합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드
        self.window = config.get('window', 4)
        self.co_ratio_range = config.get('co_ratio_range', [1.00, 1.03])
        self.over_sell_threshold = config.get('over_sell_threshold', 25)
        self.cwlc_ratio_lower = config.get('cwlc_ratio_lower', 1.04)
        self.lo_ratio_lower = config.get('lo_ratio_lower', 0.998)
        self.close_band_ratio_lower = config.get('close_band_ratio_lower', 0.3)
        self.cond3_idx = config.get('cond3_idx', 0)
        
        # 볼린저 밴드 파라미터
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """지표 사전 계산"""
        indicators = {}
        
        if talib is not None:
            indicators['bb_upper'], indicators['bb_mid'], indicators['bb_lower'] = talib.BBANDS(
                df['close'], 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_std, 
                nbdevdn=self.bb_std, 
                matype=0
            )
            
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산 로직 구현
        
        7가지 엄격한 조건을 모두 만족(AND)할 때 최종 매수 신호를 발생시킵니다.
        벡터화된 연산(Pandas/Numpy)을 사용하여 속도를 최적화했습니다.
        """
        if talib is None:
            raise ImportError("talib is required for LowBBDU2 strategy")
        
        close_val = df['close'].values
        open_val = df['open'].values
        low_val = df['low'].values
        
        n = len(df)
        
        # === 지표 계산 (캐싱된 데이터를 우선 사용) ===
        if cached_data and 'bb_upper' in cached_data:
            bb_upper = cached_data['bb_upper']
            bb_lower = cached_data['bb_lower']
            stoch_k = cached_data['stoch_k']
            stoch_d = cached_data['stoch_d']
        else:
            indicators = self.get_indicators(df)
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
        
        # === 조건 1: 캔들 몸통 크기 제한 (너무 긴 장대양봉 제외) ===
        co_ratio = df['close'] / df['open']
        cond1 = (co_ratio >= self.co_ratio_range[0]) & (co_ratio < self.co_ratio_range[1])
        
        # === 조건 2: 반등 확인 (현재 종가가 BB 하단선보다 위에 있어야 함) ===
        cond2 = df['close'] > bb_lower
        
        # === 스토캐스틱 골든크로스 계산 (Correct Direction) ===
        stoch_gc = (stoch_d < stoch_k) & (stoch_d.shift(1) > stoch_k.shift(1))
        
        # === 조건 3: 과매도권 스토캐스틱 GC (cond3_idx에 따른 변형 처리) ===
        if self.cond3_idx == 0:
            cond3 = (stoch_k < self.over_sell_threshold) & stoch_gc
        elif self.cond3_idx == 1:
            stoch_gc2 = (stoch_d < stoch_k) & (stoch_d.shift(2) > stoch_k.shift(2))
            cond3 = (stoch_k < self.over_sell_threshold) & (stoch_gc | stoch_gc2)
        elif self.cond3_idx == 2:
            cond3 = (stoch_k < self.over_sell_threshold) & stoch_gc.rolling(self.window).max().fillna(False).astype(bool)
        elif self.cond3_idx == 3:
            cond3 = df['open'] < bb_lower
        elif self.cond3_idx == 4:
            cond3 = (df['open'] < bb_lower) & (stoch_k < self.over_sell_threshold)
        elif self.cond3_idx == 5:
            cond3 = (df['open'] < bb_lower) & (stoch_k < self.over_sell_threshold) & stoch_gc.rolling(self.window).max().fillna(False).astype(bool)
        elif self.cond3_idx == 6:
            cond3 = (df['open'] < bb_lower) & (stoch_k < self.over_sell_threshold) & stoch_gc
        elif self.cond3_idx == 7:
            cond3 = (stoch_k < self.over_sell_threshold) & stoch_gc.rolling(self.window).max().fillna(False).astype(bool)
        else:
            cond3 = stoch_k < self.over_sell_threshold
        
        # === 조건 4: 최근 하락 이력 (직전 N봉 이내에 BB 하단 돌파 음봉이 있었는지 확인) ===
        cond4_raw = (df['close'] < bb_lower) & ((df['close'] / df['open']) < 1.0)
        cond4 = cond4_raw.shift(1).rolling(self.window).max().fillna(False).astype(bool)
        
        # === 조건 5: 가격 회복 탄력성 (현재가가 직점 N봉 최저가 대비 과하게 오르지 않았는지) ===
        lowest_close_prev = df['close'].shift(1).rolling(self.window).min()
        cond5 = (df['close'] / lowest_close_prev) < self.cwlc_ratio_lower
        
        # === 조건 6: 아래꼬리 확인 (저가가 시가 대비 일정 비율 이상 하락했는지) ===
        cond6 = (df['low'] / df['open']) < self.lo_ratio_lower
        
        # === 조건 7: 밴드 내 위치 (종가가 밴드 전체 폭의 하위 30% 이내에 있는지) ===
        band_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        cond7 = band_position < self.close_band_ratio_lower
        
        # === 모든 조건을 하나로 결합 ===
        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7
        
        # === 중복 신호 제거 (이미 신호가 발생한 후 N봉 이내에는 재진입 금지) ===
        prev_true_exists = cond_all.shift(1).rolling(self.window).max().fillna(False).astype(bool)
        signal = cond_all & (~prev_true_exists)
        
        # 방향 배열 (Long=1)
        direction = np.zeros(n, dtype=int)
        direction[signal.values] = 1
        
        return {
            'direction': direction,
            'signal': signal.values,
            'target_long': close_val,
            'target_short': None,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'conditions': {
                'cond1': cond1.values,
                'cond2': cond2.values,
                'cond3': cond3.values if hasattr(cond3, 'values') else cond3,
                'cond4': cond4.values,
                'cond5': cond5.values,
                'cond6': cond6.values,
                'cond7': cond7.values,
            }
        }
    
    def create_position(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        signal_type: int,
        signals: Dict[str, Any],
        available_cash: float,
        total_asset: float,
        ticker: str = 'unknown'
    ) -> Optional[PositionInfo]:
        """신호 발생 시 PositionInfo 생성"""
        if signal_type != 1:
            return None
        
        row = df.iloc[idx]
        
        max_invest = total_asset * self.max_investment_ratio
        invest_amount = min(available_cash, max_invest)
        
        if invest_amount <= 0:
            return None
        
        entry_price = row['close']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        entry_conditions = {
            'window': self.window,
            'co_ratio_range': self.co_ratio_range,
            'over_sell_threshold': self.over_sell_threshold,
            'cwlc_ratio_lower': self.cwlc_ratio_lower,
            'lo_ratio_lower': self.lo_ratio_lower,
            'close_band_ratio_lower': self.close_band_ratio_lower,
            'cond3_idx': self.cond3_idx,
            'gc_direction': 'correct',
        }
        
        if signals.get('stoch_k') is not None and len(signals['stoch_k']) > idx:
            stoch_k_val = signals['stoch_k'][idx] if hasattr(signals['stoch_k'], '__getitem__') else signals['stoch_k'].iloc[idx]
            entry_conditions['stoch_k'] = float(stoch_k_val) if not np.isnan(stoch_k_val) else None
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='low_bb_du_2',
            entry_conditions=entry_conditions,
            quantity=quantity,
            invested_amount=invest_amount,
            max_investment_ratio=self.max_investment_ratio,
            current_allocation_ratio=invest_amount / total_asset if total_asset > 0 else 0,
            current_price=entry_price,
            metadata={
                'strategy_name': self.name,
                'config': self.config
            }
        )
    
    def _get_required_config_keys(self) -> List[str]:
        return ['window', 'co_ratio_range', 'over_sell_threshold']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'low_bb_du_2',
            'window': 4,
            'co_ratio_range': [1.00, 1.03],
            'over_sell_threshold': 25,
            'cwlc_ratio_lower': 1.04,
            'lo_ratio_lower': 0.998,
            'close_band_ratio_lower': 0.3,
            'cond3_idx': 0,
            'bb_period': 20,
            'bb_std': 2,
            'max_investment_ratio': 1.0
        }
