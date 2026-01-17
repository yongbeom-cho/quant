"""
Low BB DU 3 (Bollinger Band Down Up - Reversed GC) 매수 전략

볼린저 밴드 하단에서 반등할 때 진입하는 전략입니다.
기존 sbin/strategy/strategy.py의 low_bb_du_3 함수를 리팩토링한 구현입니다.

=============================================================================
low_bb_du와의 차이점:
- 스토캐스틱 골든크로스를 역방향으로 적용
- (stoch_d > stoch_k) 일 때를 골든크로스로 판단 (잘못된 방향, 레거시 호환)
=============================================================================

전략 조건 (6가지 조건 모두 만족시 매수)
=============================================================================
1. bb_low_signal: 직전 N봉 이내에 종가가 볼린저 밴드 하단 아래로 내려간 적 있음
2. close_lb_up_signal: 현재 종가가 볼린저 밴드 하단 위에 있음
3. close_band_ratio_lower_signal: 현재 종가가 볼린저 밴드 하위 N% 이내에 위치
4. ol_hl_ratio_upper_signal: (open - low) / (high - low) > N (아래꼬리 확인)
5. close_open_ratio_upper_signal: close > open * ratio (양봉 확인)
6. over_sell_golden_cross: 스토캐스틱 골든크로스 + 과매도 (역방향)

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
    print("[WARNING] talib not installed. LowBBDU3 strategy requires talib.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class LowBBDU3Strategy(BaseBuyStrategy):
    """
    볼린저 밴드 하단 반등 매수 전략 (Low BB DU 3)
    
    low_bb_du와 동일하지만 스토캐스틱 골든크로스를 역방향으로 적용합니다.
    (stoch_d > stoch_k 일 때 골든크로스로 판단 - 레거시 호환)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.window = config.get('window', 2)
        self.close_band_ratio_lower = config.get('close_band_ratio_lower', 0.2)
        self.ol_hl_ratio_upper = config.get('ol_hl_ratio_upper', 0.3)
        self.close_open_ratio_upper = config.get('close_open_ratio_upper', 1.005)
        self.over_sell_threshold = config.get('over_sell_threshold', 20)
        
        # 볼린저 밴드 파라미터
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """지표 사전 계산 (캐싱용)"""
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
        매수 신호 계산
        
        6가지 조건을 모두 만족할 때 매수 신호 발생.
        Long only 전략 (direction: 0 또는 1)
        """
        if talib is None:
            raise ImportError("talib is required for LowBBDU3 strategy")
        
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        low_val = df['low'].values
        
        n = len(df)
        
        # === 지표 계산 ===
        if cached_data and 'bb_upper' in cached_data:
            bb_upper = cached_data['bb_upper']
            bb_mid = cached_data['bb_mid']
            bb_lower = cached_data['bb_lower']
            stoch_k = cached_data['stoch_k']
            stoch_d = cached_data['stoch_d']
        else:
            indicators = self.get_indicators(df)
            bb_upper = indicators['bb_upper']
            bb_mid = indicators['bb_mid']
            bb_lower = indicators['bb_lower']
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
        
        # === 조건: 스토캐스틱 골든크로스 (역방향 - 레거시) ===
        # stoch_d > stoch_k && 이전에는 stoch_d < stoch_k
        over_sell_golden_cross = (
            (stoch_d > stoch_k) & 
            (stoch_d.shift(1) < stoch_k.shift(1)) & 
            (stoch_k < self.over_sell_threshold)
        )
        
        # === 조건 1: 직전 N봉 이내에 BB 하단 아래로 내려간 적 있음 ===
        bb_low_signal = pd.Series(False, index=df.index)
        for i in range(1, self.window + 1):
            bb_low_signal = bb_low_signal | (df['close'].shift(i) < bb_lower.shift(i))
        
        # === 조건 2: 현재 종가가 BB 하단 위 ===
        close_lb_up_signal = df['close'] > bb_lower
        
        # === 조건 3: 현재 종가가 BB 하위 N% 이내 ===
        band_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        close_band_ratio_lower_signal = band_position < self.close_band_ratio_lower
        
        # === 조건 4: 아래꼬리 확인 ===
        hl_diff = high_val - low_val
        hl_diff_safe = np.where(hl_diff == 0, np.inf, hl_diff)
        ol_hl_ratio = (open_val - low_val) / hl_diff_safe
        ol_hl_ratio_upper_signal = (hl_diff != 0) & (ol_hl_ratio > self.ol_hl_ratio_upper)
        
        # === 조건 5: 양봉 확인 ===
        close_open_ratio_upper_signal = close_val > open_val * self.close_open_ratio_upper
        
        # === 조건 1-6 조합 ===
        prev_signal = (
            bb_low_signal & 
            close_lb_up_signal & 
            close_band_ratio_lower_signal & 
            ol_hl_ratio_upper_signal & 
            close_open_ratio_upper_signal & 
            over_sell_golden_cross
        )
        
        # === 중복 신호 제거 ===
        prev_true_exists = prev_signal.shift(1).rolling(self.window).max().fillna(0).astype(bool)
        signal = prev_signal & (~prev_true_exists)
        
        # 방향 배열 생성 (Long only)
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
            'close_band_ratio_lower': self.close_band_ratio_lower,
            'ol_hl_ratio_upper': self.ol_hl_ratio_upper,
            'close_open_ratio_upper': self.close_open_ratio_upper,
            'over_sell_threshold': self.over_sell_threshold,
            'gc_direction': 'reversed',  # 역방향 GC 표시
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
            entry_reason='low_bb_du_3',
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
        return ['window', 'close_band_ratio_lower', 'over_sell_threshold']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'low_bb_du_3',
            'window': 2,
            'close_band_ratio_lower': 0.2,
            'ol_hl_ratio_upper': 0.3,
            'close_open_ratio_upper': 1.005,
            'over_sell_threshold': 20,
            'bb_period': 20,
            'bb_std': 2,
            'max_investment_ratio': 1.0
        }
