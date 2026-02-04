"""
MACD 크로스 청산 전략 (MACD Exit Strategy)

MACD와 Signal Line의 크로스오버를 모니터링하여
추세 전환 시 청산하는 전략입니다.

수학적 원리:
- MACD가 Signal 하향 돌파: Long 청산
- MACD가 Signal 상향 돌파: Short 청산
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy

try:
    import talib
except ImportError:
    talib = None

import numpy as np
import pandas as pd


class MACDExitStrategy(BaseSellStrategy):
    """
    MACD 크로스 청산 전략
    
    청산 조건:
    1. MACD 반대 방향 크로스
    2. 다이버전스 무효화
    3. 목표 수익률 도달
    4. 고정 손절
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # MACD 파라미터
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        
        # 손절/익절 파라미터
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.02)
        self.take_profit_ratio = config.get('take_profit_ratio', 0.05)
        
        # 타임아웃
        self.timeout_bars = config.get('timeout_bars', 15)
        
        # 캐시된 지표
        self._cached_indicators = {}
    
    def update_indicators(self, df: pd.DataFrame):
        """지표 업데이트"""
        if df is None or len(df) == 0:
            return
        
        close = df['close'].values
        
        if talib is not None:
            macd, signal, hist = talib.MACD(
                close, 
                fastperiod=self.fast_period, 
                slowperiod=self.slow_period, 
                signalperiod=self.signal_period
            )
            self._cached_indicators['macd'] = macd
            self._cached_indicators['signal'] = signal
            self._cached_indicators['histogram'] = hist
        else:
            ema_fast = pd.Series(close).ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = pd.Series(close).ewm(span=self.slow_period, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=self.signal_period, adjust=False).mean()
            
            self._cached_indicators['macd'] = macd.values
            self._cached_indicators['signal'] = signal.values
            self._cached_indicators['histogram'] = (macd - signal).values
    
    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        """
        open_val = current_bar['open']
        high_val = current_bar['high']
        low_val = current_bar['low']
        close_val = current_bar['close']
        
        bars_held = current_idx - position.entry_idx
        
        # === 1. 손절매 체크 ===
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                return True, 'stop_loss', min(open_val, stop_price)
        else:
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                return True, 'stop_loss', max(open_val, stop_price)
        
        # === 2. 익절 체크 ===
        if position.direction == 1:
            take_profit_price = position.entry_price * (1 + self.take_profit_ratio)
            if high_val >= take_profit_price:
                return True, 'take_profit', max(open_val, take_profit_price)
        else:
            take_profit_price = position.entry_price * (1 - self.take_profit_ratio)
            if low_val <= take_profit_price:
                return True, 'take_profit', min(open_val, take_profit_price)
        
        # === 3. MACD 크로스오버 체크 ===
        if 'macd' in self._cached_indicators and current_idx >= 1:
            if current_idx < len(self._cached_indicators['macd']):
                macd = self._cached_indicators['macd'][current_idx]
                signal = self._cached_indicators['signal'][current_idx]
                prev_macd = self._cached_indicators['macd'][current_idx - 1]
                prev_signal = self._cached_indicators['signal'][current_idx - 1]
                
                if not np.isnan(macd) and not np.isnan(signal):
                    # Long: MACD가 Signal 하향 돌파 시 청산
                    if position.direction == 1:
                        if macd < signal and prev_macd >= prev_signal:
                            return True, 'macd_cross_down', close_val
                    # Short: MACD가 Signal 상향 돌파 시 청산
                    else:
                        if macd > signal and prev_macd <= prev_signal:
                            return True, 'macd_cross_up', close_val
        
        # === 4. 타임아웃 ===
        if bars_held >= self.timeout_bars:
            return True, 'timeout', close_val
        
        return False, 'none', 0.0
    
    def _get_required_config_keys(self) -> List[str]:
        return ['fast_period', 'slow_period', 'signal_period']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'macd_exit',
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'stop_loss_ratio': 0.02,
            'take_profit_ratio': 0.05,
            'timeout_bars': 15
        }
