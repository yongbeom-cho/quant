"""
Chandelier Exit 청산 전략 (Chandelier Exit Strategy)

ATR 기반의 동적 손절 전략으로, 최고가/최저가에서 
ATR 배수만큼 떨어진 지점을 손절선으로 사용합니다.

수학적 원리:
- Long: Chandelier_Long = Highest_High_n - k × ATR_n
- Short: Chandelier_Short = Lowest_Low_n + k × ATR_n
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


class ChandelierExitStrategy(BaseSellStrategy):
    """
    Chandelier Exit 청산 전략
    
    청산 조건:
    1. 가격 < Chandelier Exit (Long)
    2. 가격 > Chandelier Exit (Short)
    3. EMA 이탈 시 청산
    4. 고정 손절 (백업)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ATR 파라미터
        self.atr_period = config.get('atr_period', 14)
        self.exit_mult = config.get('exit_mult', 3.0)
        
        # 최고/최저 룩백
        self.lookback_period = config.get('lookback_period', 22)
        
        # EMA 청산 옵션
        self.use_ema_exit = config.get('use_ema_exit', True)
        self.ema_period = config.get('ema_period', 20)
        
        # 고정 손절 (백업)
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.05)
        
        # 캐시된 지표
        self._cached_indicators = {}
        self._position_high = {}
        self._position_low = {}
    
    def update_indicators(self, df: pd.DataFrame):
        """지표 업데이트"""
        if df is None or len(df) == 0:
            return
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if talib is not None:
            self._cached_indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
            self._cached_indicators['ema'] = talib.EMA(close, timeperiod=self.ema_period)
        else:
            tr = pd.concat([
                pd.Series(high) - pd.Series(low),
                (pd.Series(high) - pd.Series(close).shift(1)).abs(),
                (pd.Series(low) - pd.Series(close).shift(1)).abs()
            ], axis=1).max(axis=1)
            self._cached_indicators['atr'] = tr.rolling(window=self.atr_period).mean().values
            self._cached_indicators['ema'] = pd.Series(close).ewm(span=self.ema_period, adjust=False).mean().values
        
        # 롤링 최고가/최저가
        self._cached_indicators['highest_high'] = pd.Series(high).rolling(window=self.lookback_period).max().values
        self._cached_indicators['lowest_low'] = pd.Series(low).rolling(window=self.lookback_period).min().values
        
        # 롤링 고가/저가 저장
        self._cached_indicators['high'] = high
        self._cached_indicators['low'] = low
    
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
        
        position_key = f"{position.ticker}_{position.entry_idx}"
        
        # === 1. 고정 손절 체크 (백업) ===
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                self._cleanup_position(position_key)
                return True, 'fixed_stop_loss', min(open_val, stop_price)
        else:
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                self._cleanup_position(position_key)
                return True, 'fixed_stop_loss', max(open_val, stop_price)
        
        # === 2. Chandelier Exit 계산 ===
        if 'atr' in self._cached_indicators and current_idx < len(self._cached_indicators['atr']):
            atr = self._cached_indicators['atr'][current_idx]
            
            if not np.isnan(atr):
                # 포지션 진입 이후 최고/최저가 추적
                if position_key not in self._position_high:
                    self._position_high[position_key] = high_val
                    self._position_low[position_key] = low_val
                else:
                    self._position_high[position_key] = max(self._position_high[position_key], high_val)
                    self._position_low[position_key] = min(self._position_low[position_key], low_val)
                
                if position.direction == 1:
                    # Long: 최고가에서 ATR 배수만큼 하락 시 청산
                    chandelier_exit = self._position_high[position_key] - (atr * self.exit_mult)
                    if low_val <= chandelier_exit:
                        exit_price = min(open_val, chandelier_exit)
                        self._cleanup_position(position_key)
                        return True, 'chandelier_exit', exit_price
                else:
                    # Short: 최저가에서 ATR 배수만큼 상승 시 청산
                    chandelier_exit = self._position_low[position_key] + (atr * self.exit_mult)
                    if high_val >= chandelier_exit:
                        exit_price = max(open_val, chandelier_exit)
                        self._cleanup_position(position_key)
                        return True, 'chandelier_exit', exit_price
        
        # === 3. EMA 이탈 청산 (선택적) ===
        if self.use_ema_exit and 'ema' in self._cached_indicators:
            if current_idx < len(self._cached_indicators['ema']):
                ema = self._cached_indicators['ema'][current_idx]
                
                if not np.isnan(ema):
                    if position.direction == 1 and close_val < ema:
                        self._cleanup_position(position_key)
                        return True, 'ema_exit', close_val
                    elif position.direction == -1 and close_val > ema:
                        self._cleanup_position(position_key)
                        return True, 'ema_exit', close_val
        
        return False, 'none', 0.0
    
    def _cleanup_position(self, position_key: str):
        """포지션 관련 캐시 정리"""
        if position_key in self._position_high:
            del self._position_high[position_key]
        if position_key in self._position_low:
            del self._position_low[position_key]
    
    def _get_required_config_keys(self) -> List[str]:
        return ['atr_period', 'exit_mult']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'chandelier_exit',
            'atr_period': 14,
            'exit_mult': 3.0,
            'lookback_period': 22,
            'use_ema_exit': True,
            'ema_period': 20,
            'stop_loss_ratio': 0.05
        }
