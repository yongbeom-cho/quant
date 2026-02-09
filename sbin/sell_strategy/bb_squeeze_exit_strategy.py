"""
볼린저밴드 압축 청산 전략 (BB Squeeze Exit Strategy)

볼린저밴드 기반 진입 전략과 함께 사용하며,
반대 밴드 도달 또는 중심선 회귀 시 청산합니다.

수학적 원리:
- 목표가: 반대편 밴드 도달
- 부분 익절: 중심선(SMA) 회귀
- 손절: ATR 기반 동적 손절
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


class BBSqueezeExitStrategy(BaseSellStrategy):
    """
    볼린저밴드 압축 청산 전략
    
    청산 조건:
    1. 반대 밴드 터치 (목표가 도달)
    2. 중심선(SMA) 회귀 시 부분 익절
    3. ATR 기반 트레일링 스탑
    4. 타임아웃
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 볼린저밴드 파라미터
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        
        # 손절 파라미터
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.02)
        
        # ATR 트레일링 스탑
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.atr_period = config.get('atr_period', 14)
        self.trailing_atr_mult = config.get('trailing_atr_mult', 2.0)
        
        # 타임아웃
        self.timeout_bars = config.get('timeout_bars', 10)
        
        # 캐시된 지표
        self._cached_indicators = {}
        self._highest_high = {}
        self._lowest_low = {}
    
    def update_indicators(self, df: pd.DataFrame):
        """지표 업데이트"""
        if df is None or len(df) == 0:
            return
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if talib is not None:
            upper, middle, lower = talib.BBANDS(
                close, 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_std, 
                nbdevdn=self.bb_std
            )
            self._cached_indicators['upper'] = upper
            self._cached_indicators['middle'] = middle
            self._cached_indicators['lower'] = lower
            self._cached_indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
        else:
            middle = pd.Series(close).rolling(window=self.bb_period).mean()
            std = pd.Series(close).rolling(window=self.bb_period).std()
            self._cached_indicators['upper'] = (middle + (std * self.bb_std)).values
            self._cached_indicators['middle'] = middle.values
            self._cached_indicators['lower'] = (middle - (std * self.bb_std)).values
            
            tr = pd.concat([
                pd.Series(high) - pd.Series(low),
                (pd.Series(high) - pd.Series(close).shift(1)).abs(),
                (pd.Series(low) - pd.Series(close).shift(1)).abs()
            ], axis=1).max(axis=1)
            self._cached_indicators['atr'] = tr.rolling(window=self.atr_period).mean().values
    
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
        position_key = f"{position.ticker}_{position.entry_idx}"
        
        # === 1. 손절매 체크 ===
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                exit_price = min(open_val, stop_price)
                return True, 'stop_loss', exit_price
        else:
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                exit_price = max(open_val, stop_price)
                return True, 'stop_loss', exit_price
        
        # === 2. 반대 밴드 터치 (목표가 도달) ===
        if 'upper' in self._cached_indicators and current_idx < len(self._cached_indicators['upper']):
            upper = self._cached_indicators['upper'][current_idx]
            lower = self._cached_indicators['lower'][current_idx]
            middle = self._cached_indicators['middle'][current_idx]
            
            if not np.isnan(upper) and not np.isnan(lower):
                if position.direction == 1 and high_val >= upper:
                    return True, 'target_band_reached', upper
                elif position.direction == -1 and low_val <= lower:
                    return True, 'target_band_reached', lower
        
        # === 3. ATR 트레일링 스탑 ===
        if self.use_trailing_stop and 'atr' in self._cached_indicators:
            if current_idx < len(self._cached_indicators['atr']):
                atr = self._cached_indicators['atr'][current_idx]
                
                if not np.isnan(atr):
                    # 최고/최저가 추적
                    if position_key not in self._highest_high:
                        self._highest_high[position_key] = high_val
                        self._lowest_low[position_key] = low_val
                    
                    if position.direction == 1:
                        self._highest_high[position_key] = max(self._highest_high[position_key], high_val)
                        trailing_stop = self._highest_high[position_key] - (atr * self.trailing_atr_mult)
                        if low_val <= trailing_stop:
                            del self._highest_high[position_key]
                            del self._lowest_low[position_key]
                            return True, 'trailing_stop', trailing_stop
                    else:
                        self._lowest_low[position_key] = min(self._lowest_low[position_key], low_val)
                        trailing_stop = self._lowest_low[position_key] + (atr * self.trailing_atr_mult)
                        if high_val >= trailing_stop:
                            del self._highest_high[position_key]
                            del self._lowest_low[position_key]
                            return True, 'trailing_stop', trailing_stop
        
        # === 4. 타임아웃 ===
        if bars_held >= self.timeout_bars:
            return True, 'timeout', close_val
        
        return False, 'none', 0.0
    
    def _get_required_config_keys(self) -> List[str]:
        return ['bb_period', 'stop_loss_ratio']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'bb_squeeze_exit',
            'bb_period': 20,
            'bb_std': 2.0,
            'stop_loss_ratio': 0.02,
            'use_trailing_stop': True,
            'atr_period': 14,
            'trailing_atr_mult': 2.0,
            'timeout_bars': 10
        }
