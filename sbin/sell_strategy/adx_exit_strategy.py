"""
ADX 기반 청산 전략 (ADX Exit Strategy)

ADX와 DI를 모니터링하여 추세 약화 또는 방향 전환 시 청산하는 전략입니다.

수학적 원리:
- ADX < threshold: 추세 약화
- DI 역전: 추세 방향 전환
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


class ADXExitStrategy(BaseSellStrategy):
    """
    ADX 기반 청산 전략
    
    청산 조건:
    1. ADX가 threshold 이하로 하락 (추세 약화)
    2. DI 역전 (반대 방향 크로스오버)
    3. 고정 손절매
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ADX/DI 파라미터
        self.adx_period = config.get('adx_period', 14)
        self.adx_exit_threshold = config.get('adx_exit_threshold', 20)
        
        # 손절 파라미터
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.02)
        
        # 익절 파라미터
        self.take_profit_ratio = config.get('take_profit_ratio', 0.05)
        
        # DI 역전 청산
        self.exit_on_di_reversal = config.get('exit_on_di_reversal', True)
        
        # 캐시된 지표
        self._cached_indicators = {}
    
    def update_indicators(self, df: pd.DataFrame):
        """지표 업데이트 (백테스트 엔진에서 호출)"""
        if df is None or len(df) == 0:
            return
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        if talib is not None:
            self._cached_indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            self._cached_indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            self._cached_indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        else:
            self._cached_indicators['adx'] = np.ones(len(df)) * 25
            self._cached_indicators['plus_di'] = np.ones(len(df)) * 20
            self._cached_indicators['minus_di'] = np.ones(len(df)) * 20
    
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
        
        # === 1. 손절매 체크 ===
        if position.direction == 1:  # Long
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                exit_price = min(open_val, stop_price)
                return True, 'stop_loss', exit_price
        else:  # Short
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                exit_price = max(open_val, stop_price)
                return True, 'stop_loss', exit_price
        
        # === 2. 익절 체크 ===
        if position.direction == 1:  # Long
            take_profit_price = position.entry_price * (1 + self.take_profit_ratio)
            if high_val >= take_profit_price:
                exit_price = max(open_val, take_profit_price)
                return True, 'take_profit', exit_price
        else:  # Short
            take_profit_price = position.entry_price * (1 - self.take_profit_ratio)
            if low_val <= take_profit_price:
                exit_price = min(open_val, take_profit_price)
                return True, 'take_profit', exit_price
        
        # === 3. ADX 추세 약화 체크 ===
        if 'adx' in self._cached_indicators and current_idx < len(self._cached_indicators['adx']):
            adx = self._cached_indicators['adx'][current_idx]
            if not np.isnan(adx) and adx < self.adx_exit_threshold:
                return True, 'adx_weakening', close_val
        
        # === 4. DI 역전 체크 ===
        if self.exit_on_di_reversal and 'plus_di' in self._cached_indicators:
            if current_idx < len(self._cached_indicators['plus_di']):
                plus_di = self._cached_indicators['plus_di'][current_idx]
                minus_di = self._cached_indicators['minus_di'][current_idx]
                
                if not np.isnan(plus_di) and not np.isnan(minus_di):
                    if position.direction == 1 and minus_di > plus_di:
                        return True, 'di_reversal', close_val
                    elif position.direction == -1 and plus_di > minus_di:
                        return True, 'di_reversal', close_val
        
        return False, 'none', 0.0
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['adx_period', 'stop_loss_ratio']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'adx_exit',
            'adx_period': 14,
            'adx_exit_threshold': 20,
            'stop_loss_ratio': 0.02,
            'take_profit_ratio': 0.05,
            'exit_on_di_reversal': True
        }
