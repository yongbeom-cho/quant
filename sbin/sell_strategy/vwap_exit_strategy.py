"""
VWAP 목표가 청산 전략 (VWAP Target Exit Strategy)

VWAP으로 회귀하는 가격을 목표가로 설정하고,
VWAP 도달 시 청산하거나 추가 이탈 시 손절합니다.

수학적 원리:
- 목표가: VWAP (평균 회귀)
- 손절: VWAP ± 3σ (추세 확정)
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy

import numpy as np
import pandas as pd


class VWAPExitStrategy(BaseSellStrategy):
    """
    VWAP 목표가 청산 전략
    
    청산 조건:
    1. VWAP 도달 (목표가)
    2. 반대 방향 σ 밴드 도달 (확장 수익)
    3. 손절: VWAP ± 3σ 이탈 (추세 확정)
    4. 타임아웃 (일중 전략 특성)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # VWAP 파라미터
        self.vwap_period = config.get('vwap_period', 20)
        
        # 목표가/손절 레벨
        self.target_at_vwap = config.get('target_at_vwap', True)
        self.stop_std_mult = config.get('stop_std_mult', 3.0)
        self.target_std_mult = config.get('target_std_mult', 1.0)  # 반대 밴드
        
        # 고정 손절 (백업)
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.03)
        
        # 타임아웃
        self.timeout_bars = config.get('timeout_bars', 5)  # 짧은 홀딩
        
        # 캐시된 지표
        self._cached_indicators = {}
    
    def update_indicators(self, df: pd.DataFrame):
        """지표 업데이트"""
        if df is None or len(df) == 0:
            return
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # 롤링 VWAP 계산
        tp_vol = tp * volume
        cum_tp_vol = pd.Series(tp_vol).rolling(window=self.vwap_period).sum()
        cum_vol = pd.Series(volume).rolling(window=self.vwap_period).sum()
        vwap = (cum_tp_vol / cum_vol).values
        
        # VWAP 표준편차
        price_vwap_diff = close - vwap
        vwap_std = pd.Series(price_vwap_diff).rolling(window=self.vwap_period).std().values
        
        self._cached_indicators['vwap'] = vwap
        self._cached_indicators['vwap_std'] = vwap_std
        self._cached_indicators['upper_stop'] = vwap + (vwap_std * self.stop_std_mult)
        self._cached_indicators['lower_stop'] = vwap - (vwap_std * self.stop_std_mult)
        self._cached_indicators['upper_target'] = vwap + (vwap_std * self.target_std_mult)
        self._cached_indicators['lower_target'] = vwap - (vwap_std * self.target_std_mult)
    
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
        
        # === 1. 고정 손절 (백업) ===
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                return True, 'fixed_stop_loss', min(open_val, stop_price)
        else:
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                return True, 'fixed_stop_loss', max(open_val, stop_price)
        
        # === 2. VWAP 관련 청산 ===
        if 'vwap' in self._cached_indicators and current_idx < len(self._cached_indicators['vwap']):
            vwap = self._cached_indicators['vwap'][current_idx]
            upper_stop = self._cached_indicators['upper_stop'][current_idx]
            lower_stop = self._cached_indicators['lower_stop'][current_idx]
            upper_target = self._cached_indicators['upper_target'][current_idx]
            lower_target = self._cached_indicators['lower_target'][current_idx]
            
            if not np.isnan(vwap):
                if position.direction == 1:  # Long
                    # VWAP 도달 (목표가)
                    if self.target_at_vwap and high_val >= vwap:
                        return True, 'vwap_target', vwap
                    
                    # 상단 밴드 도달 (확장 수익)
                    if high_val >= upper_target:
                        return True, 'upper_band_target', upper_target
                    
                    # 추가 하락 손절 (VWAP - 3σ)
                    if low_val <= lower_stop:
                        return True, 'vwap_stop', lower_stop
                
                else:  # Short
                    # VWAP 도달 (목표가)
                    if self.target_at_vwap and low_val <= vwap:
                        return True, 'vwap_target', vwap
                    
                    # 하단 밴드 도달 (확장 수익)
                    if low_val <= lower_target:
                        return True, 'lower_band_target', lower_target
                    
                    # 추가 상승 손절 (VWAP + 3σ)
                    if high_val >= upper_stop:
                        return True, 'vwap_stop', upper_stop
        
        # === 3. 타임아웃 ===
        if bars_held >= self.timeout_bars:
            return True, 'timeout', close_val
        
        return False, 'none', 0.0
    
    def _get_required_config_keys(self) -> List[str]:
        return ['vwap_period', 'stop_std_mult']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'strategy_name': 'vwap_exit',
            'vwap_period': 20,
            'target_at_vwap': True,
            'stop_std_mult': 3.0,
            'target_std_mult': 1.0,
            'stop_loss_ratio': 0.03,
            'timeout_bars': 5
        }
