"""
Simple Ratio Sell Strategy (단순 비율 청산 전략)

손절 비율(low_limit_ratio)과 익절 비율(high_limit_ratio)만으로 청산하는 단순한 전략입니다.
기존 best_config.json의 sell_signal_config에서 사용되는 방식입니다.

=============================================================================
청산 조건
=============================================================================
1. 손절: 가격이 진입가 * low_limit_ratio 이하로 하락
2. 익절: 가격이 진입가 * high_limit_ratio 이상으로 상승

예: low_limit_ratio=0.9, high_limit_ratio=1.11
    -> -10% 손절, +11% 익절
=============================================================================
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy


class SimpleRatioSellStrategy(BaseSellStrategy):
    """
    단순 비율 청산 전략
    
    고정 비율의 손절/익절만 사용하는 가장 단순한 청산 전략입니다.
    low_bb_dru 등 볼린저 밴드 전략과 함께 사용됩니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드
        self.low_limit_ratio = config.get('low_limit_ratio', 0.95)   # 손절 비율 (예: 0.9 = -10%)
        self.high_limit_ratio = config.get('high_limit_ratio', 1.10) # 익절 비율 (예: 1.1 = +10%)
        self.is_close_lower = config.get('is_close_lower', False) # 종가로 청산 여부
    
    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        
        Returns:
            (should_exit, reason, exit_price)
        """
        open_val = current_bar['open']
        high_val = current_bar['high']
        low_val = current_bar['low']
        close_val = current_bar['close']
        is_last = current_bar.get('is_last', False)
        
        entry_price = position.entry_price
        stop_loss_price = entry_price * self.low_limit_ratio
        take_profit_price = entry_price * self.high_limit_ratio
        
        # Long 포지션만 지원 (low_bb_dru는 Long only)
        if position.direction != 1:
            return False, 'none', 0.0
        
        # 종가 기준 판단 여부에 따라 분기
        if self.is_close_lower:
            # === 종가 기준 판단 ===
            # 1. 손절 (종가가 손절선 아래로 하락)
            if close_val <= stop_loss_price:
                return True, 'stop_loss', close_val
            
            # 2. 익절 (종가가 익절선 위로 상승)
            if close_val >= take_profit_price:
                return True, 'take_profit', close_val
        else:
            # === 고가/저가 기준 판단 (기존 로직) ===
            # 1. 손절 (가격이 손절선 아래로 하락)
            if low_val <= stop_loss_price:
                # 시가가 이미 손절선 아래면 시가로 청산, 아니면 손절가로 청산
                exit_price = min(open_val, stop_loss_price)
                return True, 'stop_loss', exit_price
            
            # 2. 익절 (가격이 익절선 위로 상승)
            if high_val >= take_profit_price:
                # 시가가 이미 익절선 위면 시가로 청산, 아니면 익절가로 청산
                exit_price = max(open_val, take_profit_price)
                return True, 'take_profit', exit_price
        
        # === 3. 마지막 봉 청산 ===
        if is_last:
            return True, 'last_bar_exit', close_val
        
        return False, 'none', 0.0
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['low_limit_ratio', 'high_limit_ratio']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'simple_ratio_sell',
            'low_limit_ratio': 0.95,
            'high_limit_ratio': 1.10
        }
