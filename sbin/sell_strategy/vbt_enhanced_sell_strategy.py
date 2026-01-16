"""
VBT Enhanced 청산 전략 (고도화된 청산 전략)

ATR 기반 트레일링 스탑을 포함한 고도화된 청산 전략입니다.
기존 sbin/strategy/vbt_sell_strategy_021_enhanced.py 를 리팩토링한 구현입니다.

=============================================================================
청산 조건 (우선순위 순)
=============================================================================
1. 손절매 (Stop Loss): 진입가 대비 일정 비율 손실 시 즉시 청산
2. 트레일링 스탑 (Trailing Stop): 수익권에서 ATR 배수만큼 되돌리면 익절
3. 수익권 탈출 (Bailout Profit): 일정 기간 후 최소 수익권이면 청산
4. 타임아웃 (Time Exit): 일정 기간 후에도 수익 없으면 탈출

=============================================================================
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy


class VBTEnhancedSellStrategy(BaseSellStrategy):
    """
    고도화된 VBT 청산 전략
    
    기본 bailout 전략에 ATR 기반 트레일링 스탑을 추가한 버전입니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 손절 설정
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.02)
        
        # 트레일링 스탑 설정
        self.trailing_stop_mult = config.get('trailing_stop_mult', 3.0)
        self.min_profit_ratio = config.get('min_profit_ratio', 0.005)
        
        # 익절/타임아웃 설정
        self.bailout_profit_days = config.get('bailout_profit_days', 1)
        self.bailout_no_profit_days = config.get('bailout_no_profit_days', 4)
    
    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        
        current_bar에 'atr' 값이 있으면 사용, 없으면 high-low로 대체.
        """
        open_val = current_bar['open']
        high_val = current_bar['high']
        low_val = current_bar['low']
        close_val = current_bar['close']
        is_last = current_bar.get('is_last', False)
        
        # ATR: 제공되지 않으면 high-low로 대체 (간이 변동폭)
        atr_val = current_bar.get('atr')
        if atr_val is None or atr_val == 0:
            atr_val = high_val - low_val  # Fallback: True Range
        
        entry_price = position.entry_price
        bars_held = current_idx - position.entry_idx
        
        # === 1. 고정 비율 손절매 (Stop Loss) ===
        if position.direction == 1:  # Long
            if low_val <= entry_price * (1 - self.stop_loss_ratio):
                exit_price = min(open_val, entry_price * (1 - self.stop_loss_ratio))
                return True, 'stop_loss', exit_price
        else:  # Short
            if high_val >= entry_price * (1 + self.stop_loss_ratio):
                exit_price = max(open_val, entry_price * (1 + self.stop_loss_ratio))
                return True, 'stop_loss', exit_price
        
        # === 2. ATR 기반 트레일링 스탑 (수익 보존) ===
        if atr_val > 0:
            if position.direction == 1:  # Long
                current_pnl = (close_val / entry_price) - 1
            else:  # Short
                current_pnl = 1 - (close_val / entry_price)
            
            # 최소 수익 목표를 넘었을 때만 트레일링 스탑 활성화
            if current_pnl > self.min_profit_ratio:
                if position.direction == 1:  # Long
                    # 고점 대비 ATR 배수만큼 하락하면 익절
                    if close_val < (high_val - atr_val * self.trailing_stop_mult):
                        return True, 'trailing_stop', close_val
                else:  # Short
                    # 저점 대비 ATR 배수만큼 상승하면 익절
                    if close_val > (low_val + atr_val * self.trailing_stop_mult):
                        return True, 'trailing_stop', close_val
        
        # === 3. 수익권 탈출 (Bailout Profit) ===
        if bars_held >= self.bailout_profit_days:
            if position.direction == 1:  # Long
                if open_val > entry_price * (1 + self.min_profit_ratio):
                    return True, 'bailout_profit', open_val
            else:  # Short
                if open_val < entry_price * (1 - self.min_profit_ratio):
                    return True, 'bailout_profit', open_val
        
        # === 4. 무수익 타임아웃 탈출 ===
        if bars_held >= self.bailout_no_profit_days:
            if position.direction == 1:  # Long
                if close_val <= entry_price:
                    return True, 'bailout_timeout', close_val
            else:  # Short
                if close_val >= entry_price:
                    return True, 'bailout_timeout', close_val
        
        # === 5. 마지막 봉 청산 ===
        if is_last:
            return True, 'last_bar_exit', close_val
        
        return False, 'none', 0.0
    
    def check_reverse_signal(
        self,
        position: PositionInfo,
        reverse_to_short: bool,
        reverse_to_long: bool,
        current_bar: Dict[str, Any]
    ) -> Tuple[bool, str, float, int]:
        """
        리버스 시그널 확인 (RSI Zone Exit 기반)
        """
        open_val = current_bar['open']
        
        if position.direction == 1 and reverse_to_short:
            return True, 'reverse_to_short', open_val, -1
        elif position.direction == -1 and reverse_to_long:
            return True, 'reverse_to_long', open_val, 1
        
        return False, 'none', 0.0, 0
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['stop_loss_ratio']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'bailout_sell_enhanced',
            'stop_loss_ratio': 0.02,
            'trailing_stop_mult': 3.0,
            'min_profit_ratio': 0.005,
            'bailout_profit_days': 1,
            'bailout_no_profit_days': 4
        }
