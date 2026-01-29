"""
VBT (Volatility Breakout) 청산 전략

손절매(Stop Loss), 익절(Bailout Profit), 타임아웃 탈출을 결합한 청산 전략입니다.
기존 sbin/strategy/vbt_sell_strategy_013.py 를 리팩토링한 구현입니다.
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy


class VBTSellStrategy(BaseSellStrategy):
    """
    VBT용 청산 전략 (손절/익절/타임아웃)
    
    청산 조건:
    1. 고정 비율 손절매 (Stop Loss): 가격이 설정 비율 이상 역방향으로 움직이면 손절
    2. 수익권 조기 탈출 (Bailout Profit): 최소 보유 기간 후 시가가 진입가보다 유리하면 익절
    3. 무수익 타임아웃 (Bailout Timeout): 최대 보유 기간 후에도 수익이 없으면 탈출
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.02)
        self.bailout_profit_days = config.get('bailout_profit_days', 1)
        self.bailout_no_profit_days = config.get('bailout_no_profit_days', 4)
        self.price_flow_sluggish_threshold = config.get('price_flow_sluggish_threshold', 0.005)
    
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
        # 현재 봉 데이터
        open_val = current_bar['open']
        high_val = current_bar['high']
        low_val = current_bar['low']
        close_val = current_bar['close']
        
        # 보유 기간 계산
        bars_held = current_idx - position.entry_idx
        
        # === 1. 고정 비율 손절매 (Stop Loss) ===
        if position.direction == 1:  # Long
            # 매수 후 가격이 설정 비율 이상 떨어지면 손절
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low_val <= stop_price:
                # 손절가에 도달했으면 시가와 손절가 중 불리한 가격으로 청산
                exit_price = min(open_val, stop_price)
                return True, 'stop_loss', exit_price
        else:  # Short
            # 매도 후 가격이 설정 비율 이상 오르면 손절
            stop_price = position.entry_price * (1 + self.stop_loss_ratio)
            if high_val >= stop_price:
                exit_price = max(open_val, stop_price)
                return True, 'stop_loss', exit_price
        
        # === 2. 수익권 조기 탈출 (Bailout Profit) ===
        # 최소 보유 기간 이상 & 시가가 진입가보다 유리하면 익절
        if bars_held >= self.bailout_profit_days:
            if position.direction == 1:  # Long
                if open_val > position.entry_price:
                    return True, 'bailout_profit', open_val
            else:  # Short
                if open_val < position.entry_price:
                    return True, 'bailout_profit', open_val
        
        # === 3. 무수익 타임아웃 탈출 (Bailout No Profit / Timeout) ===
        # 최대 보유 기간 이상 & 수익이 없으면 탈출
        if bars_held >= self.bailout_no_profit_days:
            if position.direction == 1:  # Long
                if close_val <= position.entry_price:
                    return True, 'bailout_timeout', close_val
            else:  # Short
                if close_val >= position.entry_price:
                    return True, 'bailout_timeout', close_val
        
        # 아무 조건에도 해당하지 않으면 포지션 유지
        return False, 'none', 0.0
    
    def check_reverse_signal(
        self,
        position: PositionInfo,
        reverse_to_short: bool,
        reverse_to_long: bool,
        current_bar: Dict[str, Any]
    ) -> Tuple[bool, str, float, int]:
        """
        리버스 시그널 확인 (RSI 기반)
        
        Args:
            position: 현재 포지션
            reverse_to_short: 숏 전환 신호 (RSI 과매수)
            reverse_to_long: 롱 전환 신호 (RSI 과매도)
            current_bar: 현재 봉 데이터
            
        Returns:
            (should_reverse, reason, exit_price, new_direction)
        """
        open_val = current_bar['open']
        
        if position.direction == 1 and reverse_to_short:
            # Long 포지션에서 숏 전환 신호
            return True, 'reverse_to_short', open_val, -1
        elif position.direction == -1 and reverse_to_long:
            # Short 포지션에서 롱 전환 신호
            return True, 'reverse_to_long', open_val, 1
        
        return False, 'none', 0.0, 0
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['stop_loss_ratio']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'bailout_sell',
            'stop_loss_ratio': 0.02,
            'bailout_profit_days': 1,
            'bailout_no_profit_days': 4,
            'price_flow_sluggish_threshold': 0.005
        }
