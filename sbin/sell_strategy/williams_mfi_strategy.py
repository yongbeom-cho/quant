"""
Williams %R + MFI Sell Strategy (윌리엄스 %R + MFI 청산 전략)

청산 조건:
1. Overbought: Williams %R > -20 (과매수)
2. MFI Overbought: MFI > 80 (자금 과다 유입)
3. Time Cut: N일 후 강제 청산
4. Stop Loss / Take Profit
"""

from typing import Tuple, Dict, Any, List
from .base import BaseSellStrategy
from buy_strategy.position import PositionInfo


class WilliamsMFISellStrategy(BaseSellStrategy):
    """
    Williams %R + MFI 청산 전략
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 과매수 임계값
        self.willr_overbought = config.get('willr_overbought', -20)
        self.mfi_overbought = config.get('mfi_overbought', 80)
        
        # 익절
        self.take_profit_ratio = config.get('take_profit_ratio', 0.10)
        
        # 손절
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.05)
        
        # 시간 청산
        self.time_cut_period = config.get('time_cut_period', 10)

    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        """
        close = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        # 1. 익절
        if position.direction == 1:
            take_profit_price = position.entry_price * (1 + self.take_profit_ratio)
            if high >= take_profit_price:
                return True, 'take_profit', take_profit_price
        
        # 2. 손절
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low <= stop_price:
                return True, 'stop_loss', stop_price
        
        # 3. Williams %R 과매수 청산
        willr = current_bar.get('willr')
        if willr is not None and willr > self.willr_overbought:
            return True, 'willr_overbought', close
        
        # 4. MFI 과매수 청산
        mfi = current_bar.get('mfi')
        if mfi is not None and mfi > self.mfi_overbought:
            return True, 'mfi_overbought', close
        
        # 5. Time cut
        holding_period = current_idx - position.entry_idx
        if holding_period >= self.time_cut_period:
            return True, 'time_cut', close
            
        return False, 'none', 0.0

    def _get_required_config_keys(self) -> List[str]:
        return []
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'williams_mfi_exit',
            'willr_overbought': -20,
            'mfi_overbought': 80,
            'take_profit_ratio': 0.10,
            'stop_loss_ratio': 0.05,
            'time_cut_period': 10
        }
