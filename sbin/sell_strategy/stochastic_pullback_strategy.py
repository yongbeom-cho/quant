"""
Stochastic Pullback Sell Strategy (스토캐스틱 눌림목 청산 전략)

청산 조건:
1. Overbought Exit: Stochastic %K > 80 (과매수)
2. Death Cross: %K < %D (데드크로스)
3. Time Cut: N일 후 강제 청산
4. Stop Loss: 고정 손절
"""

from typing import Tuple, Dict, Any, List
from .base import BaseSellStrategy
from buy_strategy.position import PositionInfo


class StochasticPullbackSellStrategy(BaseSellStrategy):
    """
    Stochastic Pullback 청산 전략
    
    청산 조건:
    1. %K > overbought_threshold (과매수)
    2. %K < %D (데드크로스)
    3. Time cut
    4. Stop loss
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 과매수 임계값
        self.overbought_threshold = config.get('overbought_threshold', 80)
        
        # 데드크로스 청산
        self.exit_on_death_cross = config.get('exit_on_death_cross', True)
        
        # 시간 청산
        self.time_cut_period = config.get('time_cut_period', 10)
        
        # 손절
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.05)

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
        
        # 1. 손절매 체크
        if position.direction == 1:
            stop_price = position.entry_price * (1 - self.stop_loss_ratio)
            if low <= stop_price:
                return True, 'stop_loss', min(current_bar['open'], stop_price)
        
        # 2. Stochastic 과매수 청산
        stoch_k = current_bar.get('stoch_k')
        stoch_d = current_bar.get('stoch_d')
        
        if stoch_k is not None:
            # 과매수
            if stoch_k >= self.overbought_threshold:
                return True, 'overbought', close
            
            # 데드크로스
            if self.exit_on_death_cross and stoch_d is not None:
                if stoch_k < stoch_d:
                    return True, 'death_cross', close
        
        # 3. Time cut
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
            'strategy_name': 'stochastic_pullback_exit',
            'overbought_threshold': 80,
            'exit_on_death_cross': True,
            'time_cut_period': 10,
            'stop_loss_ratio': 0.05
        }
