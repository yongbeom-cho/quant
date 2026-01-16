"""
Timeseries 분할 청산 전략

상승시 분할 매도, 하락시 손절을 결합한 청산 전략입니다.
기존 sbin/strategy_timeseries_backtest/04_strategy_timeseries_backtest.py의
CoinTxManager 로직을 리팩토링한 구현입니다.
"""

from typing import Tuple, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .base import BaseSellStrategy


class TimeseriesSellStrategy(BaseSellStrategy):
    """
    Timeseries 분할 청산 전략
    
    청산 조건:
    1. 분할 익절 (Fraction Sell): 가격이 목표가에 도달하면 일부 매도
    2. 손절 (Lower Limit): 가격이 손절선 아래로 하락하면 전량 매도
    3. 마지막 봉 청산: 백테스트 마지막 날에는 전량 청산
    
    특징:
    - uppers: 분할 매도 목표 비율 리스트 (예: [1.03, 1.06, 1.10])
    - lower: 손절 비율 (예: 0.95 = -5%)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드
        self.uppers = config.get('uppers', [1.03, 1.06, 1.10])
        self.lower = config.get('lower', 0.95)
        self.close_lower_sell = config.get('close_lower_sell', False)
        
        # 현재 남은 분할 청산 단계 추적 (메타데이터로 관리)
        self.sell_fractions = self._calculate_fractions()
    
    def _calculate_fractions(self) -> List[float]:
        """분할 청산 비율 계산 (예: 3단계면 [1/3, 1/2, 1])"""
        n = len(self.uppers)
        return [1.0 / (n - i) for i in range(n)]
    
    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        
        분할 청산의 경우 partial_exit 메서드를 사용하세요.
        이 메서드는 전량 청산만 처리합니다.
        """
        open_val = current_bar['open']
        high_val = current_bar['high']
        low_val = current_bar['low']
        close_val = current_bar['close']
        is_last = current_bar.get('is_last', False)
        
        # 진입 당일에는 청산하지 않음
        entry_date = position.entry_date
        current_date = current_bar.get('date', '')
        if current_date and current_date <= entry_date:
            return False, 'none', 0.0
        
        entry_price = position.entry_price
        low_limit_price = entry_price * self.lower
        
        # === 1. 손절 (Lower Limit) ===
        if not self.close_lower_sell:
            # 장중 저가가 손절선 아래로 하락하면 즉시 손절
            if low_val < low_limit_price:
                exit_price = min(open_val, low_limit_price)
                return True, 'stop_loss_lower', exit_price
        
        # === 2. 종가 기준 손절 (close_lower_sell=True인 경우) ===
        if self.close_lower_sell and close_val < low_limit_price:
            return True, 'stop_loss_close', close_val
        
        # === 3. 마지막 봉 청산 ===
        if is_last:
            return True, 'last_bar_exit', close_val
        
        return False, 'none', 0.0
    
    def check_partial_exit(
        self,
        position: PositionInfo,
        current_bar: Dict[str, Any],
        remaining_uppers: List[float],
        remaining_fractions: List[float]
    ) -> Tuple[bool, str, float, float, List[float], List[float]]:
        """
        분할 청산 확인
        
        Args:
            position: 현재 포지션
            current_bar: 현재 봉 데이터
            remaining_uppers: 남은 목표가 비율 리스트
            remaining_fractions: 남은 분할 비율 리스트
            
        Returns:
            (should_partial_exit, reason, exit_price, sell_fraction, 
             new_remaining_uppers, new_remaining_fractions)
        """
        if not remaining_uppers:
            return False, 'none', 0.0, 0.0, [], []
        
        open_val = current_bar['open']
        high_val = current_bar['high']
        
        entry_price = position.entry_price
        upper = remaining_uppers[0]
        target_price = entry_price * upper
        
        # 고가가 목표가에 도달했는지 확인
        if high_val > target_price:
            exit_price = max(open_val, target_price)
            sell_fraction = remaining_fractions[0]
            
            # 남은 단계 업데이트
            new_uppers = remaining_uppers[1:]
            new_fractions = remaining_fractions[1:]
            
            reason = 'partial_take_profit' if new_uppers else 'full_take_profit'
            return True, reason, exit_price, sell_fraction, new_uppers, new_fractions
        
        return False, 'none', 0.0, 0.0, remaining_uppers, remaining_fractions
    
    def get_initial_state(self) -> Dict[str, Any]:
        """분할 청산을 위한 초기 상태 반환"""
        return {
            'remaining_uppers': self.uppers.copy(),
            'remaining_fractions': self._calculate_fractions(),
            'total_sold_fraction': 0.0
        }
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['uppers', 'lower']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'timeseries_sell',
            'uppers': [1.03, 1.06, 1.10],
            'lower': 0.95,
            'close_lower_sell': False
        }
