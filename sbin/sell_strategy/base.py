"""
Sell Strategy 베이스 클래스

모든 청산 전략이 상속해야 하는 추상 베이스 클래스입니다.
새로운 전략을 추가할 때 이 클래스를 상속하고 필수 메서드를 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import itertools
import sys
import os

# 상위 디렉토리의 buy_strategy 모듈을 임포트하기 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.position import PositionInfo
from .metrics import TradeRecord


class BaseSellStrategy(ABC):
    """
    모든 Sell 전략의 베이스 클래스
    
    새 전략 구현 시 필수 구현 메서드:
    - should_exit(): 청산 여부 판단
    - get_param_combinations(): 파라미터 조합 생성 (클래스 메서드)
    
    제공되는 메서드:
    - calculate_pnl(): 실현 손익 계산
    - create_trade_record(): 거래 기록 생성
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 전략 설정 딕셔너리
                - strategy_name: 전략 이름
                - 기타 전략별 파라미터
        """
        self.config = config
        self.name = config.get('strategy_name', self.__class__.__name__)
    
    @abstractmethod
    def should_exit(
        self, 
        position: PositionInfo, 
        current_bar: Dict[str, Any],
        current_idx: int
    ) -> Tuple[bool, str, float]:
        """
        청산 여부 판단
        
        Args:
            position: 현재 포지션 정보 (PositionInfo)
            current_bar: 현재 봉 데이터 딕셔너리
                - open, high, low, close, volume
                - date
            current_idx: 현재 봉 인덱스
            
        Returns:
            Tuple of:
                - should_exit: bool (청산 여부)
                - reason: str (청산 사유: 'stop_loss', 'take_profit', 'timeout', 'reverse' 등)
                - exit_price: float (청산 가격, should_exit이 False면 0.0)
        """
        pass
    
    def calculate_pnl(
        self, 
        position: PositionInfo, 
        exit_price: float,
        commission_fee: float = 0.0005,
        slippage_fee: float = 0.002
    ) -> Tuple[float, float, float]:
        """
        실현 손익 계산 (수수료, 슬리피지 포함)
        
        Args:
            position: 포지션 정보
            exit_price: 청산 가격
            commission_fee: 수수료율 (예: 0.0005 = 0.05%)
            slippage_fee: 슬리피지율 (예: 0.002 = 0.2%)
            
        Returns:
            Tuple of:
                - net_pnl: 순 손익률 (수수료, 슬리피지 포함)
                - gross_pnl: 순수 손익률 (수수료, 슬리피지 제외)
                - commission_paid: 지불한 수수료율
        """
        if position.direction == 1:  # Long
            # 슬리피지 적용 (매도시 불리하게)
            adjusted_exit = exit_price * (1.0 - slippage_fee)
            # 순수 손익
            gross_pnl = (adjusted_exit / position.entry_price) - 1.0
            # 수수료 (진입 + 청산)
            commission_total = 2 * commission_fee
            # 순 손익
            net_pnl = ((adjusted_exit / position.entry_price) * (1.0 - commission_fee)**2) - 1.0
        else:  # Short
            # 슬리피지 적용 (매수시 불리하게)
            adjusted_exit = exit_price * (1.0 + slippage_fee)
            # 순수 손익 (숏은 가격 하락시 이익)
            gross_pnl = 1.0 - (adjusted_exit / position.entry_price)
            # 수수료
            commission_total = 2 * commission_fee
            # 순 손익
            net_pnl = (2.0 - (adjusted_exit / position.entry_price) * (1.0 + commission_fee)**2) - 1.0
        
        return net_pnl, gross_pnl, commission_total
    
    def create_trade_record(
        self,
        position: PositionInfo,
        exit_price: float,
        exit_idx: int,
        exit_date: str,
        exit_reason: str,
        commission_fee: float = 0.0005,
        slippage_fee: float = 0.002
    ) -> TradeRecord:
        """
        거래 기록 생성
        
        Args:
            position: 포지션 정보
            exit_price: 청산 가격
            exit_idx: 청산 봉 인덱스
            exit_date: 청산 일자
            exit_reason: 청산 사유
            commission_fee: 수수료율
            slippage_fee: 슬리피지율
            
        Returns:
            TradeRecord 인스턴스
        """
        net_pnl, gross_pnl, commission_paid = self.calculate_pnl(
            position, exit_price, commission_fee, slippage_fee
        )
        
        return TradeRecord(
            ticker=position.ticker,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_idx=position.entry_idx,
            exit_idx=exit_idx,
            holding_bars=exit_idx - position.entry_idx,
            entry_reason=position.entry_reason,
            exit_reason=exit_reason,
            pnl=net_pnl,
            pnl_amount=net_pnl * position.invested_amount,
            gross_pnl=gross_pnl,
            commission_paid=commission_paid,
            entry_conditions=position.entry_conditions,
            metadata={
                'sell_strategy': self.name,
                'sell_config': self.config
            }
        )
    
    @classmethod
    def get_param_combinations(cls, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Config에서 파라미터 조합 생성
        
        _list suffix가 붙은 파라미터들의 Cartesian product를 생성합니다.
        
        Args:
            config: 전략 설정 (sell_signal_config)
            
        Returns:
            파라미터 딕셔너리 리스트
        """
        list_params = {}
        single_params = {}
        
        for key, value in config.items():
            if key.endswith('_list') and isinstance(value, list):
                param_name = key[:-5]  # _list 제거
                list_params[param_name] = value
            elif not key.endswith('_list'):
                single_params[key] = value
        
        if not list_params:
            return [config.copy()]
        
        param_names = list(list_params.keys())
        param_values = [list_params[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = single_params.copy()
            param_dict.update(dict(zip(param_names, combo)))
            combinations.append(param_dict)
        
        return combinations
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록 반환 (하위 클래스에서 오버라이드)"""
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
