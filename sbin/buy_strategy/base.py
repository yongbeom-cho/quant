"""
Buy Strategy 베이스 클래스

모든 매수 전략이 상속해야 하는 추상 베이스 클래스입니다.
새로운 전략을 추가할 때 이 클래스를 상속하고 필수 메서드를 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
import itertools
import inspect

from .position import PositionInfo


class BaseBuyStrategy(ABC):
    """
    모든 Buy 전략의 베이스 클래스
    
    새 전략 구현 시 필수 구현 메서드:
    - calculate_signals(): 매수 신호 계산
    - create_position(): 포지션 정보 생성
    - get_param_combinations(): 파라미터 조합 생성 (클래스 메서드)
    
    선택적 구현:
    - get_indicators(): 지표 사전 계산 (캐싱용)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 전략 설정 딕셔너리
                - strategy_name: 전략 이름
                - max_investment_ratio: 최대 투자 비율 (기본값: 1.0)
                - 기타 전략별 파라미터
        """
        self.config = config
        self.name = config.get('strategy_name', self.__class__.__name__)
        self.max_investment_ratio = config.get('max_investment_ratio', 1.0)
        self._cached_indicators: Dict[str, Any] = {}
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        OHLCV 데이터에 매수 신호 계산
        
        Args:
            df: OHLCV 데이터프레임 (컬럼: date, open, high, low, close, volume)
            cached_data: 사전 계산된 지표 데이터 (선택적)
        
        Returns:
            dict with keys:
                - 'direction': np.array (1: Long 신호, -1: Short 신호, 0: 신호 없음)
                - 'target_long': np.array (롱 진입 목표가, 선택적)
                - 'target_short': np.array (숏 진입 목표가, 선택적)
                - 'reverse_to_short': np.array (숏 전환 신호, 선택적)
                - 'reverse_to_long': np.array (롱 전환 신호, 선택적)
                - 기타 전략별 추가 시그널
        """
        pass
    
    @abstractmethod
    def create_position(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        signal_type: int,
        signals: Dict[str, Any],
        available_cash: float,
        total_asset: float,
        ticker: str = 'unknown'
    ) -> Optional[PositionInfo]:
        """
        신호 발생 시 PositionInfo 생성
        
        Args:
            df: OHLCV 데이터프레임
            idx: 현재 봉 인덱스
            signal_type: 1 (Long) 또는 -1 (Short)
            signals: calculate_signals() 결과
            available_cash: 사용 가능 현금
            total_asset: 전체 자산
            ticker: 종목 코드
            
        Returns:
            PositionInfo 또는 None (투자 한도 초과 등의 이유로 진입 불가 시)
        """
        pass
    
    @classmethod
    def get_param_combinations(cls, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Config에서 파라미터 조합 생성
        
        _list suffix가 붙은 파라미터들의 Cartesian product를 생성합니다.
        예: k_long_list: [0.4, 0.5], window_list: [5, 10]
            -> [{'k_long': 0.4, 'window': 5}, {'k_long': 0.4, 'window': 10}, ...]
        
        Args:
            config: 전략 설정 (buy_signal_config 또는 전체 config)
            
        Returns:
            파라미터 딕셔너리 리스트
        """
        # _list suffix 파라미터 찾기
        list_params = {}
        single_params = {}
        
        for key, value in config.items():
            if key.endswith('_list') and isinstance(value, list):
                param_name = key[:-5]  # _list 제거
                list_params[param_name] = value
            elif not key.endswith('_list'):
                single_params[key] = value
        
        if not list_params:
            # _list 파라미터가 없으면 원본 config 반환
            return [config.copy()]
        
        # Cartesian product 생성
        param_names = list(list_params.keys())
        param_values = [list_params[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = single_params.copy()
            param_dict.update(dict(zip(param_names, combo)))
            combinations.append(param_dict)
        
        return combinations
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        지표 사전 계산 (캐싱용)
        
        백테스트 속도 향상을 위해 반복 계산되는 지표를 미리 계산합니다.
        하위 클래스에서 필요시 오버라이드합니다.
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            계산된 지표 딕셔너리
        """
        return {}
    
    def get_entry_price(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        signal_type: int,
        signals: Dict[str, Any],
        slippage_fee: float = 0.0
    ) -> float:
        """
        진입 가격 계산 (슬리피지 적용)
        
        Args:
            df: OHLCV 데이터프레임
            idx: 현재 봉 인덱스
            signal_type: 1 (Long) 또는 -1 (Short)
            signals: calculate_signals() 결과
            slippage_fee: 슬리피지 비율
            
        Returns:
            진입 가격
        """
        open_price = df.iloc[idx]['open']
        
        if signal_type == 1:  # Long
            target_long = signals.get('target_long')
            if target_long is not None and len(target_long) > idx:
                base_price = max(open_price, target_long[idx])
            else:
                base_price = open_price
            return base_price * (1.0 + slippage_fee)
        else:  # Short
            target_short = signals.get('target_short')
            if target_short is not None and len(target_short) > idx:
                base_price = min(open_price, target_short[idx])
            else:
                base_price = open_price
            return base_price * (1.0 - slippage_fee)
    
    def validate_config(self) -> bool:
        """
        설정 유효성 검사
        
        Returns:
            유효하면 True
        """
        required_keys = self._get_required_config_keys()
        for key in required_keys:
            if key not in self.config and f'{key}_list' not in self.config:
                return False
        return True
    
    def _get_required_config_keys(self) -> List[str]:
        """
        필수 설정 키 목록 반환 (하위 클래스에서 오버라이드)
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
