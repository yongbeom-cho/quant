"""
XGBoost Buy Strategy

학습된 XGBoost 모델을 사용하는 매수 전략
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xgb_strategy.apply_model import apply_strategy_model
from .base import BaseBuyStrategy
from .position import PositionInfo


class XGBBuyStrategy(BaseBuyStrategy):
    """
    XGBoost 모델 기반 매수 전략
    
    config에 다음이 필요:
    - model_name: 모델 파일 이름
    - model_dir: 모델 디렉토리 경로
    - base_strategy_name: 기본 전략 이름 (strategy_feature 생성용)
    - base_strategy_param: 기본 전략 파라미터 (strategy_feature 생성용, 단일 딕셔너리)
    - interval: 시간 간격
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get('model_name')
        if not self.model_name:
            raise ValueError("XGBBuyStrategy requires 'model_name' in config")
        
        self.model_dir = config.get('model_dir')
        if not self.model_dir:
            raise ValueError("XGBBuyStrategy requires 'model_dir' in config")
        
        self.base_strategy_name = config.get('base_strategy_name')
        self.base_strategy_param = config.get('base_strategy_param', {})
        self.interval = config.get('interval', 'day')
    
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        XGBoost 모델을 사용하여 매수 신호 계산
        
        Returns:
            signals 딕셔너리 (direction 포함)
        """
        # xgb_strategy의 apply_strategy_model 사용
        df_with_signal = apply_strategy_model(
            df.copy(),
            self.model_name,
            self.model_dir,
            self.interval,
            self.base_strategy_name,
            self.base_strategy_param
        )
        
        # signal을 direction으로 변환 (signal이 1이면 direction도 1)
        direction = np.zeros(len(df), dtype=int)
        if 'signal' in df_with_signal.columns:
            # 원본 df의 인덱스와 매칭
            signal_series = df_with_signal['signal']
            if len(signal_series) == len(df):
                direction = signal_series.values
            else:
                # 인덱스가 다른 경우 재정렬
                direction = signal_series.reindex(df.index, fill_value=0).values
        
        return {
            'direction': direction
        }
    
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
        """
        if signal_type != 1:  # Long only 전략
            return None
        
        row = df.iloc[idx]
        
        # 투자 금액 계산
        max_invest = total_asset * self.max_investment_ratio
        invest_amount = min(available_cash, max_invest)
        
        if invest_amount <= 0:
            return None
        
        # 진입 가격 (종가 기준)
        entry_price = row['close']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'model_name': self.model_name,
            'base_strategy_name': self.base_strategy_name,
            'base_strategy_param': self.base_strategy_param
        }
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='pb_rebound_xgb_model',
            entry_conditions=entry_conditions,
            quantity=quantity,
            invested_amount=invest_amount,
            max_investment_ratio=self.max_investment_ratio,
            current_allocation_ratio=invest_amount / total_asset if total_asset > 0 else 0,
            current_price=entry_price,
            metadata={
                'strategy_name': self.model_name,
                'config': self.config
            }
        )

    def _get_required_config_keys(self) -> list:
        """필수 설정 키 목록"""
        return ['model_name', 'model_dir', 'base_strategy_name', 'base_strategy_param']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'xgb_buy',
            'model_name': '',
            'model_dir': '',
            'base_strategy_name': 'pb_rebound',
            'base_strategy_param': {},
            'interval': 'day'
        }

