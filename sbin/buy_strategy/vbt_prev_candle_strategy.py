"""
VBT Prev Candle (Volatility Breakout Previous Candle) 매수 전략

직전 봉의 레인지를 기반으로 변동성 돌파를 판단하는 가장 기본적인 전략입니다.
기존 sbin/strategy/strategy.py의 volatility_breakout_prev_candle 함수를 리팩토링한 구현입니다.

=============================================================================
전략 조건
=============================================================================
1. 직전 봉의 레인지(high - low) 계산
2. 목표가 = 시가 + (직전 봉 레인지 * k)
3. 종가가 목표가 이상이면 매수 신호

=============================================================================

이식전 청산 전략: bailoss_rout_sell
- stop_latio: [0.02, 0.03]
- bailout_profit_days: [1, 2]
- bailout_no_profit_days: [3, 5]
- price_flow_sluggish_threshold: [0.005]
=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .base import BaseBuyStrategy
from .position import PositionInfo


class VBTPrevCandleStrategy(BaseBuyStrategy):
    """
    변동성 돌파 매수 전략 (직전 봉 레인지 기준)
    
    가장 기본적인 변동성 돌파 전략입니다.
    직전 봉의 고가-저가 범위를 변동성으로 사용합니다.
    
    주요 특징:
    - 단순하고 직관적인 로직
    - 직전 1봉의 레인지만 사용
    - k 파라미터로 돌파 강도 조절
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 (기본값 설정)
        self.k = config.get('k', 0.5)
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        종가가 목표가(시가 + 직전 봉 레인지 * k) 이상이면 매수.
        Long only 전략 (direction: 0 또는 1)
        """
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        low_val = df['low'].values
        
        n = len(df)
        
        # === 직전 봉 레인지 계산 ===
        prev_range = (df['high'] - df['low']).shift(1).values
        
        # === 목표가 계산 ===
        bo_tp = open_val + prev_range * self.k
        
        # === 매수 신호: 종가 >= 목표가 ===
        signal = close_val > bo_tp
        
        # 방향 배열 생성 (Long only)
        direction = np.zeros(n, dtype=int)
        direction[signal] = 1
        
        return {
            'direction': direction,
            'signal': signal,
            'target_long': bo_tp,  # 목표가 기준 진입
            'target_short': None,
            'bo_tp': bo_tp,
            'prev_range': prev_range,
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
        
        # 진입 가격: 목표가 또는 종가 중 높은 값
        bo_tp = signals.get('bo_tp')
        if bo_tp is not None and len(bo_tp) > idx:
            entry_price = max(row['open'], bo_tp[idx])
        else:
            entry_price = row['close']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'k': self.k,
        }
        
        # 목표가 및 직전 레인지 기록
        if signals.get('bo_tp') is not None and len(signals['bo_tp']) > idx:
            entry_conditions['bo_tp'] = float(signals['bo_tp'][idx])
        if signals.get('prev_range') is not None and len(signals['prev_range']) > idx:
            entry_conditions['prev_range'] = float(signals['prev_range'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='volatility_breakout_prev_candle',
            entry_conditions=entry_conditions,
            quantity=quantity,
            invested_amount=invest_amount,
            max_investment_ratio=self.max_investment_ratio,
            current_allocation_ratio=invest_amount / total_asset if total_asset > 0 else 0,
            current_price=entry_price,
            metadata={
                'strategy_name': self.name,
                'config': self.config
            }
        )
    
    def _get_required_config_keys(self) -> List[str]:
        """필수 설정 키 목록"""
        return ['k']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'volatility_breakout_prev_candle',
            'k': 0.5,
            'max_investment_ratio': 1.0
        }
