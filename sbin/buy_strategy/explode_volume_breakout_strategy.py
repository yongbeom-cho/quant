"""
Explode Volume Breakout 매수 전략

거래량 폭발 + 가격 돌파 전략입니다.
기존 sbin/strategy/strategy.py의 explode_volume_breakout 함수를 리팩토링한 구현입니다.

=============================================================================
전략 조건 (3가지 조건 모두 만족시 매수)
=============================================================================
1. volume_signal: 거래량이 N봉 trimmed mean 대비 vol_ratio배 이상
2. price_signal: 종가가 시가 * co_ratio 이상 (가격 돌파)
3. utr_signal: (close - open) / (high - open) > utr (상승 기여율)

=============================================================================

이식전 청산 전략: simple_ratio_sell
- low_limit_ratio: [0.99, 0.98, 0.97]
- high_limit_ratio: [1.02, 1.03, 1.04, 1.05, 1.06]
=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .base import BaseBuyStrategy
from .position import PositionInfo


def trimmed_mean(x: np.ndarray, prev_top_vol_del_ratio: float) -> float:
    """상위 거래량 제거 후 평균 계산"""
    n = len(x)
    k = int(n * prev_top_vol_del_ratio)  # 제거할 개수
    x_sorted = np.sort(x)
    return x_sorted[:n-k].mean()  # 상위 k개 제거


class ExplodeVolumeBreakoutStrategy(BaseBuyStrategy):
    """
    거래량 폭발 + 가격 돌파 매수 전략
    
    거래량이 과거 평균 대비 크게 증가하고,
    가격이 시가 대비 일정 비율 이상 상승할 때 진입합니다.
    
    주요 특징:
    - Trimmed mean 사용 (상위 거래량 이상치 제거)
    - 가격 돌파 확인 (close >= open * co_ratio)
    - 상승 기여율(UTR) 확인
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.window = config.get('window', 240)
        self.prev_top_vol_del_ratio = config.get('prev_top_vol_del_ratio', 0.1)
        self.vol_ratio = config.get('vol_ratio', 50)
        self.co_ratio = config.get('co_ratio', 1.02)
        self.utr = config.get('utr', 0.8)
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        3가지 조건을 모두 만족할 때 매수 신호 발생.
        Long only 전략 (direction: 0 또는 1)
        """
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        volume_val = df['volume'].values
        
        n = len(df)
        
        # === 조건 1: 거래량 폭발 (trimmed mean 대비) ===
        # 이전 봉 기준으로 rolling window의 trimmed mean 계산
        top_vol_trimmed_mean = df['volume'].shift(1).rolling(window=self.window).apply(
            lambda x: trimmed_mean(x, self.prev_top_vol_del_ratio), raw=True
        )
        volume_signal = volume_val > (top_vol_trimmed_mean.values * self.vol_ratio)
        
        # === 조건 2: 가격 돌파 (close >= open * co_ratio) ===
        bo_tp = open_val * self.co_ratio
        price_signal = close_val >= bo_tp
        
        # === 조건 3: 상승 기여율 (UTR) ===
        # (close - open) / (high - open) > utr
        high_open_diff = high_val - open_val
        # 0으로 나누는 것 방지
        high_open_diff = np.where(high_open_diff == 0, np.inf, high_open_diff)
        utr_ratio = (close_val - open_val) / high_open_diff
        utr_signal = utr_ratio > self.utr
        
        # === 종합 신호 ===
        signal = volume_signal & price_signal & utr_signal
        
        # 방향 배열 생성 (Long only)
        direction = np.zeros(n, dtype=int)
        direction[signal] = 1
        
        return {
            'direction': direction,
            'signal': signal,
            'target_long': close_val,  # 종가 기준 진입
            'target_short': None,
            'bo_tp': bo_tp,
            'volume_signal': volume_signal,
            'price_signal': price_signal,
            'utr_signal': utr_signal,
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
            'window': self.window,
            'prev_top_vol_del_ratio': self.prev_top_vol_del_ratio,
            'vol_ratio': self.vol_ratio,
            'co_ratio': self.co_ratio,
            'utr': self.utr,
        }
        
        # 목표가 기록
        if signals.get('bo_tp') is not None and len(signals['bo_tp']) > idx:
            entry_conditions['bo_tp'] = float(signals['bo_tp'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='explode_volume_breakout',
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
        return ['window', 'vol_ratio', 'co_ratio', 'utr']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'explode_volume_breakout',
            'window': 240,
            'prev_top_vol_del_ratio': 0.1,
            'vol_ratio': 50,
            'co_ratio': 1.02,
            'utr': 0.8,
            'max_investment_ratio': 1.0
        }
