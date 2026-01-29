"""
Explode Volume Breakout V2 매수 전략

거래량 폭발 v2 + 가격 돌파 전략입니다.
기존 sbin/strategy/strategy.py의 explode_volume_breakout_2 함수를 리팩토링한 구현입니다.

=============================================================================
V1과의 차이점:
- Trimmed mean 대신 단순 평균 사용
- 장기 + 단기 거래량 두 가지 조건 모두 확인
=============================================================================

전략 조건 (4가지 조건 모두 만족시 매수)
=============================================================================
1. volume_signal: 장기 거래량이 N봉 평균 대비 vol_ratio배 이상
2. short_volume_signal: 단기 거래량이 M봉 평균 대비 short_vol_ratio배 이상
3. price_signal: 종가가 시가 * co_ratio 이상 (가격 돌파)
4. utr_signal: (close - open) / (high - open) > utr (상승 기여율)

=============================================================================

이식전 청산 전략: simple_ratio_sell
- low_limit_ratio: [0.98, 0.97, 0.96]
- high_limit_ratio: [1.02, 1.03, 1.04, 1.05, 1.06]
=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .base import BaseBuyStrategy
from .position import PositionInfo


class ExplodeVolumeBreakout2Strategy(BaseBuyStrategy):
    """
    거래량 폭발 V2 + 가격 돌파 매수 전략
    
    장기/단기 두 가지 거래량 조건을 모두 확인하고,
    가격이 시가 대비 일정 비율 이상 상승할 때 진입합니다.
    
    주요 특징:
    - 장기 거래량 평균 (window) 대비 확인
    - 단기 거래량 평균 (short_window) 대비 확인
    - 가격 돌파 확인 (close >= open * co_ratio)
    - 상승 기여율(UTR) 확인
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.window = config.get('window', 240)
        self.vol_ratio = config.get('vol_ratio', 100)
        self.short_window = config.get('short_window', 2)
        self.short_vol_ratio = config.get('short_vol_ratio', 50)
        self.co_ratio = config.get('co_ratio', 1.02)
        self.utr = config.get('utr', 0.9)
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        4가지 조건을 모두 만족할 때 매수 신호 발생.
        Long only 전략 (direction: 0 또는 1)
        """
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        volume_val = df['volume'].values
        
        n = len(df)
        
        # === 조건 1: 장기 거래량 폭발 ===
        vol_mean = df['volume'].shift(1).rolling(window=self.window).mean()
        volume_signal = volume_val > (vol_mean.values * self.vol_ratio)
        
        # === 조건 2: 단기 거래량 폭발 ===
        short_vol_mean = df['volume'].shift(1).rolling(window=self.short_window).mean()
        short_volume_signal = volume_val > (short_vol_mean.values * self.short_vol_ratio)
        
        # === 조건 3: 가격 돌파 (close >= open * co_ratio) ===
        bo_tp = open_val * self.co_ratio
        price_signal = close_val >= bo_tp
        
        # === 조건 4: 상승 기여율 (UTR) ===
        high_open_diff = high_val - open_val
        high_open_diff = np.where(high_open_diff == 0, np.inf, high_open_diff)
        utr_ratio = (close_val - open_val) / high_open_diff
        utr_signal = utr_ratio > self.utr
        
        # === 종합 신호 ===
        signal = volume_signal & short_volume_signal & price_signal & utr_signal
        
        # 방향 배열 생성 (Long only)
        direction = np.zeros(n, dtype=int)
        direction[signal] = 1
        
        return {
            'direction': direction,
            'signal': signal,
            'target_long': close_val,
            'target_short': None,
            'bo_tp': bo_tp,
            'volume_signal': volume_signal,
            'short_volume_signal': short_volume_signal,
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
            'vol_ratio': self.vol_ratio,
            'short_window': self.short_window,
            'short_vol_ratio': self.short_vol_ratio,
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
            entry_reason='explode_volume_breakout_2',
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
        return ['window', 'vol_ratio', 'short_window', 'short_vol_ratio', 'co_ratio', 'utr']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'explode_volume_breakout_2',
            'window': 240,
            'vol_ratio': 100,
            'short_window': 2,
            'short_vol_ratio': 50,
            'co_ratio': 1.02,
            'utr': 0.9,
            'max_investment_ratio': 1.0
        }
