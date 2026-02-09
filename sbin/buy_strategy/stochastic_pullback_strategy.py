"""
Stochastic Pullback Strategy (스토캐스틱 눌림목 전략)

Stochastic 오실레이터의 과매도 신호와 골든크로스를 활용한 눌림목 진입 전략입니다.

수학적 근거:
- %K = 100 × (Close - LowestLow_N) / (HighestHigh_N - LowestLow_N)
- %D = SMA(%K, 3)
- 논리: 상승 추세에서 %K < 20 (과매도) 후 %K가 %D를 상향 돌파하면 반등 시작

알고리즘 로직:
1. Trend Filter: Close > EMA(50) (상승 추세)
2. Oversold: Stochastic %K < 20
3. Golden Cross: %K crosses above %D
4. Volume Filter: Volume < Average Volume (약한 매도세)
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None

from .base import BaseBuyStrategy
from .position import PositionInfo


class StochasticPullbackBuyStrategy(BaseBuyStrategy):
    """
    Stochastic Pullback 매수 전략
    
    진입 조건:
    - Close > EMA (상승 추세)
    - Stochastic %K < oversold_threshold (과매도)
    - %K > %D (골든크로스 또는 상태)
    - Volume < Volume MA (약한 매도세)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Stochastic 파라미터
        self.stoch_k_period = config.get('stoch_k_period', 14)
        self.stoch_d_period = config.get('stoch_d_period', 3)
        self.stoch_slowing = config.get('stoch_slowing', 3)
        
        # 과매도 임계값
        self.oversold_threshold = config.get('oversold_threshold', 20)
        
        # 추세 필터
        self.ema_period = config.get('ema_period', 50)
        
        # 거래량 필터
        self.vol_ma_period = config.get('vol_ma_period', 20)
        self.use_volume_filter = config.get('use_volume_filter', True)
        
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Stochastic 기반 눌림목 신호 계산
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. Stochastic 계산
        if talib is not None:
            stoch_k, stoch_d = talib.STOCH(
                high, low, close,
                fastk_period=self.stoch_k_period,
                slowk_period=self.stoch_slowing,
                slowk_matype=0,
                slowd_period=self.stoch_d_period,
                slowd_matype=0
            )
        else:
            # Pandas 기반 계산
            lowest_low = low.rolling(window=self.stoch_k_period).min()
            highest_high = high.rolling(window=self.stoch_k_period).max()
            
            # Fast %K
            fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            
            # Slow %K (Fast %K의 SMA)
            stoch_k = fast_k.rolling(window=self.stoch_slowing).mean()
            
            # %D (Slow %K의 SMA)
            stoch_d = stoch_k.rolling(window=self.stoch_d_period).mean()
        
        # 2. EMA 추세 필터
        if talib is not None:
            ema = talib.EMA(close, timeperiod=self.ema_period)
        else:
            ema = close.ewm(span=self.ema_period, adjust=False).mean()
        
        # 3. 거래량 필터
        vol_ma = volume.rolling(window=self.vol_ma_period).mean()
        
        # 4. 신호 조건
        # 상승 추세
        trend_condition = close > ema
        
        # 과매도 상태 또는 과매도에서 회복 중
        oversold_condition = stoch_k < self.oversold_threshold
        
        # 골든크로스: %K가 %D 위로
        golden_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
        
        # 또는 과매도에서 %K > %D 상태
        recovery_condition = oversold_condition & (stoch_k > stoch_d)
        
        # 거래량 필터
        if self.use_volume_filter:
            volume_condition = volume < vol_ma
        else:
            volume_condition = pd.Series(True, index=df.index)
        
        # 최종 신호: 추세 + (골든크로스 또는 과매도 회복) + 거래량
        buy_signal = trend_condition & (golden_cross | recovery_condition) & volume_condition
        
        signals = {
            'direction': np.where(buy_signal, 1, 0),
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'ema': ema,
            'vol_ma': vol_ma
        }
        
        return signals

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
        row = df.iloc[idx]
        
        # 투자 금액 계산 (최대 투자 비율 적용)
        max_invest = total_asset * self.max_investment_ratio
        invest_amount = min(available_cash, max_invest)
        
        if invest_amount <= 0:
            return None
        
        # 진입 가격 (시가 기준)
        entry_price = row['open']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'oversold_threshold': self.oversold_threshold,
            'ema_period': self.ema_period,
        }
        
        # 현재 지표값 기록
        if signals.get('stoch_k') is not None and len(signals['stoch_k']) > idx:
            entry_conditions['stoch_k'] = float(signals['stoch_k'].iloc[idx] if hasattr(signals['stoch_k'], 'iloc') else signals['stoch_k'][idx])
        if signals.get('stoch_d') is not None and len(signals['stoch_d']) > idx:
            entry_conditions['stoch_d'] = float(signals['stoch_d'].iloc[idx] if hasattr(signals['stoch_d'], 'iloc') else signals['stoch_d'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason=f'Stoch(%K={entry_conditions.get("stoch_k", 0):.1f}) Oversold Recovery',
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
        return ['stoch_k_period', 'oversold_threshold']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'stochastic_pullback',
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_slowing': 3,
            'oversold_threshold': 20,
            'ema_period': 50,
            'vol_ma_period': 20,
            'use_volume_filter': True,
            'max_investment_ratio': 1.0
        }
