"""
Williams %R + MFI Strategy (윌리엄스 %R + 자금흐름 지표 전략)

Williams %R과 MFI(Money Flow Index)의 이중 과매도 신호를 활용한 전략입니다.

수학적 근거:
- Williams %R = (HighestHigh_N - Close) / (HighestHigh_N - LowestLow_N) × -100
  - 범위: -100 ~ 0, -80 미만 = 과매도
- MFI = 100 - (100 / (1 + Positive Money Flow / Negative Money Flow))
  - 범위: 0 ~ 100, 20 미만 = 과매도
  - RSI의 거래량 가중 버전

알고리즘 로직:
1. Double Oversold: Williams %R < -80 AND MFI < 20
2. Trend Filter: Close > EMA(50)
3. Recovery: Williams %R가 -80 위로 상승
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


class WilliamsMFIBuyStrategy(BaseBuyStrategy):
    """
    Williams %R + MFI 매수 전략
    
    진입 조건:
    - Williams %R < -80 (과매도)
    - MFI < 20 (자금 유출 극심)
    - Close > EMA (상승 추세)
    - Williams %R 회복 중
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Williams %R 파라미터
        self.willr_period = config.get('willr_period', 14)
        self.willr_oversold = config.get('willr_oversold', -80)
        
        # MFI 파라미터
        self.mfi_period = config.get('mfi_period', 14)
        self.mfi_oversold = config.get('mfi_oversold', 20)
        
        # 추세 필터
        self.ema_period = config.get('ema_period', 50)
        
        # 이중 과매도 필수 여부
        self.require_double_oversold = config.get('require_double_oversold', True)
        
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Williams %R + MFI 신호 계산
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. Williams %R 계산
        if talib is not None:
            willr = talib.WILLR(high, low, close, timeperiod=self.willr_period)
        else:
            highest_high = high.rolling(window=self.willr_period).max()
            lowest_low = low.rolling(window=self.willr_period).min()
            willr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # 2. MFI 계산
        if talib is not None:
            mfi = talib.MFI(high, low, close, volume, timeperiod=self.mfi_period)
        else:
            # Typical Price
            typical_price = (high + low + close) / 3
            
            # Raw Money Flow
            raw_mf = typical_price * volume
            
            # Positive / Negative Money Flow
            tp_diff = typical_price.diff()
            positive_mf = np.where(tp_diff > 0, raw_mf, 0)
            negative_mf = np.where(tp_diff < 0, raw_mf, 0)
            
            # Rolling sum
            positive_mf_sum = pd.Series(positive_mf).rolling(window=self.mfi_period).sum()
            negative_mf_sum = pd.Series(negative_mf).rolling(window=self.mfi_period).sum()
            
            # MFI
            money_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)  # 0으로 나누기 방지
            mfi = 100 - (100 / (1 + money_ratio))
        
        # 3. EMA 추세 필터
        if talib is not None:
            ema = talib.EMA(close, timeperiod=self.ema_period)
        else:
            ema = close.ewm(span=self.ema_period, adjust=False).mean()
        
        # 4. 신호 조건
        # Williams %R 과매도
        willr_oversold = willr < self.willr_oversold
        
        # MFI 과매도
        mfi_oversold = mfi < self.mfi_oversold
        
        # 이중 과매도
        if self.require_double_oversold:
            double_oversold = willr_oversold & mfi_oversold
        else:
            double_oversold = willr_oversold | mfi_oversold
        
        # Williams %R 회복 중 (이전보다 상승)
        willr_recovering = willr > willr.shift(1)
        
        # 추세 필터
        trend_condition = close > ema
        
        # 최종 신호
        buy_signal = double_oversold & willr_recovering & trend_condition
        
        signals = {
            'direction': np.where(buy_signal, 1, 0),
            'willr': willr,
            'mfi': mfi,
            'ema': ema
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
        
        max_invest = total_asset * self.max_investment_ratio
        invest_amount = min(available_cash, max_invest)
        
        if invest_amount <= 0:
            return None
        
        entry_price = row['open']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'willr_period': self.willr_period,
            'mfi_period': self.mfi_period,
            'willr_oversold': self.willr_oversold,
            'mfi_oversold': self.mfi_oversold,
        }
        
        # 현재 지표값 기록
        if signals.get('willr') is not None and len(signals['willr']) > idx:
            val = signals['willr']
            entry_conditions['willr'] = float(val.iloc[idx] if hasattr(val, 'iloc') else val[idx])
        if signals.get('mfi') is not None and len(signals['mfi']) > idx:
            val = signals['mfi']
            entry_conditions['mfi'] = float(val.iloc[idx] if hasattr(val, 'iloc') else val[idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason=f'WillR({entry_conditions.get("willr", 0):.1f}) + MFI({entry_conditions.get("mfi", 0):.1f}) Double Oversold',
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
        return ['willr_period', 'mfi_period']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'williams_mfi',
            'willr_period': 14,
            'willr_oversold': -80,
            'mfi_period': 14,
            'mfi_oversold': 20,
            'ema_period': 50,
            'require_double_oversold': True,
            'max_investment_ratio': 1.0
        }
