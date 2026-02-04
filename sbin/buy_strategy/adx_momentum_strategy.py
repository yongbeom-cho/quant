"""
ADX 모멘텀 필터 매수 전략 (ADX Momentum Filter Strategy)

ADX(Average Directional Index)와 +DI/-DI를 활용하여 
추세의 강도와 방향을 확인 후 진입하는 전략입니다.

수학적 원리:
- ADX = Smoothed(DX), DX = |+DI - (-DI)| / (+DI + (-DI)) × 100
- ADX > threshold: 강한 추세 존재
- +DI > -DI: 상승 추세, +DI < -DI: 하락 추세
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. ADX Momentum strategy may not work correctly.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class ADXMomentumStrategy(BaseBuyStrategy):
    """
    ADX 모멘텀 필터 매수 전략
    
    진입 조건:
    - Long: ADX > threshold, +DI > -DI, +DI가 -DI 상향 돌파, 가격 > EMA
    - Short: ADX > threshold, -DI > +DI, -DI가 +DI 상향 돌파, 가격 < EMA
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ADX/DI 관련 파라미터
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        
        # EMA 필터 파라미터
        self.ema_period = config.get('ema_period', 20)
        self.use_ema_filter = config.get('use_ema_filter', True)
        
        # 크로스오버 확인 옵션
        self.require_crossover = config.get('require_crossover', True)
        
        # 거래량 필터
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.0)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ADX, +DI, -DI, EMA 지표 사전 계산
        """
        indicators = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if talib is not None:
            # talib을 사용한 계산
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            indicators['ema'] = talib.EMA(close, timeperiod=self.ema_period)
        else:
            # talib 없을 경우 pandas 기반 대체 계산
            indicators['adx'] = self._calculate_adx_pandas(df)
            indicators['plus_di'] = np.zeros(len(df))  # 간소화된 대체
            indicators['minus_di'] = np.zeros(len(df))
            indicators['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean().values
        
        return indicators
    
    def _calculate_adx_pandas(self, df: pd.DataFrame) -> np.ndarray:
        """pandas 기반 간소화된 ADX 계산 (talib 대체용)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=self.adx_period).mean()
        
        # 간소화된 DX 계산 (정확한 ADX와 다를 수 있음)
        price_change = close.diff()
        dx = (price_change.abs() / atr * 100).fillna(0)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx.values
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ADX 모멘텀 기반 매수 신호 계산
        """
        # 지표 로드 또는 계산
        if cached_data and 'adx' in cached_data:
            adx = cached_data['adx']
            plus_di = cached_data['plus_di']
            minus_di = cached_data['minus_di']
            ema = cached_data['ema']
        else:
            indicators = self.get_indicators(df)
            adx = indicators['adx']
            plus_di = indicators['plus_di']
            minus_di = indicators['minus_di']
            ema = indicators['ema']
        
        close = df['close'].values
        volume = df['volume'].values
        
        # numpy array로 변환
        if hasattr(adx, 'values'):
            adx = adx
        if hasattr(plus_di, 'values'):
            plus_di = plus_di
        if hasattr(minus_di, 'values'):
            minus_di = minus_di
        if hasattr(ema, 'values'):
            ema = ema
        
        # === 1. ADX 조건: 추세 강도 확인 ===
        adx_condition = adx > self.adx_threshold
        
        # === 2. DI 방향 조건 ===
        long_direction = plus_di > minus_di    # 상승 추세
        short_direction = minus_di > plus_di   # 하락 추세
        
        # === 3. DI 크로스오버 (선택적) ===
        if self.require_crossover:
            # 직전 봉과 비교하여 크로스오버 확인
            prev_plus_di = np.roll(plus_di, 1)
            prev_minus_di = np.roll(minus_di, 1)
            prev_plus_di[0] = np.nan
            prev_minus_di[0] = np.nan
            
            # Long: +DI가 -DI를 상향 돌파
            long_crossover = (plus_di > minus_di) & (prev_plus_di <= prev_minus_di)
            # Short: -DI가 +DI를 상향 돌파
            short_crossover = (minus_di > plus_di) & (prev_minus_di <= prev_plus_di)
        else:
            long_crossover = long_direction
            short_crossover = short_direction
        
        # === 4. EMA 필터 (선택적) ===
        if self.use_ema_filter:
            ema_long = close > ema
            ema_short = close < ema
        else:
            ema_long = np.ones(len(df), dtype=bool)
            ema_short = np.ones(len(df), dtype=bool)
        
        # === 5. 거래량 필터 ===
        avg_volume = pd.Series(volume).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = volume > (avg_volume * self.volume_mult)
        
        # === 6. 최종 신호 결합 ===
        signal_long = adx_condition & long_crossover & ema_long & volume_filter
        signal_short = adx_condition & short_crossover & ema_short & volume_filter
        
        # NaN 처리
        signal_long = np.nan_to_num(signal_long, nan=False).astype(bool)
        signal_short = np.nan_to_num(signal_short, nan=False).astype(bool)
        
        # 방향 배열 생성
        direction = np.zeros(len(df), dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': None,  # 시가 진입
            'target_short': None,
            'reverse_to_short': np.zeros(len(df), dtype=bool),
            'reverse_to_long': np.zeros(len(df), dtype=bool),
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'ema': ema
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
        row = df.iloc[idx]
        
        # 투자 금액 계산
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
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'ema_period': self.ema_period,
        }
        
        # 현재 지표값 기록
        if signals.get('adx') is not None and len(signals['adx']) > idx:
            entry_conditions['adx'] = float(signals['adx'][idx])
        if signals.get('plus_di') is not None and len(signals['plus_di']) > idx:
            entry_conditions['plus_di'] = float(signals['plus_di'][idx])
        if signals.get('minus_di') is not None and len(signals['minus_di']) > idx:
            entry_conditions['minus_di'] = float(signals['minus_di'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='adx_momentum_long' if signal_type == 1 else 'adx_momentum_short',
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
        return ['adx_period', 'adx_threshold']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'adx_momentum',
            'adx_period': 14,
            'adx_threshold': 25,
            'ema_period': 20,
            'use_ema_filter': True,
            'require_crossover': True,
            'volume_window': 20,
            'volume_mult': 1.0,
            'max_investment_ratio': 1.0
        }
