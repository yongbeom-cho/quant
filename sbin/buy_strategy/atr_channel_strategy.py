"""
ATR 채널 돌파 매수 전략 (ATR Channel Breakout Strategy)

ATR(Average True Range)을 기반으로 한 Keltner Channel 변형으로,
가격이 "정상 범위"를 벗어나는 움직임을 포착하는 전략입니다.

수학적 원리:
- Upper = EMA_n + k × ATR_n
- Lower = EMA_n - k × ATR_n
- 돌파: 종가가 채널 상단/하단을 벗어날 때 진입
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. ATR Channel strategy may use alternative calculations.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class ATRChannelStrategy(BaseBuyStrategy):
    """
    ATR 채널 돌파 매수 전략
    
    진입 조건:
    - Long: 종가 > Upper Channel, ATR 증가 추세, 거래량 확인
    - Short: 종가 < Lower Channel, ATR 증가 추세, 거래량 확인
    - ADX > threshold (추세 존재)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 채널 파라미터
        self.ema_period = config.get('ema_period', 20)
        self.atr_period = config.get('atr_period', 14)
        self.channel_mult = config.get('channel_mult', 2.0)
        
        # ADX 필터
        self.use_adx_filter = config.get('use_adx_filter', True)
        self.adx_threshold = config.get('adx_threshold', 20)
        
        # ATR 증가 추세 확인
        self.require_atr_expansion = config.get('require_atr_expansion', True)
        self.atr_ma_period = config.get('atr_ma_period', 10)
        
        # 거래량 필터
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.0)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        EMA, ATR, ADX 지표 사전 계산
        """
        indicators = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if talib is not None:
            indicators['ema'] = talib.EMA(close, timeperiod=self.ema_period)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
            if self.use_adx_filter:
                indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.atr_period)
        else:
            # pandas 기반 계산
            indicators['ema'] = pd.Series(close).ewm(span=self.ema_period, adjust=False).mean().values
            
            # ATR 계산
            tr = pd.concat([
                pd.Series(high) - pd.Series(low),
                (pd.Series(high) - pd.Series(close).shift(1)).abs(),
                (pd.Series(low) - pd.Series(close).shift(1)).abs()
            ], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(window=self.atr_period).mean().values
            
            if self.use_adx_filter:
                indicators['adx'] = np.ones(len(df)) * 25  # 기본값
        
        # 채널 계산
        ema = indicators['ema']
        atr = indicators['atr']
        indicators['upper_channel'] = ema + (atr * self.channel_mult)
        indicators['lower_channel'] = ema - (atr * self.channel_mult)
        
        # ATR 이동평균 (확장 추세 확인용)
        indicators['atr_ma'] = pd.Series(atr).rolling(window=self.atr_ma_period).mean().values
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ATR 채널 돌파 신호 계산
        """
        # 지표 로드 또는 계산
        if cached_data and 'atr' in cached_data:
            ema = cached_data['ema']
            atr = cached_data['atr']
            upper = cached_data['upper_channel']
            lower = cached_data['lower_channel']
            atr_ma = cached_data['atr_ma']
            adx = cached_data.get('adx')
        else:
            indicators = self.get_indicators(df)
            ema = indicators['ema']
            atr = indicators['atr']
            upper = indicators['upper_channel']
            lower = indicators['lower_channel']
            atr_ma = indicators['atr_ma']
            adx = indicators.get('adx')
        
        close = df['close'].values
        volume = df['volume'].values
        n = len(df)
        
        # === 1. 채널 돌파 조건 ===
        breakout_long = close > upper   # 상단 돌파
        breakout_short = close < lower  # 하단 돌파
        
        # === 2. ATR 확장 추세 (선택적) ===
        if self.require_atr_expansion:
            atr_expanding = atr > atr_ma
        else:
            atr_expanding = np.ones(n, dtype=bool)
        
        # === 3. ADX 필터 ===
        if self.use_adx_filter and adx is not None:
            adx_condition = adx > self.adx_threshold
        else:
            adx_condition = np.ones(n, dtype=bool)
        
        # === 4. 거래량 필터 ===
        avg_volume = pd.Series(volume).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = volume > (avg_volume * self.volume_mult)
        
        # === 5. 최종 신호 결합 ===
        signal_long = breakout_long & atr_expanding & adx_condition & volume_filter
        signal_short = breakout_short & atr_expanding & adx_condition & volume_filter
        
        # NaN 처리
        signal_long = np.nan_to_num(signal_long, nan=False).astype(bool)
        signal_short = np.nan_to_num(signal_short, nan=False).astype(bool)
        
        # 방향 배열 생성
        direction = np.zeros(n, dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': upper,
            'target_short': lower,
            'reverse_to_short': np.zeros(n, dtype=bool),
            'reverse_to_long': np.zeros(n, dtype=bool),
            'ema': ema,
            'atr': atr,
            'upper_channel': upper,
            'lower_channel': lower,
            'adx': adx
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
        
        # 진입 가격 결정
        if signal_type == 1:
            entry_price = max(row['open'], signals['upper_channel'][idx]) if signals.get('upper_channel') is not None else row['open']
        else:
            entry_price = min(row['open'], signals['lower_channel'][idx]) if signals.get('lower_channel') is not None else row['open']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'ema_period': self.ema_period,
            'atr_period': self.atr_period,
            'channel_mult': self.channel_mult,
        }
        
        # 현재 지표값 기록
        if signals.get('atr') is not None and len(signals['atr']) > idx:
            entry_conditions['atr'] = float(signals['atr'][idx])
        if signals.get('upper_channel') is not None and len(signals['upper_channel']) > idx:
            entry_conditions['upper_channel'] = float(signals['upper_channel'][idx])
        if signals.get('lower_channel') is not None and len(signals['lower_channel']) > idx:
            entry_conditions['lower_channel'] = float(signals['lower_channel'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='atr_channel_long' if signal_type == 1 else 'atr_channel_short',
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
        return ['ema_period', 'atr_period', 'channel_mult']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'atr_channel',
            'ema_period': 20,
            'atr_period': 14,
            'channel_mult': 2.0,
            'use_adx_filter': True,
            'adx_threshold': 20,
            'require_atr_expansion': True,
            'atr_ma_period': 10,
            'volume_window': 20,
            'volume_mult': 1.0,
            'max_investment_ratio': 1.0
        }
