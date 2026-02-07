"""
VWAP 회귀 매수 전략 (VWAP Mean Reversion Strategy)

VWAP(Volume Weighted Average Price)에서 크게 벗어난 가격이
평균으로 회귀하는 경향을 이용한 전략입니다.

수학적 원리:
- VWAP = Σ(TP × V) / Σ(V), TP = (H + L + C) / 3
- 가격 < VWAP - 2σ: 과매도 → Long
- 가격 > VWAP + 2σ: 과매수 → Short
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. VWAP Reversion strategy may use alternative calculations.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class VWAPReversionStrategy(BaseBuyStrategy):
    """
    VWAP 회귀 매수 전략
    
    진입 조건:
    - Long: 가격 < VWAP - 2σ, RSI 과매도, 반등 캔들 확인
    - Short: 가격 > VWAP + 2σ, RSI 과매수, 하락 캔들 확인
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # VWAP 파라미터
        self.vwap_period = config.get('vwap_period', 20)  # 롤링 VWAP 기간
        self.vwap_std_mult = config.get('vwap_std_mult', 2.0)
        
        # RSI 필터
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.use_rsi_filter = config.get('use_rsi_filter', True)
        
        # 캔들 확인
        self.require_reversal_candle = config.get('require_reversal_candle', True)
        
        # 손절 레벨
        self.stop_std_mult = config.get('stop_std_mult', 3.0)
        
        # 거래량 필터
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.0)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        VWAP, VWAP 밴드, RSI 지표 사전 계산
        """
        indicators = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # 롤링 VWAP 계산
        tp_vol = tp * volume
        cum_tp_vol = pd.Series(tp_vol).rolling(window=self.vwap_period).sum()
        cum_vol = pd.Series(volume).rolling(window=self.vwap_period).sum()
        vwap = (cum_tp_vol / cum_vol).values
        
        # VWAP 표준편차 (가격과 VWAP의 차이 기반)
        price_vwap_diff = close - vwap
        vwap_std = pd.Series(price_vwap_diff).rolling(window=self.vwap_period).std().values
        
        indicators['vwap'] = vwap
        indicators['vwap_std'] = vwap_std
        indicators['upper_band'] = vwap + (vwap_std * self.vwap_std_mult)
        indicators['lower_band'] = vwap - (vwap_std * self.vwap_std_mult)
        indicators['stop_upper'] = vwap + (vwap_std * self.stop_std_mult)
        indicators['stop_lower'] = vwap - (vwap_std * self.stop_std_mult)
        
        # RSI 계산
        if talib is not None:
            indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)
        else:
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).values
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        VWAP 회귀 신호 계산
        """
        # 지표 로드 또는 계산
        if cached_data and 'vwap' in cached_data:
            vwap = cached_data['vwap']
            upper_band = cached_data['upper_band']
            lower_band = cached_data['lower_band']
            rsi = cached_data.get('rsi')
        else:
            indicators = self.get_indicators(df)
            vwap = indicators['vwap']
            upper_band = indicators['upper_band']
            lower_band = indicators['lower_band']
            rsi = indicators.get('rsi')
        
        close = df['close'].values
        volume = df['volume'].values
        n = len(df)
        
        # === 1. VWAP 이탈 조건 ===
        oversold = close < lower_band   # 하단 이탈 (과매도)
        overbought = close > upper_band  # 상단 이탈 (과매수)
        
        # === 2. RSI 필터 ===
        if self.use_rsi_filter and rsi is not None:
            rsi_long = rsi < self.rsi_oversold
            rsi_short = rsi > self.rsi_overbought
        else:
            rsi_long = np.ones(n, dtype=bool)
            rsi_short = np.ones(n, dtype=bool)
        
        # === 3. 반전 캔들 확인 (선택적) ===
        if self.require_reversal_candle:
            prev_close = np.roll(close, 1)
            prev_close[0] = np.nan
            # Long: 현재 종가 > 이전 종가 (반등)
            reversal_long = close > prev_close
            # Short: 현재 종가 < 이전 종가 (반락)
            reversal_short = close < prev_close
        else:
            reversal_long = np.ones(n, dtype=bool)
            reversal_short = np.ones(n, dtype=bool)
        
        # === 4. 거래량 필터 ===
        avg_volume = pd.Series(volume).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = volume > (avg_volume * self.volume_mult)
        
        # === 5. 최종 신호 결합 ===
        signal_long = oversold & rsi_long & reversal_long & volume_filter
        signal_short = overbought & rsi_short & reversal_short & volume_filter
        
        # NaN 처리
        signal_long = np.nan_to_num(signal_long, nan=False).astype(bool)
        signal_short = np.nan_to_num(signal_short, nan=False).astype(bool)
        
        # 방향 배열 생성
        direction = np.zeros(n, dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': vwap,   # VWAP이 목표가
            'target_short': vwap,
            'reverse_to_short': overbought,
            'reverse_to_long': oversold,
            'vwap': vwap,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'rsi': rsi
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
            'vwap_period': self.vwap_period,
            'vwap_std_mult': self.vwap_std_mult,
            'rsi_period': self.rsi_period,
        }
        
        # 현재 지표값 기록
        if signals.get('vwap') is not None and len(signals['vwap']) > idx:
            entry_conditions['vwap'] = float(signals['vwap'][idx])
        if signals.get('rsi') is not None and len(signals['rsi']) > idx:
            entry_conditions['rsi'] = float(signals['rsi'][idx])
        if signals.get('upper_band') is not None and len(signals['upper_band']) > idx:
            entry_conditions['upper_band'] = float(signals['upper_band'][idx])
        if signals.get('lower_band') is not None and len(signals['lower_band']) > idx:
            entry_conditions['lower_band'] = float(signals['lower_band'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='vwap_reversion_long' if signal_type == 1 else 'vwap_reversion_short',
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
        return ['vwap_period', 'vwap_std_mult']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'vwap_reversion',
            'vwap_period': 20,
            'vwap_std_mult': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'use_rsi_filter': True,
            'require_reversal_candle': True,
            'stop_std_mult': 3.0,
            'volume_window': 20,
            'volume_mult': 1.0,
            'max_investment_ratio': 1.0
        }
