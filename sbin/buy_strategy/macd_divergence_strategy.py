"""
MACD 다이버전스 매수 전략 (MACD Divergence Strategy)

가격과 MACD 지표가 반대 방향으로 움직이는 다이버전스를 감지하여
추세 전환 초기에 진입하는 전략입니다.

수학적 원리:
- MACD = EMA_12 - EMA_26
- Signal = EMA_9(MACD)
- Bullish Divergence: 가격 저점 ↓, MACD 저점 ↑
- Bearish Divergence: 가격 고점 ↑, MACD 고점 ↓
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. MACD Divergence strategy may use alternative calculations.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class MACDDivergenceStrategy(BaseBuyStrategy):
    """
    MACD 다이버전스 매수 전략
    
    진입 조건:
    - Long (Bullish Divergence): 가격 신저점 but MACD 고점 상승, MACD 상향 크로스
    - Short (Bearish Divergence): 가격 신고점 but MACD 저점 하락, MACD 하향 크로스
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # MACD 파라미터
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        
        # 다이버전스 감지 파라미터
        self.divergence_lookback = config.get('divergence_lookback', 10)
        
        # RSI 필터
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_threshold_long = config.get('rsi_threshold_long', 40)
        self.rsi_threshold_short = config.get('rsi_threshold_short', 60)
        self.use_rsi_filter = config.get('use_rsi_filter', True)
        
        # MACD 크로스오버 확인
        self.require_macd_cross = config.get('require_macd_cross', True)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        MACD, Signal, Histogram, RSI 지표 사전 계산
        """
        indicators = {}
        close = df['close'].values
        
        if talib is not None:
            macd, signal, hist = talib.MACD(
                close, 
                fastperiod=self.fast_period, 
                slowperiod=self.slow_period, 
                signalperiod=self.signal_period
            )
            indicators['macd'] = macd
            indicators['signal'] = signal
            indicators['histogram'] = hist
            indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)
        else:
            # pandas 기반 계산
            ema_fast = pd.Series(close).ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = pd.Series(close).ewm(span=self.slow_period, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=self.signal_period, adjust=False).mean()
            hist = macd - signal
            
            indicators['macd'] = macd.values
            indicators['signal'] = signal.values
            indicators['histogram'] = hist.values
            
            # RSI 계산
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).values
        
        return indicators
    
    def _find_local_extremes(self, data: np.ndarray, lookback: int) -> tuple:
        """
        로컬 최대/최소값 인덱스 탐지
        """
        n = len(data)
        local_max_idx = []
        local_min_idx = []
        
        for i in range(lookback, n - lookback):
            window = data[i - lookback:i + lookback + 1]
            if not np.any(np.isnan(window)):
                mid = lookback
                if data[i] == np.max(window):
                    local_max_idx.append(i)
                if data[i] == np.min(window):
                    local_min_idx.append(i)
        
        return local_max_idx, local_min_idx
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        MACD 다이버전스 신호 계산
        """
        # 지표 로드 또는 계산
        if cached_data and 'macd' in cached_data:
            macd = cached_data['macd']
            signal = cached_data['signal']
            histogram = cached_data['histogram']
            rsi = cached_data.get('rsi')
        else:
            indicators = self.get_indicators(df)
            macd = indicators['macd']
            signal = indicators['signal']
            histogram = indicators['histogram']
            rsi = indicators.get('rsi')
        
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        n = len(df)
        
        # 신호 배열 초기화
        bullish_divergence = np.zeros(n, dtype=bool)
        bearish_divergence = np.zeros(n, dtype=bool)
        
        # === 다이버전스 탐지 ===
        lookback = self.divergence_lookback
        
        for i in range(lookback * 2, n):
            # 최근 lookback 기간의 데이터
            price_window = close[i - lookback * 2:i + 1]
            macd_window = histogram[i - lookback * 2:i + 1]
            
            if np.any(np.isnan(macd_window)):
                continue
            
            # Bullish Divergence: 가격 신저점 but MACD 고점 상승
            price_min_idx = np.argmin(price_window[-lookback:])
            if price_min_idx == 0 or price_min_idx == lookback - 1:
                # 가격이 최근 window에서 저점
                prev_price_min = np.min(price_window[:lookback])
                curr_price_min = price_window[-1]
                prev_macd_min = np.min(macd_window[:lookback])
                curr_macd_min = np.min(macd_window[-lookback:])
                
                # 가격은 하락, MACD는 상승 = Bullish Divergence
                if curr_price_min < prev_price_min and curr_macd_min > prev_macd_min:
                    bullish_divergence[i] = True
            
            # Bearish Divergence: 가격 신고점 but MACD 저점 하락
            price_max_idx = np.argmax(price_window[-lookback:])
            if price_max_idx == 0 or price_max_idx == lookback - 1:
                prev_price_max = np.max(price_window[:lookback])
                curr_price_max = price_window[-1]
                prev_macd_max = np.max(macd_window[:lookback])
                curr_macd_max = np.max(macd_window[-lookback:])
                
                # 가격은 상승, MACD는 하락 = Bearish Divergence
                if curr_price_max > prev_price_max and curr_macd_max < prev_macd_max:
                    bearish_divergence[i] = True
        
        # === MACD 크로스오버 확인 ===
        if self.require_macd_cross:
            prev_macd = np.roll(macd, 1)
            prev_signal = np.roll(signal, 1)
            prev_macd[0] = np.nan
            prev_signal[0] = np.nan
            
            macd_cross_up = (macd > signal) & (prev_macd <= prev_signal)
            macd_cross_down = (macd < signal) & (prev_macd >= prev_signal)
        else:
            macd_cross_up = macd > signal
            macd_cross_down = macd < signal
        
        # === RSI 필터 ===
        if self.use_rsi_filter and rsi is not None:
            rsi_long = rsi < self.rsi_threshold_long
            rsi_short = rsi > self.rsi_threshold_short
        else:
            rsi_long = np.ones(n, dtype=bool)
            rsi_short = np.ones(n, dtype=bool)
        
        # === 최종 신호 결합 ===
        signal_long = bullish_divergence & macd_cross_up & rsi_long
        signal_short = bearish_divergence & macd_cross_down & rsi_short
        
        # NaN 처리
        signal_long = np.nan_to_num(signal_long, nan=False).astype(bool)
        signal_short = np.nan_to_num(signal_short, nan=False).astype(bool)
        
        # 방향 배열 생성
        direction = np.zeros(n, dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': None,
            'target_short': None,
            'reverse_to_short': bearish_divergence,
            'reverse_to_long': bullish_divergence,
            'macd': macd,
            'signal': signal,
            'histogram': histogram,
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
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'divergence_lookback': self.divergence_lookback,
        }
        
        # 현재 지표값 기록
        if signals.get('macd') is not None and len(signals['macd']) > idx:
            entry_conditions['macd'] = float(signals['macd'][idx])
        if signals.get('histogram') is not None and len(signals['histogram']) > idx:
            entry_conditions['histogram'] = float(signals['histogram'][idx])
        if signals.get('rsi') is not None and len(signals['rsi']) > idx:
            entry_conditions['rsi'] = float(signals['rsi'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='macd_divergence_long' if signal_type == 1 else 'macd_divergence_short',
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
        return ['fast_period', 'slow_period', 'signal_period']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'macd_divergence',
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'divergence_lookback': 10,
            'rsi_period': 14,
            'rsi_threshold_long': 40,
            'rsi_threshold_short': 60,
            'use_rsi_filter': True,
            'require_macd_cross': True,
            'max_investment_ratio': 1.0
        }
