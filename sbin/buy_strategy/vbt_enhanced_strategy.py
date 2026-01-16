"""
VBT Enhanced (고도화된 변동성 돌파) 매수 전략

기본 VBT 전략에 ADX(추세 강도), ATR(평균 변동폭) 지표를 추가한 고도화 버전입니다.
기존 sbin/strategy/vbt_strategy_020_enhanced.py 를 리팩토링한 구현입니다.

=============================================================================
전략 특징 (기본 VBT 대비 추가된 요소)
=============================================================================
1. ATR 기반 돌파: Range 대신 ATR(변동폭)을 기준으로 돌파 판단
2. ADX 필터: 추세 강도가 일정 이상일 때만 진입 (횡보장 회피)
3. RSI Zone Exit: 과매수 구간 '탈출' 시점에 리버스 (더 정밀한 반전)
4. 강한 추세 추종: ADX가 매우 높고 상승 중이면 리버스 억제

=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. VBT Enhanced strategy requires talib.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class VBTEnhancedBuyStrategy(BaseBuyStrategy):
    """
    고도화된 변동성 돌파 (VBT Enhanced) 매수 전략
    
    기본 VBT에 ADX, ATR 지표를 추가하여 정밀도를 높인 전략입니다.
    
    추가 필터:
    - ADX 필터: 추세 강도 확인 (25 이상이면 강한 추세)
    - ATR 기반 돌파: 변동폭을 기준으로 돌파 판단
    - RSI Zone Exit: 과매수/과매도 구간 탈출 시 리버스
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.k_long = config.get('k_long', 0.5)
        self.k_short = config.get('k_short', 0.5)
        
        # EMA 설정
        self.ema_confirm = config.get('ema_confirm', True)
        self.ema_period = config.get('ema_period', 20)
        
        # RSI 설정
        self.rsi_confirm = config.get('rsi_confirm', False)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_upper = config.get('rsi_upper', 70)
        self.rsi_lower = config.get('rsi_lower', 30)
        
        # ADX 설정 (추세 강도)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 20.0)
        
        # ATR 설정 (변동폭)
        self.atr_period = config.get('atr_period', 14)
        self.atr_mult = config.get('atr_mult', 1.0)
        
        # 거래량 설정
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.0)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        지표 사전 계산 (캐싱용)
        
        EMA, RSI, ADX, ATR을 미리 계산합니다.
        """
        indicators = {}
        
        if talib is not None:
            close = df['close']
            high = df['high']
            low = df['low']
            
            indicators['ema'] = talib.EMA(close, timeperiod=self.ema_period)
            indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        Returns:
            dict with:
                - direction: 1 (Long), -1 (Short), 0 (None)
                - target_long: 롱 진입 목표가
                - target_short: 숏 진입 목표가
                - reverse_to_short: 숏 전환 신호
                - reverse_to_long: 롱 전환 신호
                - atr: ATR 값 (sell 전략에서 trailing stop에 사용)
        """
        if talib is None:
            raise ImportError("talib is required for VBT Enhanced strategy")
        
        open_val = df['open'].values
        close_val = df['close'].values
        high_val = df['high'].values
        low_val = df['low'].values
        vol_val = df['volume'].values
        
        # === 1. 지표 계산/로드 ===
        if cached_data and 'ema' in cached_data:
            ema_val = cached_data['ema']
            rsi_val = cached_data['rsi']
            adx_val = cached_data['adx']
            atr_val = cached_data['atr']
        else:
            indicators = self.get_indicators(df)
            ema_val = indicators['ema']
            rsi_val = indicators['rsi']
            adx_val = indicators['adx']
            atr_val = indicators['atr']
        
        # numpy array로 통일
        if hasattr(ema_val, 'values'):
            ema_val = ema_val.values
        if hasattr(rsi_val, 'values'):
            rsi_val = rsi_val.values
        if hasattr(adx_val, 'values'):
            adx_val = adx_val.values
        if hasattr(atr_val, 'values'):
            atr_val = atr_val.values
        
        # === 2. ATR 기반 돌파 타겟 설정 ===
        prev_atr = pd.Series(atr_val).shift(1).values
        target_long = open_val + prev_atr * self.k_long * self.atr_mult
        target_short = open_val - prev_atr * self.k_short * self.atr_mult
        
        # === 3. 필터 로직 ===
        # ADX 필터: 추세 강도가 설정값보다 커야 함
        adx_filter = adx_val > self.adx_threshold
        
        # 거래량 필터
        avg_volume = pd.Series(vol_val).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = vol_val > (avg_volume * self.volume_mult)
        
        # EMA 필터
        if self.ema_confirm:
            long_trend = close_val > ema_val
            short_trend = close_val < ema_val
        else:
            long_trend = np.ones(len(df), dtype=bool)
            short_trend = np.ones(len(df), dtype=bool)
        
        # RSI 필터
        if self.rsi_confirm:
            long_rsi = rsi_val < self.rsi_upper
            short_rsi = rsi_val > self.rsi_lower
        else:
            long_rsi = np.ones(len(df), dtype=bool)
            short_rsi = np.ones(len(df), dtype=bool)
        
        # === 4. 진입 신호 결합 ===
        signal_long = (high_val >= target_long) & adx_filter & volume_filter & long_trend & long_rsi
        signal_short = (low_val <= target_short) & adx_filter & volume_filter & short_trend & short_rsi
        
        # === 5. 리버스 신호 고도화 (RSI Zone Exit) ===
        prev_rsi = pd.Series(rsi_val).shift(1).values
        prev_adx = pd.Series(adx_val).shift(1).values
        
        # RSI Zone Exit: 전봉에서 과매수/과매도였다가 벗어날 때
        rev_short_base = (prev_rsi >= self.rsi_upper) & (rsi_val < self.rsi_upper)
        rev_long_base = (prev_rsi <= self.rsi_lower) & (rsi_val > self.rsi_lower)
        
        # 강한 추세 추종 필터 (ADX가 매우 높고 상승 중이면 리버스 억제)
        strong_trend_rising = (adx_val > prev_adx) & (adx_val > self.adx_threshold * 1.8)
        
        reverse_to_short = rev_short_base & ~strong_trend_rising
        reverse_to_long = rev_long_base & ~strong_trend_rising
        
        # === 6. 방향 배열 생성 ===
        direction = np.zeros(len(df), dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': target_long,
            'target_short': target_short,
            'reverse_to_short': reverse_to_short,
            'reverse_to_long': reverse_to_long,
            'ema': ema_val,
            'rsi': rsi_val,
            'adx': adx_val,
            'atr': atr_val  # trailing stop에 사용
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
        
        # 진입 가격 계산
        entry_price = self.get_entry_price(df, idx, signal_type, signals)
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'k': self.k_long if signal_type == 1 else self.k_short,
            'ema_period': self.ema_period,
            'rsi_period': self.rsi_period,
            'adx_threshold': self.adx_threshold,
            'atr_period': self.atr_period,
        }
        
        # 지표 값 기록
        if signals.get('rsi') is not None and len(signals['rsi']) > idx:
            entry_conditions['rsi'] = float(signals['rsi'][idx])
        if signals.get('adx') is not None and len(signals['adx']) > idx:
            entry_conditions['adx'] = float(signals['adx'][idx])
        if signals.get('atr') is not None and len(signals['atr']) > idx:
            entry_conditions['atr'] = float(signals['atr'][idx])
        
        # 목표가 기록
        if signal_type == 1 and signals.get('target_long') is not None:
            entry_conditions['target_price'] = float(signals['target_long'][idx])
        elif signal_type == -1 and signals.get('target_short') is not None:
            entry_conditions['target_price'] = float(signals['target_short'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='vbt_enhanced_long' if signal_type == 1 else 'vbt_enhanced_short',
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
        return ['k_long', 'k_short']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'vbt_with_filters_enhanced',
            'k_long': 0.5,
            'k_short': 0.5,
            'ema_confirm': True,
            'ema_period': 20,
            'rsi_confirm': False,
            'rsi_period': 14,
            'rsi_upper': 70,
            'rsi_lower': 30,
            'adx_period': 14,
            'adx_threshold': 20.0,
            'atr_period': 14,
            'atr_mult': 1.0,
            'volume_window': 20,
            'volume_mult': 1.0,
            'max_investment_ratio': 1.0
        }
