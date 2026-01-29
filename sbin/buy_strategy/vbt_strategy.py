"""
VBT (Volatility Breakout) 매수 전략

래리 윌리엄스의 변동성 돌파 전략을 기반으로 EMA, RSI, 거래량 필터를 추가한 버전입니다.
기존 sbin/strategy/vbt_strategy_012.py 를 리팩토링한 구현입니다.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. Some indicators may not work.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class VBTBuyStrategy(BaseBuyStrategy):
    """
    변동성 돌파 (Volatility Breakout) 매수 전략
    
    기본 원리:
    - 변동폭(Range) = 전일 고가 - 전일 저가
    - 돌파 기준 = 오늘 시가 + (변동폭 평균 * k)
    - 가격이 돌파 기준을 넘어서면 추세가 시작된 것으로 보고 매수
    
    추가 필터:
    - EMA (지수이동평균) 필터: 종가가 EMA 위/아래인지 확인
    - RSI 필터: 과매수/과매도 상태에서 리버스 시그널 생성
    - 거래량 필터: 평균 거래량 대비 현재 거래량 확인
    - 변동성 필터: 최소 변동성 이상인지 확인
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.window = config.get('window', 5)
        self.k_long = config.get('k_long', 0.5)
        self.k_short = config.get('k_short', 0.5)
        self.ema_period = config.get('ema_period', 15)
        self.rsi_period = config.get('rsi_period', 8)
        self.rsi_upper = config.get('rsi_upper', 70)
        self.rsi_lower = config.get('rsi_lower', 30)
        self.use_std = config.get('use_std', False)
        self.std_mult = config.get('std_mult', 1.0)
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.0)
        self.volatility_window = config.get('volatility_window', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.0)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        지표 사전 계산 (캐싱용)
        
        EMA, RSI를 미리 계산하여 백테스트 속도를 높입니다.
        """
        indicators = {}
        
        if talib is not None:
            indicators['ema'] = talib.EMA(df['close'], timeperiod=self.ema_period)
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        else:
            # talib 없을 경우 대체 계산
            indicators['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean().values
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).values
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        Returns:
            dict with:
                - direction: 1 (Long), -1 (Short), 0 (None)
                - target_long: 롱 진입 목표가
                - target_short: 숏 진입 목표가
                - reverse_to_short: RSI 과매수 -> 숏 전환 신호
                - reverse_to_long: RSI 과매도 -> 롱 전환 신호
        """
        open_val = df['open'].values
        close_val = df['close'].values
        high_val = df['high'].values
        low_val = df['low'].values
        vol_val = df['volume'].values
        
        # === 1. 가격 돌파 타겟 설정 (Range 계산) ===
        range_val = high_val - low_val
        shifted_range = pd.Series(range_val).shift(1)
        avg_range = shifted_range.rolling(window=self.window).mean().values
        
        if self.use_std:
            std_range = shifted_range.rolling(window=self.window).std().values
            target_long = open_val + (avg_range + std_range * self.std_mult) * self.k_long
            target_short = open_val - (avg_range + std_range * self.std_mult) * self.k_short
        else:
            target_long = open_val + avg_range * self.k_long
            target_short = open_val - avg_range * self.k_short
        
        # === 2. EMA 및 RSI 지표 로드/계산 ===
        if cached_data and 'ema' in cached_data:
            ema_val = cached_data['ema']
            rsi_val = cached_data['rsi']
        else:
            indicators = self.get_indicators(df)
            ema_val = indicators['ema']
            rsi_val = indicators['rsi']
        
        # numpy array로 통일
        if hasattr(ema_val, 'values'):
            ema_val = ema_val.values
        if hasattr(rsi_val, 'values'):
            rsi_val = rsi_val.values
        
        # === 3. 거래량 및 변동성 필터 ===
        avg_volume = pd.Series(vol_val).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = vol_val > (avg_volume * self.volume_mult)
        
        curr_range = high_val - low_val
        avg_volatility = pd.Series(curr_range).shift(1).rolling(window=self.volatility_window).mean().values
        volatility_filter = avg_volatility > (open_val * self.volatility_threshold)
        
        # === 4. 진입 신호 결합 ===
        # 롱 조건: (현재 고가 >= 목표가) AND (종가 > EMA) AND (필터 통과)
        signal_long = (high_val >= target_long) & (close_val > ema_val) & volume_filter & volatility_filter
        # 숏 조건: (현재 저가 <= 목표가) AND (종가 < EMA) AND (필터 통과)
        signal_short = (low_val <= target_short) & (close_val < ema_val) & volume_filter & volatility_filter
        
        # 리버스 조건: RSI 과매수/과매도
        reverse_to_short = rsi_val >= self.rsi_upper
        reverse_to_long = rsi_val <= self.rsi_lower
        
        # 방향 배열 생성
        vbt_direction = np.zeros(len(df), dtype=int)
        vbt_direction[signal_long] = 1
        vbt_direction[signal_short] = -1
        
        return {
            'direction': vbt_direction,
            'target_long': target_long,
            'target_short': target_short,
            'reverse_to_short': reverse_to_short,
            'reverse_to_long': reverse_to_long,
            'ema': ema_val,
            'rsi': rsi_val
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
        
        # 투자 금액 계산 (최대 투자 비율 적용)
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
            'window': self.window,
            'ema_period': self.ema_period,
            'rsi_period': self.rsi_period,
        }
        
        # RSI 값이 있으면 기록
        if signals.get('rsi') is not None and len(signals['rsi']) > idx:
            entry_conditions['rsi'] = float(signals['rsi'][idx])
        
        # EMA 값이 있으면 기록
        if signals.get('ema') is not None and len(signals['ema']) > idx:
            entry_conditions['ema'] = float(signals['ema'][idx])
        
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
            entry_reason='vbt_long_breakout' if signal_type == 1 else 'vbt_short_breakout',
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
        return ['window', 'k_long', 'k_short']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'vbt_with_filters',
            'window': 5,
            'k_long': 0.5,
            'k_short': 0.5,
            'ema_period': 15,
            'rsi_period': 8,
            'rsi_upper': 70,
            'rsi_lower': 30,
            'use_std': False,
            'std_mult': 1.0,
            'volume_window': 20,
            'volume_mult': 1.0,
            'volatility_window': 20,
            'volatility_threshold': 0.0,
            'max_investment_ratio': 1.0
        }
