"""
볼린저밴드 압축 돌파 매수 전략 (Bollinger Squeeze Breakout Strategy)

볼린저밴드가 압축(Squeeze)된 후 확장할 때 큰 가격 움직임이 
발생하는 원리를 활용한 전략입니다.

수학적 원리:
- BBW (Bollinger Band Width) = (Upper - Lower) / Middle
- Squeeze 감지: BBW가 최근 N일 최저치에 도달
- 돌파: 가격이 압축 상태에서 밴드를 돌파할 때 진입
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. BB Squeeze strategy may use alternative calculations.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class BBSqueezeStrategy(BaseBuyStrategy):
    """
    볼린저밴드 압축 돌파 매수 전략
    
    진입 조건:
    - BBW가 최근 squeeze_lookback 기간의 하위 squeeze_percentile% 이내 (압축 상태)
    - Long: 종가 > Upper Band (상단 돌파)
    - Short: 종가 < Lower Band (하단 돌파)
    - 거래량 확인
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 볼린저밴드 파라미터
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        
        # 압축 감지 파라미터
        self.squeeze_percentile = config.get('squeeze_percentile', 20)
        self.squeeze_lookback = config.get('squeeze_lookback', 50)
        
        # 거래량 필터
        self.volume_window = config.get('volume_window', 20)
        self.volume_mult = config.get('volume_mult', 1.5)
        
        # ADX 필터 (선택적)
        self.use_adx_filter = config.get('use_adx_filter', False)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 20)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        볼린저밴드 및 관련 지표 사전 계산
        """
        indicators = {}
        close = df['close'].values
        
        if talib is not None:
            upper, middle, lower = talib.BBANDS(
                close, 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_std, 
                nbdevdn=self.bb_std
            )
            indicators['upper'] = upper
            indicators['middle'] = middle
            indicators['lower'] = lower
            
            if self.use_adx_filter:
                high = df['high'].values
                low = df['low'].values
                indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
        else:
            # pandas 기반 계산
            middle = df['close'].rolling(window=self.bb_period).mean()
            std = df['close'].rolling(window=self.bb_period).std()
            upper = middle + (std * self.bb_std)
            lower = middle - (std * self.bb_std)
            
            indicators['upper'] = upper.values
            indicators['middle'] = middle.values
            indicators['lower'] = lower.values
            
            if self.use_adx_filter:
                indicators['adx'] = np.zeros(len(df))  # 간소화된 대체
        
        # BBW (Bollinger Band Width) 계산
        bbw = (indicators['upper'] - indicators['lower']) / indicators['middle']
        indicators['bbw'] = bbw
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        볼린저밴드 압축 돌파 신호 계산
        """
        # 지표 로드 또는 계산
        if cached_data and 'bbw' in cached_data:
            upper = cached_data['upper']
            middle = cached_data['middle']
            lower = cached_data['lower']
            bbw = cached_data['bbw']
            adx = cached_data.get('adx')
        else:
            indicators = self.get_indicators(df)
            upper = indicators['upper']
            middle = indicators['middle']
            lower = indicators['lower']
            bbw = indicators['bbw']
            adx = indicators.get('adx')
        
        close = df['close'].values
        volume = df['volume'].values
        n = len(df)
        
        # === 1. 압축(Squeeze) 상태 감지 ===
        # 롤링 윈도우에서 BBW의 하위 percentile 계산
        squeeze_condition = np.zeros(n, dtype=bool)
        
        for i in range(self.squeeze_lookback, n):
            window_bbw = bbw[i - self.squeeze_lookback:i]
            valid_bbw = window_bbw[~np.isnan(window_bbw)]
            if len(valid_bbw) > 0:
                threshold = np.percentile(valid_bbw, self.squeeze_percentile)
                squeeze_condition[i] = bbw[i] <= threshold
        
        # === 2. 돌파 조건 ===
        breakout_long = close > upper   # 상단 돌파
        breakout_short = close < lower  # 하단 돌파
        
        # === 3. 거래량 필터 ===
        avg_volume = pd.Series(volume).shift(1).rolling(window=self.volume_window).mean().values
        volume_filter = volume > (avg_volume * self.volume_mult)
        
        # === 4. ADX 필터 (선택적) ===
        if self.use_adx_filter and adx is not None:
            adx_condition = adx > self.adx_threshold
        else:
            adx_condition = np.ones(n, dtype=bool)
        
        # === 5. 최종 신호 결합 ===
        # 압축 상태에서 돌파 시 진입
        signal_long = squeeze_condition & breakout_long & volume_filter & adx_condition
        signal_short = squeeze_condition & breakout_short & volume_filter & adx_condition
        
        # NaN 처리
        signal_long = np.nan_to_num(signal_long, nan=False).astype(bool)
        signal_short = np.nan_to_num(signal_short, nan=False).astype(bool)
        
        # 방향 배열 생성
        direction = np.zeros(n, dtype=int)
        direction[signal_long] = 1
        direction[signal_short] = -1
        
        return {
            'direction': direction,
            'target_long': upper,   # 상단 밴드가 목표가
            'target_short': lower,  # 하단 밴드가 목표가
            'reverse_to_short': np.zeros(n, dtype=bool),
            'reverse_to_long': np.zeros(n, dtype=bool),
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bbw': bbw,
            'squeeze': squeeze_condition
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
            # Long: 상단 밴드 돌파 가격 또는 시가
            entry_price = max(row['open'], signals['upper'][idx]) if signals.get('upper') is not None else row['open']
        else:
            # Short: 하단 밴드 돌파 가격 또는 시가
            entry_price = min(row['open'], signals['lower'][idx]) if signals.get('lower') is not None else row['open']
        
        if entry_price <= 0:
            return None
        
        quantity = invest_amount / entry_price
        
        # 진입 조건 기록
        entry_conditions = {
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'squeeze_percentile': self.squeeze_percentile,
        }
        
        # 현재 지표값 기록
        if signals.get('bbw') is not None and len(signals['bbw']) > idx:
            entry_conditions['bbw'] = float(signals['bbw'][idx])
        if signals.get('upper') is not None and len(signals['upper']) > idx:
            entry_conditions['upper_band'] = float(signals['upper'][idx])
        if signals.get('lower') is not None and len(signals['lower']) > idx:
            entry_conditions['lower_band'] = float(signals['lower'][idx])
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='bb_squeeze_long' if signal_type == 1 else 'bb_squeeze_short',
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
        return ['bb_period', 'bb_std']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'bb_squeeze',
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_percentile': 20,
            'squeeze_lookback': 50,
            'volume_window': 20,
            'volume_mult': 1.5,
            'use_adx_filter': False,
            'adx_period': 14,
            'adx_threshold': 20,
            'max_investment_ratio': 1.0
        }
