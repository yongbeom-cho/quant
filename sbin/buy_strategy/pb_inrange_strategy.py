"""
Percent B inrange 매수 전략

볼린저 밴드 percent b의 특정 범위 밖에서 특정 범위 내로 들어올때 진입하는 전략입니다.

=============================================================================
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
except ImportError:
    talib = None
    print("[WARNING] talib not installed. LowBBDRU strategy requires talib.")

from .base import BaseBuyStrategy
from .position import PositionInfo


class PBInrangeBuyStrategy(BaseBuyStrategy):
    """
    볼린저 밴드 Percent B 특정범위 밖에서 특정범위 내로 들어올때 매수 전략 (PB Inrange)
    
    주요 특징:
    - 
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.pb_sma_period = config.get('pb_sma_period', 20)
        self.pb_sma_up = config.get('pb_sma_up', 0.7)
        self.pb_inrange = config.get('pb_inrange', [0.65, 0.7])
        
        # 볼린저 밴드 파라미터
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        지표 사전 계산 (캐싱용)
        
        볼린저 밴드, 스토캐스틱을 미리 계산합니다.
        """
        indicators = {}

        if talib is not None:
            # 볼린저 밴드
            indicators['bb_upper'], indicators['bb_mid'], indicators['bb_lower'] = talib.BBANDS(
                df['close'], 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_std, 
                nbdevdn=self.bb_std, 
                matype=0
            )
            
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        Long only 전략 (direction: 0 또는 1)
        """
        if talib is None:
            raise ImportError("talib is required for PBRebound strategy")
        
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        low_val = df['low'].values
        
        # === 지표 계산 ===
        if cached_data and 'bb_upper' in cached_data:
            bb_upper = cached_data['bb_upper']
            bb_mid = cached_data['bb_mid']
            bb_lower = cached_data['bb_lower']
        else:
            indicators = self.get_indicators(df)
            bb_upper = indicators['bb_upper']
            bb_mid = indicators['bb_mid']
            bb_lower = indicators['bb_lower']
        

        pb = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        pb_sma_raw = talib.SMA(pb.values if hasattr(pb, 'values') else pb, timeperiod=self.pb_sma_period)
        # talib는 numpy 배열 반환 → pandas Series와 &/| 시 타입 오류 방지를 위해 동일 인덱스 Series로 통일
        pb_sma = pd.Series(pb_sma_raw, index=df.index)
        # pb_inrange [0.6, 0.7] → 0.01 단위 (low, high) 쌍: (0.6,0.61), (0.61,0.62), ... 마다 조건 구한 뒤 OR
        range_conditions = []
        step = 0.05
        r0, r1 = self.pb_inrange[0], self.pb_inrange[1]
        for low in np.arange(r0, r1, step):
            high = min(low + step, r1)
            prev_out = (pb.shift(1) < low) | (pb.shift(1) > high)
            now_in = (pb > low) & (pb < high)
            range_conditions.append(prev_out & now_in)
        combined = range_conditions[0] if range_conditions else pd.Series(False, index=df.index)
        for cond in range_conditions[1:]:
            combined = combined | cond
        signal = (pb_sma > self.pb_sma_up) & combined
        signal_ok = signal.fillna(False).astype(bool)
        
        # 방향 배열 생성 (Long only)
        direction = np.zeros(len(df), dtype=int)
        direction[signal_ok.values] = 1
        
        return {
            'direction': direction,
            'signal': signal.values,
            'target_long': close_val,  # 종가 기준 진입
            'target_short': None,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'pb': pb,
            'pb_sma': pb_sma
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
            'pb_sma_period': self.pb_sma_period,
            'pb_sma_up': self.pb_sma_up,
            'pb_inrange': self.pb_inrange
        }
        
        # 볼린저 밴드 값 기록
        if signals.get('pb') is not None and len(signals['pb']) > idx:
            pb_val = signals['pb'][idx] if hasattr(signals['pb'], '__getitem__') else signals['pb'].iloc[idx]
            entry_conditions['pb'] = float(pb_val) if np.isfinite(pb_val) else None
        
        if signals.get('pb_sma') is not None and len(signals['pb_sma']) > idx:
            pb_sma_val = signals['pb_sma'][idx] if hasattr(signals['pb_sma'], '__getitem__') else signals['pb_sma'].iloc[idx]
            entry_conditions['pb_sma'] = float(pb_sma_val) if np.isfinite(pb_sma_val) else None
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='pb_inrange',
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
        return ['pb_sma_period', 'pb_sma_up', 'pb_inrange']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'pb_inrange',
            'pb_sma_period': 20,
            'pb_sma_up': 0.7,
            'pb_inrange': [0.6, 0.7],
            'bb_period': 20,
            'bb_std': 2,
            'max_investment_ratio': 1.0
        }
