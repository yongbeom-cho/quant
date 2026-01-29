"""
Low BB DRU (Bollinger Band Down Rebound Up) 매수 전략

볼린저 밴드 하단에서 반등할 때 진입하는 전략입니다.
기존 sbin/strategy/strategy.py의 low_bb_dru 함수를 리팩토링한 구현입니다.

=============================================================================
전략 조건 (6가지 조건 모두 만족시 매수)
=============================================================================
1. cond1: 현재 봉이 양봉 (close/open >= co_upper)
2. cond2: 현재 봉의 시가/종가가 볼린저 밴드 하단 위에 있음
3. cond3: 직전 봉이 하락봉이면서 아래꼬리가 있음
4. cond4: 직전 N봉 이내에 볼린저 밴드 아래로 하락한 음봉 존재
5. cond5: 현재 종가가 볼린저 밴드 하위 N% 이내에 위치
6. cond6: 스토캐스틱 과매도 + 추가 조건 (cond6_idx에 따라 변경)

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


class LowBBDRUBuyStrategy(BaseBuyStrategy):
    """
    볼린저 밴드 하단 반등 매수 전략 (Low BB DRU)
    
    볼린저 밴드 하단에서 반등 시그널이 발생할 때 매수합니다.
    스토캐스틱 과매도 지표와 함께 사용하여 신뢰도를 높입니다.
    
    주요 특징:
    - 볼린저 밴드 기반 지지선 확인
    - 스토캐스틱 과매도 구간 확인
    - 직전 봉의 하락 패턴 확인
    - 6가지 조건의 AND 조합
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 전략 파라미터 로드 (기본값 설정)
        self.window = config.get('window', 3)
        self.co_upper = config.get('co_upper', 1.002)
        self.close_band_ratio_lower = config.get('close_band_ratio_lower', 0.25)
        self.prev_co_and_lc_range = config.get('prev_co_and_lc_range', [1.002, 0.98])
        self.over_sell_threshold = config.get('over_sell_threshold', 25)
        self.cond6_idx = config.get('cond6_idx', 0)
        
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
            
            # 스토캐스틱
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
        
        return indicators
    
    def calculate_signals(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        매수 신호 계산
        
        6가지 조건을 모두 만족할 때 매수 신호 발생.
        Long only 전략 (direction: 0 또는 1)
        """
        if talib is None:
            raise ImportError("talib is required for LowBBDRU strategy")
        
        close_val = df['close'].values
        open_val = df['open'].values
        high_val = df['high'].values
        low_val = df['low'].values
        
        # === 지표 계산 ===
        if cached_data and 'bb_upper' in cached_data:
            bb_upper = cached_data['bb_upper']
            bb_mid = cached_data['bb_mid']
            bb_lower = cached_data['bb_lower']
            stoch_k = cached_data['stoch_k']
            stoch_d = cached_data['stoch_d']
        else:
            indicators = self.get_indicators(df)
            bb_upper = indicators['bb_upper']
            bb_mid = indicators['bb_mid']
            bb_lower = indicators['bb_lower']
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
        
        # === 조건 1: 현재 봉이 양봉 (close/open >= co_upper) ===
        cond1 = (df['close'] / df['open']) >= self.co_upper
        
        # === 조건 2: 현재 봉의 시가/종가가 볼린저 밴드 하단 위 ===
        cond2 = (df['close'] > bb_lower) & (df['open'] > bb_lower)
        
        # === 조건 3: 직전 봉이 하락봉 + 아래꼬리 ===
        prev_co = df['close'].shift(1) / df['open'].shift(1)
        prev_lc = df['low'].shift(1) / df['close'].shift(1)
        cond3 = (prev_co < self.prev_co_and_lc_range[0]) & (prev_lc < self.prev_co_and_lc_range[1])
        
        # === 조건 4: 직전 N봉 이내에 볼린저밴드 아래 음봉 존재 ===
        cond4_raw = (df['close'] < bb_lower) & ((df['close'] / df['open']) < 1.0)
        cond4 = cond4_raw.shift(1).rolling(self.window).max().fillna(False).astype(bool)
        
        # === 조건 5: 현재 종가가 볼린저 밴드 하위 N% 이내 ===
        band_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        cond5 = band_position < self.close_band_ratio_lower
        
        # === 조건 6: 스토캐스틱 과매도 + 추가 조건 ===
        stoch_gc = (stoch_d < stoch_k) & (stoch_d.shift(1) > stoch_k.shift(1))
        prev_close_above_bb = df['close'].shift(1) > bb_lower.shift(1)
        prev_close_above_bb_curr = df['close'].shift(1) > bb_lower
        
        if self.cond6_idx == 0:
            cond6 = (stoch_k < self.over_sell_threshold) & prev_close_above_bb
        elif self.cond6_idx == 1:
            cond6 = (stoch_k < self.over_sell_threshold) & \
                    stoch_gc.rolling(self.window).max().fillna(False).astype(bool) & \
                    prev_close_above_bb
        elif self.cond6_idx == 2:
            cond6 = stoch_gc.rolling(self.window).max().fillna(False).astype(bool) & \
                    prev_close_above_bb
        elif self.cond6_idx == 3:
            cond6 = (stoch_k < self.over_sell_threshold) & prev_close_above_bb_curr
        elif self.cond6_idx == 4:
            cond6 = (stoch_k < self.over_sell_threshold) & \
                    stoch_gc.rolling(self.window).max().fillna(False).astype(bool) & \
                    prev_close_above_bb_curr
        else:
            cond6 = stoch_gc.rolling(self.window).max().fillna(False).astype(bool) & \
                    prev_close_above_bb_curr
        
        # === 조건 1-6 조합 ===
        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        
        # === 중복 신호 제거 (직전 N봉 이내에 신호가 없었어야 함) ===
        prev_true_exists = cond_all.shift(1).rolling(self.window).max().fillna(False).astype(bool)
        signal = cond_all & (~prev_true_exists)
        
        # 방향 배열 생성 (Long only)
        direction = np.zeros(len(df), dtype=int)
        direction[signal.values] = 1
        
        return {
            'direction': direction,
            'signal': signal.values,
            'target_long': close_val,  # 종가 기준 진입
            'target_short': None,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'conditions': {
                'cond1': cond1.values,
                'cond2': cond2.values,
                'cond3': cond3.values,
                'cond4': cond4.values,
                'cond5': cond5.values,
                'cond6': cond6.values
            }
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
            'window': self.window,
            'co_upper': self.co_upper,
            'close_band_ratio_lower': self.close_band_ratio_lower,
            'over_sell_threshold': self.over_sell_threshold,
            'cond6_idx': self.cond6_idx
        }
        
        # 스토캐스틱 값 기록
        if signals.get('stoch_k') is not None and len(signals['stoch_k']) > idx:
            stoch_k_val = signals['stoch_k'][idx] if hasattr(signals['stoch_k'], '__getitem__') else signals['stoch_k'].iloc[idx]
            entry_conditions['stoch_k'] = float(stoch_k_val) if not np.isnan(stoch_k_val) else None
        
        # 볼린저 밴드 값 기록
        if signals.get('bb_lower') is not None and len(signals['bb_lower']) > idx:
            bb_lower_val = signals['bb_lower'][idx] if hasattr(signals['bb_lower'], '__getitem__') else signals['bb_lower'].iloc[idx]
            entry_conditions['bb_lower'] = float(bb_lower_val) if not np.isnan(bb_lower_val) else None
        
        return PositionInfo(
            ticker=ticker,
            direction=signal_type,
            entry_price=entry_price,
            entry_idx=idx,
            entry_date=str(row['date']),
            entry_reason='low_bb_dru_rebound',
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
        return ['window', 'co_upper', 'close_band_ratio_lower', 'over_sell_threshold']
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'strategy_name': 'low_bb_dru',
            'window': 3,
            'co_upper': 1.002,
            'close_band_ratio_lower': 0.25,
            'prev_co_and_lc_range': [1.002, 0.98],
            'over_sell_threshold': 25,
            'cond6_idx': 0,
            'bb_period': 20,
            'bb_std': 2,
            'max_investment_ratio': 1.0
        }
