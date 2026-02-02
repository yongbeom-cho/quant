"""
Strategy Feature Module

buy_strategy를 이용하여 strategy_feature (매수 신호)를 생성합니다.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# buy_strategy 모듈 import를 위한 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from buy_strategy.registry import get_buy_strategy
from buy_strategy.base import BaseBuyStrategy


def get_strategy_feature_from_buy_strategy(
    df: pd.DataFrame,
    buy_strategy_name: str,
    buy_params: Dict[str, Any],
    interval: str
) -> pd.DataFrame:
    """
    buy_strategy를 사용하여 strategy_feature 생성
    
    Args:
        df: OHLCV DataFrame
        buy_strategy_name: buy 전략 이름
        buy_params: buy 전략 파라미터
        interval: 시간 간격
    
    Returns:
        'strategy_feature' 컬럼이 추가된 DataFrame
    """
    # buy_strategy 인스턴스 생성
    buy_strategy = get_buy_strategy(buy_strategy_name, buy_params)
    
    # buy_strategy의 calculate_signals 호출
    signals = buy_strategy.calculate_signals(df)
    direction = signals.get('direction', np.zeros(len(df), dtype=int))
    
    # direction이 1인 경우를 strategy_feature로 설정
    df['strategy_feature'] = (direction == 1).astype(bool)
    
    return df

