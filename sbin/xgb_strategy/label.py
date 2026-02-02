"""
Label Module

sell_strategy를 이용하여 label을 생성합니다.
각 label은 다른 upper/lower 비율을 가진 sell_strategy에 해당합니다.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import sys
import os

# sell_strategy 모듈 import를 위한 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sell_strategy.registry import get_sell_strategy
from sell_strategy.base import BaseSellStrategy
from buy_strategy.position import PositionInfo


def get_labels_from_sell_strategy(
    df: pd.DataFrame,
    sell_strategy_params_list: List[Dict[str, Any]],
    max_cnt: int = 9999
) -> pd.DataFrame:
    """
    sell_strategy를 사용하여 label 생성
    
    Args:
        df: strategy_feature가 True인 행들을 포함한 DataFrame
        sell_strategy_params_list: sell 전략 파라미터 리스트 (각각이 하나의 label)
        max_cnt: 최대 보유 기간 (봉 개수)
    
    Returns:
        label0, label1, label2, ... 컬럼이 추가된 DataFrame
    """
    labels = {}
    
    # strategy_feature가 True인 인덱스 찾기
    buy_indices = df.index[df['strategy_feature'] == True].tolist()
    
    for label_idx, sell_strategy_params in enumerate(sell_strategy_params_list):
        label_name = f'label{label_idx}'
        label_values = np.full(len(df), np.nan)
        
        # sell_strategy 인스턴스 생성
        sell_strategy = get_sell_strategy(sell_strategy_params['strategy_name'], sell_strategy_params['params'])
        
        for buy_idx in buy_indices:
            buy_close = df.at[buy_idx, 'close']
            buy_date = df.at[buy_idx, 'date']
            
            # buy_idx 이후의 데이터만 확인
            future_df = df.loc[buy_idx+1:].head(max_cnt)
            
            if len(future_df) == 0:
                continue
            
            # PositionInfo 생성 (가상의 포지션)
            position = PositionInfo(
                ticker=df.at[buy_idx, 'ticker'] if 'ticker' in df.columns else 'UNKNOWN',
                entry_price=buy_close,
                entry_idx=buy_idx,
                entry_date=str(buy_date),
                direction=1,  # Long only
                invested_amount=1.0,  # 더미 값
                entry_reason='strategy_feature',
                entry_conditions={},  # 진입 조건 상세 (label 생성 시에는 불필요)
                quantity=1.0,  # 더미 수량
                max_investment_ratio=1.0,  # 더미 값
                current_allocation_ratio=0.0  # 더미 값
            )
            
            # 각 future bar에 대해 청산 조건 확인
            for future_idx, future_row in future_df.iterrows():
                current_bar = {
                    'open': future_row['open'],
                    'high': future_row['high'],
                    'low': future_row['low'],
                    'close': future_row['close'],
                    'volume': future_row['volume'],
                    'date': str(future_row['date']),
                    'is_last': future_idx == df.index[-1]
                }
                
                should_exit, exit_reason, exit_price = sell_strategy.should_exit(
                    position, current_bar, future_idx
                )
                
                if should_exit:
                    # 손절인지 익절인지 판단
                    if exit_reason == 'stop_loss':
                        label_values[buy_idx] = 0
                    elif exit_reason == 'take_profit':
                        label_values[buy_idx] = 1
                    else:
                        # last_bar_exit 등은 종가 기준으로 판단
                        if exit_price >= buy_close:
                            label_values[buy_idx] = 1
                        else:
                            label_values[buy_idx] = 0
                    break
        
        labels[label_name] = label_values
    
    # DataFrame에 label 컬럼 추가
    for label_name, label_values in labels.items():
        df[label_name] = label_values
    
    return df


def get_label_params_from_sell_strategy(
    sell_strategy_name: str,
    sell_params: Dict[str, Any]
) -> Tuple[float, float]:
    """
    sell_strategy의 파라미터에서 upper/lower 비율 추출
    
    Returns:
        (upper_ratio, lower_ratio)
    """
    if sell_strategy_name == 'simple_ratio_sell':
        upper = sell_params.get('high_limit_ratio', 1.1)
        lower = sell_params.get('low_limit_ratio', 0.9)
        return upper, lower
    else:
        # 다른 sell_strategy의 경우 기본값 반환
        return 1.1, 0.9

