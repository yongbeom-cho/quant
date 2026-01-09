import pandas as pd
import numpy as np

def bailout_sell_strategy(
    entry_price,
    entry_idx,
    current_idx,
    position_type, # 'long' (매수) or 'short' (매도/공매도)
    low_val,
    high_val,
    open_val,
    close_val,
    stop_loss_ratio = 0.02,
    bailout_profit_days = 1,
    bailout_no_profit_days = 4,
    price_flow_sluggish_threshold = 0.005
):
    """
    [기본 청산(Exit) 전략 로직]
    손절매(Stop Loss)와 시점 기반의 탈출(Bailout)을 결합한 방식입니다.
    """
    # 보유 기간 (현재 봉 인덱스 - 진입 봉 인덱스)
    bars_held = current_idx - entry_idx
    
    # 1. 고정 비율 손절매 (Stop Loss)
    if position_type == 'long':
        # 매수 후 가격이 설정된 비율(예: 2%) 이상 떨어지면 손절
        if low_val <= entry_price * (1 - stop_loss_ratio):
            return True, 'stop_loss'
    else: # short
        # 매도 후 가격이 설정된 비율 이상 오르면 손절
        if high_val >= entry_price * (1 + stop_loss_ratio):
            return True, 'stop_loss'
            
    # 2. 수익권 조기 탈출 (Bailout Profit)
    # 최소 n일(봉) 이상 보유했고, 시가가 진입가보다 높으면(익절 상태면) 청산하여 수익 확보
    if bars_held >= bailout_profit_days:
        if (position_type == 'long' and open_val > entry_price) or \
           (position_type == 'short' and open_val < entry_price):
            return True, 'bailout_profit'
                
    # 3. 무수익 타임아웃 탈출 (Bailout No Profit / Timeout)
    # 최대 n일 이상 보유했는데도 수익이 나지 않으면(본전 이하) 시간 낭비를 줄이기 위해 탈출
    if bars_held >= bailout_no_profit_days:
        if (position_type == 'long' and close_val <= entry_price) or \
           (position_type == 'short' and close_val >= entry_price):
            return True, 'bailout_timeout'
                
    # 아무 조건에도 해당하지 않으면 포지션 유지
    return False, 'none'
