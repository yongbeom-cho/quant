import pandas as pd
import numpy as np

def bailout_sell_strategy_enhanced(
    entry_price,
    entry_idx,
    current_idx,
    position_type, # 'long' (매수) or 'short' (매도/공매도)
    low_val,
    high_val,
    open_val,
    close_val,
    atr_val = 0.0, # 현재 시점의 ATR(변동성) 값
    stop_loss_ratio = 0.02,
    trailing_stop_mult = 3.0, # ATR 배수 기반 트레일링 스탑 설정
    bailout_profit_days = 1,
    bailout_no_profit_days = 4,
    min_profit_ratio = 0.005 # 수수료와 슬리피지를 고려한 최소 익절 목표치
):
    """
    [고도화된 청산(Sell/Exit) 전략 로직]
    1. 손절매(Stop Loss): 진입가 대비 설정된 비율(예: 2%) 이상 손실 발생 시 즉시 청산
    2. 트레일링 스탑(Trailing Stop): 수익이 나기 시작하면 고점 대비 ATR 배수만큼 하락 시 익절하여 수익 보존
    3. 최소 수익 기간 보호: 수수료를 감안한 최소 수익권에 도달하기 전까지 성급한 탈출 방지
    4. 타임아웃(Time Exit): 일정 기간(예: 4봉) 동안 수익이 나지 않으면 시장가 탈출
    """
    # 진입 후 경과한 봉(Bar)의 수
    bars_held = current_idx - entry_idx
    
    # 1. 고정 비율 손절매 (Stop Loss)
    # 롱(매수) 포지션일 때
    if position_type == 'long':
        if low_val <= entry_price * (1 - stop_loss_ratio):
            return True, 'stop_loss' # 진입가보다 일정 비율 하락하면 손절
    # 숏(매도) 포지션일 때
    else:
        if high_val >= entry_price * (1 + stop_loss_ratio):
            return True, 'stop_loss' # 진입가보다 일정 비율 상승하면 손절

    # 2. ATR 기반 트레일링 스탑 (수익 보존 및 추세 추종)
    # 현재 수익률 계산 (롱: 가격 상승이 이득, 숏: 가격 하락이 이득)
    current_pnl = (close_val / entry_price - 1) if position_type == 'long' else (1 - close_val / entry_price)
    
    # 최소 수익 목표(예: 0.5%)를 넘었을 때만 트레일링 스탑 활성화
    if current_pnl > min_profit_ratio:
        if position_type == 'long':
            # 롱 익절: '이번 봉 고점 - (변동폭 * 배수)' 아래로 종가가 떨어지면 추세 꺾임으로 보고 익절
            if close_val < (high_val - atr_val * trailing_stop_mult):
                return True, 'trailing_stop'
        else: # short
            # 숏 익절: '이번 봉 저점 + (변동폭 * 배수)' 위로 종가가 올라오면 익절
            if close_val > (low_val + atr_val * trailing_stop_mult):
                return True, 'trailing_stop'

    # 3. 수익권 탈출 (Bailout Profit)
    # 일정 기간 이상 보유했고, 시가가 진입가 대비 최소 수익권 위에 있다면 청산 (익절 시나리오)
    if bars_held >= bailout_profit_days:
        if position_type == 'long' and open_val > entry_price * (1 + min_profit_ratio):
            return True, 'bailout_profit'
        elif position_type == 'short' and open_val < entry_price * (1 - min_profit_ratio):
            return True, 'bailout_profit'
                
    # 4. 무수익 타임아웃 탈출 (Bailout No Profit / Time Exit)
    # 너무 오래 들고 있었는데도 수익이 나지 않는다면(본전 이하) 미련 없이 탈출
    if bars_held >= bailout_no_profit_days:
        if (position_type == 'long' and close_val <= entry_price) or \
           (position_type == 'short' and close_val >= entry_price):
            return True, 'bailout_timeout'
                
    # 아무 조건에도 해당하지 않으면 포지션 유지
    return False, 'none'
