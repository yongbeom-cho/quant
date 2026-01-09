import numpy as np
import pandas as pd
import talib

def get_vbt_indicators_enhanced(df: pd.DataFrame, ema_period: int = 20, rsi_period: int = 14, adx_period: int = 14, atr_period: int = 14):
    """
    [고도화 전략 전용 지표 계산 함수]
    전략 실행 전에 EMA, RSI, ADX, ATR 지표를 미리 계산하여 성능을 높입니다.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    indicators = {}
    # EMA (지수 이동 평균): 최근 가격에 가중치를 두는 이동 평균선
    indicators['ema'] = talib.EMA(close, timeperiod=ema_period)
    # RSI (상대강도지수): 과매수/과매도 여부를 판단하는 지표 (0~100)
    indicators['rsi'] = talib.RSI(close, timeperiod=rsi_period)
    # ADX (평균 방향성 지표): 현재 추세의 강도가 얼마나 센지 측정 (25 이상이면 강한 추세)
    indicators['adx'] = talib.ADX(high, low, close, timeperiod=adx_period)
    # ATR (평균 실질 변동폭): 최근 가격이 얼마나 크게 움직였는지 보여주는 변동성 지표
    indicators['atr'] = talib.ATR(high, low, close, timeperiod=atr_period)
    return indicators

def vbt_with_filters_enhanced(
    df: pd.DataFrame,
    k_long: float = 0.5,
    k_short: float = 0.5,
    ema_confirm: bool = True,
    ema_period: int = 20,
    rsi_confirm: bool = False,
    rsi_upper: float = 70,
    rsi_lower: float = 30,
    adx_threshold: float = 20.0,
    atr_mult: float = 1.0,
    volume_window: int = 20,
    volume_mult: float = 1.0,
    cached_indicators: dict = None,
    cached_ranges: dict = None
):
    """
    [고도화된 변동성 돌파 전략 로직]
    1. 변동성 돌파: 이전 봉의 변동폭(ATR)을 기준으로 일정 수준 이상 돌파하면 진입
    2. ADX 필터: 시장에 추세가 어느 정도 있을 때만 진입하여 횡보장의 수수료 낭비 방지
    3. EMA 필터: 현재 가격이 이동평균선 위에 있을 때만 롱, 아래일 때만 숏 진입 (정배열 확인)
    4. RSI Zone Exit: 단순히 과매수 시점이 아니라, 과매수 구간을 '탈출'할 때 반전(Reverse) 시도
    """
    open_val = df['open'].values
    close_val = df['close'].values
    high_val = df['high'].values
    low_val = df['low'].values
    vol_val = df['volume'].values

    # 1. 지표 데이터 로드 (캐시된 데이터가 있으면 사용, 없으면 새로 계산)
    if cached_indicators:
        ema_val = cached_indicators.get('ema')
        rsi_val = cached_indicators.get('rsi')
        adx_val = cached_indicators.get('adx')
        atr_val = cached_indicators.get('atr')
    else:
        ema_val = talib.EMA(df['close'], timeperiod=ema_period)
        rsi_val = talib.RSI(df['close'], timeperiod=14)
        adx_val = talib.ADX(high_val, low_val, close_val, timeperiod=14)
        atr_val = talib.ATR(high_val, low_val, close_val, timeperiod=14)

    # 2. 타겟 가격 설정 (변동성 돌파 기준)
    # 이전 봉의 ATR(변동폭)에 k값을 곱해 돌파 기준을 잡습니다.
    prev_atr = pd.Series(atr_val).shift(1).values
    target_long = open_val + prev_atr * k_long   # 현재가 + (변동폭 * k) 이상이면 상방 돌파
    target_short = open_val - prev_atr * k_short # 현재가 - (변동폭 * k) 이하이면 하방 돌파

    # 3. 각종 필터 로직 계산
    # ADX 필터: 추세 강도가 설정값(예: 25)보다 커야 '거래할 만한 장'으로 판단
    adx_filter = adx_val > adx_threshold
    
    # 거래량 필터: 현재 거래량이 이전 n봉 평균보다 높아야 '힘 실린 움직임'으로 판단
    avg_volume = pd.Series(vol_val).shift(1).rolling(window=volume_window).mean().values
    volume_filter = vol_val > (avg_volume * volume_mult)
    
    # EMA 필터: 롱 진입시는 가격이 이평선 위, 숏 진입시는 이평선 아래여야 함
    long_trend = (close_val > ema_val) if ema_confirm else True
    short_trend = (close_val < ema_val) if ema_confirm else True

    # RSI 필터: 롱 진입시 너무 과매수 상태(예: 70 이상)라면 진입 자제
    if rsi_confirm:
        long_rsi = rsi_val < rsi_upper
        short_rsi = rsi_val > rsi_lower
    else:
        long_rsi = True
        short_rsi = True

    # 4. 진입 신호 결합 (모든 조건이 참이어야 진입)
    # 가격 돌파 AND 추세 강도 AND 거래량 AND 방향성 AND 과매수여부
    signal_long = (high_val >= target_long) & adx_filter & volume_filter & long_trend & long_rsi
    signal_short = (low_val <= target_short) & adx_filter & volume_filter & short_trend & short_rsi
    
    # 5. 리버스 신호 고도화 (RSI Zone Exit + 추세 저항)
    prev_rsi = pd.Series(rsi_val).shift(1).values
    prev_adx = pd.Series(adx_val).shift(1).values
    
    # RSI Zone Exit: 전봉에서 70 이상(과매수)이었다가 이번 봉에서 70 미만으로 떨어질 때가 진짜 '꺾이는 지점'
    rev_short_base = (prev_rsi >= rsi_upper) & (rsi_val < rsi_upper)
    rev_long_base = (prev_rsi <= rsi_lower) & (rsi_val > rsi_lower)
    
    # 강한 추세 추종 필터: ADX가 매우 높고 계속 상승 중이라면 RSI가 높더라도 리버스하지 않고 더 버팀
    # (고성능 추세장에서는 RSI만 보고 반대로 거는 것이 위험하기 때문)
    strong_trend_rising = (adx_val > prev_adx) & (adx_val > adx_threshold * 1.8)
    
    reverse_to_short = rev_short_base & ~strong_trend_rising
    reverse_to_long = rev_long_base & ~strong_trend_rising
    
    # 6. 최종 방향성 결정 (1: 롱, -1: 숏, 0: 관망)
    vbt_direction = np.zeros(len(df), dtype=int)
    vbt_direction[signal_long] = 1
    vbt_direction[signal_short] = -1
    
    return {
        'vbt_direction': vbt_direction,
        'reverse_to_short': reverse_to_short,
        'reverse_to_long': reverse_to_long,
        'target_long': target_long,
        'target_short': target_short
    }

VBT_STRATEGY_REGISTRY = {
    "vbt_with_filters_enhanced": vbt_with_filters_enhanced
}
