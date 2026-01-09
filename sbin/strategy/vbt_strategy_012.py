import numpy as np
import pandas as pd
import talib

def get_vbt_indicators(df: pd.DataFrame, ema_period: int = 15, rsi_period: int = 8):
    """
    [기본 지표 계산 및 캐싱]
    EMA와 RSI를 미리 계산하여 백테스트 속도를 높입니다.
    """
    indicators = {}
    indicators['ema'] = talib.EMA(df['close'], timeperiod=ema_period).values
    indicators['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period).values
    return indicators

def numpy_rolling_mean(a, window):
    """Numpy 기반 고속 이동 평균 (성능 최적화용)"""
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return np.concatenate(([np.nan]*(window-1), ret[window-1:] / window))

def numpy_rolling_std(a, window):
    """Numpy 기반 고속 이동 표준편차 (성능 최적화용)"""
    a2 = a**2
    m1 = numpy_rolling_mean(a, window)
    m2 = numpy_rolling_mean(a2, window)
    return np.sqrt(np.maximum(0, m2 - m1**2))

def vbt_with_filters(
    df: pd.DataFrame,
    window: int = 5,
    k_long: float = 0.5,
    k_short: float = 0.5,
    ema_period: int = 15,
    rsi_period: int = 8,
    rsi_upper: float = 70,
    rsi_lower: float = 30,
    use_std: bool = False,
    std_mult: float = 1.0,
    volume_window: int = 20,
    volume_mult: float = 1.0,
    volatility_window: int = 20,
    volatility_threshold: float = 0.0,
    cached_indicators: dict = None,
    cached_ranges: dict = None
):
    """
    [기본 변동성 돌파(VBT) 전략 로직]
    래리 윌리엄스의 변동성 돌파 전략을 기반으로 EMA, RSI, 거래량 필터를 추가한 버전입니다.
    
    기본 원리:
    - 변동폭(Range): 전일 고가 - 전일 저가
    - 돌파 기준: 오늘 시가 + (변동폭 * k)
    - 가격이 돌파 기준을 넘어서면 추세가 시작된 것으로 보고 매수
    """
    open_val = df['open'].values
    close_val = df['close'].values
    high_val = df['high'].values
    low_val = df['low'].values
    vol_val = df['volume'].values
    
    # 1. 가격 돌파 타겟 설정 (Range 계산)
    if cached_ranges and window in cached_ranges:
        avg_range = cached_ranges[window]['avg']
        std_range = cached_ranges[window].get('std')
    else:
        # Range = High - Low (하루 동안의 변동폭)
        range_val = (high_val - low_val)
        # Shift(1)을 통해 '어제'의 변동폭을 가져옵니다.
        shifted_range = pd.Series(range_val).shift(1)
        # 설정된 window 기간 동안의 평균 변동폭을 구합니다.
        avg_range = shifted_range.rolling(window=window).mean().values
        std_range = shifted_range.rolling(window=window).std().values if use_std else None
    
    # 돌파 목표가 계산: 시가 + (평균 변동폭 * k)
    if use_std and std_range is not None:
        # 표준편차를 가산하여 변동성이 확대될 때만 진입하도록 보정 가능
        target_long = open_val + (avg_range + std_range * std_mult) * k_long
        target_short = open_val - (avg_range + std_range * std_mult) * k_short
    else:
        target_long = open_val + avg_range * k_long
        target_short = open_val - avg_range * k_short
        
    # 2. 이동평균선(EMA) 및 RSI 지표 로드
    if cached_indicators and 'ema' in cached_indicators:
        ema_val = cached_indicators['ema']
        rsi_val = cached_indicators['rsi']
    else:
        ema_val = talib.EMA(df['close'], timeperiod=ema_period).values
        rsi_val = talib.RSI(df['close'], timeperiod=rsi_period).values

    # 3. 거래량 및 변동성 필터
    # 현재 거래량이 과거 n일 평균보다 높아야 '신뢰할 수 있는 돌파'로 간주
    avg_volume = pd.Series(vol_val).shift(1).rolling(window=volume_window).mean().values
    volume_filter = vol_val > (avg_volume * volume_mult)
    
    # 최소 변동성 필터: 너무 조용한 장에서는 돌파 신호가 나와도 속임수일 확률이 높으므로 제외
    curr_range = high_val - low_val
    avg_volatility = pd.Series(curr_range).shift(1).rolling(window=volatility_window).mean().values
    volatility_filter = avg_volatility > (open_val * volatility_threshold)

    # 4. 진입 신호 결합
    # 롱 조건: (현재 고가 >= 목표가) AND (종가 > 이평선) AND (거래량 필터) AND (변동성 필터)
    signal_long = (high_val >= target_long) & (close_val > ema_val) & volume_filter & volatility_filter
    # 숏 조건: (현재 저가 <= 목표가) AND (종가 < 이평선) AND (거래량 필터) AND (변동성 필터)
    signal_short = (low_val <= target_short) & (close_val < ema_val) & volume_filter & volatility_filter
    
    # 리버스 조건: RSI가 과하게 높거나 낮을 때 반대 포지션으로 스위칭 고려
    reverse_to_short = (rsi_val >= rsi_upper)
    reverse_to_long = (rsi_val <= rsi_lower)
    
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

# 전략 레지스트리 (고도화 백테스터에서 참조)
VBT_STRATEGY_REGISTRY = {
    "vbt_with_filters": vbt_with_filters
}

def get_vbt_strategy_params_list(strategy_name, config):
    import itertools
    import inspect
    
    if strategy_name not in VBT_STRATEGY_REGISTRY:
        raise KeyError(f"{strategy_name} not in VBT_STRATEGY_REGISTRY")
        
    func = VBT_STRATEGY_REGISTRY[strategy_name]
    sig = inspect.signature(func)
    arg_names = [name for name in sig.parameters.keys() if name != "df"]

    param_values = []
    for name in arg_names:
        list_key = name + '_list'
        if list_key not in config:
             if name in config:
                 param_values.append([config[name]])
             else:
                 # Default if optional
                 default = sig.parameters[name].default
                 if default is not inspect.Parameter.empty:
                     param_values.append([default])
                 else:
                     raise KeyError(f"Config missing {list_key} or {name}")
        else:
            param_values.append(config[list_key])
            
    params_list = []
    for combo in itertools.product(*param_values):
        params = dict(zip(arg_names, combo))
        params_list.append(params)
        
    return params_list
