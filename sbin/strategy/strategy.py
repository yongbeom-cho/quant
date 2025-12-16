import numpy as np
import inspect
import itertools
import talib

def apply_strategy(df, strategy_name, params):
    if strategy_name == "explode_volume_breakout":
        return explode_volume_breakout(df, params['window'], params['prev_top_vol_del_ratio'], params['vol_ratio'], params['co_ratio'], params['utr'])
    elif strategy_name == 'explode_volume_volatility_breakout':
        return explode_volume_volatility_breakout(df, params['window'], params['prev_top_vol_del_ratio'], params['vol_ratio'], params['k'], params['utr'])
    elif strategy_name == 'explode_volume_breakout_2':
        return explode_volume_breakout_2(df, params['window'], params['vol_ratio'], params['short_window'], params['short_vol_ratio'], params['co_ratio'], params['utr'])
    elif strategy_name == 'explode_volume_volatility_breakout_2':
        return explode_volume_volatility_breakout_2(df, params['window'], params['vol_ratio'], params['short_window'], params['short_vol_ratio'], params['k'], params['utr'])
    elif strategy_name == 'low_bb_du':
        return low_bb_du(df, params['window'], params['close_band_ratio_lower'], params['ol_hl_ratio_upper'], params['close_open_ratio_upper'], params['over_sell_threshold'])
    elif strategy_name == 'larry_williams_vb':
        return larry_williams_vb(df, params['k'])
    elif strategy_name == 'larry_williams_vb_pro':
        return larry_williams_vb_pro(df, params['k'], params['ma_window'], params['dip_ma_window'])
    else:
        return None


def get_strategy_params_list(strategy_name, config):
    # 문자열 이름으로 함수 가져오기
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(f"{strategy_name} 함수가 registry에 없음")

    func = STRATEGY_REGISTRY[strategy_name]

    # 함수 signature 읽기
    sig = inspect.signature(func)
    arg_names = [name for name in sig.parameters.keys() if name != "df"]

    # config에서 param 값 읽기
    param_values = []
    for name in arg_names:
        if name + '_list' not in config:
            raise KeyError(f"config에 '{name}'_list 항목이 없음")
        param_values.append(config[name+'_list'])

    # 조합 생성
    params_list = []
    for combo in itertools.product(*param_values):
        params = dict(zip(arg_names, combo))
        params_list.append(params)

    return params_list

def get_sell_strategy_params_list(strategy_name, config):
    # 문자열 이름으로 함수 가져오기
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(f"{strategy_name} 함수가 registry에 없음")

    arg_names = [name[:-5] for name in config.keys()]


    # config에서 param 값 읽기
    param_values = []
    for name in arg_names:
        if name + '_list' not in config:
            raise KeyError(f"config에 '{name}'_list 항목이 없음")
        param_values.append(config[name+'_list'])

    # 조합 생성
    params_list = []
    for combo in itertools.product(*param_values):
        params = dict(zip(arg_names, combo))
        params_list.append(params)

    return params_list

def volatility_breakout_prev_candle(df, k=0.5):
    df["range"] = df["high"] - df["low"]
    df["bo_tp"] = df["open"] + df["range"].shift(1) * k 
    df["signal"] = df["close"] > df["bo_tp"]
    return df

def larry_williams_vb(df, k):
    """
    래리 윌리엄스 변동성 돌파 전략
    - range: 이전 봉의 (고가 - 저가)
    - entry_target: 현재 봉의 시가 + (이전 봉의 range * k)
    - signal: 현재 봉의 종가가 entry_target 보다 높으면 True
    """
    df['range'] = df['high'].shift(1) - df['low'].shift(1)
    df['entry_target'] = df['open'] + df['range'] * k
    df['signal'] = df['close'] > df['entry_target']
    return df

def larry_williams_vb_pro(df, k, ma_window, dip_ma_window):
    """
    Larry Williams Volatility Breakout Pro
    - 진입 조건1 (돌파): 상승 추세(MA) + 변동성 돌파
    - 진입 조건2 (추가매수/딥매수): 상승 추세(MA) + 단기 이평선 터치
    - 익절/손절: 백테스터에서 처리
    """
    # 기본 변동성 돌파 계산
    df['range'] = df['high'].shift(1) - df['low'].shift(1)
    df['entry_target'] = df['open'] + df['range'] * k

    # 추세 필터 (장기 이동평균)
    df['ma_long'] = talib.SMA(df['close'], timeperiod=ma_window)
    is_uptrend = df['close'] > df['ma_long']

    # 조건 1: 변동성 돌파 신호
    breakout_signal = (df['close'] > df['entry_target']) & is_uptrend

    # 조건 2: 딥매수 신호 (단기 이동평균)
    df['ma_dip'] = talib.SMA(df['close'], timeperiod=dip_ma_window)
    dip_buy_signal = (df['low'] < df['ma_dip']) & (df['close'] > df['ma_dip']) & is_uptrend
    
    # 최종 신호: 돌파 또는 딥매수
    df['signal'] = breakout_signal | dip_buy_signal
    
    return df


def trimmed_mean(x, prev_top_vol_del_ratio):
    n = len(x)
    k = int(n * prev_top_vol_del_ratio)  # 제거할 개수
    x_sorted = np.sort(x)
    return x_sorted[:n-k].mean()   # 상위 k개 제거

def explode_volume_breakout(
        df, 
        window=240, 
        prev_top_vol_del_ratio=0.1, 
        vol_ratio=50, 
        co_ratio=1.02, 
        utr=0.8
):
    top_vol_trimmed_mean = df['volume'].shift(1).rolling(window=window).apply(lambda x: trimmed_mean(x, prev_top_vol_del_ratio), raw=True)
    df['volume_signal'] = df['volume'] > (top_vol_trimmed_mean * vol_ratio)

    df['bo_tp'] = (df['open'] * co_ratio)
    df['price_signal'] = (df['close'] >= df['bo_tp'])

    df['utr_signal'] = ((df['close'] - df['open']) / (df['high'] - df['open'])) > utr
    
    df['signal'] = df['volume_signal'] & df['price_signal'] & df['utr_signal']
    
    return df

def explode_volume_volatility_breakout(
        df, 
        window=240, 
        prev_top_vol_del_ratio=0.1, 
        vol_ratio=50, 
        k=3.0, 
        utr=0.8
):
    top_vol_trimmed_mean = df['volume'].shift(1).rolling(window=window).apply(lambda x: trimmed_mean(x, prev_top_vol_del_ratio), raw=True)
    df['volume_signal'] = df['volume'] > (top_vol_trimmed_mean * vol_ratio)

    df["range"] = (df["high"].shift(1) - df["low"].shift(1)).rolling(window=window).mean()
    df["bo_tp"] = df["open"] + df["range"] * k 
    df['price_signal'] = (df['close'] >= df['bo_tp'])

    df['utr_signal'] = ((df['close'] - df['open']) / (df['high'] - df['open'])) > utr
    
    df['signal'] = df['volume_signal'] & df['price_signal'] & df['utr_signal']
    
    return df

def explode_volume_breakout_2(
        df, 
        window=240, 
        vol_ratio=100, 
        short_window=2, 
        short_vol_ratio=50, 
        co_ratio=1.02, 
        utr=0.9
):
    vol_mean = df['volume'].shift(1).rolling(window=window).mean()
    df['volume_signal'] = df['volume'] > (vol_mean * vol_ratio)

    short_vol_mean = df['volume'].shift(1).rolling(window=short_window).mean()
    df['short_volume_signal'] = df['volume'] > (short_vol_mean * short_vol_ratio)

    df['bo_tp'] = (df['open'] * co_ratio)
    df['price_signal'] = (df['close'] >= df['bo_tp'])

    df['utr_signal'] = ((df['close'] - df['open']) / (df['high'] - df['open'])) > utr
    
    df['signal'] = df['volume_signal'] &  df['short_volume_signal'] & df['price_signal'] & df['utr_signal']
    
    return df

def explode_volume_volatility_breakout_2(
        df, 
        window=240, 
        vol_ratio=50, 
        short_window=2, 
        short_vol_ratio=50, 
        k=3.0, 
        utr=0.8
):
    vol_mean = df['volume'].shift(1).rolling(window=window).mean()
    df['volume_signal'] = df['volume'] > (vol_mean * vol_ratio)

    short_vol_mean = df['volume'].shift(1).rolling(window=short_window).mean()
    df['short_volume_signal'] = df['volume'] > (short_vol_mean * short_vol_ratio)

    df["range"] = (df["high"].shift(1) - df["low"].shift(1)).rolling(window=window).mean()
    df["bo_tp"] = df["open"] + df["range"] * k 
    df['price_signal'] = (df['close'] >= df['bo_tp'])

    df['utr_signal'] = ((df['close'] - df['open']) / (df['high'] - df['open'])) > utr
    
    df['signal'] = df['volume_signal'] &  df['short_volume_signal'] & df['price_signal'] & df['utr_signal']
    
    return df

def low_bb_du(
    df,
    window=2,
    close_band_ratio_lower=0.2,
    ol_hl_ratio_upper=0.3,
    close_open_ratio_upper=1.005,
    over_sell_threshold=20
):
    
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['high'],
        df['low'],
        df['close'],
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )

    df['over_sell_golden_cross'] = (df['stoch_d'] > df['stoch_k']) & (df['stoch_d'].shift(1) < df['stoch_k'].shift(1)) & (df['stoch_k'] < over_sell_threshold)

    df['bb_low_signal'] = False
    for i in range(1, window+1):
        df['bb_low_signal'] |= df['close'].shift(i) < df['bb_lower'].shift(i)

    df['close_band_ratio_lower_signal'] = (((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) < close_band_ratio_lower)
    df['close_lb_up_signal'] = (df['close'] > df['bb_lower'])
    df['ol_hl_ratio_upper_signal'] = (
        (df['high'] != df['low']) &
        (((df['open'] - df['low']) / (df['high'] - df['low'])) > ol_hl_ratio_upper)
    )

    df['close_open_ratio_upper_signal'] = df['close'] > df['open'] * close_open_ratio_upper

    df['prev_signal'] = df['bb_low_signal'] & df['close_lb_up_signal'] & df['close_band_ratio_lower_signal'] & df['ol_hl_ratio_upper_signal'] & df['close_open_ratio_upper_signal'] & df['over_sell_golden_cross']
    df['signal'] = df['prev_signal'] & ~(
        df['prev_signal'].shift(1).rolling(window).max().fillna(0).astype(bool)
    )
    
    return df



# 이곳에 strategy를 등록해야 unit test가 자동으로 된다.
STRATEGY_REGISTRY = {
    "volatility_breakout_prev_candle": volatility_breakout_prev_candle,
    "explode_volume_breakout": explode_volume_breakout,
    "explode_volume_volatility_breakout": explode_volume_volatility_breakout,
    "explode_volume_breakout_2": explode_volume_breakout_2,
    "explode_volume_volatility_breakout_2": explode_volume_volatility_breakout_2,
    "low_bb_du": low_bb_du,
    "larry_williams_vb": larry_williams_vb,
    "larry_williams_vb_pro": larry_williams_vb_pro
}