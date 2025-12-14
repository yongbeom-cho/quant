import numpy as np
import inspect
import itertools


def apply_strategy(df, strategy_name, params):
    if strategy_name == "explode_volume_breakout":
        return explode_volume_breakout(df, params['window'], params['prev_top_vol_del_ratio'], params['vol_ratio'], params['co_ratio'], params['utr'])
    elif strategy_name == 'explode_volume_volatility_breakout':
        return explode_volume_volatility_breakout(df, params['window'], params['prev_top_vol_del_ratio'], params['vol_ratio'], params['k'], params['utr'])
    elif strategy_name == 'explode_volume_breakout_2':
        return explode_volume_breakout_2(df, params['window'], params['vol_ratio'], params['short_window'], params['short_vol_ratio'], params['k'], params['utr'])
    elif strategy_name == 'explode_volume_volatility_breakout_2':
        return explode_volume_volatility_breakout_2(df, params['window'], params['vol_ratio'], params['short_window'], params['short_vol_ratio'], params['k'], params['utr'])
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


# 이곳에 strategy를 등록해야 unit test가 자동으로 된다.
STRATEGY_REGISTRY = {
    "volatility_breakout_prev_candle": volatility_breakout_prev_candle,
    "explode_volume_breakout": explode_volume_breakout,
    "explode_volume_volatility_breakout": explode_volume_volatility_breakout,
    "explode_volume_breakout_2": explode_volume_breakout_2,
    "explode_volume_volatility_breakout_2": explode_volume_volatility_breakout_2
}