import talib
import numpy as np
from xgboost import XGBClassifier

def apply_strategy_xgb_feats(df, interval, strategy_feature_name, expanding_cache=None):
    if strategy_feature_name == 'pb_du':
        if interval == 'day':
            return day_pb_du(df, expanding_cache)
        elif interval == 'minute60':
            return minute60_pb_du(df, expanding_cache)
        elif interval == 'minute240':
            return minute240_pb_du(df, expanding_cache)
    return None


def apply_strategy_xgb(df, model_name, model_input_path, expanding_cache=None):
    #1. feat apply
    dummy, market, interval, target_strategy, label_name, min_precision, threshold, str_feat = model_name.split('-')
    threshold = float(threshold)
    df = apply_strategy_xgb_feats(df, interval, target_strategy, expanding_cache)
    feats = ['feat'+feat for feat in str_feat.split('f')[1:] ]
    df['signal'] = 0

    #2. model apply
    model = XGBClassifier()
    model.load_model(model_input_path)

    mask_strategy = df['strategy_feature'] == True
    mask_notna = df[feats].notna().all(axis=1)
    mask_notinf = ~df[feats].isin([np.inf, -np.inf]).any(axis=1)
    mask = mask_strategy & mask_notna & mask_notinf

    X = df.loc[mask, feats]
    if len(X) > 0:
        proba = model.predict_proba(X)[:, 1]
        df.loc[mask, 'signal'] = (proba >= threshold).astype(int)
    return df

def day_pb_du(df, expanding_cache=None):
    """
    Day interval pb_du strategy
    Required features: feat2, feat5, feat7, feat17, feat28, feat32, feat33, feat37, feat40, feat41, feat44, feat46
    expanding_cache: {'bb_std': float, 'feat0_expanding_mean': float, 'feat0_expanding_std': float}
    """
    # 기본 계산 (strategy_feature에 필요)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['pb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['pb_sma20'] = talib.SMA(df['pb'], timeperiod=20)
    
    pb_sma_period = 20
    pb_sma = 0.7
    pb = 0.8
    df['strategy_feature'] = (df['pb'] > df['pb'].shift(1)) & (df['pb'].shift(1) < pb) & (df['pb'] > pb) & (df['pb_sma20'] > pb_sma)
    
    # 필요한 중간 계산들
    # feat2, feat5, feat7, feat17, feat28, feat32, feat33, feat37, feat40, feat41, feat44, feat46
    # 의존성: feat2, feat5(stoch), feat7(stoch), feat17(sma40), feat28(sma60), feat32(pb_sma20), feat33(pb_sma40), feat37(feat33), feat40(donchain), feat41(donchain), feat44(bb_range), feat46(feat0)
    
    # stoch 계산 (feat5, feat7 필요)
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
    
    # sma40 계산 (feat17 필요)
    sma40 = talib.SMA(df['close'], timeperiod=40)
    
    # sma60 계산 (feat28 필요)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    
    # pb_sma40 계산 (feat33, feat37 필요)
    df['pb_sma40'] = talib.SMA(df['pb'], timeperiod=40)
    
    # volume 계산 (feat46 -> feat0 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    
    # donchain 계산 (feat40, feat41 필요)
    window = 20
    donchain_min = (df["close"].shift(1).rolling(window).min())
    donchain_min_w1 = donchain_min.shift(window)
    donchain_min_w2 = donchain_min.shift(window*2)
    
    # bb_range 계산 (feat44 필요)
    bb_range = df["bb_upper"] - df["bb_lower"]
    bb_range_std = (bb_range.shift(1).rolling(window).std())
    
    # expanding 값 사용 (캐시가 있으면 사용, 없으면 계산)
    if expanding_cache and expanding_cache.get('bb_std') is not None:
        bb_std = expanding_cache['bb_std']
    else:
        bb_std = bb_range.shift(1).expanding().std().iloc[-1] if len(bb_range) > 0 else 1.0
    
    # 모든 feat를 0으로 초기화
    for i in range(47):
        df[f'feat{i}'] = 0.0
    
    # 필요한 feature만 계산
    df['feat0'] = v2 / v20.shift(5)  # feat46 의존성
    df['feat2'] = (df['low'] / df['open'])
    df['feat5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['feat7'] = df['stoch_k']
    df['feat17'] = sma40 / sma40.shift(1)
    df['feat28'] = sma60 / sma60.shift(5)
    df['feat32'] = df['pb_sma20'].shift(1)
    df['feat33'] = df['pb_sma40'].shift(1)
    df['feat37'] = df['pb'] / df['feat33']
    df['feat40'] = donchain_min_w1 / donchain_min_w2
    df['feat41'] = donchain_min / donchain_min_w1
    df['feat44'] = bb_range_std / bb_std
    
    # feat46 계산 (expanding 값 사용)
    if expanding_cache and expanding_cache.get('feat0_expanding_mean') is not None and expanding_cache.get('feat0_expanding_std') is not None:
        feat0_mean = expanding_cache['feat0_expanding_mean']
        feat0_std = expanding_cache['feat0_expanding_std']
        df['feat46'] = (df['feat0'] - feat0_mean) / feat0_std if feat0_std != 0 else 0.0
    else:
        df['feat46'] = (df['feat0'] - df['feat0'].shift(1).expanding().mean()) / df['feat0'].shift(1).expanding().std()
    
    return df

def minute60_pb_du(df, expanding_cache=None):
    """
    Minute60 interval pb_du strategy
    Required features: feat1, feat2, feat4, feat5, feat6, feat8, feat13, feat15, feat16, feat21, feat24, feat25, feat27, feat30, feat35, feat42, feat46
    expanding_cache: {'feat0_expanding_mean': float, 'feat0_expanding_std': float}
    """
    # 기본 계산 (strategy_feature에 필요)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['pb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['pb_sma100'] = talib.SMA(df['pb'], timeperiod=100)
    
    pb_sma_period = 100
    pb_sma = 0.7
    pb = 0.7
    df['strategy_feature'] = (df['pb'] > df['pb'].shift(1)) & (df['pb'].shift(1) < pb) & (df['pb'] > pb) & (df['pb_sma100'] > pb_sma)
    
    # 필요한 중간 계산들
    # feat1, feat2, feat4, feat5, feat6, feat8, feat13, feat15, feat16, feat21, feat24, feat25, feat27, feat30, feat35, feat42, feat46
    # 의존성: feat1, feat2, feat4(stoch), feat5(stoch), feat6(stoch), feat8, feat13(pb_sma60), feat15(RSI), feat16(sma20), feat21(sma40,sma60), feat24(sma20), feat25(sma20), feat27(sma40), feat30(sma100), feat35(pb_sma100), feat42(donchain), feat46(feat0)
    
    # stoch 계산 (feat4, feat5, feat6 필요)
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
    
    # sma20 계산 (feat16, feat24, feat25 필요)
    sma20 = df['bb_mid']
    
    # sma40 계산 (feat21, feat27 필요)
    sma40 = talib.SMA(df['close'], timeperiod=40)
    
    # sma60 계산 (feat21 필요)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    
    # sma100 계산 (feat30 필요)
    sma100 = talib.SMA(df['close'], timeperiod=100)
    
    # pb_sma60 계산 (feat13 필요)
    df['pb_sma60'] = talib.SMA(df['pb'], timeperiod=60)
    
    # volume 계산 (feat46 -> feat0 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    
    # donchain 계산 (feat42 필요)
    window = 20
    donchain_max = (df["close"].shift(1).rolling(window).max())
    donchain_max_w1 = donchain_max.shift(window)
    donchain_max_w2 = donchain_max.shift(window*2)
    
    # 모든 feat를 0으로 초기화
    for i in range(47):
        df[f'feat{i}'] = 0.0
    
    # 필요한 feature만 계산
    df['feat0'] = v2 / v20.shift(5)  # feat46 의존성
    df['feat1'] = (df['close'] / df['open'])
    df['feat2'] = (df['low'] / df['open'])
    df['feat4'] = (df['stoch_d'] < df['stoch_k'])
    df['feat5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['feat6'] = (df['stoch_d'].shift(2) > df['stoch_k'].shift(2))
    df['feat8'] = (df['close'] / df['bb_lower'])
    df['feat13'] = df['pb_sma60']
    df['feat15'] = talib.RSI(df['close'], timeperiod=14)
    df['feat16'] = sma20 / sma20.shift(1)
    df['feat21'] = sma40 / sma60
    df['feat24'] = sma20 / sma20.shift(5)
    df['feat25'] = sma20 / sma20.shift(20)
    df['feat27'] = sma40 / sma40.shift(20)
    df['feat30'] = sma100 / sma100.shift(5)
    df['feat35'] = df['pb_sma100'].shift(1)
    df['feat42'] = donchain_max_w1 / donchain_max_w2
    
    # feat46 계산 (expanding 값 사용)
    if expanding_cache and expanding_cache.get('feat0_expanding_mean') is not None and expanding_cache.get('feat0_expanding_std') is not None:
        feat0_mean = expanding_cache['feat0_expanding_mean']
        feat0_std = expanding_cache['feat0_expanding_std']
        df['feat46'] = (df['feat0'] - feat0_mean) / feat0_std if feat0_std != 0 else 0.0
    else:
        df['feat46'] = (df['feat0'] - df['feat0'].shift(1).expanding().mean()) / df['feat0'].shift(1).expanding().std()
    
    return df

def minute240_pb_du(df, expanding_cache=None):
    """
    Minute240 interval pb_du strategy
    Required features: feat2, feat5, feat7, feat10, feat11, feat12, feat15, feat19, feat22, feat24, feat25, feat27, feat29, feat31, feat34, feat35, feat36, feat37, feat39, feat40, feat41, feat46
    expanding_cache: {'feat0_expanding_mean': float, 'feat0_expanding_std': float}
    """
    # 기본 계산 (strategy_feature에 필요)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['pb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['pb_sma60'] = talib.SMA(df['pb'], timeperiod=60)
    
    pb_sma_period = 60
    pb_sma = 0.75
    pb = 0.8
    df['strategy_feature'] = (df['pb'] > df['pb'].shift(1)) & (df['pb'].shift(1) < pb) & (df['pb'] > pb) & (df['pb_sma60'] > pb_sma)
    
    # 필요한 중간 계산들
    # feat2, feat5, feat7, feat10, feat11, feat12, feat15, feat19, feat22, feat24, feat25, feat27, feat29, feat31, feat34, feat35, feat36, feat37, feat39, feat40, feat41, feat46
    # 의존성: feat2, feat5(stoch), feat7(stoch), feat10(pb), feat11(pb_sma20), feat12(pb_sma40), feat15(RSI), feat19(sma100), feat22(sma60,sma100), feat24(sma20), feat25(sma20), feat27(sma40), feat29(sma60), feat31(sma100), feat34(pb_sma60), feat35(pb_sma100), feat36(feat32), feat37(feat33), feat39(feat35), feat40(donchain), feat41(donchain), feat46(feat0)
    
    # stoch 계산 (feat5, feat7 필요)
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
    
    # sma20 계산 (feat24, feat25 필요)
    sma20 = df['bb_mid']
    
    # sma40 계산 (feat27 필요)
    sma40 = talib.SMA(df['close'], timeperiod=40)
    
    # sma60 계산 (feat22, feat29 필요)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    
    # sma100 계산 (feat19, feat22, feat31 필요)
    sma100 = talib.SMA(df['close'], timeperiod=100)
    
    # pb_sma 계산 (feat11, feat12, feat34, feat35, feat36, feat37, feat39 필요)
    df['pb_sma20'] = talib.SMA(df['pb'], timeperiod=20)
    df['pb_sma40'] = talib.SMA(df['pb'], timeperiod=40)
    df['pb_sma100'] = talib.SMA(df['pb'], timeperiod=100)
    
    # volume 계산 (feat46 -> feat0 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    
    # donchain 계산 (feat40, feat41 필요)
    window = 20
    donchain_min = (df["close"].shift(1).rolling(window).min())
    donchain_min_w1 = donchain_min.shift(window)
    donchain_min_w2 = donchain_min.shift(window*2)
    
    # 모든 feat를 0으로 초기화
    for i in range(47):
        df[f'feat{i}'] = 0.0
    
    # 필요한 feature만 계산
    df['feat0'] = v2 / v20.shift(5)  # feat46 의존성
    df['feat2'] = (df['low'] / df['open'])
    df['feat5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['feat7'] = df['stoch_k']
    df['feat10'] = pb
    df['feat11'] = df['pb_sma20']
    df['feat12'] = df['pb_sma40']
    df['feat15'] = talib.RSI(df['close'], timeperiod=14)
    df['feat19'] = sma100 / sma100.shift(1)
    df['feat22'] = sma60 / sma100
    df['feat24'] = sma20 / sma20.shift(5)
    df['feat25'] = sma20 / sma20.shift(20)
    df['feat27'] = sma40 / sma40.shift(20)
    df['feat29'] = sma60 / sma60.shift(20)
    df['feat31'] = sma100 / sma100.shift(20)
    df['feat34'] = df['pb_sma60'].shift(1)
    df['feat35'] = df['pb_sma100'].shift(1)
    df['feat32'] = df['pb_sma20'].shift(1)  # feat36 의존성
    df['feat33'] = df['pb_sma40'].shift(1)  # feat37 의존성
    df['feat36'] = df['pb'] / df['feat32']
    df['feat37'] = df['pb'] / df['feat33']
    df['feat39'] = df['pb'] / df['feat35']
    df['feat40'] = donchain_min_w1 / donchain_min_w2
    df['feat41'] = donchain_min / donchain_min_w1
    
    # feat46 계산 (expanding 값 사용)
    if expanding_cache and expanding_cache.get('feat0_expanding_mean') is not None and expanding_cache.get('feat0_expanding_std') is not None:
        feat0_mean = expanding_cache['feat0_expanding_mean']
        feat0_std = expanding_cache['feat0_expanding_std']
        df['feat46'] = (df['feat0'] - feat0_mean) / feat0_std if feat0_std != 0 else 0.0
    else:
        df['feat46'] = (df['feat0'] - df['feat0'].shift(1).expanding().mean()) / df['feat0'].shift(1).expanding().std()
    
    return df
