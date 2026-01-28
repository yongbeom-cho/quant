import talib
import numpy as np
from xgboost import XGBClassifier

def apply_strategy_xgb_feats(df, interval, strategy_feature_name):
    if strategy_feature_name == 'low_bb_du':
        return low_bb_du(df, interval)
    elif strategy_feature_name == 'pb_du':
        return pb_du(df, interval)
    return None

def get_feats_num(strategy_feature_name):
    if strategy_feature_name == 'low_bb_du':
        return 44
    elif strategy_feature_name == 'pb_du':
        return 47
    return 0


def apply_strategy_xgb(df, model_name, model_input_path):
    #1. feat apply
    dummy, market, interval, target_strategy, label_name, min_precision, threshold, str_feat = model_name.split('-')
    threshold = float(threshold)
    df = apply_strategy_xgb_feats(df, interval, target_strategy)
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

def low_bb_du(df, interval):
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    sma20 = df['bb_mid']
    sma40 = talib.SMA(df['close'], timeperiod=40)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    sma100 = talib.SMA(df['close'], timeperiod=100)

    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    
    cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))

    if interval == 'minute60':
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool)) & (sma60 > sma60.shift(4))
    elif interval == 'day':
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool)) & (sma100 > sma60.shift(24))
    else:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool)) & (sma100 > sma100.shift(4))

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

    df['feat0'] = v2 / v20.shift(5)
    
    df['feat1'] = (df['close'] / df['open'])
    df['feat2'] = (df['low'] / df['open'])
    df['feat3'] = (df['high'] / df['open'])

    df['feat4'] = (df['stoch_d'] < df['stoch_k'])
    df['feat5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['feat6'] = (df['stoch_d'].shift(2) > df['stoch_k'].shift(2))
    df['feat7'] = df['stoch_k']
    df['feat8'] = (df['close'] / df['bb_lower']) 
    df['feat9'] = df['close'] / (df['close'].shift(1).rolling(5).min())
    
    bb_range = df["bb_upper"] - df["bb_lower"]
    pb = ((df['close'] - df['bb_lower']) / (bb_range))
    df['feat10'] = pb
    pb20 = talib.SMA(df['feat10'], 20)
    pb40 = talib.SMA(df['feat10'], 40)
    pb60 = talib.SMA(df['feat10'], 60)
    pb100 = talib.SMA(df['feat10'], 100)
    
    
    df['feat11'] = False
    for i in range(1, 6):
        df['feat11'] |= df['close'].shift(i) < df['bb_lower'].shift(i)
    
    df['feat12'] = talib.RSI(df['close'], timeperiod=14)
    
    df['feat13'] = sma20 / sma20.shift(1)
    df['feat14'] = sma40 / sma40.shift(1)
    df['feat15'] = sma60 / sma60.shift(1)
    df['feat16'] = sma100 / sma100.shift(1)

    df['feat17'] = sma20 / sma40
    df['feat18'] = sma40 / sma60
    df['feat19'] = sma60 / sma100

    df['feat20'] = (df['feat15'] - df['feat15'].expanding().mean()) / df['feat15'].expanding().std()
    df['feat21'] = sma20 / sma20.shift(5)
    df['feat22'] = sma20 / sma20.shift(20)
    df['feat23'] = sma40 / sma40.shift(5)
    df['feat24'] = sma40 / sma40.shift(20)
    df['feat25'] = sma60 / sma60.shift(5)
    df['feat26'] = sma60 / sma60.shift(20)
    df['feat27'] = sma100 / sma100.shift(5)
    df['feat28'] = sma100 / sma100.shift(20)

    df['feat29'] = pb20.shift(1)
    df['feat30'] = pb40.shift(1)
    df['feat31'] = pb60.shift(1)
    df['feat32'] = pb100.shift(1)
    df['feat33'] = pb / df['feat29']
    df['feat34'] = pb / df['feat30']
    df['feat35'] = pb / df['feat31']
    df['feat36'] = pb / df['feat32']
    
    window = 20
    donchain_min = (df["close"].shift(1).rolling(window).min())
    donchain_max = (df["close"].shift(1).rolling(window).max())
    donchain_min_w1 = donchain_min.shift(window)
    donchain_max_w1 = donchain_max.shift(window)
    donchain_min_w2 = donchain_min.shift(window*2)
    donchain_max_w2 = donchain_max.shift(window*2)

    df['feat37'] = donchain_min_w1 / donchain_min_w2
    df['feat38'] = donchain_min / donchain_min_w1
    df['feat39'] = donchain_max_w1 / donchain_max_w2
    df['feat40'] = donchain_max / donchain_max_w1
    
    bb_range_std = (bb_range.shift(1).rolling(window).std())
    bb_std = bb_range.shift(1).expanding().std()
    df['feat41'] = bb_range_std / bb_std
    df['feat42'] = bb_range_std.shift(5) / bb_std
    df['feat43'] = (df['feat0'] - df['feat0'].shift(1).expanding().mean()) / df['feat0'].shift(1).expanding().std()
    return df


def pb_du(df, interval):
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['pb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['pb_sma20'] = talib.SMA(df['pb'], timeperiod=20)
    df['pb_sma40'] = talib.SMA(df['pb'], timeperiod=40)
    df['pb_sma60'] = talib.SMA(df['pb'], timeperiod=60)
    df['pb_sma100'] = talib.SMA(df['pb'], timeperiod=100)
    df['pb_sma200'] = talib.SMA(df['pb'], timeperiod=200)
    
    sma20 = df['bb_mid']
    sma40 = talib.SMA(df['close'], timeperiod=40)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    sma100 = talib.SMA(df['close'], timeperiod=100)

    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    
    pb_sma_period = 60
    pb_sma = 0.75
    pb = 0.8
    if interval == 'minute60':
        pb_sma_period = 100
        pb_sma = 0.7
        pb = 0.7
    elif interval == 'day':
        pb_sma_period = 20
        pb_sma = 0.7
        pb = 0.8

    df['strategy_feature'] = (df['pb'] > df['pb'].shift(1)) & (df['pb'].shift(1) < pb) & (df['pb'] > pb) & (df['pb_sma' + str(pb_sma_period)] > pb_sma)
    

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

    df['feat0'] = v2 / v20.shift(5)
    
    df['feat1'] = (df['close'] / df['open'])
    df['feat2'] = (df['low'] / df['open'])
    df['feat3'] = (df['high'] / df['open'])

    df['feat4'] = (df['stoch_d'] < df['stoch_k'])
    df['feat5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['feat6'] = (df['stoch_d'].shift(2) > df['stoch_k'].shift(2))
    df['feat7'] = df['stoch_k']
    df['feat8'] = (df['close'] / df['bb_lower']) 
    df['feat9'] = df['close'] / (df['close'].shift(1).rolling(5).min())
    
    
    df['feat10'] = pb
    df['feat11'] = df['pb_sma20']
    df['feat12'] = df['pb_sma40']
    df['feat13'] = df['pb_sma60']
    df['feat14'] = df['pb_sma100']
    
    
    df['feat15'] = talib.RSI(df['close'], timeperiod=14)
    
    df['feat16'] = sma20 / sma20.shift(1)
    df['feat17'] = sma40 / sma40.shift(1)
    df['feat18'] = sma60 / sma60.shift(1)
    df['feat19'] = sma100 / sma100.shift(1)

    df['feat20'] = sma20 / sma40
    df['feat21'] = sma40 / sma60
    df['feat22'] = sma60 / sma100

    df['feat23'] = (df['feat15'] - df['feat15'].expanding().mean()) / df['feat15'].expanding().std()
    df['feat24'] = sma20 / sma20.shift(5)
    df['feat25'] = sma20 / sma20.shift(20)
    df['feat26'] = sma40 / sma40.shift(5)
    df['feat27'] = sma40 / sma40.shift(20)
    df['feat28'] = sma60 / sma60.shift(5)
    df['feat29'] = sma60 / sma60.shift(20)
    df['feat30'] = sma100 / sma100.shift(5)
    df['feat31'] = sma100 / sma100.shift(20)

    df['feat32'] = df['pb_sma20'].shift(1)
    df['feat33'] = df['pb_sma40'].shift(1)
    df['feat34'] = df['pb_sma60'].shift(1)
    df['feat35'] = df['pb_sma100'].shift(1)
    df['feat36'] = df['pb'] / df['feat32']
    df['feat37'] = df['pb'] / df['feat33']
    df['feat38'] = df['pb'] / df['feat34']
    df['feat39'] = df['pb'] / df['feat35']
    
    window = 20
    donchain_min = (df["close"].shift(1).rolling(window).min())
    donchain_max = (df["close"].shift(1).rolling(window).max())
    donchain_min_w1 = donchain_min.shift(window)
    donchain_max_w1 = donchain_max.shift(window)
    donchain_min_w2 = donchain_min.shift(window*2)
    donchain_max_w2 = donchain_max.shift(window*2)

    df['feat40'] = donchain_min_w1 / donchain_min_w2
    df['feat41'] = donchain_min / donchain_min_w1
    df['feat42'] = donchain_max_w1 / donchain_max_w2
    df['feat43'] = donchain_max / donchain_max_w1
    
    bb_range = df["bb_upper"] - df["bb_lower"]
    bb_range_std = (bb_range.shift(1).rolling(window).std())
    bb_std = bb_range.shift(1).expanding().std()
    df['feat44'] = bb_range_std / bb_std
    df['feat45'] = bb_range_std.shift(5) / bb_std
    df['feat46'] = (df['feat0'] - df['feat0'].shift(1).expanding().mean()) / df['feat0'].shift(1).expanding().std()
    return df