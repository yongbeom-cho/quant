"""
Feature Calculation Module

XGBoost 모델 학습을 위한 feature 계산 로직
"""
import talib
import numpy as np
import pandas as pd
from typing import Optional


def get_features(df: pd.DataFrame, interval: str, strategy_name: Optional[str] = None) -> pd.DataFrame:
    """
    전략별 feature 계산
    
    Args:
        df: OHLCV DataFrame
        interval: 시간 간격 (day, minute60, minute240)
        strategy_name: 전략 이름 (선택적)
    
    Returns:
        feature가 추가된 DataFrame
    """
    if strategy_name == 'pb_rebound' or strategy_name is None:
        return pb_rebound_features(df, interval)
    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")


def pb_rebound_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """PB Rebound 전략 feature 계산"""
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
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
    
    df['feat10'] = df['pb']
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
