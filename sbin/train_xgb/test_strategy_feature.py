import numpy as np
import argparse
import os
import sys
import sqlite3
import pandas as pd
import talib

def get_tickers(db_path, table_name):
    column = "ticker"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {column} FROM {table_name}")
    tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers

def load_ohlcv(db_path, table_name, ticker):
    df = None
    max_retries = 3
    retry_delay = 10
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE ticker = ?
        ORDER BY date ASC
    """

    for attempt in range(1, max_retries + 1):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()
            return df

        except Exception as e:
            print(f"[ERROR] DB Load 실패 (attempt {attempt}/{max_retries}): {e}")

            # 마지막 시도 실패면 예외 던짐
            if attempt == max_retries:
                raise

            # 대기 후 재시도
            time.sleep(retry_delay)

    # 논리적으로 여기까지 오면 안 오지만 안정성 위해
    raise RuntimeError("DB load retry failed unexpectedly.")

    return df

def label_df(df, label_name, upper, lower):
    labels = np.full(len(df), np.nan)

    buy_indices = df.index[df['strategy_feature'] == True]

    for idx in buy_indices:
        buy_close = df.at[idx, 'close']

        future = df.loc[idx+1:]

        # 손절 먼저 체크
        stop_loss = future[future['low'] < buy_close * lower]
        take_profit = future[future['high'] > buy_close * upper]

        if not stop_loss.empty and not take_profit.empty:
            # 둘 다 발생하면 더 먼저 발생한 것
            if stop_loss.index[0] <= take_profit.index[0]:
                labels[idx] = 0
            else:
                labels[idx] = 1

        elif not stop_loss.empty:
            labels[idx] = 0

        elif not take_profit.empty:
            labels[idx] = 1
        # else:
        #     # 끝까지 갔을 때
        #     last_close = future.iloc[-1]['close']
        #     labels[idx] = 1 if last_close > buy_close else 0

    df[label_name] = labels
    return df

def test(df, idx):
    
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
    sma20 = talib.SMA(df['close'], timeperiod=20)
    sma40 = talib.SMA(df['close'], timeperiod=40)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    sma100 = talib.SMA(df['close'], timeperiod=100)
    # df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower'])
    if idx == 0:
        df['strategy_feature'] = True
    elif idx == 1:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower'])
    elif idx == 2:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond'])
    elif idx == 3:
        df['strategy_feature'] = (df['stoch_k'] < 30) & (df['close'] > df['bb_lower'])
    elif idx == 4:
        df['strategy_feature'] = (df['stoch_k'] < 30) & (df['close'] < df['bb_lower'])
    elif idx == 5:
        df['strategy_feature'] = (df['stoch_k'] < 30) & ((df['stoch_d'] < df['stoch_k']) & (df['stoch_d'].shift(1) > df['stoch_k'].shift(1)))
    elif idx == 6:
        df['strategy_feature'] = (df['stoch_k'] < 30) & ((df['stoch_d'] > df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_k'].shift(1)))
    elif idx == 7:
        df['strategy_feature'] = (df['stoch_k'] < 30) & ((df['stoch_d'] > df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_k'].shift(1))) & (df['close'] < df['bb_lower'])
    elif idx == 8:
        df['strategy_feature'] = (df['stoch_k'] < 30) & ((df['stoch_d'] > df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_k'].shift(1))) & (df['close'] > df['bb_lower'])
    elif idx == 9:
        df['aa'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower'])
        cond_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = df['aa'] & df['aa'].shift(1) & (df['cond'])
    elif idx == 10:
        df['aa'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower'])
        cond_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = df['aa'] & df['aa'].shift(1) & (df['cond']) & (df['stoch_k'] < 30)
    elif idx == 11:
        df['aa'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower'])
        cond_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = df['aa'] & df['aa'].shift(1) & (df['cond']) & (df['close'] > df['open'])
    elif idx == 12:
        df['aa'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower'])
        cond_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = df['aa'] & df['aa'].shift(1) & (df['cond']) & (df['close'] < df['bb_mid'])
    elif idx == 13:
        df['aa'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower'])
        cond_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['under'] = df['bb_lower'] + (df['bb_mid'] - df['bb_lower']) / 2
        df['strategy_feature'] = df['aa'] & df['aa'].shift(1) & (df['cond']) & (df['close'] < df['under'])
    elif idx == 14:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma100 > sma100.shift(4))
    elif idx == 15:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma100 > sma100.shift(14))
    elif idx == 16:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma100 > sma100.shift(24))
    elif idx == 17:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma60 > sma60.shift(4))
    elif idx == 18:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma60 > sma60.shift(14))
    elif idx == 19:
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower']) & (sma60 > sma60.shift(24))
    elif idx == 20:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma100 > sma100.shift(4))
    elif idx == 21:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma100 > sma100.shift(14))
    elif idx == 22:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma100 > sma100.shift(24))
    elif idx == 23:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma60 > sma60.shift(4))
    elif idx == 24:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma60 > sma100.shift(14))
    elif idx == 25:
        cond4_raw = ((df['close'] < df['bb_lower']) & ((df['close'] / df['open']) < 1.0))
        df['cond'] = (cond4_raw.shift(1).rolling(2).max().fillna(False).astype(bool))
        df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] > df['bb_lower']) & (df['cond']) & (sma100 > sma60.shift(24))


    df = label_df(df, 'label', 1.12, 0.9)
    filtered_df = df[df['strategy_feature']]
    pos_cnt = (df['label'] == 1).sum()
    neg_cnt = (df['label'] == 0).sum()
    return pos_cnt, neg_cnt



parser = argparse.ArgumentParser(description='08_test_strategy_feature')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
# parser.add_argument('--interval', type=str, default="minute60")

args = parser.parse_args()


for interval in ['minute60', 'minute240', 'day']:
    table_name = f'{args.market}_ohlcv_{interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    xgb_dir = os.path.join(args.root_dir, "var/xgb_data")
    os.makedirs(xgb_dir, exist_ok=True)

    tickers = get_tickers(db_path, table_name)
    for idx in range(1, 26):
        pos_cnt = 0
        neg_cnt = 0
        for ticker in tickers:
            df = load_ohlcv(db_path, table_name, ticker)
            if len(df) <= 50:
                continue
            p_cnt, n_cnt = test(df, idx)
            pos_cnt += p_cnt
            neg_cnt += n_cnt
        print(idx, interval, pos_cnt, neg_cnt, "%.3f" %(pos_cnt/float(pos_cnt+neg_cnt)))