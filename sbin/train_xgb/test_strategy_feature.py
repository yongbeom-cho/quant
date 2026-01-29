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

def test(df, target_idx):
    
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['pb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['pb_sma20'] = talib.SMA(df['pb'], timeperiod=20)
    df['pb_sma40'] = talib.SMA(df['pb'], timeperiod=40)
    df['pb_sma60'] = talib.SMA(df['pb'], timeperiod=60)
    df['pb_sma100'] = talib.SMA(df['pb'], timeperiod=100)
    df['pb_sma200'] = talib.SMA(df['pb'], timeperiod=200)

    pb_sma_periods = [20, 40, 60, 100, 200] #5
    pb_smas =[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] #7
    pbs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3] #6
    
    idx = -1
    for pb_sma_period in pb_sma_periods:
        for pb_sma in pb_smas:
            for pb in pbs:
                idx += 1
                if idx != target_idx:
                    continue
                param_name = str(pb_sma_period) + '_' + str(pb_sma) + '_' + str(pb)
                df['strategy_feature'] = (df['pb'] > df['pb'].shift(1)) & (df['pb'].shift(1) < pb) & (df['pb'] > pb) & (df['pb_sma' + str(pb_sma_period)] > pb_sma)
                df = label_df(df, 'label', 1.12, 0.9)
                filtered_df = df[df['strategy_feature']]
                pos_cnt = (df['label'] == 1).sum()
                neg_cnt = (df['label'] == 0).sum()
                return pos_cnt, neg_cnt, param_name



parser = argparse.ArgumentParser(description='08_test_strategy_feature')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute60")
parser.add_argument('--parallel', type=int, default=8)
parser.add_argument('--part', type=int, default=0)


args = parser.parse_args()


interval = args.interval

table_name = f'{args.market}_ohlcv_{interval}'
db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
tickers = get_tickers(db_path, table_name)

for idx in range(210):
    if idx % args.parallel != args.part:
        continue
    pos_cnt = 0
    neg_cnt = 0
    for ticker in tickers:
        df = load_ohlcv(db_path, table_name, ticker)
        if len(df) <= 220:
            continue
        p_cnt, n_cnt, param_name = test(df, idx)
        pos_cnt += p_cnt
        neg_cnt += n_cnt
    print(idx, param_name, interval, pos_cnt, neg_cnt, "%.3f" %(pos_cnt/float(pos_cnt+neg_cnt)))