import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
import os
import mplfinance as mpf
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.strategy import apply_strategy, get_strategy_params_list, get_sell_strategy_params_list

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def view_chart(df):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot = df_plot.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    })

    bb_upper = mpf.make_addplot(df_plot['bb_upper'], color='red', width=1)
    bb_mid   = mpf.make_addplot(df_plot['bb_mid'],   color='green', width=1)
    bb_lower = mpf.make_addplot(df_plot['bb_lower'], color='blue', width=1)

    signal_y = np.where(df_plot['signal'], df_plot['Low'] * 0.995, np.nan)

    signal_plot = mpf.make_addplot(
        signal_y,
        type='scatter',
        marker='^',
        markersize=120,
        color='black'
    )

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style='yahoo',
        addplot=[bb_upper, bb_mid, bb_lower, signal_plot],
        volume=False,
        figsize=(16, 8),
        returnfig=True
    )

    # 🔑 axes[0] = price axis
    ax = axes[0]

    ax.set_title('Candlestick with Bollinger Bands & Signal')

    plt.show()

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


def apply_buy_signal_strategy(
    db_path,
    table_name,
    ticker,
    strategy_name,
    params
):
    """
    db_path: DB path
    table_name: {market}_ohlcv_{interval}
    ticker: 종목 (코인명))
    strategy_name: 전략 이름
    params: 전략에 따른 parameter들
    """
    df = load_ohlcv(db_path, table_name, ticker)
    df = apply_strategy(df, strategy_name, params)
    return df
    

parser = argparse.ArgumentParser(description='02_strategy_unit_backtest')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute1")
parser.add_argument('--target_ticker', type=str, default="all") #KRW-BTC
parser.add_argument('--target_strategy', type=str, default="all") #explode_volume_breakout

args = parser.parse_args()

if __name__ == "__main__":
    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    strategy_config_path = os.path.join(args.root_dir, 'sbin/strategy/best_config.json')
    with open(strategy_config_path, 'r', encoding='utf-8') as f:
        strategy_config_list = json.load(f)
    
    tickers = get_tickers(db_path, table_name)
    if args.target_ticker != 'all':
        if args.target_ticker in tickers:
            tickers = [args.target_ticker]
    
    
    for strategy_config in strategy_config_list:
        strategy_name = strategy_config['strategy_name']
        if args.target_strategy != 'all' and args.target_strategy != strategy_name:
            continue
        buy_params = strategy_config['buy_signal_config']
        sell_params = strategy_config['sell_signal_config']
        for ticker in tickers:
            df = apply_buy_signal_strategy(
                    db_path=db_path,
                    table_name=table_name,
                    ticker=ticker,
                    strategy_name=strategy_name,
                    params=buy_params
                )
            view_chart(df)

                