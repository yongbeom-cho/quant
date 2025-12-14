import os
import sys
import sqlite3
import pandas as pd
import argparse
import json
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.strategy import apply_strategy, get_strategy_params_list, get_sell_strategy_params_list

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

def get_tickers(db_path, table_name):
    column = "ticker"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {column} FROM {table_name}")
    tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


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

def apply_sell_signal_strategy(
    df,
    params
):
    low_limit_ratio = params['low_limit_ratio']
    high_limit_ratio = params['high_limit_ratio']

    strategy_indices = df.index[df['signal'] == True].tolist()
    df['result'] = 0
    for idx in strategy_indices:
        entry_close = df.loc[idx, "close"]
        stop_price = entry_close * low_limit_ratio
        take_price = entry_close * high_limit_ratio

        # 이후 row 탐색
        for j in range(idx + 1, len(df)):
            low_j = df.loc[j, "low"]
            high_j = df.loc[j, "high"]

            hit_stop = low_j < stop_price
            hit_take = high_j > take_price

            if hit_stop:
                df.loc[idx, "result"] = -1
                break
            elif hit_take:
                df.loc[idx, "result"] = 1
                break
            elif j == len(df) - 1:
                df.loc[idx, "result"] = -1
    return df

parser = argparse.ArgumentParser(description='02_strategy_unit_backtest')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute1")
args = parser.parse_args()

# 사용 예시
if __name__ == "__main__":
    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    strategy_config_path = os.path.join(args.root_dir, 'sbin/strategy/config.json')
    with open(strategy_config_path, 'r', encoding='utf-8') as f:
        strategy_config_list = json.load(f)
    tickers = get_tickers(db_path, table_name)

    for strategy_config in strategy_config_list:
        strategy_name = strategy_config['strategy_name']
        buy_params_list = get_strategy_params_list(strategy_name, strategy_config['buy_signal_config'])
        sell_params_list = get_sell_strategy_params_list(strategy_name, strategy_config['sell_signal_config'])
        params_win_loses = {}
        params_expected_returns = {}
        print("@@@ STRATEGY BACKTEST START @@@")
        print("%s" %(strategy_name))
        for buy_params in buy_params_list:
            params_ticker_win_loses = defaultdict(dict)
            for ticker in tickers:
                df = apply_buy_signal_strategy(
                        db_path=db_path,
                        table_name=table_name,
                        ticker=ticker,
                        strategy_name=strategy_name,
                        params=buy_params
                    )
                for sell_params in sell_params_list:
                    str_params = str(buy_params) + '-' + str(sell_params)
                    low_limit_ratio = sell_params['low_limit_ratio']
                    high_limit_ratio = sell_params['high_limit_ratio']
                    rrr = (high_limit_ratio-1.0) / (1.0-low_limit_ratio)
                    
                    df = apply_sell_signal_strategy(df, sell_params)

                    n_win  = (df['result'] == 1).sum()
                    n_lose = (df['result'] == -1).sum()
                    win_ratio = n_win/float(n_win+n_lose) if n_win + n_lose > 0 else 0.0
                    params_ticker_win_loses[str_params][ticker] = (n_win, n_lose)
                    # print("%s  -  win count : %d, loss count : %d, win_ratio : %.2f, risk-reward-ratio : %.1f" %(ticker, n_win, n_lose, win_ratio, rrr))
                    # win_indices = df.index[df['result'] == 1].tolist()
                    # loss_indices = df.index[df['result'] == -1].tolist()
                    # print(df.head(30))

            for dic_str_params, ticker_win_loses in params_ticker_win_loses.items():
                params_n_win = sum(v[0] for v in ticker_win_loses.values())
                params_n_lose = sum(v[1] for v in ticker_win_loses.values())
                params_win_ratio = params_n_win/float(params_n_win+params_n_lose) if params_n_win + params_n_lose > 0 else 0.0
                buy_str_params, sell_str_params = dic_str_params.split('-')
                buy_params = json.loads(buy_str_params)
                sell_params = json.loads(sell_str_params)
                low_limit_ratio = sell_params['low_limit_ratio']
                high_limit_ratio = sell_params['high_limit_ratio']
                rrr = (high_limit_ratio-1.0) / (1.0-low_limit_ratio)
                print("%s  -  win count : %d, loss count : %d, win_ratio : %.2f, risk-reward-ratio : %.1f" %(dic_str_params, params_n_win, params_n_lose, params_win_ratio, rrr))
                params_win_loses[dic_str_params] = (params_n_win, params_n_lose)
                params_expected_returns[dic_str_params] = (params_win_ratio*rrr) - (1-params_win_ratio)
        
        expected_return_mean = sum(params_expected_returns.values()) / len(params_expected_returns)
        print("### STRATEGY_RESULT START ###")
        print("%s" %(strategy_name))
        print("%s\t%.2f" %(strategy_name, expected_return_mean))
        print("### STRATEGY_PARAMS_RESULT START ###")
        sorted_results = sorted(params_expected_returns.items(), key=lambda x: x[1], reverse=True)
        for params_str, expected_return in sorted_results:
            print(params_str, expected_return)
        print("### STRATEGY_PARAMS_RESULT END ###")
        print("### STRATEGY_RESULT END ###")
