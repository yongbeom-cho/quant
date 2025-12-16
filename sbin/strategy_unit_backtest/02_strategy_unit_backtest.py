import os
import sys
import sqlite3
import pandas as pd
import argparse
import json
import ast
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
    df['result'] = 0
    df['return_ratio'] = 0.0
    strategy_indices = df.index[df['signal'] == True].tolist()

    # Determine which cut rates to use
    loss_cut_rate = params.get('loss_cut_rate', params.get('low_limit_ratio'))
    profit_cut_rate = params.get('profit_cut_rate', params.get('high_limit_ratio'))

    if loss_cut_rate is not None and profit_cut_rate is not None:
        for idx in strategy_indices:
            entry_close = df.loc[idx, "close"]
            stop_price = entry_close * loss_cut_rate
            take_price = entry_close * profit_cut_rate

            # 이후 row 탐색
            for j in range(idx + 1, len(df)):
                low_j = df.loc[j, "low"]
                high_j = df.loc[j, "high"]
                
                # 다음 signal이 나오면 현재 포지션은 종료된 것으로 간주 (현재 봉에서 매도)
                if j in strategy_indices:
                    sell_price = df.loc[j, "open"] # 다음 봉 시가에 매도
                    df.loc[idx, "result"] = 1 if sell_price > entry_close else -1
                    df.loc[idx, "return_ratio"] = sell_price / entry_close
                    break

                hit_stop = low_j < stop_price
                hit_take = high_j > take_price

                if hit_stop and hit_take: # 하루에 익절/손절 모두 도달
                    open_j = df.loc[j, "open"]
                    if open_j > entry_close: # 갭상승으로 시작하면 익절로 처리
                         df.loc[idx, "result"] = 1
                         df.loc[idx, "return_ratio"] = profit_cut_rate
                    else: # 아니면 손절로 처리
                        df.loc[idx, "result"] = -1
                        df.loc[idx, "return_ratio"] = loss_cut_rate
                    break
                elif hit_stop:
                    df.loc[idx, "result"] = -1
                    df.loc[idx, "return_ratio"] = loss_cut_rate
                    break
                elif hit_take:
                    df.loc[idx, "result"] = 1
                    df.loc[idx, "return_ratio"] = profit_cut_rate
                    break
                elif j == len(df) - 1: # 탐색이 끝까지 갔으면
                    sell_price = df.loc[j, "close"] # 종가에 매도
                    df.loc[idx, "result"] = 1 if sell_price > entry_close else -1
                    df.loc[idx, "return_ratio"] = sell_price / entry_close
                    break
    
    elif 'hold_period_candles' in params:
        hold_period_candles = params['hold_period_candles']
        for idx in strategy_indices:
            entry_close = df.loc[idx, "close"]
            sell_idx = idx + hold_period_candles
            
            # 다음 signal이 보유기간 내에 나오면 거기서 매도
            next_signal_indices = [i for i in strategy_indices if i > idx]
            if next_signal_indices and next_signal_indices[0] <= sell_idx:
                sell_idx = next_signal_indices[0]

            if sell_idx < len(df):
                sell_close = df.loc[sell_idx, "close"]
                df.loc[idx, "result"] = 1 if sell_close > entry_close else -1
                df.loc[idx, "return_ratio"] = sell_close / entry_close
            else: # 기간 못채우고 끝나면 손실로 처리
                sell_close = df.iloc[-1]["close"]
                df.loc[idx, "result"] = -1 
                df.loc[idx, "return_ratio"] = sell_close / entry_close

    else:
        raise ValueError("Sell strategy parameters are missing or invalid.")

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
    strategy_config_path = os.path.join(args.root_dir, 'sbin/strategy/config.json')
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
        buy_params_list = get_strategy_params_list(strategy_name, strategy_config['buy_signal_config'])
        sell_params_list = get_sell_strategy_params_list(strategy_name, strategy_config['sell_signal_config'])
        params_win_loses = {}
        params_expected_returns = {}
        params_total_rors = {}

        print("@@@ STRATEGY BACKTEST START @@@")
        print("%s" %(strategy_name))
        for buy_params in buy_params_list:
            params_ticker_results = defaultdict(lambda: {'win': 0, 'lose': 0, 'ror': []})

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
                    
                    df_result = apply_sell_signal_strategy(df.copy(), sell_params)

                    n_win  = (df_result['result'] == 1).sum()
                    n_lose = (df_result['result'] == -1).sum()
                    rors = df_result['return_ratio'][df_result['return_ratio'] != 0].tolist()

                    params_ticker_results[str_params]['win'] += n_win
                    params_ticker_results[str_params]['lose'] += n_lose
                    params_ticker_results[str_params]['ror'].extend(rors)


            for str_params, results in params_ticker_results.items():
                params_n_win = results['win']
                params_n_lose = results['lose']
                params_win_ratio = params_n_win/float(params_n_win+params_n_lose) if params_n_win + params_n_lose > 0 else 0.0
                
                buy_str_params, sell_str_params = str_params.split('-')
                sell_params = ast.literal_eval(sell_str_params)
                
                loss_cut_rate = sell_params.get('loss_cut_rate', sell_params.get('low_limit_ratio'))
                profit_cut_rate = sell_params.get('profit_cut_rate', sell_params.get('high_limit_ratio'))

                if loss_cut_rate and profit_cut_rate:
                    rrr = (profit_cut_rate-1.0) / (1.0-loss_cut_rate) if loss_cut_rate != 1.0 else 0
                    params_expected_returns[str_params] = (params_win_ratio*rrr) - (1-params_win_ratio)
                    print("%s  -  win count : %d, loss count : %d, win_ratio : %.2f, risk-reward-ratio : %.2f" %(str_params, params_n_win, params_n_lose, params_win_ratio, rrr))
                else:
                    params_expected_returns[str_params] = None
                    print("%s  -  win count : %d, loss count : %d, win_ratio : %.2f" %(str_params, params_n_win, params_n_lose, params_win_ratio))

                params_win_loses[str_params] = (params_n_win, params_n_lose)
                
                if results['ror']:
                    total_ror = pd.Series(results['ror']).prod()
                    params_total_rors[str_params] = total_ror

        
        expected_return_mean = sum(v for v in params_expected_returns.values() if v is not None) / len(params_expected_returns) if params_expected_returns else 0.0
        print("### STRATEGY_RESULT START ###")
        print("%s" %(strategy_name))
        print("%s\t%.2f" %(strategy_name, expected_return_mean))
        print("### STRATEGY_PARAMS_RESULT START ###")

        print("######### Expected Return Sorted #########")
        sorted_results = sorted(params_expected_returns.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
        for params_str, expected_return in sorted_results[:50]:
            if expected_return is not None:
                print(params_str, expected_return, " | ", params_win_loses[params_str])

        print("######### Total ROR Sorted #########")
        sorted_ror_results = sorted(params_total_rors.items(), key=lambda x: x[1], reverse=True)
        for params_str, total_ror in sorted_ror_results[:50]:
            print(params_str, total_ror, " | ", params_win_loses[params_str])

        print("### STRATEGY_PARAMS_RESULT END ###")
        print("### STRATEGY_RESULT END ###")