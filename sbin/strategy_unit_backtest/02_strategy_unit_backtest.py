import os
import sys
import sqlite3
import pandas as pd
import argparse
import json
import ast
import time
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.strategy import apply_strategy, get_strategy_params_list, get_sell_strategy_params_list

def run_larry_williams_backtest(df, buy_params, sell_params):
    """
    래리 윌리엄스 변동성 돌파 전략을 위한 상세 백테스터
    """
    # 파라미터 추출
    k = buy_params['k']
    stop_loss_pct = sell_params['stop_loss_pct']
    profit_target_pct = sell_params['profit_target_pct']
    hold_period_candles = sell_params['hold_period_candles']

    trades = []
    position = None

    # 데이터프레임의 각 캔들(row)을 순회하며 시뮬레이션
    for i, candle in df.iterrows():
        # 포지션이 있는 경우, 청산 조건 확인
        if position:
            # 스탑로스 확인
            if candle['low'] <= position['stop_loss_price']:
                exit_price = position['stop_loss_price']
                position['exit_price'] = exit_price
                position['pnl'] = (exit_price - position['entry_price']) / position['entry_price']
                position['exit_time'] = candle['date']
                trades.append(position)
                position = None
                continue

            # 익절 확인
            if candle['high'] >= position['profit_target_price']:
                exit_price = position['profit_target_price']
                position['exit_price'] = exit_price
                position['pnl'] = (exit_price - position['entry_price']) / position['entry_price']
                position['exit_time'] = candle['date']
                trades.append(position)
                position = None
                continue
            
            # 보유 기간 만료 확인
            if (i - position['entry_index']) >= hold_period_candles:
                exit_price = candle['close']
                position['exit_price'] = exit_price
                position['pnl'] = (exit_price - position['entry_price']) / position['entry_price']
                position['exit_time'] = candle['date']
                trades.append(position)
                position = None
                continue

        # 포지션이 없는 경우, 진입 조건 확인
        if not position:
            entry_target = candle['entry_target']
            # entry_target이 유효하고, 현재 캔들의 고가가 목표가를 돌파했는지 확인
            if pd.notna(entry_target) and candle['high'] > entry_target:
                entry_price = entry_target
                position = {
                    'entry_price': entry_price,
                    'entry_time': candle['date'],
                    'entry_index': i,
                    'stop_loss_price': entry_price * (1 - stop_loss_pct),
                    'profit_target_price': entry_price * (1 + profit_target_pct),
                }

    # 최종 결과 집계
    if not trades:
        return 0, 0 # (승, 패)

    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len(trades) - wins
    return wins, losses


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
            # 날짜 형식에 맞게 인덱스 설정
            df = pd.read_sql(query, conn, params=(ticker,))
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
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
        start_index = df.index.get_loc(idx)
        for j in range(start_index + 1, len(df)):
            low_j = df.iloc[j]["low"]
            high_j = df.iloc[j]["high"]

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
        print("@@@ STRATEGY BACKTEST START @@@")
        print("%s" % (strategy_name))

        params_win_loses = {}
        params_expected_returns = {}
        # 모든 buy-sell 파라미터 조합에 대해 루프
        for buy_params in buy_params_list:
            for sell_params in sell_params_list:
                total_wins = 0
                total_losses = 0

                for ticker in tickers:
                    if strategy_name == 'larry_williams_vb':
                        # 데이터 로드 및 전략 적용 (entry_target 계산)
                        df = load_ohlcv(db_path, table_name, ticker)
                        df = apply_strategy(df, strategy_name, buy_params)

                        # 상세 백테스팅 실행
                        wins, losses = run_larry_williams_backtest(df.reset_index(), buy_params, sell_params)
                        total_wins += wins
                        total_losses += losses
                    else:
                        df = apply_buy_signal_strategy(
                            db_path=db_path,
                            table_name=table_name,
                            ticker=ticker,
                            strategy_name=strategy_name,
                            params=buy_params
                        )
                        df = apply_sell_signal_strategy(df, sell_params)
                        n_win = (df['result'] == 1).sum()
                        n_lose = (df['result'] == -1).sum()
                        total_wins += n_win
                        total_losses += n_lose

                # 결과 집계
                str_params = str(buy_params) + '-' + str(sell_params)
                win_ratio = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0

                expected_return = 0
                if strategy_name == 'larry_williams_vb':
                    # 기대수익률 계산
                    profit_target = sell_params['profit_target_pct']
                    stop_loss = sell_params['stop_loss_pct']
                    risk_reward_ratio = profit_target / stop_loss if stop_loss > 0 else 0
                    expected_return = (win_ratio * risk_reward_ratio) - (1 - win_ratio)
                    print(f"{str_params} - Wins: {total_wins}, Losses: {total_losses}, Win Ratio: {win_ratio:.2f}, RRR: {risk_reward_ratio:.2f}, Expected Return: {expected_return:.2f}")
                else:
                    low_limit_ratio = sell_params['low_limit_ratio']
                    high_limit_ratio = sell_params['high_limit_ratio']
                    # rrr calculation might result in division by zero
                    denominator = 1.0 - low_limit_ratio
                    rrr = (high_limit_ratio - 1.0) / denominator if denominator != 0 else 0
                    expected_return = (win_ratio * rrr) - (1 - win_ratio)
                    print(f"{str_params}  -  win count : {total_wins}, loss count : {total_losses}, win_ratio : {win_ratio:.2f}, risk-reward-ratio : {rrr:.1f}")

                params_win_loses[str_params] = (total_wins, total_losses)
                params_expected_returns[str_params] = expected_return
        
        # ------------------ 최종 결과 출력 (공통) ------------------
        if not params_expected_returns:
            print("No backtest results to display.")
            continue

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
