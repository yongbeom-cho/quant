import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.xgb_strategy import apply_strategy_xgb

class CoinTxManager:
    def __init__(self, cash, max_tx_cnt, uppers, lower, commision_fee, slippage_fee):
        self.uppers = uppers
        self.lower = lower
        self.commision_fee = commision_fee
        self.slippage_fee = slippage_fee
        
        self.max_tx_cnt = max_tx_cnt
        
        # coin_info = { "KRW-BTC" : {"avg_buy_price": 0.0, "balance": 0.0, "buy_date": ""} }
        # coin_info = { "KRW-BTC" : [ {"buy_price": 0.0, "balance": 0.0, "buy_date": ""} ]}
        self.coin_infos = {}
        self.cash = cash
        self.win_cnt = 0
        self.lose_cnt = 0

    def buy(self, coin, ohlcv, code_prices):
        if ohlcv is None:
            return False
        elif not ohlcv['signal']:
            return False
        
        now_price = ohlcv['close']
        today = ohlcv['date']
        buy_price = now_price * 1
        
        total_asset = self.get_total_asset(code_prices)
        alloc_asset = total_asset / self.max_tx_cnt
        max_fracs = self.max_tx_cnt * len(self.uppers)
        total_fracs = sum(
            len(elem['sell_fractions'])
            for values in self.coin_infos.values()
            for elem in values
        )
        if max_fracs == total_fracs:
            return False

        alloc_cash = self.cash / ((max_fracs - total_fracs) / len(self.uppers))
        alloc_cash = min(alloc_cash, alloc_asset)
        
        balance = (alloc_cash * (1.00-self.commision_fee-self.slippage_fee)) / buy_price
        
        self.cash -= alloc_cash
        new_coin_info = {
                "avg_buy_price": buy_price, 
                "balance": balance, 
                "buy_date": today, 
                "sell_uppers": self.uppers.copy(),
                "sell_fractions": [ 1/(len(self.uppers)-i) for i in range(len(self.uppers)) ]
            }
        if coin in self.coin_infos:
            self.coin_infos[coin].append(new_coin_info)
        else:
            self.coin_infos[coin] = [new_coin_info]
        print("###BUY", coin)
        
        return True

    def sell(self, coin, ohlcv, is_last_date, close_lower_sell=False):
        # close lower version sell
        # realtime lower version sell possible

        if ohlcv is None:
            return 0.0
        elif coin not in self.coin_infos:
            return 0.0

        today = ohlcv['date']
        coin_info_list = self.coin_infos[coin]
        
        
        del_idxes = []
        win_cnt = 0
        lose_cnt = 0
        for ci_idx in range(len(coin_info_list)):
            coin_info = coin_info_list[ci_idx]
            buy_date = coin_info['buy_date']
            if today <= buy_date:
                continue
            avg_buy_price = coin_info['avg_buy_price']
            low_limit_price = avg_buy_price * self.lower



            if not close_lower_sell and (ohlcv['low'] < low_limit_price):
                lose_cnt += len(coin_info['sell_uppers'])
                balance = coin_info['balance']
                sell_price = min(ohlcv['open'], low_limit_price) * (1.00)
                self.cash += balance * sell_price * (1.00-self.commision_fee-self.slippage_fee)
                del_idxes.append(ci_idx)
                print("###Sell Lower", coin, win_cnt, lose_cnt, len(self.uppers))
                continue

            
            upper = coin_info['sell_uppers'][0]
            while ohlcv['high'] > upper * avg_buy_price:
                win_cnt += 1
                balance = coin_info['balance']
                sell_price = max(ohlcv['open'], upper * avg_buy_price) * (1)
                sell_fraction = coin_info['sell_fractions'][0]
                sell_shares = sell_fraction * balance
                self.cash += sell_shares * sell_price * (1.00 - self.commision_fee - self.slippage_fee)
                if 0 == len(coin_info['sell_uppers']) - 1:
                    # del self.coin_infos[coin]
                    del_idxes.append(ci_idx)
                    print("###SELL ALL", coin)
                    break
                else:    
                    coin_info['balance'] -= sell_shares
                    coin_info['sell_uppers'] = coin_info['sell_uppers'][1:]
                    coin_info['sell_fractions'] = coin_info['sell_fractions'][1:]
                    self.coin_infos[coin][ci_idx] = coin_info
                    upper = coin_info['sell_uppers'][0]
                    print("###Sell Fraction", coin)

            if ci_idx in del_idxes:
                continue

            if ((close_lower_sell) and (ohlcv['close'] < low_limit_price)) or (is_last_date):
                lose_cnt += len(coin_info['sell_uppers'])
                balance = coin_info['balance']
                sell_price = ohlcv['close'] * (1)
                self.cash += balance * sell_price * (1.00-self.commision_fee-self.slippage_fee)
                # del self.coin_infos[coin]
                del_idxes.append(ci_idx)
                if is_last_date:
                    print("###Sell At Lastday", coin, win_cnt, lose_cnt, len(self.uppers))
                else:
                    print("###Sell Lower", coin, win_cnt, lose_cnt, len(self.uppers))
                continue
        self.win_cnt += win_cnt
        self.lose_cnt += lose_cnt
        sell_ratio = float(win_cnt + lose_cnt) / (len(self.uppers) * len(coin_info_list))
        del_idxes = set(del_idxes)
        coin_info_list = [v for i, v in enumerate(coin_info_list) if i not in del_idxes]
        if len(coin_info_list) == 0:
            del self.coin_infos[coin]
        else:
            self.coin_infos[coin] = coin_info_list
        
        return sell_ratio

    def get_total_asset(self, code_prices):
        coin_asset = 0.0
        for coin, info_list in self.coin_infos.items():
            for info in info_list:
                coin_asset += info['balance'] * code_prices[coin]
        return self.cash + coin_asset
    
    def to_string(self):
        str_coin_infos = json.dumps(self.coin_infos)
        return "cash:%.2f" %(self.cash) + ", " + str_coin_infos    
        

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


def apply_buy_signal_xgb(
    db_path,
    table_name,
    ticker,
    model_name,
    model_input_path
):
    """
    db_path: DB path
    table_name: {market}_ohlcv_{interval}
    ticker: 종목 (코인명))
    model_input_path: model_input_path
    """
    df = load_ohlcv(db_path, table_name, ticker)
    df = apply_strategy_xgb(df, model_name, model_input_path)
    return df

def timeseries_backtest(args, ts_backtest_cfg):
    blacklist_tickers = ['KRW-USDT', 'KRW-USDC', 'KRW-USD1']
    start_date = ts_backtest_cfg['start_date']
    end_date = ts_backtest_cfg['end_date']

    dummy, market, interval, target_strategy, label_name, threshold, str_feat = args.model_name.split('-')

    table_name = f'{market}_ohlcv_{interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')

    tickers = get_tickers(db_path, table_name)
    
    
    code_dfs = {}
    date_list = set()
    xgb_model_dir = os.path.join(args.root_dir, 'var/xgb_model')
    for ticker in tickers:
        if ticker in blacklist_tickers:
            continue
        df = apply_buy_signal_xgb(
                db_path=db_path,
                table_name=table_name,
                ticker=ticker,
                model_name=args.model_name,
                model_input_path=os.path.join(xgb_model_dir, args.model_name)
            )
        df = df[
            (df['date'] >= start_date) &
            (df['date'] <= end_date)
        ].sort_values('date').reset_index(drop=True).set_index('date', drop=False)
        if len(df) <= 50:
            continue

        code_dfs[ticker] = df
        date_list |= set(df['date'])
    date_list = sorted(date_list)
    start_date = date_list[0]
    end_date = date_list[-1]

    rors = []
    for code, df in code_dfs.items():
        rors.append(df.iloc[-1]['close']/df.iloc[0]['open'])
    avg_buy_and_hold_ror = sum(rors)/len(rors)

    for max_tx_cnt in ts_backtest_cfg['max_tx_cnts']:
        for uppers in ts_backtest_cfg['uppers_list']:
            for lower in ts_backtest_cfg['lower_list']:
                for close_lower_sell in [True, False]:
                    str_uppers = '|'.join(['%.2f' %(upper) for upper in uppers])
                    str_lower = '%.3f' %(lower)
                    tx_manager = CoinTxManager(cash=1.0,
                                            max_tx_cnt=max_tx_cnt,
                                            uppers=uppers,
                                            lower=lower,
                                            commision_fee=args.commision_fee,
                                            slippage_fee=args.slippage_fee)
                    
                    buy_cnt = 0
                    sell_cnt = 0.0
                    max_asset = 1.0
                    mdd = 1.0
                    not_buy_coin_cnts = []
                    code_prices = {}
                    for cur_date in date_list:
                        buy_cand_code_volumes = Counter()
                        buy_cand_code_ohlcvs = {}

                        for code, df in code_dfs.items():
                            if cur_date in df.index:
                                # ohlcv = df.loc[cur_date].iloc[0]
                                ohlcv = df.loc[cur_date]
                                code_prices[code] = ohlcv['close']
                                if ohlcv['signal'] == True:
                                    buy_cand_code_volumes[code] = ohlcv['volume']
                                    buy_cand_code_ohlcvs[code] = ohlcv
                                is_last_date = False
                                if df.iloc[-1]['date'] == cur_date:
                                    is_last_date = True
                                sell_balance_ratio = tx_manager.sell(coin=code, ohlcv=ohlcv, is_last_date=is_last_date, close_lower_sell=close_lower_sell)
                                sell_cnt += sell_balance_ratio
                        
                        not_buy_coin_cnt = 0
                        for buy_cand_code, volume in buy_cand_code_volumes.most_common():
                            ohlcv = buy_cand_code_ohlcvs[buy_cand_code]
                            if tx_manager.buy(coin=buy_cand_code, ohlcv=ohlcv, code_prices=code_prices):
                                buy_cnt += 1
                            else:
                                not_buy_coin_cnt += 1
                        not_buy_coin_cnts.append(not_buy_coin_cnt)
                        cur_asset = tx_manager.get_total_asset(code_prices)
                        mdd = min(mdd, cur_asset/max_asset)
                        max_asset = max(max_asset, cur_asset)
                        print("start_date:%s\tend_date:%s\tdate:%s\tuppers:%s\tlower:%s\tclose_sell:%d\ttx_cnt:%d\tcur_asset:%.2f\tmdd:%.2f\tbah_ror:%.2f\tbuy_cnt:%d\tsell_cnt:%d\twin_cnt:%d\tlose_cnt:%d\tnbcc:%.2f" %(start_date, end_date, cur_date, str_uppers, str_lower, int(close_lower_sell), max_tx_cnt, cur_asset, mdd, avg_buy_and_hold_ror, buy_cnt, sell_cnt, tx_manager.win_cnt, tx_manager.lose_cnt, sum(not_buy_coin_cnts)/len(not_buy_coin_cnts)))
                    
                    print("############\tstart_date:%s\tend_date:%s\tdate:%s\tuppers:%s\tlower:%s\tclose_sell:%d\ttx_cnt:%d\tcur_asset:%.2f\tmdd:%.2f\tbah_ror:%.2f\tbuy_cnt:%d\tsell_cnt:%d\twin_cnt:%d\tlose_cnt:%d\tnbcc:%.2f\t############" %(start_date, end_date, cur_date, str_uppers, str_lower, int(close_lower_sell), max_tx_cnt, cur_asset, mdd, avg_buy_and_hold_ror, buy_cnt, sell_cnt, tx_manager.win_cnt, tx_manager.lose_cnt, sum(not_buy_coin_cnts)/len(not_buy_coin_cnts)))
            
            

parser = argparse.ArgumentParser(description='04_real_backtest')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--model_name', type=str, default="xgb-coin-day-low_bb_du-label0-0.56676245-f3f8f31f14f22f34f40f20f36f7f24f38f5f23")
# parser.add_argument('--market', type=str, default="coin")
# parser.add_argument('--interval', type=str, default="minute60")
# parser.add_argument('--target_strategy', type=str, default="low_bb_du_2")
parser.add_argument('--commision_fee', type=float, default=0.0005)
parser.add_argument('--slippage_fee', type=float, default=0.002)



args = parser.parse_args()

if __name__ == "__main__":
    dummy, market, interval, target_strategy, label_name, threshold, str_feat = args.model_name.split('-')
    
    ts_bt_cfg_path = os.path.join(args.root_dir, 'sbin/strategy_timeseries_backtest/config.json')
    with open(ts_bt_cfg_path, 'r', encoding='utf-8') as f:
        ts_cfg = json.load(f)

    
    ts_backtest_cfg = ts_cfg[interval+'-'+target_strategy+'-'+label_name]

    timeseries_backtest(args, ts_backtest_cfg)

        
