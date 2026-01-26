import pyupbit
import json
import os
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import sys
import traceback
from decimal import Decimal, getcontext, ROUND_DOWN

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
getcontext().prec = 20

# 주문은 초당 8회, 분당 200회 / 주문 외 요청은 초당 30회, 분당 900회 사용 가능
class CoinTxManager:
    
    def load_config(self):        
        load_dotenv()
        access = os.getenv('access')
        secret = os.getenv('secret')
        self.upbit = pyupbit.Upbit(access=access, secret=secret)
        self.set_not_warning_krw_tickers()
        self.query_time = 0.03334
        self.order_time = 0.125
        self.retry = 3
        self.coin_infos = {}
    
    def renew_coin_infos_by_upbit(self):
        time.sleep(self.query_time)    
        balance_infos = self.upbit.get_balances()
        tickers = set(['KRW-' + balance_info['currency'] for balance_info in balance_infos])
        for key in list(self.coin_infos.keys()):
            if key not in tickers:
                del self.coin_infos[key]

        for balance_info in balance_infos:
            ticker = balance_info['currency']
            balance = Decimal(balance_info['balance'])
                
            # balance = float(balance_info['balance'])
            avg_buy_price = float(balance_info['avg_buy_price'])
            avg_buy_price = Decimal(balance_info['avg_buy_price'])

            
            if ticker == 'KRW':
                self.cash = balance
                continue
            
            if avg_buy_price == 0:
                continue
            krw_ticker = 'KRW-' + ticker
            time.sleep(self.order_time)
            price = self.get_current_price(krw_ticker)
            price = Decimal(price)
            if price * balance < 5000:
                continue
            
            if krw_ticker not in self.coin_infos:
                self.coin_infos[krw_ticker] = {
                    'avg_buy_price': avg_buy_price,
                    'balance': balance
                    }
            if self.coin_infos[krw_ticker]['avg_buy_price'] != avg_buy_price:
                self.coin_infos[krw_ticker]['avg_buy_price'] = avg_buy_price
            if self.coin_infos[krw_ticker]['balance'] != balance:
                self.coin_infos[krw_ticker]['balance'] = balance

    def __init__(self):
        print("INIT START ", datetime.now().strftime('%Y%m%d %H%M%S'))
        self.load_config()
        self.renew_coin_infos_by_upbit()
        print("INIT END ", datetime.now().strftime('%Y%m%d %H%M%S'))
        sys.stdout.flush()
    
    
    def get_current_price(self, coin):
        price = 0
        retry = self.retry
        while (price == 0) and (retry > 0):
                try:
                    price = pyupbit.get_current_price(coin)
                except:
                    retry -= 1
                    time.sleep(self.order_time)
        return price
        
    
    def set_not_warning_krw_tickers(self):
        self.krw_tickers = set()
        krw_ticker_infos = pyupbit.get_tickers('KRW', True)
        for krw_ticker_info in krw_ticker_infos:
            krw_ticker = krw_ticker_info['market']
            if not krw_ticker_info['market_event']['warning']:
                self.krw_tickers.add(krw_ticker)
        
    

    def sell_market_order(self, krw_ticker, balance):
        result = self.upbit.sell_market_order(krw_ticker, balance)
        time.sleep(self.order_time)
        balance_infos = self.upbit.get_balances()
        remain_balance = balance
        cash = self.cash
        for balance_info in balance_infos:
            if 'KRW-' + balance_info['currency'] == krw_ticker:
                remain_balance = Decimal(balance_info['balance'])
            elif 'KRW' == balance_info['currency']:
                cash = Decimal(balance_info['balance'])
        return cash, remain_balance
    

        
    def sell_all(self):
        plus_cash = 0
        sell_cand_coins = self.coin_infos.copy()

        for krw_ticker, coin_info in sell_cand_coins.items():
            try:
                plus_cash += self.sell_coin(krw_ticker, coin_info)
            except:
                traceback.print_exc()
                sys.stdout.flush()

        return plus_cash
    
    
    def sell_coin(self, krw_ticker, coin_info):
        cash = self.cash
        balance = coin_info['balance']
        print(balance)
        self.cash, remain_balance = self.sell_market_order(krw_ticker, balance)
        print("SELL ALL BY UPPER %s" %(krw_ticker))
        del self.coin_infos[krw_ticker]
        return self.cash - cash
        


if __name__ == "__main__":
    coin_tx_manager = CoinTxManager()
    coin_tx_manager.sell_all()
    