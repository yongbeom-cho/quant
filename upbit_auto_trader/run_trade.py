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
import numpy as np
from decimal import Decimal, getcontext, ROUND_DOWN

# Import strategy modules from upbit_auto_trader directory
from strategy.xgb_strategy import apply_strategy_xgb
from expanding_cache import precompute_all_expanding_values, load_expanding_values, check_expanding_cache_exists

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

load_dotenv()

class StrategyManager:
    """각 전략(day, minute60, minute240)을 관리하는 클래스"""
    
    def __init__(self, strategy_name, model_name, model_path, upper, lower, close_lower_sell, max_tx_cnt, 
                 commision_fee, upbit, config, root_dir=None):
        self.strategy_name = strategy_name
        self.model_name = model_name
        self.model_path = model_path
        self.uppers = [upper]  # 리스트로 변환
        self.lower = lower
        self.close_lower_sell = close_lower_sell
        self.max_tx_cnt = max_tx_cnt
        self.commision_fee = commision_fee
        self.upbit = upbit
        self.config = config
        self.root_dir = root_dir
        
        # coin_infos = { "KRW-BTC" : [ {"buy_price": 0.0, "balance": 0.0, "buy_date": "", "sell_uppers": [], "sell_fractions": []} ]}
        self.coin_infos = {}
        self.cash = 0.0  # 이 전략에 할당된 현금
        self.win_cnt = 0
        self.lose_cnt = 0
        
        # interval 설정 (strategy_name에서 추출: 'day-pb_du' -> 'day')
        interval_part = strategy_name.split('-')[0]
        if interval_part == 'day':
            self.interval = 'day'
        elif interval_part == 'minute60':
            self.interval = 'minute60'
        elif interval_part == 'minute240':
            self.interval = 'minute240'
        else:
            raise ValueError(f"Unknown interval in strategy_name: {strategy_name}")
        
        # expanding cache 경로 설정
        if root_dir:
            self.expanding_cache_path = os.path.join(root_dir, 'upbit_auto_trader', 'expanding_cache.db')
        else:
            self.expanding_cache_path = None
    
    def get_ohlcv_df(self, krw_ticker, count=200):
        """OHLCV 데이터 가져오기"""
        time.sleep(self.config['query_time'])
        df = pyupbit.get_ohlcv(krw_ticker, interval=self.interval, count=count)
        retry = self.config['retry']
        while (df is None) and (retry > 0):
            try_cnt = (self.config['retry'] - retry)
            time.sleep(1 * try_cnt * try_cnt + 1 + self.config['query_time'])
            df = pyupbit.get_ohlcv(krw_ticker, interval=self.interval, count=count)
            if df is not None:
                df = df.drop_duplicates(keep='first')
                df = df.sort_index()
                df = df.drop('value', axis=1)
                if len(df) < count - 1:
                    df = None
            retry -= 1
        
        if df is None:
            return None
        
        df['date'] = df.index
        df['date'] = df['date'].apply(lambda d: d.strftime('%Y%m%d%H%M'))
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
        return df
    
    def apply_buy_signal_xgb(self, krw_ticker):
        """XGB 모델을 사용하여 매수 시그널 생성"""
        df = self.get_ohlcv_df(krw_ticker)
        if df is None or len(df) < 50:
            return None
        
        # expanding cache 로드
        expanding_cache = None
        if self.expanding_cache_path:
            try:
                expanding_cache = load_expanding_values(self.expanding_cache_path, self.interval, krw_ticker)
            except Exception as e:
                print(f"[WARNING] Expanding cache 로드 실패 ({krw_ticker}): {e}")
        
        df = apply_strategy_xgb(df, self.model_name, self.model_path, expanding_cache)
        return df
    
    def is_over_spread_limit(self, krw_ticker):
        """스프레드 제한 확인"""
        time.sleep(self.config['query_time'])
        orderbook = pyupbit.get_orderbook(krw_ticker)
        bid_price = orderbook['orderbook_units'][0]['bid_price']
        ask_price = orderbook['orderbook_units'][0]['ask_price']
        spread = (ask_price - bid_price) / bid_price * 100
        return spread > self.config['spread_limit']
    
    def buy_market_order(self, krw_ticker, buy_money):
        """시장가 매수 주문"""
        retry = self.config['retry']
        is_over_spread_limit = self.is_over_spread_limit(krw_ticker)
        while is_over_spread_limit and retry > 0:
            time.sleep(1)
            is_over_spread_limit = self.is_over_spread_limit(krw_ticker)
            retry -= 1
        
        if is_over_spread_limit:
            return 0, 0
        
        # 매수 전 정보 확인
        balance_infos_before = self.upbit.get_balances()
        cash_before = 0.0
        coin_balance_before = 0.0
        for balance_info in balance_infos_before:
            if 'KRW' == balance_info['currency']:
                cash_before = float(Decimal(str(balance_info['balance'])))
            elif 'KRW-' + balance_info['currency'] == krw_ticker:
                coin_balance_before = float(Decimal(str(balance_info['balance'])))
        
        result = self.upbit.buy_market_order(krw_ticker, buy_money)
        time.sleep(self.config['order_time'])
        
        # 매수 후 정보 확인
        balance_infos_after = self.upbit.get_balances()
        cash_after = 0.0
        coin_balance_after = 0.0
        for balance_info in balance_infos_after:
            if 'KRW' == balance_info['currency']:
                cash_after = float(Decimal(str(balance_info['balance'])))
            elif 'KRW-' + balance_info['currency'] == krw_ticker:
                coin_balance_after = float(Decimal(str(balance_info['balance'])))
        
        # 이번에 산 balance 계산
        bought_balance = coin_balance_after - coin_balance_before
        if bought_balance <= 0:
            return 0, 0
        
        # 이 전략에서 사용된 현금 계산
        used_cash = cash_before - cash_after
        if used_cash <= 0:
            return 0, 0
        
        # 이번에 산 가격 계산 (사용한 현금 / 산 balance)
        buy_price = used_cash / bought_balance
        self.cash -= used_cash
        
        return buy_price, bought_balance
    
    def sell_market_order(self, krw_ticker, balance):
        """시장가 매도 주문"""
        balance_infos = self.upbit.get_balances()
        for balance_info in balance_infos:
            if 'KRW-' + balance_info['currency'] == krw_ticker:
                now_balance = Decimal(str(balance_info['balance']))
                if Decimal(str(balance)) > now_balance:
                    balance = now_balance
        
        # 매도 전 현금 확인
        balance_infos_before = self.upbit.get_balances()
        cash_before = 0.0
        for balance_info in balance_infos_before:
            if 'KRW' == balance_info['currency']:
                cash_before = float(Decimal(str(balance_info['balance'])))
        
        result = self.upbit.sell_market_order(krw_ticker, balance)
        time.sleep(self.config['order_time'])
        
        # 매도 후 현금 확인
        balance_infos_after = self.upbit.get_balances()
        cash_after = 0.0
        remain_balance = 0.0
        for balance_info in balance_infos_after:
            if 'KRW-' + balance_info['currency'] == krw_ticker:
                remain_balance = float(Decimal(str(balance_info['balance'])))
            elif 'KRW' == balance_info['currency']:
                cash_after = float(Decimal(str(balance_info['balance'])))
        
        # 이 전략에 추가된 현금 계산 (전체 현금 증가분을 이 전략에 반영)
        added_cash = cash_after - cash_before
        self.cash += added_cash
        
        return self.cash, remain_balance
    
    def get_current_price(self, coin):
        """현재 가격 가져오기"""
        price = 0
        retry = self.config['retry']
        while (price == 0) and (retry > 0):
            try:
                price = pyupbit.get_current_price(coin)
            except:
                retry -= 1
                time.sleep(self.config['order_time'])
        return price
    
    def get_total_asset(self, code_prices):
        """총 자산 계산"""
        coin_asset = 0.0
        for coin, info_list in self.coin_infos.items():
            for info in info_list:
                coin_asset += info['balance'] * code_prices.get(coin, 0)
        return self.cash + coin_asset
    
    def buy(self, coin, ohlcv, code_prices):
        """매수 로직 (07_strategy_timeseries_backtest.py의 buy 메서드와 동일)"""
        if ohlcv is None:
            print(f"[{self.strategy_name}] ✗ {coin} 매수 실패: ohlcv 데이터가 None입니다")
            return False
        elif not ohlcv.get('signal', False):
            print(f"[{self.strategy_name}] ✗ {coin} 매수 실패: 매수 시그널이 False입니다 (signal={ohlcv.get('signal', None)})")
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
            print(f"[{self.strategy_name}] ✗ {coin} 매수 실패: 최대 매수 횟수에 도달했습니다 (max_fracs={max_fracs}, total_fracs={total_fracs})")
            return False
        
        alloc_cash = self.cash / ((max_fracs - total_fracs) / len(self.uppers))
        alloc_cash = min(alloc_cash, alloc_asset)
        
        if alloc_cash <= 0:
            print(f"[{self.strategy_name}] ✗ {coin} 매수 실패: 할당된 현금이 0 이하입니다 (alloc_cash={alloc_cash:.2f}, cash={self.cash:.2f}, total_asset={total_asset:.2f})")
            return False
        
        # 실제 주문 실행 (buy_market_order 내부에서 self.cash 업데이트됨)
        buy_price, bought_balance = self.buy_market_order(coin, alloc_cash)
        if buy_price == 0 or bought_balance == 0:
            print(f"[{self.strategy_name}] ✗ {coin} 매수 실패: 매수 주문 실패 (buy_price={buy_price}, bought_balance={bought_balance:.8f}, alloc_cash={alloc_cash:.2f})")
            return False
        
        new_coin_info = {
            "buy_price": buy_price,
            "balance": bought_balance,
            "buy_date": today,
            "sell_uppers": self.uppers.copy(),
            "sell_fractions": [1/(len(self.uppers)-i) for i in range(len(self.uppers))]
        }
        if coin in self.coin_infos:
            self.coin_infos[coin].append(new_coin_info)
        else:
            self.coin_infos[coin] = [new_coin_info]
        print(f"[{self.strategy_name}] ###BUY {coin} at {buy_price:.2f}, balance: {bought_balance:.8f}")
        
        return True
    
    def sell(self, coin, ohlcv, is_last_date=False):
        """매도 로직 (07_strategy_timeseries_backtest.py의 sell 메서드와 동일)"""
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
            buy_price = coin_info['buy_price']
            low_limit_price = buy_price * self.lower
            
            # low 가격으로 손절 (close_lower_sell이 False일 때)
            if not self.close_lower_sell and (ohlcv['low'] < low_limit_price):
                lose_cnt += len(coin_info['sell_uppers'])
                balance = coin_info['balance']
                # 실제 매도 주문 실행 (sell_market_order 내부에서 self.cash 업데이트됨)
                updated_cash, remain_balance = self.sell_market_order(coin, balance)
                del_idxes.append(ci_idx)
                print(f"[{self.strategy_name}] ###Sell Lower {coin}, win: {win_cnt}, lose: {lose_cnt}")
                continue
            
            # upper 가격으로 분할 매도
            if len(coin_info['sell_uppers']) > 0:
                upper = coin_info['sell_uppers'][0]
                while ohlcv['high'] > upper * buy_price:
                    win_cnt += 1
                    balance = coin_info['balance']
                    sell_fraction = coin_info['sell_fractions'][0]
                    sell_shares = sell_fraction * balance
                    # 실제 매도 주문 실행 (sell_market_order 내부에서 self.cash 업데이트됨)
                    updated_cash, remain_balance = self.sell_market_order(coin, sell_shares)
                    
                    if len(coin_info['sell_uppers']) == 1:
                        del_idxes.append(ci_idx)
                        print(f"[{self.strategy_name}] ###SELL ALL {coin}")
                        break
                    else:
                        coin_info['balance'] -= sell_shares
                        coin_info['sell_uppers'] = coin_info['sell_uppers'][1:]
                        coin_info['sell_fractions'] = coin_info['sell_fractions'][1:]
                        self.coin_infos[coin][ci_idx] = coin_info
                        if len(coin_info['sell_uppers']) > 0:
                            upper = coin_info['sell_uppers'][0]
                        print(f"[{self.strategy_name}] ###Sell Fraction {coin}")
            
            if ci_idx in del_idxes:
                continue
            
            # close 가격으로 손절 (close_lower_sell이 True일 때) 또는 마지막 날
            if ((self.close_lower_sell and (ohlcv['close'] < low_limit_price)) or is_last_date):
                lose_cnt += len(coin_info['sell_uppers'])
                balance = coin_info['balance']
                # 실제 매도 주문 실행 (sell_market_order 내부에서 self.cash 업데이트됨)
                updated_cash, remain_balance = self.sell_market_order(coin, balance)
                del_idxes.append(ci_idx)
                if is_last_date:
                    print(f"[{self.strategy_name}] ###Sell At Lastday {coin}, win: {win_cnt}, lose: {lose_cnt}")
                else:
                    print(f"[{self.strategy_name}] ###Sell Lower {coin}, win: {win_cnt}, lose: {lose_cnt}")
                continue
        
        self.win_cnt += win_cnt
        self.lose_cnt += lose_cnt
        sell_ratio = float(win_cnt + lose_cnt) / (len(self.uppers) * len(coin_info_list)) if len(coin_info_list) > 0 else 0.0
        del_idxes = set(del_idxes)
        coin_info_list = [v for i, v in enumerate(coin_info_list) if i not in del_idxes]
        if len(coin_info_list) == 0:
            del self.coin_infos[coin]
        else:
            self.coin_infos[coin] = coin_info_list
        
        return sell_ratio
    
    def check_realtime_sell(self, coin):
        """실시간 가격 확인하여 매도 조건 체크"""
        if coin not in self.coin_infos:
            return
        
        try:
            # 실시간 현재 가격 가져오기
            current_price = self.get_current_price(coin)
            if current_price == 0:
                return
            
            coin_info_list = self.coin_infos[coin]
            del_idxes = []
            
            for ci_idx in range(len(coin_info_list)):
                coin_info = coin_info_list[ci_idx]
                buy_price = coin_info['buy_price']
                upper_price = buy_price * self.uppers[0]  # 첫 번째 upper 사용
                low_limit_price = buy_price * self.lower
                
                # 1. 모든 전략: close가 upper 이상이면 매도
                if current_price >= upper_price:
                    balance = coin_info['balance']
                    if len(coin_info['sell_uppers']) > 0:
                        # 분할 매도
                        sell_fraction = coin_info['sell_fractions'][0] if len(coin_info['sell_fractions']) > 0 else 1.0
                        sell_shares = sell_fraction * balance
                        updated_cash, remain_balance = self.sell_market_order(coin, sell_shares)
                        
                        if len(coin_info['sell_uppers']) == 1:
                            del_idxes.append(ci_idx)
                            print(f"[{self.strategy_name}] ###SELL ALL by Upper (realtime) {coin} at {current_price:.2f}")
                        else:
                            coin_info['balance'] -= sell_shares
                            coin_info['sell_uppers'] = coin_info['sell_uppers'][1:]
                            coin_info['sell_fractions'] = coin_info['sell_fractions'][1:]
                            self.coin_infos[coin][ci_idx] = coin_info
                            print(f"[{self.strategy_name}] ###Sell Fraction by Upper (realtime) {coin} at {current_price:.2f}")
                    continue
                
                # 2. minute60 전략만: close_lower_sell=False이므로 실시간으로 low < lower면 매도
                if not self.close_lower_sell:
                    # 현재 가격이 lower보다 낮으면 매도
                    if current_price < low_limit_price:
                        balance = coin_info['balance']
                        updated_cash, remain_balance = self.sell_market_order(coin, balance)
                        del_idxes.append(ci_idx)
                        print(f"[{self.strategy_name}] ###Sell Lower (realtime) {coin} at {current_price:.2f}, low_limit: {low_limit_price:.2f}")
                        continue
            
            # 삭제된 항목 제거
            if len(del_idxes) > 0:
                del_idxes = set(del_idxes)
                coin_info_list = [v for i, v in enumerate(coin_info_list) if i not in del_idxes]
                if len(coin_info_list) == 0:
                    del self.coin_infos[coin]
                else:
                    self.coin_infos[coin] = coin_info_list
                    
        except Exception as e:
            print(f"[ERROR] 실시간 매도 체크 실패 ({coin}): {e}")
            traceback.print_exc()
    
    def to_string(self):
        """상태 문자열 반환"""
        str_coin_infos = json.dumps(self.coin_infos, indent=2)
        return f"[{self.strategy_name}] cash: {self.cash:.2f}, win: {self.win_cnt}, lose: {self.lose_cnt}\n{str_coin_infos}"


class MultiStrategyTrader:
    """여러 전략을 동시에 실행하는 메인 클래스"""
    
    def __init__(self, config_path, root_dir):
        self.root_dir = root_dir
        self.load_config(config_path)
        
        load_dotenv()
        access = os.getenv('access')
        secret = os.getenv('secret')
        self.upbit = pyupbit.Upbit(access=access, secret=secret)
        
        # 전략 설정을 config에서 읽어서 동적으로 생성
        model_dir = os.path.join(root_dir, 'upbit_auto_trader', 'xgb_pb_du_model')
        self.strategy_managers = {}
        
        if 'strategies' not in self.config:
            raise ValueError("config.json에 'strategies' 섹션이 없습니다.")
        
        for strategy_key, strategy_config in self.config['strategies'].items():
            model_name = strategy_config['model_name']
            strategy_name = strategy_config['strategy_name']
            upper = strategy_config['upper']
            lower = strategy_config['lower']
            close_lower_sell = strategy_config['close_lower_sell']
            max_tx_cnt = strategy_config['max_tx_cnt']
            
            self.strategy_managers[strategy_key] = StrategyManager(
                strategy_name=strategy_name,
                model_name=model_name,
                model_path=os.path.join(model_dir, model_name),
                upper=upper,
                lower=lower,
                close_lower_sell=close_lower_sell,
                max_tx_cnt=max_tx_cnt,
                commision_fee=self.config['commision_fee'],
                upbit=self.upbit,
                config=self.config,
                root_dir=root_dir
            )
            print(f"[INFO] Strategy '{strategy_key}' initialized: model={model_name}, upper={upper}, lower={lower}, close_lower_sell={close_lower_sell}, max_tx_cnt={max_tx_cnt}")
        
        self.set_not_warning_krw_tickers()
        
        # 저장된 coin_infos 로드
        self.load_all_coin_infos()
        
        # 로드된 coin_infos 상태를 저장 (변경 감지용)
        self.saved_coin_infos_state = self._get_coin_infos_state()
        
        self.initialize_cash()
    
    def load_config(self, cfg_path):
        """설정 파일 로드"""
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def set_not_warning_krw_tickers(self):
        """경고 없는 KRW 티커 목록 설정"""
        self.krw_tickers = set()
        krw_ticker_infos = pyupbit.get_tickers('KRW', True)
        for krw_ticker_info in krw_ticker_infos:
            krw_ticker = krw_ticker_info['market']
            if not krw_ticker_info['market_event']['warning']:
                self.krw_tickers.add(krw_ticker)
        
        # 거래량 기준으로 정렬
        tickers_tx_money = Counter()
        for krw_ticker in self.krw_tickers:
            try:
                df = pyupbit.get_ohlcv(krw_ticker, interval='day', count=1)
                if df is not None and len(df) > 0:
                    volume_money = df.iloc[-1]['close'] * df.iloc[-1]['volume']
                    tickers_tx_money[krw_ticker] = volume_money
                else:
                    tickers_tx_money[krw_ticker] = 0
            except:
                tickers_tx_money[krw_ticker] = 0
        
        self.krw_tickers = [ticker for ticker, count in tickers_tx_money.most_common()]
    
    def initialize_cash(self):
        """각 전략에 현금 할당 (코인 가치를 고려하여 균등 분배)"""
        total_cash = self.upbit.get_balance('KRW')
        if total_cash is None:
            total_cash = 0.0
        
        num_strategies = len(self.strategy_managers)
        if num_strategies == 0:
            print(f"Total cash: {total_cash:.2f}, Number of strategies: 0")
            return
        
        # 각 전략의 코인 가치 계산 (buy_price * balance)
        strategy_coin_values = {}
        total_coin_value = 0.0
        
        for strategy_key, strategy_manager in self.strategy_managers.items():
            coin_value = 0.0
            for coin, coin_info_list in strategy_manager.coin_infos.items():
                for coin_info in coin_info_list:
                    coin_value += coin_info['buy_price'] * coin_info['balance']
            strategy_coin_values[strategy_key] = coin_value
            total_coin_value += coin_value
        
        # 전체 자산 = 전체 현금 + 모든 코인 가치
        total_asset = total_cash + total_coin_value
        
        # 각 전략에 할당할 자산 (전체 자산을 전략 수로 나눔)
        asset_per_strategy = total_asset / num_strategies
        
        # 각 전략에 할당할 현금 = 할당 자산 - 해당 전략의 코인 가치
        strategy_cash_allocations = {}
        for strategy_key, strategy_manager in self.strategy_managers.items():
            coin_value = strategy_coin_values[strategy_key]
            allocated_cash = asset_per_strategy - coin_value
            # 현금은 0 이상이어야 함
            allocated_cash = max(0.0, allocated_cash)
            strategy_cash_allocations[strategy_key] = allocated_cash
        
        # 정수로 변환 (내림 처리)
        strategy_cash_int = {}
        total_allocated_int = 0
        for strategy_key, cash in strategy_cash_allocations.items():
            cash_int = int(cash)
            strategy_cash_int[strategy_key] = cash_int
            total_allocated_int += cash_int
        
        # 남은 현금을 전략들에 1원씩 분배
        remaining_cash = int(total_cash) - total_allocated_int
        if remaining_cash > 0:
            strategy_keys = list(strategy_cash_int.keys())
            for i in range(remaining_cash):
                strategy_cash_int[strategy_keys[i % len(strategy_keys)]] += 1
        
        # 각 전략에 현금 할당
        for strategy_key, strategy_manager in self.strategy_managers.items():
            strategy_manager.cash = float(strategy_cash_int[strategy_key])
        
        # 로그 출력
        print(f"Total cash: {total_cash:.2f}, Total coin value: {total_coin_value:.2f}, Total asset: {total_asset:.2f}")
        print(f"Number of strategies: {num_strategies}, Asset per strategy: {asset_per_strategy:.2f}")
        for strategy_key, strategy_manager in self.strategy_managers.items():
            coin_value = strategy_coin_values[strategy_key]
            allocated_cash = strategy_manager.cash
            print(f"  [{strategy_key}] coin_value: {coin_value:.2f}, allocated_cash: {allocated_cash:.2f}, total: {coin_value + allocated_cash:.2f}")
    
    def renew_cash(self):
        """현금 재할당 (코인 가치를 고려하여 균등 분배)"""
        total_cash = self.upbit.get_balance('KRW')
        if total_cash is None:
            total_cash = 0.0
        
        time.sleep(self.config['query_time'])
        
        num_strategies = len(self.strategy_managers)
        if num_strategies == 0:
            return
        
        # 각 전략의 코인 가치 계산 (buy_price * balance)
        strategy_coin_values = {}
        total_coin_value = 0.0
        
        for strategy_key, strategy_manager in self.strategy_managers.items():
            coin_value = 0.0
            for coin, coin_info_list in strategy_manager.coin_infos.items():
                for coin_info in coin_info_list:
                    coin_value += coin_info['buy_price'] * coin_info['balance']
            strategy_coin_values[strategy_key] = coin_value
            total_coin_value += coin_value
        
        # 전체 자산 = 전체 현금 + 모든 코인 가치
        total_asset = total_cash + total_coin_value
        
        # 각 전략에 할당할 자산 (전체 자산을 전략 수로 나눔)
        asset_per_strategy = total_asset / num_strategies
        
        # 각 전략에 할당할 현금 = 할당 자산 - 해당 전략의 코인 가치
        strategy_cash_allocations = {}
        for strategy_key, strategy_manager in self.strategy_managers.items():
            coin_value = strategy_coin_values[strategy_key]
            allocated_cash = asset_per_strategy - coin_value
            # 현금은 0 이상이어야 함
            allocated_cash = max(0.0, allocated_cash)
            strategy_cash_allocations[strategy_key] = allocated_cash
        
        # 정수로 변환 (내림 처리)
        strategy_cash_int = {}
        total_allocated_int = 0
        for strategy_key, cash in strategy_cash_allocations.items():
            cash_int = int(cash)
            strategy_cash_int[strategy_key] = cash_int
            total_allocated_int += cash_int
        
        # 남은 현금을 전략들에 1원씩 분배
        remaining_cash = int(total_cash) - total_allocated_int
        if remaining_cash > 0:
            strategy_keys = list(strategy_cash_int.keys())
            for i in range(remaining_cash):
                strategy_cash_int[strategy_keys[i % len(strategy_keys)]] += 1
        
        # 각 전략에 현금 할당
        for strategy_key, strategy_manager in self.strategy_managers.items():
            strategy_manager.cash = float(strategy_cash_int[strategy_key])
    
    def process_strategy(self, strategy_name):
        """각 전략 처리 (매수/매도 로직 실행)"""
        strategy_manager = self.strategy_managers[strategy_name]
        now = datetime.now()
        print(f"\n{'='*80}")
        print(f"[{strategy_name}] 전략 처리 시작 - {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # 매도 처리
        code_prices = {}
        for coin in list(strategy_manager.coin_infos.keys()):
            try:
                df = strategy_manager.get_ohlcv_df(coin, count=2)
                if df is not None and len(df) > 0:
                    ohlcv = df.iloc[-1]
                    code_prices[coin] = ohlcv['close']
                    strategy_manager.sell(coin, ohlcv, is_last_date=False)
            except:
                traceback.print_exc()
        
        # 매수 처리
        buy_cand_code_tx_amounts = Counter()
        buy_cand_code_ohlcvs = {}
        buy_cand_code_dfs = {}
        
        print(f"\n[{strategy_name}] 매수 후보 탐색 중...")
        for krw_ticker in self.krw_tickers:
            if krw_ticker in self.config['black_coins']:
                continue
            
            try:
                df = strategy_manager.apply_buy_signal_xgb(krw_ticker)
                if df is not None and len(df) > 0:
                    ohlcv = df.iloc[-1]
                    code_prices[krw_ticker] = ohlcv['close']
                    
                    if ohlcv.get('signal', False):
                        buy_cand_code_tx_amounts[krw_ticker] = ohlcv['volume'] * ohlcv['close']
                        buy_cand_code_ohlcvs[krw_ticker] = ohlcv
                        buy_cand_code_dfs[krw_ticker] = df
                        tx_amount = ohlcv['volume'] * ohlcv['close']

                        # df 상태 로그 (현재 기준 3개 row, 모든 컬럼)
                        print(f"\n[{strategy_name}] {krw_ticker} - DF 상태 (최근 3개):")
                        df_display = df.tail(3).copy()
                        print(df_display.to_string(index=False))
                        
                        print(f"[{strategy_name}] {krw_ticker} - 매수 시그널 발생! (거래대금: {tx_amount:.2f} KRW, 거래량: {ohlcv['volume']:.2f}, 가격: {ohlcv['close']:.2f})")
                    else:
                        print(f"[{strategy_name}] {krw_ticker} - 매수 시그널 없음")
            except Exception as e:
                print(f"[{strategy_name}] {krw_ticker} - 오류 발생: {e}")
                traceback.print_exc()
        
        # 거래대금 순으로 매수
        print(f"\n[{strategy_name}] 거래대금 순 매수 시도 시작...")
        print(f"[{strategy_name}] 매수 후보 개수: {len(buy_cand_code_tx_amounts)}")
        
        bought_tickers = []
        failed_tickers = []
        
        for buy_cand_code, tx_amount in buy_cand_code_tx_amounts.most_common():
            ohlcv = buy_cand_code_ohlcvs[buy_cand_code]
            buy_price_before = ohlcv['close']
            
            print(f"\n[{strategy_name}] {buy_cand_code} 매수 시도 중... (거래대금: {tx_amount:.2f} KRW, 현재가: {buy_price_before:.2f})")
            
            # 매수 전 현금 상태
            cash_before = strategy_manager.cash
            print(f"[{strategy_name}] {buy_cand_code} - 매수 전 현금: {cash_before:.2f} KRW")
            
            # 매수 시도
            success = strategy_manager.buy(coin=buy_cand_code, ohlcv=ohlcv, code_prices=code_prices)
            
            if success:
                # 매수 성공 시 상세 정보
                cash_after = strategy_manager.cash
                used_cash = cash_before - cash_after
                
                # coin_infos에서 최근 매수 정보 가져오기
                if buy_cand_code in strategy_manager.coin_infos:
                    latest_coin_info = strategy_manager.coin_infos[buy_cand_code][-1]
                    actual_buy_price = latest_coin_info['buy_price']
                    bought_balance = latest_coin_info['balance']
                    
                    bought_tickers.append({
                        'ticker': buy_cand_code,
                        'buy_price': actual_buy_price,
                        'balance': bought_balance,
                        'used_cash': used_cash,
                        'tx_amount': tx_amount
                    })
                    
                    print(f"[{strategy_name}] ✓ {buy_cand_code} 매수 성공!")
                    print(f"  - 매수가: {actual_buy_price:.2f} KRW")
                    print(f"  - 매수량: {bought_balance:.8f}")
                    print(f"  - 사용 현금: {used_cash:.2f} KRW")
                    print(f"  - 거래대금: {tx_amount:.2f} KRW")
                else:
                    print(f"[{strategy_name}] ⚠ {buy_cand_code} 매수 성공했으나 coin_infos에 정보 없음")
            else:
                failed_tickers.append({
                    'ticker': buy_cand_code,
                    'reason': 'buy() 메서드 실패',
                    'tx_amount': tx_amount,
                    'price': buy_price_before
                })
                print(f"[{strategy_name}] ✗ {buy_cand_code} 매수 실패")
        
        # 매수 결과 요약
        print(f"\n{'='*80}")
        print(f"[{strategy_name}] 매수 결과 요약")
        print(f"{'='*80}")
        print(f"성공한 매수: {len(bought_tickers)}개")
        for item in bought_tickers:
            print(f"  - {item['ticker']}: {item['balance']:.8f} @ {item['buy_price']:.2f} KRW (사용: {item['used_cash']:.2f} KRW)")
        
        print(f"\n실패한 매수: {len(failed_tickers)}개")
        for item in failed_tickers:
            print(f"  - {item['ticker']}: {item['reason']} (거래대금: {item['tx_amount']:.2f} KRW, 가격: {item['price']:.2f})")
        
        print(f"{'='*80}\n")
    
    def is_in_time_range(self, hour, minute_start, minute_end):
        """특정 시간 범위 내에 있는지 확인"""
        now = datetime.now()
        return now.hour == hour and now.minute in range(minute_start, minute_end + 1)
    
    def save_all_coin_infos(self):
        """모든 전략의 coin_infos를 하나의 JSON 파일로 저장"""
        save_dir = os.path.join(self.root_dir, 'upbit_auto_trader')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, 'coin_infos.json')
        
        try:
            all_coin_infos = {}
            for strategy_key, strategy_manager in self.strategy_managers.items():
                all_coin_infos[strategy_key] = strategy_manager.coin_infos
            
            data = {
                'coin_infos': all_coin_infos,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 저장 후 상태 업데이트
            self.saved_coin_infos_state = self._get_coin_infos_state()
            
            print(f"[INFO] All coin_infos saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] coin_infos 저장 실패: {e}")
            traceback.print_exc()
    
    def load_all_coin_infos(self):
        """저장된 모든 전략의 coin_infos를 JSON 파일에서 로드"""
        load_path = os.path.join(self.root_dir, 'upbit_auto_trader', 'coin_infos.json')
        
        if not os.path.exists(load_path):
            print(f"[INFO] 저장된 coin_infos 파일이 없습니다: {load_path}")
            return
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_coin_infos = data.get('coin_infos', {})
            saved_at = data.get('saved_at', 'Unknown')
            
            # 각 전략의 coin_infos 복원
            for strategy_key, coin_infos in all_coin_infos.items():
                if strategy_key in self.strategy_managers:
                    self.strategy_managers[strategy_key].coin_infos = coin_infos
                    print(f"[INFO] {strategy_key} coin_infos loaded: {len(coin_infos)} coins")
            
            print(f"[INFO] All coin_infos loaded from {load_path} (saved at: {saved_at})")
        except Exception as e:
            print(f"[ERROR] coin_infos 로드 실패: {e}")
            traceback.print_exc()
    
    def _get_coin_infos_state(self):
        """현재 메모리의 coin_infos 상태를 반환 (비교용)"""
        state = {}
        for strategy_key, strategy_manager in self.strategy_managers.items():
            state[strategy_key] = json.dumps(strategy_manager.coin_infos, sort_keys=True)
        return state
    
    def _has_coin_infos_changed(self):
        """coin_infos가 변경되었는지 확인"""
        current_state = self._get_coin_infos_state()
        return current_state != self.saved_coin_infos_state
    
    def should_run_minute240(self, now, last_minute240_check):
        """minute240 전략 실행 여부 확인"""
        # 8:50~8:59 사이에 한 번 실행
        if now.hour == 8 and now.minute in range(50, 60):
            if last_minute240_check is None or last_minute240_check.date() != now.date() or last_minute240_check.hour != 8:
                return True
        
        # 4시간마다 실행 (12:50~12:59, 16:50~16:59, 20:50~20:59, 0:50~0:59, 4:50~4:59)
        target_hours = [12, 16, 20, 0, 4]
        if now.hour in target_hours and now.minute in range(50, 60):
            if last_minute240_check is None:
                return True
            # 마지막 실행 시간이 4시간 이상 지났는지 확인
            time_diff = (now - last_minute240_check).total_seconds()
            if time_diff >= 14400:  # 4시간 = 14400초
                return True
        
        return False
    
    def run(self):
        """메인 실행 루프"""
        print(f"############################### START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 시작 시 expanding cache 확인 및 초기 계산
        if not check_expanding_cache_exists(self.root_dir):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Expanding cache가 없습니다. 초기 계산을 시작합니다...")
            try:
                precompute_all_expanding_values(self.root_dir)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Expanding values 초기 계산 완료")
            except Exception as e:
                print(f"[ERROR] Expanding 값 초기 계산 실패: {e}")
                traceback.print_exc()
                print("[WARNING] Expanding cache 없이 계속 진행합니다...")
        
        last_day_check = None
        last_minute240_check = None
        last_minute60_check = None
        last_expanding_precompute = None
        last_realtime_check = None
        
        while True:
            try:
                now = datetime.now()
                
                # 실시간 가격 모니터링 (모든 전략의 보유 코인 체크)
                # 매 초마다 체크하되, API 호출 제한을 고려하여 적절한 간격 유지
                if last_realtime_check is None or (now - last_realtime_check).total_seconds() >= 2:
                    for strategy_name, strategy_manager in self.strategy_managers.items():
                        for coin in list(strategy_manager.coin_infos.keys()):
                            strategy_manager.check_realtime_sell(coin)
                    last_realtime_check = now
                
                # 매일 06:15에 expanding 값 미리 계산
                if now.hour == 6 and now.minute == 15:
                    if last_expanding_precompute is None or last_expanding_precompute.date() != now.date():
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Precomputing expanding values...")
                        try:
                            precompute_all_expanding_values(self.root_dir)
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Expanding values precomputed successfully")
                        except Exception as e:
                            print(f"[ERROR] Expanding 값 계산 실패: {e}")
                            traceback.print_exc()
                        last_expanding_precompute = now
                
                # Day strategy: 매일 오전 8:50~8:59 사이에 한 번 실행
                if now.hour == 8 and now.minute in range(50, 60):
                    if last_day_check is None or last_day_check.date() != now.date() or last_day_check.hour != 8:
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Processing day strategy...")
                        self.renew_cash()
                        self.process_strategy('day-pb_du')
                        last_day_check = now
                        print(self.strategy_managers['day-pb_du'].to_string())
                
                # Minute240 strategy: 
                # - 매일 오전 8:50~8:59 사이에 한 번
                # - 그 다음 매 4시간마다 한 번씩 (12:50~12:59, 16:50~16:59, 20:50~20:59, 0:50~0:59, 4:50~4:59)
                if self.should_run_minute240(now, last_minute240_check):
                    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Processing minute240 strategy...")
                    self.renew_cash()
                    self.process_strategy('minute240-pb_du')
                    last_minute240_check = now
                    print(self.strategy_managers['minute240-pb_du'].to_string())
                
                # Minute60 strategy: 매시 x시:55~x시:59 사이에 한 번씩
                if now.minute in range(55, 60):
                    if last_minute60_check is None or last_minute60_check.hour != now.hour or last_minute60_check.date() != now.date():
                        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Processing minute60 strategy...")
                        self.renew_cash()
                        self.process_strategy('minute60-pb_du')
                        last_minute60_check = now
                        print(self.strategy_managers['minute60-pb_du'].to_string())
                
                time.sleep(self.config['order_time'])
                
            except:
                traceback.print_exc()
                sys.stdout.flush()
                time.sleep(30)
            finally:
                # 매수/매도 후 coin_infos 변경 확인 및 자동 저장
                if self._has_coin_infos_changed():
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] coin_infos 변경 감지 - 자동 저장 중...")
                    try:
                        self.save_all_coin_infos()
                    except Exception as e:
                        print(f"[ERROR] coin_infos 자동 저장 실패: {e}")
                        traceback.print_exc()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, 'upbit_auto_trader', 'config.json')
    
    trader = MultiStrategyTrader(config_path, root_dir)
    trader.run()

