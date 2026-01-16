import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import itertools
import inspect

# [유틸리티] 진행 상황을 보기 위한 tqdm 임포트
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 전략 모듈 경로 추가 (sbin 디렉토리에서 실행 시 상위 디렉토리 참조)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.vbt_strategy_012 import vbt_with_filters, get_vbt_indicators, get_vbt_strategy_params_list
from strategy.vbt_sell_strategy_013 import bailout_sell_strategy
from strategy.vbt_strategy_020_enhanced import vbt_with_filters_enhanced, get_vbt_indicators_enhanced
from strategy.vbt_sell_strategy_021_enhanced import bailout_sell_strategy_enhanced

# [병렬 처리 최적화] 자식 프로세스들이 공유할 전역 데이터
GLOBAL_OHLCV = {}
GLOBAL_CACHED_DATA = {}

def init_worker():
    """워커 프로세스 시작 시 초기화 (필요시 사용)"""
    pass

def load_one_ticker(args_tuple):
    """DB에서 특정 종목의 OHLCV 데이터를 읽어오는 함수"""
    db_path, table_name, ticker = args_tuple
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date ASC", conn, params=(ticker,))
        conn.close()
        return ticker, df
    except: return ticker, None

def calc_pre_data(args_tuple):
    """
    [지표 사전 계산] 
    백테스트 도중 반복 계산되는 지표(EMA, RSI, ATR 등)를 미리 계산하여 캐싱합니다.
    """
    ticker, df, ema_rsi_list, window_list, vol_win_list, volatility_win_list = args_tuple
    res = {'indicators': {}, 'ranges': {}, 'volumes': {}, 'volatilities': {}}
    
    range_val = (df['high'] - df['low']).values
    shifted_range = pd.Series(range_val).shift(1)
    
    for ema, rsi in ema_rsi_list:
        # 일반 전략용 지표
        res['indicators'][(ema, rsi)] = get_vbt_indicators(df, ema, rsi)
        # 고도화 전략용 지표 (ADX, ATR 포함)
        res['indicators'][f'enhanced_{ema}_{rsi}'] = get_vbt_indicators_enhanced(df, ema_period=ema, rsi_period=rsi)
    
    # 윈도우 기반 이동 평균 변동폭 사전 계산
    for w in window_list:
        res['ranges'][w] = {
            'avg': shifted_range.rolling(window=w).mean().values,
            'std': shifted_range.rolling(window=w).std().values
        }
    
    # 거래량 및 변동성 필터용 데이터 사전 계산
    vol_val = df['volume'].values
    shifted_vol = pd.Series(vol_val).shift(1)
    for vw in vol_win_list:
        res['volumes'][vw] = shifted_vol.rolling(window=vw).mean().values
        
    for vvw in volatility_win_list:
        res['volatilities'][vvw] = shifted_range.rolling(window=vvw).mean().values
        
    return ticker, res

def get_params_list(func, config):
    """설정 파일(JSON)을 바탕으로 테스트할 파라미터 조합을 생성하는 함수"""
    sig = inspect.signature(func)
    # 데이터와 관련된 인자들은 조합 생성에서 제외
    exclude = ['df', 'entry_price', 'entry_idx', 'current_idx', 'position_type',
               'low_val', 'high_val', 'open_val', 'close_val', 'cached_indicators', 'cached_ranges', 'atr_val']
    arg_names = [name for name in sig.parameters.keys() if name not in exclude]
    param_values = []
    for name in arg_names:
        list_key = name + '_list'
        val = config.get(list_key, [config.get(name, sig.parameters[name].default)])
        param_values.append(val)
    # 카테시안 곱(Cartesian Product)을 통해 모든 가능한 조합 생성
    return [dict(zip(arg_names, combo)) for combo in itertools.product(*param_values)]

def run_vbt_backtest_core(df, entry_params, sell_params, strategy_name, cached_inds, cached_ranges, commision_fee=0.0, slippage_fee=0.0):
    """
    [백테스트 핵심 엔진]
    특정 파라미터 조합에 대해 시계열 데이터를 돌며 진입/청산/수익률을 계산합니다.
    """
    # 1. 전략 함수 및 청산(Sell) 로직 선택
    if strategy_name == 'vbt_with_filters_enhanced':
        res = vbt_with_filters_enhanced(df, **entry_params, cached_indicators=cached_inds)
        sell_func = bailout_sell_strategy_enhanced
        atr_val = cached_inds.get('atr') if cached_inds else None
    else:
        res = vbt_with_filters(df, **entry_params, cached_indicators=cached_inds, cached_ranges=cached_ranges)
        sell_func = bailout_sell_strategy
        atr_val = None

    # 전략에서 계산된 시그널 로드
    direc, r_short, r_long = res['vbt_direction'], res['reverse_to_short'], res['reverse_to_long']
    target_longs, target_shorts = res['target_long'], res['target_short']
    
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    position, entry_price, entry_idx = 0, 0.0, 0
    pnls = []
    
    # 데이터 전체를 봉 단위로 순회
    for i in range(1, len(df)):
        # ** 보유 중인 포지션이 있을 때 **
        if position != 0:
            # A. 리버스 시그널(RSI 등) 발생 시 즉시 반대 포지션으로 전환
            if (position == 1 and r_short[i]) or (position == -1 and r_long[i]):
                base_exit_p = opens[i] # 다음 봉 시가 기준
                if position == 1:
                    exit_p = base_exit_p * (1.0 - slippage_fee)
                    # 수수료(진입시/청산시 2회)와 슬리피지를 적용한 순익
                    net_return = ((exit_p / entry_price) * (1.0 - commision_fee)**2) - 1.0
                else: # Short Exit
                    exit_p = base_exit_p * (1.0 + slippage_fee)
                    net_return = (2.0 - (exit_p / entry_price) * (1.0 + commision_fee)**2) - 1.0
                
                pnls.append(net_return)
                
                # 반대 포지션으로 바로 재진입 (Reverse Entry)
                if position == 1: # Long -> Short
                    position, entry_price, entry_idx = -1, base_exit_p * (1.0 - slippage_fee), i
                else: # Short -> Long
                    position, entry_price, entry_idx = 1, base_exit_p * (1.0 + slippage_fee), i
                continue

            # B. 일반 청산 로직(손절, 트레일링 스탑 등) 체크
            extra_args = {}
            if strategy_name == 'vbt_with_filters_enhanced' and atr_val is not None:
                extra_args['atr_val'] = atr_val[i]

            exit_sig, reason = sell_func(entry_price, entry_idx, i, 'long' if position == 1 else 'short',
                                                   lows[i], highs[i], opens[i], closes[i], **sell_params, **extra_args)
            if exit_sig:
                # 익절 사유면 시가, 아니면 종가 기준 (단순화된 규칙)
                base_exit_p = opens[i] if 'profit' in reason else closes[i]
                if position == 1:
                    exit_p = base_exit_p * (1.0 - slippage_fee)
                    net_return = ((exit_p / entry_price) * (1.0 - commision_fee)**2) - 1.0
                else:
                    exit_p = base_exit_p * (1.0 + slippage_fee)
                    net_return = (2.0 - (exit_p / entry_price) * (1.0 + commision_fee)**2) - 1.0
                
                pnls.append(net_return)
                position = 0 # 전량 청산

        # ** 보유 중인 포지션이 없을 때 (신규 진입) **
        else:
            if direc[i] == 1: # 롱 진입 신호
                # 시가가 목표가보다 위면 시가, 아니면 목표가에 지정가 체결로 가정
                base_entry_p = max(opens[i], target_longs[i])
                entry_price = base_entry_p * (1.0 + slippage_fee) # 슬리피지 적용
                position, entry_price, entry_idx = 1, entry_price, i
            elif direc[i] == -1: # 숏 진입 신호
                base_entry_p = min(opens[i], target_shorts[i])
                entry_price = base_entry_p * (1.0 - slippage_fee)
                position, entry_price, entry_idx = -1, entry_price, i
                
    return pnls

def worker_task(chunk_dict):
    """멀티프로세싱 워커: 파라미터 묶음을 받아서 결과를 계산해 반환"""
    if not GLOBAL_OHLCV: return []
    results = []
    chunk = chunk_dict['chunk']
    commision_fee = chunk_dict['commision_fee']
    slippage_fee = chunk_dict['slippage_fee']
    
    for entry_p, sell_p in chunk:
        ema, rsi = entry_p.get('ema_period', 20), entry_p.get('rsi_period', 14)
        strategy_name = chunk_dict['strategy_name']
        all_pnls = []
        for ticker, df in GLOBAL_OHLCV.items():
            pre = GLOBAL_CACHED_DATA[ticker]
            # 지표 종류에 맞는 캐시 키 생성
            cache_key = (ema, rsi) if strategy_name != 'vbt_with_filters_enhanced' else f'enhanced_{ema}_{rsi}'
            all_pnls.extend(run_vbt_backtest_core(df, entry_p, sell_p, strategy_name, pre['indicators'].get(cache_key), pre['ranges'], commision_fee, slippage_fee))
        
        # 성과 지표 산출
        if all_pnls:
            trades_count = len(all_pnls)
            wins = [p for p in all_pnls if p > 0]
            loses = [p for p in all_pnls if p <= 0]
            win_rate = len(wins) / trades_count if trades_count > 0 else 0
            
            # MDD (최대 낙폭) 계산: 누적 자산 곡선의 고점 대비 현재가 얼마나 떨어졌는가
            cum_pnl = 1.0 # 초깃값 1.0 (100%)
            max_asset = 1.0
            mdd = 1.0
            for p in all_pnls:
                cum_pnl *= (1 + p)
                if cum_pnl > max_asset:
                    max_asset = cum_pnl
                mdd = min(mdd, cum_pnl / max_asset)
            
            results.append({
                'entry': entry_p, 
                'sell': sell_p, 
                'trades': trades_count, 
                'win_rate': win_rate, 
                'win_cnt': len(wins),
                'lose_cnt': len(loses),
                'total_pnl': cum_pnl - 1.0,
                'mdd': mdd
            })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Breakout Strategy (VBT) Backtest Engine")
    parser.add_argument('--root_dir', type=str, default=os.getcwd())
    parser.add_argument('--market', type=str, default="coin")
    parser.add_argument('--interval', type=str, default="minute60")
    parser.add_argument('--processes', type=int, default=cpu_count())
    parser.add_argument('--ticker', type=str, default=None, help="Specific ticker to test (comma separated for multiple)")
    parser.add_argument('--commision_fee', type=float, default=0.0005, help="Commission fee (e.g., 0.0005 for 0.05%)")
    parser.add_argument('--slippage_fee', type=float, default=0.002, help="Slippage factor (e.g., 0.002 for 0.2%)")
    parser.add_argument('--config_idx', type=int, default=0, help="Strategy configuration index (default: 0)")
    args = parser.parse_args()

    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    config_path = os.path.join(args.root_dir, 'sbin/strategy/vbt_config.json')

    # 1. 데이터 로드 (고속 처리)
    print("--- [1/3] 데이터 병렬 로드 ---")
    conn = sqlite3.connect(db_path); cur = conn.cursor(); cur.execute(f"SELECT DISTINCT ticker FROM {table_name}"); all_tickers = [r[0] for r in cur.fetchall()]; conn.close()
    
    if args.ticker:
        target_tickers = [t.strip() for t in args.ticker.split(',')]
        all_tickers = [t for t in all_tickers if t in target_tickers]
        print(f"Target tickers: {all_tickers}")
    
    ohlcv_dict = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        it = ex.map(load_one_ticker, [(db_path, table_name, t) for t in all_tickers])
        for ticker, df in tqdm(it, total=len(all_tickers), desc="Loading") if tqdm else it:
            if df is not None: ohlcv_dict[ticker] = df

    # 2. 지표 사전 계산 (중복 계산 방지)
    print("\n--- [2/3] 지표 사전 계산 ---")
    with open(config_path, 'r', encoding='utf-8') as f: 
        configs = json.load(f)
        if args.config_idx >= len(configs):
            print(f"Error: Config index {args.config_idx} out of range (Total: {len(configs)})")
            sys.exit(1)
        config = configs[args.config_idx]
        
    ema_list = config['buy_signal_config'].get('ema_period_list', [config['buy_signal_config'].get('ema_period', 15)])
    rsi_list = config['buy_signal_config'].get('rsi_period_list', [config['buy_signal_config'].get('rsi_period', 8)])
    window_list = config['buy_signal_config'].get('window_list', [config['buy_signal_config'].get('window', 5)])
    vol_win_list = config['buy_signal_config'].get('volume_window_list', [20])
    volatility_win_list = config['buy_signal_config'].get('volatility_window_list', [20])
    ema_rsi_combi = list(itertools.product(ema_list, rsi_list))
    
    cached_dict = {}
    with ThreadPoolExecutor(max_workers=args.processes * 2) as ex:
        it = ex.map(calc_pre_data, [(t, df, ema_rsi_combi, window_list, vol_win_list, volatility_win_list) for t, df in ohlcv_dict.items()])
        for t, res in tqdm(it, total=len(ohlcv_dict), desc="Caching") if tqdm else it:
            cached_dict[t] = res

    # 3. 브루트 포스(Brute-force) 시뮬레이션
    print("\n--- [3/3] 백테스트 시뮬레이션 ---")
    # 전역 변수에 로드된 데이터 주입 (리눅스 fork 방식의 효율적 메모리 공유 활용)
    GLOBAL_OHLCV.update(ohlcv_dict)
    GLOBAL_CACHED_DATA.update(cached_dict)

    strategy_name = config.get('strategy_name', 'vbt_with_filters')
    if strategy_name == 'vbt_with_filters_enhanced':
        entry_combi = get_params_list(vbt_with_filters_enhanced, config['buy_signal_config'])
    else:
        entry_combi = get_params_list(vbt_with_filters, config['buy_signal_config'])
        
    sell_combi = get_params_list(bailout_sell_strategy if strategy_name != 'vbt_with_filters_enhanced' else bailout_sell_strategy_enhanced, config['sell_signal_config'])
    all_combis = list(itertools.product(entry_combi, sell_combi))
    
    chunk_size = max(50, len(all_combis) // (args.processes * 100))
    chunks = [all_combis[i:i+chunk_size] for i in range(0, len(all_combis), chunk_size)]
    
    print(f"전체 조합: {len(all_combis)}, 프로세스: {args.processes}, 전략: {strategy_name}")
    
    final_results = []
    with Pool(args.processes) as pool:
        chunk_dicts = [{'chunk': c, 'commision_fee': args.commision_fee, 'slippage_fee': args.slippage_fee, 'strategy_name': strategy_name} for c in chunks]
        it = pool.imap_unordered(worker_task, chunk_dicts)
        if tqdm: it = tqdm(it, total=len(chunks), desc="Testing")
        for res_list in it:
            if res_list: final_results.extend(res_list)

    # 4. 결과 출력
    if final_results:
        # PnL(수익률) 기준 내림차순 정렬
        sorted_res = sorted(final_results, key=lambda x: x['total_pnl'], reverse=True)
        print("\n### TOP 10 RESULTS ###")
        for r in sorted_res[:10]:
            print(f"PnL: {r['total_pnl']:.4f} | MDD: {r['mdd']:.4f} | Win: {r['win_rate']:.2f} | Trades: {r['trades']} (W:{r['win_cnt']} L:{r['lose_cnt']}) | Entry: {r['entry']} | Sell: {r['sell']}")
    else:
        print("거래 내역이 없습니다. (No trades found)")
