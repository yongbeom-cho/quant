import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
import mplfinance as mpf
import matplotlib.pyplot as plt

# 전략 모듈 경로 추가 (sbin 디렉토리에서 실행 시 상위 디렉토리 참조)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.vbt_strategy_012 import vbt_with_filters, get_vbt_indicators
from strategy.vbt_sell_strategy_013 import bailout_sell_strategy
from strategy.vbt_strategy_020_enhanced import vbt_with_filters_enhanced, get_vbt_indicators_enhanced
from strategy.vbt_sell_strategy_021_enhanced import bailout_sell_strategy_enhanced
import inspect
import itertools

def get_params_list(func, config):
    """설정 파일에서 파라미터 조합 생성 (백테스터와 동일한 로직)"""
    sig = inspect.signature(func)
    exclude = ['df', 'entry_price', 'entry_idx', 'current_idx', 'position_type',
               'low_val', 'high_val', 'open_val', 'close_val', 'cached_indicators', 'cached_ranges', 'atr_val']
    arg_names = [name for name in sig.parameters.keys() if name not in exclude]
    param_values = []
    for name in arg_names:
        list_key = name + '_list'
        val = config.get(list_key, [config.get(name, sig.parameters[name].default)])
        param_values.append(val)
    return [dict(zip(arg_names, combo)) for combo in itertools.product(*param_values)]

def load_ohlcv(db_path, table_name, ticker):
    """DB에서 캔들 데이터를 읽어와 Pandas DataFrame으로 변환"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date ASC", conn, params=(ticker,))
    conn.close()
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True) # 차트 가로축을 위한 인덱스 설정
    return df

def simulate_vbt_with_history(df, entry_params, sell_params, strategy_name='vbt_with_filters', commision_fee=0.0005, slippage_fee=0.002, initial_balance=1000.0):
    """
    [시각화를 위한 시뮬레이션]
    백테스트 엔진과 동일한 로직을 수행하되, 차트에 표시할 마커(진입/청산 지점)와 거래 내역을 기록합니다.
    """
    if strategy_name == 'vbt_with_filters_enhanced':
        res = vbt_with_filters_enhanced(df, **entry_params)
        sell_func = bailout_sell_strategy_enhanced
        # 시각화 도구에서는 단순화를 위해 ATR을 직접 계산해서 사용
        import talib
        atr_val = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    else:
        res = vbt_with_filters(df, **entry_params)
        sell_func = bailout_sell_strategy
        atr_val = None

    direc = res['vbt_direction']
    r_short, r_long = res['reverse_to_short'], res['reverse_to_long']
    target_longs, target_shorts = res['target_long'], res['target_short']
    
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    times = df.index
    
    position = 0
    entry_price = 0.0
    entry_idx = 0
    trades = []
    current_balance = initial_balance
    
    # 차트에 표시할 시세 마커 (해당 인덱스에 가격 값 주입)
    long_entry = np.full(len(df), np.nan)
    long_exit = np.full(len(df), np.nan)
    short_entry = np.full(len(df), np.nan)
    short_exit = np.full(len(df), np.nan)
    
    for i in range(1, len(df)):
        if position != 0:
            exit_sig = False
            reason = 'none'
            
            # A. 리버스 시그널 체크
            if (position == 1 and r_short[i]) or (position == -1 and r_long[i]):
                exit_sig = True
                reason = 'reverse'
                base_exit_p = opens[i]
            # B. 일반 청산 로직 체크
            else:
                extra_args = {}
                if strategy_name == 'vbt_with_filters_enhanced' and atr_val is not None:
                    extra_args['atr_val'] = atr_val[i]

                exit_sig, reason = sell_func(entry_price, entry_idx, i, 'long' if position == 1 else 'short',
                                                       lows[i], highs[i], opens[i], closes[i], **sell_params, **extra_args)
                base_exit_p = opens[i] if 'profit' in reason else closes[i]

            if exit_sig:
                # 수익률 및 잔고 업데이트
                if position == 1:
                    exit_p = base_exit_p * (1.0 - slippage_fee)
                    pnl = ((exit_p / entry_price) * (1.0 - commision_fee)**2) - 1.0
                else:
                    exit_p = base_exit_p * (1.0 + slippage_fee)
                    pnl = (2.0 - (exit_p / entry_price) * (1.0 + commision_fee)**2) - 1.0
                
                current_balance *= (1.0 + pnl)
                # 거래 기록 저장
                trades.append({
                    'type': 'long' if position == 1 else 'short',
                    'entry_time': times[entry_idx],
                    'exit_time': times[i],
                    'entry_price': entry_price,
                    'exit_price': exit_p,
                    'pnl': pnl,
                    'balance': current_balance,
                    'reason': reason
                })
                
                # 차트 마커 기록
                if position == 1:
                    long_exit[i] = base_exit_p
                else:
                    short_exit[i] = base_exit_p
                
                # 리버스인 경우 반대 방향으로 재진입
                if reason == 'reverse':
                    if position == 1: # Switch to Short
                        position, entry_price, entry_idx = -1, base_exit_p * (1.0 - slippage_fee), i
                        short_entry[i] = base_exit_p
                    else: # Switch to Long
                        position, entry_price, entry_idx = 1, base_exit_p * (1.0 + slippage_fee), i
                        long_entry[i] = base_exit_p
                else:
                    position = 0
        else:
            # 신규 진입 시그널 체크
            if direc[i] == 1:
                base_entry_p = max(opens[i], target_longs[i])
                entry_price = base_entry_p * (1.0 + slippage_fee)
                position, entry_idx = 1, i
                long_entry[i] = base_entry_p
            elif direc[i] == -1:
                base_entry_p = min(opens[i], target_shorts[i])
                entry_price = base_entry_p * (1.0 - slippage_fee)
                position, entry_idx = -1, i
                short_entry[i] = base_entry_p
                
    return trades, (long_entry, short_entry, long_exit, short_exit)

def visualize_trades(df, trades, markers, ticker, strategy_name, output_path=None):
    """진입/청산 마커와 거래 내역 표가 포함된 차트 생성"""
    df_plot = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    long_entry, short_entry, long_exit, short_exit = markers
    
    # 보조 지표 (EMA) 추가
    import talib
    ema = talib.EMA(df['close'], timeperiod=15)
    
    add_plots = []
    if ema is not None:
        add_plots.append(mpf.make_addplot(ema, color='blue', width=0.7))
    
    # 롱 진입 (매수): 빨간색 위쪽 화살표
    if not np.all(np.isnan(long_entry)):
        add_plots.append(mpf.make_addplot(long_entry, type='scatter', marker='^', markersize=100, color='red', label='Long Entry'))
    
    # 숏 진입 (매도): 파란색 아래쪽 화살표
    if not np.all(np.isnan(short_entry)):
        add_plots.append(mpf.make_addplot(short_entry, type='scatter', marker='v', markersize=100, color='blue', label='Short Entry'))

    # 롱 청산 (매수 종료): 주황색 아래쪽 화살표
    if not np.all(np.isnan(long_exit)):
        add_plots.append(mpf.make_addplot(long_exit, type='scatter', marker='v', markersize=100, color='orange', label='Long Exit'))

    # 숏 청산 (매도 종료): 보라색 위쪽 화살표
    if not np.all(np.isnan(short_exit)):
        add_plots.append(mpf.make_addplot(short_exit, type='scatter', marker='^', markersize=100, color='purple', label='Short Exit'))

    # 성과 요약 텍스트 계산
    total_pnl = 1.0
    max_asset = 1.0
    mdd = 1.0
    for t in trades:
        total_pnl *= (1 + t['pnl'])
        if total_pnl > max_asset:
            max_asset = total_pnl
        mdd = min(mdd, total_pnl / max_asset)
    
    summary_text = f"Total PnL: {total_pnl-1.0:+.2%} | Max Drawdown: {-(1.0-mdd):.2%}"
    
    # 메인 차트 그리기
    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        title=f"VBT Trades: {ticker} ({strategy_name})\n{summary_text}",
        ylabel='Price',
        addplot=add_plots,
        volume=True,
        figsize=(16, 12),
        returnfig=True,
        panel_ratios=(6, 2)
    )
    
    # 차트 하단에 거래 내역 표(Table) 추가
    if trades:
        header = ['Type', 'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'PnL', 'Balance', 'Reason']
        table_data = []
        for t in trades[:15]: # 공간 효율상 최근 15개 거래만 표시
            table_data.append([
                t['type'],
                str(t['entry_time'].strftime('%m-%d %H:%M')),
                str(t['exit_time'].strftime('%m-%d %H:%M')),
                f"{t['entry_price']:,.0f}",
                f"{t['exit_price']:,.0f}",
                f"{t['pnl']:+.2%}",
                f"{t['balance']:,.0f}",
                t['reason']
            ])
        
        if len(trades) > 15:
            table_data.append(['...', '...', '...', '...', '...', '...', '...', f'And {len(trades)-15} more...'])

        # 표를 위한 별도 축(Axes) 생성
        table_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15]) 
        table_ax.axis('off')
        
        the_table = table_ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 1.2)
        
        # 헤더 스타일 설정 (굵게, 배경색)
        for j in range(len(header)):
            the_table[0, j].get_text().set_weight('bold')
            the_table[0, j].set_facecolor('#f0f0f0')

    # 결과물 저장 또는 화면 출력
    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Chart with trade table saved to: {output_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not show plot: {e}. Try specifying --output to save to a file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBT Trade Visualizer with Chart and Metrics")
    parser.add_argument('--root_dir', type=str, default=os.getcwd())
    parser.add_argument('--market', type=str, default="coin")
    parser.add_argument('--interval', type=str, default="minute60")
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--config_idx', type=int, default=0, help="Index in the config file (root list)")
    parser.add_argument('--combo_idx', type=int, default=0, help="Specific combination index within the chosen config")
    parser.add_argument('--list_combos', action='store_true', help="List all available combinations for the selected strategy")
    parser.add_argument('--output', type=str, default=None, help="Path to save the chart image (e.g., vbt_chart.png)")
    parser.add_argument('--initial_balance', type=float, default=1000.0, help="Initial balance (default: 1000)")
    parser.add_argument('--commision_fee', type=float, default=0.0005)
    parser.add_argument('--slippage_fee', type=float, default=0.002)
    args = parser.parse_args()

    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    config_path = os.path.join(args.root_dir, 'sbin/strategy/vbt_config.json')

    # 1. 시세 데이터 로드
    df = load_ohlcv(db_path, table_name, args.ticker)
    if df is None:
        print(f"No data for {args.ticker}")
        sys.exit(1)

    # 2. 전략 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)[args.config_idx]

    # 전략 종류에 따른 파라미터 조합 생성
    strategy_name = config.get('strategy_name', 'vbt_with_filters')
    if strategy_name == 'vbt_with_filters_enhanced':
        entry_combos = get_params_list(vbt_with_filters_enhanced, config['buy_signal_config'])
        sell_combos = get_params_list(bailout_sell_strategy_enhanced, config['sell_signal_config'])
    else:
        entry_combos = get_params_list(vbt_with_filters, config['buy_signal_config'])
        sell_combos = get_params_list(bailout_sell_strategy, config['sell_signal_config'])
    
    all_combos = list(itertools.product(entry_combos, sell_combos))

    # 파라미터 조합 목록만 출력하고 종료하는 모드
    if args.list_combos:
        print(f"\nAvailable combinations for config index {args.config_idx} (Strategy: {strategy_name}):")
        for i, (e, s) in enumerate(all_combos):
            print(f"[{i}] Entry: {e} | Sell: {s}")
        sys.exit(0)

    # 특정 파라미터 조합 선택
    if args.combo_idx < 0 or args.combo_idx >= len(all_combos):
        print(f"Invalid combo_idx {args.combo_idx}. Total available: {len(all_combos)}")
        sys.exit(1)

    entry_params, sell_params = all_combos[args.combo_idx]
    
    print(f"Using Combo [{args.combo_idx}]:")
    print(f"  Strategy: {strategy_name}")
    print(f"  Entry: {entry_params}")
    print(f"  Sell:  {sell_params}")
    print(f"  Initial Balance: {args.initial_balance}")

    # 3. 상세 거래 시뮬레이션 실행 (기록용)
    trades, markers = simulate_vbt_with_history(
        df, entry_params, sell_params, strategy_name, args.commision_fee, args.slippage_fee, args.initial_balance
    )
    
    # 4. 텍스트 로그 출력
    print("\n" + "="*80)
    print(f"{'TYPE':<6} | {'ENTRY TIME':<20} | {'EXIT TIME':<20} | {'PNL':<8} | {'BALANCE':<10} | {'REASON'}")
    print("-"*80)
    total_pnl = 1.0
    max_asset = 1.0
    mdd = 1.0
    for t in trades:
        print(f"{t['type']:<6} | {str(t['entry_time']):<20} | {str(t['exit_time']):<20} | {t['pnl']:>8.2%} | {t['balance']:>10,.0f} | {t['reason']}")
        total_pnl *= (1 + t['pnl'])
        if total_pnl > max_asset:
            max_asset = total_pnl
        mdd = min(mdd, total_pnl / max_asset)
        
    print("-" * 80)
    print(f"TOTAL PNL: {total_pnl-1.0:+.2%} | MAX DRAWDOWN: {-(1.0-mdd):.2%}")
    print("="*80 + "\n")

    # 5. 차트 시각화
    visualize_trades(df, trades, markers, args.ticker, strategy_name, args.output)
