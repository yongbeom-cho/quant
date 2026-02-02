#!/usr/bin/env python3
"""
통합 백테스트 실행기 (CLI)

모든 Buy/Sell 전략 조합을 테스트하고 결과를 정렬하여 출력합니다.

=============================================================================
사용 예시 (분리된 config 방식 - 권장)
=============================================================================

# 1. 기본 사용 (첫번째 config만 사용)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --market coin --interval minute60

# 2. 모든 buy config + 모든 sell config 조합 (전체 테스트)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx all \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx all \\
    --market coin --interval minute60

# 3. 특정 config 인덱스 지정
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx 0 1 \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx 0 2 \\
    --market coin --interval minute60

# 4. 결과를 승률/MDD 기준으로 정렬
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --sort_by win_ratio --top_n 20

# 5. 특정 종목만 테스트
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --ticker KRW-BTC,KRW-ETH

# 6. 결과를 CSV로 저장
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --output results.csv

=============================================================================
사용 예시 (기존 방식 - 통합 config)
=============================================================================

python sbin/backtest/backtest_runner.py \\
    --buy_strategy vbt_with_filters \\
    --sell_strategy bailout_sell \\
    --config sbin/strategy/vbt_config.json \\
    --market coin --interval minute60

=============================================================================
Config 인덱스 옵션
=============================================================================

--buy_config_idx all      : 모든 buy config 사용
--sell_config_idx all     : 모든 sell config 사용
--buy_config_idx 0 1 2    : 특정 인덱스만 사용
(생략시)                   : 기본값 0 사용

=============================================================================
병렬 처리 옵션
=============================================================================

# 멀티프로세스 병렬 처리 (기본 워커 수)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx 0 \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx 0 \\
    --parallel

# 워커 수 지정 (4개)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/buy_config.json --buy_config_idx all \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx all \\
    --parallel --workers 4

=============================================================================
"""

import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Any

# 모듈 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SBIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SBIN_DIR)

from buy_strategy.registry import get_buy_strategy, get_all_buy_param_combinations
from sell_strategy.registry import get_sell_strategy, get_all_sell_param_combinations
from backtest.engine import UnifiedBacktestEngine
from backtest.data_loader import load_ohlcv_data, get_db_path, get_table_name


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 리스트가 아니면 리스트로 래핑
    if not isinstance(config, list):
        config = [config]
    
    return config


def parse_xgb_model_name(model_name: str, market: str, interval: str) -> Dict[str, Any]:
    """
    XGBoost 모델 이름을 파싱하여 base_strategy_name과 base_strategy_param 추출
    
    모델 이름 형식:
    xgb-{market}-{interval}-{strategy_name}-{str_buy_strategy_params}-{label_col}-{min_precision}-{best_threshold}-{str_feats}
    
    str_buy_strategy_params 형식: key=value^key=value^...
    
    예시:
    xgb-coin-day-pb_rebound-pb_rebound_line=0.7^pb_sma_period=20^pb_sma_up=0.6-label0-0.55-0.5-f0f1f2
    
    Returns:
        {
            'base_strategy_name': str,
            'base_strategy_param': dict
        }
    """
    prefix = f"xgb-{market}-{interval}-"
    if not model_name.startswith(prefix):
        raise ValueError(f"Model name '{model_name}' does not match expected format: {prefix}...")
    
    # prefix 제거
    remaining = model_name[len(prefix):]
    parts = remaining.split('-')
    
    if len(parts) < 1:
        raise ValueError(f"Model name '{model_name}' has insufficient parts after parsing")
    
    # 첫 번째 부분은 strategy_name
    base_strategy_name = parts[0]
    
    # 두 번째 부분부터 label_col을 찾기 (label로 시작하는 부분)
    label_idx = parts[2]
    
    # str_buy_strategy_params 추출 (strategy_name과 label_col 사이)
    base_strategy_param = {}
        
    # parts[1] 파싱 (key=value^key=value^... 형식)
    # ^로 split하여 각 key=value 쌍 추출
    
    param_pairs = parts[1].split('^')
    for pair in param_pairs:
        if '=' in pair:
            k, v = pair.split('=', 1)
            # 값이 숫자면 숫자로 변환
            try:
                if '.' in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass  # 문자열로 유지
            base_strategy_param[k] = v
    
    return {
        'base_strategy_name': base_strategy_name,
        'base_strategy_param': base_strategy_param
    }


def create_buy_strategies_from_separate_config(config_path: str, config_idx: int = 0, root_dir: str = None, market: str = 'coin', interval: str = 'day') -> tuple:
    """
    buy_strategy/config 폴더의 설정 파일에서 전략 인스턴스 생성
    
    1. JSON 설정 파일을 로드합니다.
    2. 지정된 인덱스의 설정을 가져옵니다.
    3. Registry를 통해 해당 전략의 모든 파라미터 조합을 생성합니다.
    4. 각 조합별로 전략 객체를 생성하여 리스트로 반환합니다.
    
    Returns:
        (strategies_list, strategy_name)
    """
    configs = load_config(config_path)
    if config_idx >= len(configs):
        raise ValueError(f"Config index {config_idx} out of range (Total: {len(configs)})")
    
    config = configs[config_idx]
    strategy_name = config.get('strategy_name')
    if not strategy_name:
        raise ValueError("buy config must contain 'strategy_name'")
    
    buy_config = config.get('buy_signal_config', config)
    
    # model_name_list가 있으면 XGBBuyStrategy 사용
    if 'model_dir' in buy_config:
        # root_dir 기본값 설정
        if root_dir is None:
            root_dir = os.getcwd()
        model_dir = os.path.join(root_dir, buy_config['model_dir'])
        
        # model_dir에 있는 xgb-{market}-{interval}로 시작하는 모든 파일 찾기
        pattern = os.path.join(model_dir, f'xgb-{market}-{interval}*')
        model_files = glob.glob(pattern)
        
        # 파일명에서 확장자를 제거한 모델 이름 리스트 생성
        model_name_list = []
        for file_path in model_files:
            model_name = os.path.basename(file_path)
            model_name_list.append(model_name)
        
        # 정렬하여 일관성 유지
        model_name_list.sort()

        if not model_name_list:
            raise ValueError(f"No model files found matching pattern: xgb-{market}-{interval}* in {model_dir}")
        
        strategies = []
        for model_name in model_name_list:
            # 모델 이름 파싱
            try:
                parsed = parse_xgb_model_name(model_name, market, interval)
            except ValueError as e:
                print(f"Warning: Failed to parse model name '{model_name}': {e}")
                continue
            
            # xgb_params 생성
            xgb_params = {
                'strategy_name': 'xgb_buy',
                'model_name': model_name,
                'model_dir': model_dir,
                'base_strategy_name': parsed['base_strategy_name'],
                'base_strategy_param': parsed['base_strategy_param'],
                'interval': interval
            }

            # 최대 투자 비율 설정 전달 (자산 관리용)
            if 'max_investment_ratio' in buy_config:
                xgb_params['max_investment_ratio'] = buy_config['max_investment_ratio']
            strategy = get_buy_strategy('xgb_buy', xgb_params)
            strategies.append(strategy)
        
        return strategies, 'xgb_buy'
    
    # 일반 전략 처리
    # Registry에서 파라미터 리스트를 기반으로 모든 가능한 조합(Cartesian Product)을 생성
    combinations = get_all_buy_param_combinations(strategy_name, buy_config)
    
    strategies = []
    for params in combinations:
        params['strategy_name'] = strategy_name
        # 최대 투자 비율 설정 전달 (자산 관리용)
        if 'max_investment_ratio' in buy_config:
            params['max_investment_ratio'] = buy_config['max_investment_ratio']
        # 실제 전략 인스턴스 생성
        strategy = get_buy_strategy(strategy_name, params)
        strategies.append(strategy)
    
    return strategies, strategy_name


def create_sell_strategies_from_separate_config(config_path: str, config_idx: int = 0) -> tuple:
    """
    sell_strategy/config 폴더의 설정 파일에서 청산 전략 인스턴스 생성
    
    Returns:
        (strategies_list, strategy_name)
    """
    configs = load_config(config_path)
    if config_idx >= len(configs):
        raise ValueError(f"Config index {config_idx} out of range (Total: {len(configs)})")
    
    config = configs[config_idx]
    strategy_name = config.get('strategy_name')
    if not strategy_name:
        raise ValueError("sell config must contain 'strategy_name'")
    
    sell_config = config.get('sell_signal_config', config)
    combinations = get_all_sell_param_combinations(strategy_name, sell_config)
    
    strategies = []
    for params in combinations:
        params['strategy_name'] = strategy_name
        strategy = get_sell_strategy(strategy_name, params)
        strategies.append(strategy)
    
    return strategies, strategy_name


def create_buy_strategies(strategy_names: List[str], config: Dict[str, Any]) -> list:
    """매수 전략 인스턴스 생성 (기존 방식)"""
    strategies = []
    
    buy_config = config.get('buy_signal_config', config)
    
    for name in strategy_names:
        combinations = get_all_buy_param_combinations(name, buy_config)
        for params in combinations:
            params['strategy_name'] = name
            strategy = get_buy_strategy(name, params)
            strategies.append(strategy)
    
    return strategies


def create_sell_strategies(strategy_names: List[str], config: Dict[str, Any]) -> list:
    """청산 전략 인스턴스 생성 (기존 방식)"""
    strategies = []
    
    sell_config = config.get('sell_signal_config', config)
    
    for name in strategy_names:
        combinations = get_all_sell_param_combinations(name, sell_config)
        for params in combinations:
            params['strategy_name'] = name
            strategy = get_sell_strategy(name, params)
            strategies.append(strategy)
    
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description='Unified Strategy Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 분리된 config 사용 (권장)
  python backtest_runner.py \\
      --buy_config ../buy_strategy/config/vbt_config.json \\
      --sell_config ../sell_strategy/config/sell_config.json \\
      --market coin --interval minute60

  # 기존 방식 (통합 config)
  python backtest_runner.py --buy_strategy vbt_with_filters --sell_strategy bailout_sell \\
      --config ../strategy/vbt_config.json

  # Cross-combination test
  python backtest_runner.py \\
      --buy_config ../buy_strategy/config/vbt_config.json --buy_config_idx 0 \\
      --sell_config ../sell_strategy/config/sell_config.json --sell_config_idx 0 2
        """
    )
    
    # === 분리된 config 옵션 (권장) ===
    parser.add_argument('--buy_config', type=str, default=None,
                        help='Path to buy strategy config JSON file')
    parser.add_argument('--buy_config_idx', type=str, nargs='+', default=['0'],
                        help='Buy config indices to use. Use "all" for all configs, or specify indices like "0 1 2"')
    parser.add_argument('--sell_config', type=str, default=None,
                        help='Path to sell strategy config JSON file')
    parser.add_argument('--sell_config_idx', type=str, nargs='+', default=['0'],
                        help='Sell config indices to use. Use "all" for all configs, or specify indices like "0 1 2"')
    
    # === 기존 방식 옵션 ===
    parser.add_argument('--buy_strategy', type=str, nargs='+', default=None,
                        help='Buy strategy names (legacy mode)')
    parser.add_argument('--sell_strategy', type=str, nargs='+', default=None,
                        help='Sell strategy names (legacy mode)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to unified config JSON file (legacy mode)')
    parser.add_argument('--config_idx', type=int, default=0,
                        help='Config index in JSON array (legacy mode)')
    
    # 데이터 설정
    parser.add_argument('--root_dir', type=str, default=os.getcwd(),
                        help='Project root directory')
    parser.add_argument('--market', type=str, default='coin',
                        help='Market type (default: coin)')
    parser.add_argument('--interval', type=str, default='day',
                        help='Interval (default: day)')
    parser.add_argument('--ticker', type=str, default=None,
                        help='Specific tickers to test (comma separated)')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    
    # 백테스트 설정
    parser.add_argument('--commission_fee', type=float, default=0.0005,
                        help='Commission fee (default: 0.0005 = 0.05%%)')
    parser.add_argument('--slippage_fee', type=float, default=0.002,
                        help='Slippage fee (default: 0.002 = 0.2%%)')
    parser.add_argument('--initial_capital', type=float, default=1.0,
                        help='Initial capital (default: 1.0)')
    # 다중 포지션 설정 (매수)
    parser.add_argument('--max_position_cnts', type=str, default='1',
                        help='Maximum number of concurrent positions per ticker (comma-separated, default: 1)')
    parser.add_argument('--is_timeseries_backtest', action='store_true',
                        help='Use timeseries backtest mode (default: False)')
    
    # 병렬 처리 설정
    parser.add_argument('--parallel', action='store_true',
                        help='Use multiprocessing for parallel execution')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count, max 8)')
    
    # 결과 설정
    parser.add_argument('--sort_by', type=str, default='total_pnl',
                        choices=['total_pnl', 'win_ratio', 'mdd', 'sharpe_ratio', 'trade_count'],
                        help='Sort results by (default: total_pnl)')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top results to show (default: 10)')
    parser.add_argument('--min_trades', type=int, default=1,
                        help='Minimum trades filter (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (optional)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save results every N combinations (default: 100, only for parallel mode)')
    
    args = parser.parse_args()
    
    # === 1. 설정 모드 확인 ===
    use_separate_config = args.buy_config is not None and args.sell_config is not None
    use_legacy_config = args.buy_strategy is not None and args.sell_strategy is not None and args.config is not None
    
    if not use_separate_config and not use_legacy_config:
        parser.error("Either (--buy_config + --sell_config) or (--buy_strategy + --sell_strategy + --config) must be specified")
    
    # === 2. 설정 로드 ===
    print("=== [1/4] Loading Configuration ===")
    
    if use_separate_config:
        print(f"Buy config: {args.buy_config}")
        print(f"Sell config: {args.sell_config}")
        
        # 인덱스 파싱 (all 또는 숫자)
        buy_configs = load_config(args.buy_config)
        sell_configs = load_config(args.sell_config)
        
        # Buy config 인덱스 처리
        if 'all' in args.buy_config_idx:
            buy_indices = list(range(len(buy_configs)))
            print(f"  Using ALL buy configs ({len(buy_indices)} total)")
        else:
            buy_indices = [int(idx) for idx in args.buy_config_idx]
        
        # Sell config 인덱스 처리
        if 'all' in args.sell_config_idx:
            sell_indices = list(range(len(sell_configs)))
            print(f"  Using ALL sell configs ({len(sell_indices)} total)")
        else:
            sell_indices = [int(idx) for idx in args.sell_config_idx]
        
        # Buy 전략 생성
        buy_strategies = []
        for idx in buy_indices:
            strats, name = create_buy_strategies_from_separate_config(args.buy_config, idx, args.root_dir, args.market, args.interval)
            buy_strategies.extend(strats)
            print(f"  Buy[{idx}]: {name} ({len(strats)} combinations)")
        
        # Sell 전략 생성
        sell_strategies = []
        for idx in sell_indices:
            strats, name = create_sell_strategies_from_separate_config(args.sell_config, idx)
            sell_strategies.extend(strats)
            print(f"  Sell[{idx}]: {name} ({len(strats)} combinations)")
    else:
        # 기존 방식
        configs = load_config(args.config)
        if args.config_idx >= len(configs):
            print(f"Error: Config index {args.config_idx} out of range (Total: {len(configs)})")
            sys.exit(1)
        config = configs[args.config_idx]
        print(f"Loaded config: {config.get('strategy_name', 'unknown')}")
        
        buy_strategies = create_buy_strategies(args.buy_strategy, config)
        sell_strategies = create_sell_strategies(args.sell_strategy, config)
    
    # === 3. 데이터 로드 ===
    print("\n=== [2/4] Loading OHLCV Data ===")
    table_name = get_table_name(args.market, args.interval)
    db_path = get_db_path(args.root_dir, args.market, args.interval)
    
    print(f"DB path: {db_path}")
    print(f"Table: {table_name}")
    
    tickers = None
    if args.ticker:
        tickers = [t.strip() for t in args.ticker.split(',')]
        print(f"Target tickers: {tickers}")
    
    data = load_ohlcv_data(
        db_path=db_path,
        table_name=table_name,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    print(f"Loaded {len(data)} tickers")
    
    if not data:
        print("Error: No data loaded. Check DB path and parameters.")
        sys.exit(1)
    
    # === 4. max_position_cnts 파싱 ===
    max_position_cnts = [int(x.strip()) for x in args.max_position_cnts.split(',')]
    
    # === 5. 전략 조합 정보 ===
    print("\n=== [3/4] Strategy Combinations ===")
    print(f"Buy strategies: {len(buy_strategies)} combinations")
    print(f"Sell strategies: {len(sell_strategies)} combinations")
    print(f"max_position_cnts: {max_position_cnts} ({len(max_position_cnts)} values)")
    total_combinations = len(buy_strategies) * len(sell_strategies) * len(max_position_cnts)
    print(f"Total combinations: {total_combinations}")
    
    # === 6. 백테스트 실행 (핵심 시뮬레이션 환경 구축) ===
    print("\n=== [4/4] Running Backtest ===")
    print(f"Settings: max_position_cnts={max_position_cnts}")
    engine = UnifiedBacktestEngine(
        commission_fee=args.commission_fee,
        slippage_fee=args.slippage_fee,
        initial_capital=args.initial_capital
    )
    
    # 병렬 또는 순차 실행 선택
    # 병렬 실행 시 중간 결과를 checkpoint_file에 저장하여 OOM(메모리 부족) 방지
    if args.parallel:
        print(f"Using parallel mode (workers: {args.workers or 'auto'})")
        results = engine.run_cross_combination_test_parallel(
            data=data,
            buy_strategies=buy_strategies,
            sell_strategies=sell_strategies,
            max_position_cnts=max_position_cnts,
            use_reverse_signal=True,
            is_timeseries_backtest=args.is_timeseries_backtest,
            n_workers=args.workers,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_file=args.output  # 중간 결과를 output 파일에 저장
        )
    else:
        results = engine.run_cross_combination_test(
            data=data,
            buy_strategies=buy_strategies,
            sell_strategies=sell_strategies,
            max_position_cnts=max_position_cnts,
            use_reverse_signal=True,
            is_timeseries_backtest=args.is_timeseries_backtest
        )
    
    # === 6. 결과 출력 및 파일 저장 ===
    if results:
        # 병렬 모드에서 체크포인트를 사용한 경우 파일에서 상위 결과 로드 및 출력
        if args.parallel and args.output:
            # CSV 파일에서 결과 읽어서 top_n 출력
            import pandas as pd
            df = pd.read_csv(args.output)
            df = df.sort_values(args.sort_by, ascending=False)
            
            print(f"\n{'='*80}")
            print(f"TOP {min(args.top_n, len(df))} RESULTS (from {len(df)} total)")
            print(f"{'='*80}")
            
            for i, (_, row) in enumerate(df.head(args.top_n).iterrows(), 1):
                print(f"\n[{i}] {row['buy_strategy_name']} + {row['sell_strategy_name']}")
                print(f"    PnL: {row['total_pnl']:.4f} | Win: {row['win_ratio']:.2%} | "
                      f"Trades: {int(row['trade_count'])} | MDD: {row['max_drawdown_pct']:.2f}%")
                # buy_params, sell_params, max_position_cnt 출력
                print(f"    Buy Params: {row['buy_params']}")
                print(f"    Sell Params: {row['sell_params']}")
                print(f"    Max Position Cnt: {int(row['max_position_cnt'])}")
        else:
            # 순차 모드에서는 메모리에 있는 결과를 바로 정렬하여 출력
            top_results = engine.get_top_results(
                results,
                sort_by=args.sort_by,
                top_n=args.top_n,
                min_trades=args.min_trades
            )
            
            engine.print_results(top_results, args.top_n)
            
            # 최종 결과를 CSV로 저장 (사용자 요청 시)
            if args.output:
                df = engine.results_to_dataframe(results)
                df = df.sort_values(args.sort_by, ascending=False)
                df.to_csv(args.output, index=False)
                print(f"\nAll {len(results)} results saved to: {args.output}")
    else:
        print("No results found. Check your data and configuration.")


if __name__ == "__main__":
    main()
