#!/usr/bin/env python3
"""
통합 백테스트 실행기 (CLI)

모든 Buy/Sell 전략 조합을 테스트하고 결과를 정렬하여 출력합니다.

=============================================================================
사용 예시 (분리된 config 방식 - 권장)
=============================================================================

# 1. 기본 사용 (첫번째 config만 사용)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --market coin --interval minute60

# 2. 모든 buy config + 모든 sell config 조합 (전체 테스트)
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json --buy_config_idx all \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx all \\
    --market coin --interval minute60

# 3. 특정 config 인덱스 지정
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json --buy_config_idx 0 1 \\
    --sell_config sbin/sell_strategy/config/sell_config.json --sell_config_idx 0 2 \\
    --market coin --interval minute60

# 4. 결과를 승률/MDD 기준으로 정렬
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --sort_by win_ratio --top_n 20

# 5. 특정 종목만 테스트
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json \\
    --sell_config sbin/sell_strategy/config/sell_config.json \\
    --ticker KRW-BTC,KRW-ETH

# 6. 결과를 CSV로 저장
python sbin/backtest/backtest_runner.py \\
    --buy_config sbin/buy_strategy/config/vbt_config.json \\
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


def create_buy_strategies_from_separate_config(config_path: str, config_idx: int = 0) -> tuple:
    """
    buy_strategy/config 폴더의 설정 파일에서 전략 인스턴스 생성
    
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
    combinations = get_all_buy_param_combinations(strategy_name, buy_config)
    
    strategies = []
    for params in combinations:
        params['strategy_name'] = strategy_name
        # max_investment_ratio 전달
        if 'max_investment_ratio' in buy_config:
            params['max_investment_ratio'] = buy_config['max_investment_ratio']
        strategy = get_buy_strategy(strategy_name, params)
        strategies.append(strategy)
    
    return strategies, strategy_name


def create_sell_strategies_from_separate_config(config_path: str, config_idx: int = 0) -> tuple:
    """
    sell_strategy/config 폴더의 설정 파일에서 전략 인스턴스 생성
    
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
    parser.add_argument('--interval', type=str, default='minute60',
                        help='Interval (default: minute60)')
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
            strats, name = create_buy_strategies_from_separate_config(args.buy_config, idx)
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
    
    # === 4. 전략 조합 정보 ===
    print("\n=== [3/4] Strategy Combinations ===")
    print(f"Buy strategies: {len(buy_strategies)} combinations")
    print(f"Sell strategies: {len(sell_strategies)} combinations")
    print(f"Total combinations: {len(buy_strategies) * len(sell_strategies)}")
    
    # === 5. 백테스트 실행 ===
    print("\n=== [4/4] Running Backtest ===")
    engine = UnifiedBacktestEngine(
        commission_fee=args.commission_fee,
        slippage_fee=args.slippage_fee
    )
    
    # 병렬 또는 순차 실행
    if args.parallel:
        print(f"Using parallel mode (workers: {args.workers or 'auto'})")
        results = engine.run_cross_combination_test_parallel(
            data=data,
            buy_strategies=buy_strategies,
            sell_strategies=sell_strategies,
            n_workers=args.workers
        )
    else:
        results = engine.run_cross_combination_test(
            data=data,
            buy_strategies=buy_strategies,
            sell_strategies=sell_strategies
        )
    
    # === 6. 결과 출력 ===
    if results:
        top_results = engine.get_top_results(
            results,
            sort_by=args.sort_by,
            top_n=args.top_n,
            min_trades=args.min_trades
        )
        
        engine.print_results(top_results, args.top_n)
        
        # CSV 출력 (옵션)
        if args.output:
            df = engine.results_to_dataframe(results)
            df = df.sort_values(args.sort_by, ascending=False)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    else:
        print("No results found. Check your data and configuration.")


if __name__ == "__main__":
    main()
