"""
통합 백테스트 엔진

모든 Buy/Sell 전략 조합을 테스트할 수 있는 통합 백테스트 엔진입니다.
Cross-combination을 지원하여 VBTBuy + TimeseriesSell 같은 조합도 가능합니다.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

# 상위 디렉토리 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_strategy.base import BaseBuyStrategy
from buy_strategy.position import PositionInfo
from buy_strategy.registry import get_buy_strategy
from sell_strategy.base import BaseSellStrategy
from sell_strategy.registry import get_sell_strategy
from sell_strategy.metrics import PerformanceMetrics, TradeRecord


def _run_single_backtest_worker(
    buy_config: Dict[str, Any],
    sell_config: Dict[str, Any],
    buy_name: str,
    sell_name: str,
    kwargs: Dict[str, Any]
) -> PerformanceMetrics:
    """
    멀티프로세스용 워커 함수
    
    각 프로세스에서 독립적으로 백테스트를 실행합니다.
    전략 인스턴스는 프로세스 내에서 새로 생성 (피클링 문제 방지).
    """
    # 전략 인스턴스 생성 (각 프로세스 내에서)
    buy_strategy = get_buy_strategy(buy_name, buy_config)
    sell_strategy = get_sell_strategy(sell_name, sell_config)
    
    # 엔진 생성 및 백테스트 실행
    engine = UnifiedBacktestEngine(
        commission_fee=kwargs['commission_fee'],
        slippage_fee=kwargs['slippage_fee'],
        initial_capital=kwargs['initial_capital']
    )
    
    return engine.run_single_backtest(
        kwargs['data'],
        buy_strategy,
        sell_strategy,
        kwargs['use_reverse_signal']
    )


class UnifiedBacktestEngine:
    """
    통합 백테스트 엔진
    
    모든 Buy/Sell 전략 조합을 지원합니다.
    
    사용 예:
    ```python
    engine = UnifiedBacktestEngine(commission_fee=0.0005, slippage_fee=0.002)
    
    # 단일 조합 테스트
    result = engine.run_single_backtest(data, buy_strategy, sell_strategy)
    
    # Cross-combination 테스트
    results = engine.run_cross_combination_test(data, buy_strategies, sell_strategies)
    
    # 결과 정렬
    top_results = engine.get_top_results(results, sort_by='total_pnl', top_n=10)
    ```
    """
    
    def __init__(
        self,
        commission_fee: float = 0.0005,
        slippage_fee: float = 0.002,
        initial_capital: float = 1.0
    ):
        """
        Args:
            commission_fee: 수수료율 (예: 0.0005 = 0.05%)
            slippage_fee: 슬리피지율 (예: 0.002 = 0.2%)
            initial_capital: 초기 자본
        """
        self.commission_fee = commission_fee
        self.slippage_fee = slippage_fee
        self.initial_capital = initial_capital
    
    def run_single_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategy: BaseBuyStrategy,
        sell_strategy: BaseSellStrategy,
        use_reverse_signal: bool = True
    ) -> PerformanceMetrics:
        """
        단일 전략 조합 백테스트 실행
        
        Args:
            data: {ticker: ohlcv_df} 딕셔너리
            buy_strategy: 매수 전략 인스턴스
            sell_strategy: 청산 전략 인스턴스
            use_reverse_signal: 리버스 시그널 사용 여부
            
        Returns:
            PerformanceMetrics 인스턴스
        """
        all_trades: List[TradeRecord] = []
        
        for ticker, df in data.items():
            ticker_trades = self._run_ticker_backtest(
                ticker, df, buy_strategy, sell_strategy, use_reverse_signal
            )
            all_trades.extend(ticker_trades)
        
        return PerformanceMetrics.from_trades(
            trades=all_trades,
            initial_capital=self.initial_capital,
            buy_strategy_name=buy_strategy.name,
            buy_params=buy_strategy.config,
            sell_strategy_name=sell_strategy.name,
            sell_params=sell_strategy.config
        )
    
    def _run_ticker_backtest(
        self,
        ticker: str,
        df: pd.DataFrame,
        buy_strategy: BaseBuyStrategy,
        sell_strategy: BaseSellStrategy,
        use_reverse_signal: bool
    ) -> List[TradeRecord]:
        """
        단일 종목 백테스트 실행
        """
        trades: List[TradeRecord] = []
        
        # 1. 매수 신호 계산
        signals = buy_strategy.calculate_signals(df)
        direction = signals.get('direction', np.zeros(len(df)))
        reverse_to_short = signals.get('reverse_to_short', np.zeros(len(df), dtype=bool))
        reverse_to_long = signals.get('reverse_to_long', np.zeros(len(df), dtype=bool))
        
        # 2. 백테스트 시뮬레이션
        position: Optional[PositionInfo] = None
        cash = self.initial_capital
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            current_bar = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'date': str(row['date']),
                'is_last': i == len(df) - 1
            }
            
            # Buy 전략에서 계산한 지표를 Sell 전략에 전달 (Cross-combination 호환성)
            # 각 Sell 전략은 필요한 지표가 없을 때 자체적으로 기본값 처리
            for indicator_key in ['atr', 'rsi', 'adx', 'ema', 'stoch_k', 'bb_lower', 'bb_upper']:
                if signals.get(indicator_key) is not None:
                    indicator = signals[indicator_key]
                    if hasattr(indicator, '__len__') and len(indicator) > i:
                        val = indicator[i]
                        current_bar[indicator_key] = float(val) if not np.isnan(val) else None
                    else:
                        current_bar[indicator_key] = None
            
            # === 포지션 보유 중 ===
            if position is not None:
                # A. 리버스 시그널 확인 (VBT 전략 특화)
                if use_reverse_signal and hasattr(sell_strategy, 'check_reverse_signal'):
                    should_reverse, rev_reason, rev_price, new_dir = sell_strategy.check_reverse_signal(
                        position, reverse_to_short[i], reverse_to_long[i], current_bar
                    )
                    if should_reverse:
                        # 기존 포지션 청산
                        trade = sell_strategy.create_trade_record(
                            position, rev_price, i, current_bar['date'], rev_reason,
                            self.commission_fee, self.slippage_fee
                        )
                        trades.append(trade)
                        
                        # 반대 포지션으로 재진입
                        position = buy_strategy.create_position(
                            df, i, new_dir, signals, cash, cash, ticker
                        )
                        continue
                
                # B. 일반 청산 로직
                should_exit, exit_reason, exit_price = sell_strategy.should_exit(
                    position, current_bar, i
                )
                
                if should_exit:
                    trade = sell_strategy.create_trade_record(
                        position, exit_price, i, current_bar['date'], exit_reason,
                        self.commission_fee, self.slippage_fee
                    )
                    trades.append(trade)
                    
                    # 자본 업데이트
                    cash *= (1 + trade.pnl)
                    position = None
            
            # === 포지션 없음 - 신규 진입 ===
            else:
                if direction[i] != 0:
                    position = buy_strategy.create_position(
                        df, i, int(direction[i]), signals, cash, cash, ticker
                    )
        
        return trades
    
    def run_cross_combination_test(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategies: List[BaseBuyStrategy],
        sell_strategies: List[BaseSellStrategy],
        use_reverse_signal: bool = True
    ) -> List[PerformanceMetrics]:
        """
        모든 Buy/Sell 전략 조합 테스트
        
        예: VBTBuy + VBTSell, VBTBuy + TimeseriesSell,
            LowBBDUBuy + VBTSell, LowBBDUBuy + TimeseriesSell
        
        Args:
            data: {ticker: ohlcv_df} 딕셔너리
            buy_strategies: 매수 전략 리스트
            sell_strategies: 청산 전략 리스트
            use_reverse_signal: 리버스 시그널 사용 여부
            
        Returns:
            모든 조합의 PerformanceMetrics 리스트
        """
        results = []
        total = len(buy_strategies) * len(sell_strategies)
        count = 0
        
        for buy_strat in buy_strategies:
            for sell_strat in sell_strategies:
                count += 1
                result = self.run_single_backtest(
                    data, buy_strat, sell_strat, use_reverse_signal
                )
                results.append(result)
                
                # 진행 상황 출력
                if count % 10 == 0 or count == total:
                    print(f"Progress: {count}/{total} combinations tested")
        
        return results
    
    def run_cross_combination_test_parallel(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategies: List[BaseBuyStrategy],
        sell_strategies: List[BaseSellStrategy],
        use_reverse_signal: bool = True,
        n_workers: int = None
    ) -> List[PerformanceMetrics]:
        """
        모든 Buy/Sell 전략 조합 테스트 (멀티프로세스)
        
        Args:
            data: {ticker: ohlcv_df} 딕셔너리
            buy_strategies: 매수 전략 리스트
            sell_strategies: 청산 전략 리스트
            use_reverse_signal: 리버스 시그널 사용 여부
            n_workers: 워커 수 (None이면 CPU 코어 수)
            
        Returns:
            모든 조합의 PerformanceMetrics 리스트
        """
        from multiprocessing import Pool, cpu_count
        from functools import partial
        
        if n_workers is None:
            n_workers = min(cpu_count(), 8)  # 최대 8개 프로세스
        
        # 모든 조합 생성
        combinations = [
            (buy_strat, sell_strat)
            for buy_strat in buy_strategies
            for sell_strat in sell_strategies
        ]
        
        total = len(combinations)
        print(f"Starting parallel backtest with {n_workers} workers for {total} combinations...")
        
        # 워커 함수에서 사용할 데이터와 설정 준비
        worker_kwargs = {
            'data': data,
            'use_reverse_signal': use_reverse_signal,
            'commission_fee': self.commission_fee,
            'slippage_fee': self.slippage_fee,
            'initial_capital': self.initial_capital
        }
        
        # 멀티프로세스 실행
        results = []
        with Pool(processes=n_workers) as pool:
            async_results = []
            for i, (buy_strat, sell_strat) in enumerate(combinations):
                async_result = pool.apply_async(
                    _run_single_backtest_worker,
                    args=(buy_strat.config, sell_strat.config, 
                          buy_strat.name, sell_strat.name, worker_kwargs)
                )
                async_results.append(async_result)
            
            # 결과 수집
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=300)  # 5분 타임아웃
                    results.append(result)
                except Exception as e:
                    print(f"Error in combination {i}: {e}")
                
                # 진행 상황 출력
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(f"Progress: {i + 1}/{total} combinations completed")
        
        return results
    
    @staticmethod
    def get_top_results(
        results: List[PerformanceMetrics],
        sort_by: str = 'total_pnl',
        top_n: int = 10,
        ascending: bool = False,
        min_trades: int = 0
    ) -> List[PerformanceMetrics]:
        """
        결과를 정렬하여 상위 N개 반환
        
        Args:
            results: PerformanceMetrics 리스트
            sort_by: 정렬 기준 ('total_pnl', 'win_ratio', 'mdd', 'sharpe_ratio')
            top_n: 반환할 개수
            ascending: 오름차순 정렬 여부
            min_trades: 최소 거래 횟수 필터
            
        Returns:
            상위 N개의 PerformanceMetrics 리스트
        """
        # 최소 거래 횟수 필터
        filtered = [r for r in results if r.trade_count >= min_trades]
        
        if not filtered:
            return []
        
        # MDD는 높을수록 좋음 (1.0 = 낙폭 없음)
        if sort_by == 'mdd':
            reverse = not ascending  # mdd는 높을수록 좋으므로 reverse
        else:
            reverse = not ascending
        
        sorted_results = sorted(
            filtered,
            key=lambda x: getattr(x, sort_by, 0),
            reverse=reverse
        )
        
        return sorted_results[:top_n]
    
    @staticmethod
    def print_results(results: List[PerformanceMetrics], top_n: int = 10) -> None:
        """
        결과 출력
        
        Args:
            results: PerformanceMetrics 리스트
            top_n: 출력할 개수
        """
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(results))} RESULTS")
        print(f"{'='*80}")
        
        for i, r in enumerate(results[:top_n], 1):
            print(f"\n[{i}] {r.summary_string()}")
            print(f"    Buy Params: {r.buy_params}")
            print(f"    Sell Params: {r.sell_params}")
    
    @staticmethod
    def results_to_dataframe(results: List[PerformanceMetrics]) -> pd.DataFrame:
        """
        결과를 DataFrame으로 변환
        
        Args:
            results: PerformanceMetrics 리스트
            
        Returns:
            결과 DataFrame
        """
        records = [r.to_dict() for r in results]
        return pd.DataFrame(records)
