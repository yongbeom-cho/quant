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
        initial_capital=kwargs['initial_capital'],
        max_positions=kwargs.get('max_positions', 1)
    )
    
    return engine.run_single_backtest(
        kwargs['data'],
        buy_strategy,
        sell_strategy,
        kwargs['use_reverse_signal'],
        kwargs.get('is_timeseries_backtest', False)
    )


def _run_single_backtest_worker_wrapper(args: tuple) -> PerformanceMetrics:
    """imap_unordered용 wrapper (단일 튜플 인자를 언패킹)"""
    buy_config, sell_config, buy_name, sell_name, kwargs = args
    return _run_single_backtest_worker(buy_config, sell_config, buy_name, sell_name, kwargs)


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
        initial_capital: float = 1.0,
        max_positions: int = 1
    ):
        """
        Args:
            commission_fee: 수수료율 (예: 0.0005 = 0.05%)
            slippage_fee: 슬리피지율 (예: 0.002 = 0.2%)
            initial_capital: 초기 자본
            max_positions: 동시에 보유 가능한 최대 포지션 수 (기본값: 1)
        """
        self.commission_fee = commission_fee
        self.slippage_fee = slippage_fee
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.entry_capital_ratio = 1.0 / max_positions
    
    def run_single_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategy: BaseBuyStrategy,
        sell_strategy: BaseSellStrategy,
        use_reverse_signal: bool = True,
        is_timeseries_backtest: bool = False
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
        
        if is_timeseries_backtest:
            return self._run_timeseries_backtest(
                data, buy_strategy, sell_strategy, use_reverse_signal
            )
        
        ticker_mdds: List[float] = []
        for ticker, df in data.items():
            ticker_trades, ticker_mdd = self._run_ticker_backtest(
                ticker, df, buy_strategy, sell_strategy, use_reverse_signal
            )
            all_trades.extend(ticker_trades)
            ticker_mdds.append(ticker_mdd)
        
        # 모든 ticker의 MDD 중 최소값 (가장 낮은 값 = 최악의 MDD)
        min_ticker_mdd = min(ticker_mdds) if ticker_mdds else 1.0
        
        return PerformanceMetrics.from_trades(
            trades=all_trades,
            initial_capital=self.initial_capital,
            buy_strategy_name=buy_strategy.name,
            buy_params=buy_strategy.config,
            sell_strategy_name=sell_strategy.name,
            sell_params=sell_strategy.config,
            mdd=min_ticker_mdd
        )


    def _run_timeseries_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategy: BaseBuyStrategy,
        sell_strategy: BaseSellStrategy,
        use_reverse_signal: bool
    ) -> PerformanceMetrics:
        """
        data에 있는 모든 ticker들의 시간들을 모두 모아 가장 과거부터 최신으로 순회하며 backtest를 실행한다.
        1. 먼저 각 ticker별로 매수 signal을 모두 구하고, 
        2. 과거부터 최신으로 순회하면서 모든 ticker들의 signal을 확인한다. 
        3. 이때, len(active_positions) < self.max_positions: 일때만 포지션에 진입할 수 있다.
        4. 같은 시점에 여러개의 ticker에서 동시에 signal이 발생 할 수 있는데, 이때는 거래대금 순으로 매수한다.
        5. mdd도 전체 자산에 대해서 계산해야한다. (각 시간마다 total_asset를 계산하고, 최고 total_asset을 기록하고 있어서 매 시간마다 mdd를 계산해야한다.)
        """
        all_trades: List[TradeRecord] = []
        
        # 1. 각 ticker별로 매수 signal 계산
        ticker_signals: Dict[str, Dict[str, Any]] = {}
        ticker_dfs: Dict[str, pd.DataFrame] = {}
        
        for ticker, df in data.items():
            signals = buy_strategy.calculate_signals(df)
            ticker_signals[ticker] = signals
            ticker_dfs[ticker] = df.copy()
        
        # 2. 모든 ticker의 모든 시간을 모아서 정렬 (과거부터 최신 순)
        time_events: List[Tuple[str, int, pd.Series]] = []  # (ticker, row_index, row)
        
        for ticker, df in ticker_dfs.items():
            for i in range(len(df)):
                row = df.iloc[i]
                time_events.append((ticker, i, row))
        
        # date 기준으로 정렬 (과거부터 최신 순)
        time_events.sort(key=lambda x: x[2]['date'])
        
        # 3. 같은 date로 그룹화하여 처리 (같은 시점의 이벤트들을 함께 처리)
        from collections import defaultdict
        events_by_date = defaultdict(list)
        for ticker, row_idx, row in time_events:
            date_key = str(row['date'])
            events_by_date[date_key].append((ticker, row_idx, row))
        
        # date 순으로 정렬된 키 리스트
        sorted_dates = sorted(events_by_date.keys())
        
        # 4. 백테스트 시뮬레이션
        active_positions: List[PositionInfo] = []  # 현재 보유 중인 포지션 리스트
        cash = self.initial_capital
        total_asset = self.initial_capital
        max_total_asset = self.initial_capital  # MDD 계산용 최고 자산
        mdd = 1.0  # MDD (1.0 = 낙폭 없음)
        
        # 각 시점마다 모든 ticker의 현재 가격을 저장 (전체 자산 계산용)
        ticker_prices: Dict[str, float] = {}
        
        for date_key in sorted_dates:
            same_time_events = events_by_date[date_key]
            
            # 같은 시점의 모든 ticker 가격 업데이트
            for ticker, row_idx, row in same_time_events:
                ticker_prices[ticker] = row['close']
            
            # 같은 시점의 이벤트들에 대해 먼저 청산 처리
            positions_to_remove = []
            
            for ticker, row_idx, row in same_time_events:
                current_bar = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'date': str(row['date']),
                    'is_last': row_idx == len(ticker_dfs[ticker]) - 1
                }
                
                signals = ticker_signals[ticker]
                for indicator_key in ['atr', 'rsi', 'adx', 'ema', 'stoch_k', 'bb_lower', 'bb_upper']:
                    if signals.get(indicator_key) is not None:
                        indicator = signals[indicator_key]
                        if hasattr(indicator, '__len__') and len(indicator) > row_idx:
                            val = indicator[row_idx]
                            current_bar[indicator_key] = float(val) if not np.isnan(val) else None
                        else:
                            current_bar[indicator_key] = None
                
                # 해당 ticker의 포지션들에 대해 청산 확인
                for pos_idx, position in enumerate(active_positions):
                    if position.ticker != ticker:
                        continue
                    
                    # 리버스 시그널 확인
                    if use_reverse_signal and hasattr(sell_strategy, 'check_reverse_signal'):
                        reverse_to_short = signals.get('reverse_to_short', np.zeros(len(ticker_dfs[ticker]), dtype=bool))
                        reverse_to_long = signals.get('reverse_to_long', np.zeros(len(ticker_dfs[ticker]), dtype=bool))
                        
                        should_reverse, rev_reason, rev_price, new_dir = sell_strategy.check_reverse_signal(
                            position, reverse_to_short[row_idx], reverse_to_long[row_idx], current_bar
                        )
                        if should_reverse:
                            trade = sell_strategy.create_trade_record(
                                position, rev_price, row_idx, current_bar['date'], rev_reason,
                                self.commission_fee, self.slippage_fee
                            )
                            all_trades.append(trade)
                            if pos_idx not in positions_to_remove:
                                positions_to_remove.append(pos_idx)
                            cash += position.invested_amount * (1 + trade.pnl)
                            continue
                    
                    # 일반 청산 조건 확인
                    should_exit, exit_reason, exit_price = sell_strategy.should_exit(
                        position, current_bar, row_idx
                    )
                    
                    if should_exit:
                        trade = sell_strategy.create_trade_record(
                            position, exit_price, row_idx, current_bar['date'], exit_reason,
                            self.commission_fee, self.slippage_fee
                        )
                        all_trades.append(trade)
                        if pos_idx not in positions_to_remove:
                            positions_to_remove.append(pos_idx)
                        cash += position.invested_amount * (1 + trade.pnl)
            
            # 청산된 포지션 제거
            for pos_idx in sorted(positions_to_remove, reverse=True):
                active_positions.pop(pos_idx)
            
            # 전체 자산 재계산 (모든 포지션의 현재 가격 기준)
            total_position_value = 0.0
            for pos in active_positions:
                # 해당 ticker의 현재 시점 가격 사용 (같은 시점에 없으면 마지막 알려진 가격 사용)
                pos_price = ticker_prices.get(pos.ticker, pos.entry_price)
                pos_value = pos.invested_amount * (pos_price / pos.entry_price)
                total_position_value += pos_value
            
            total_asset = cash + total_position_value
            
            # MDD 계산
            if total_asset > max_total_asset:
                max_total_asset = total_asset
            current_dd = total_asset / max_total_asset if max_total_asset > 0 else 1.0
            mdd = min(mdd, current_dd)
            
            # 같은 시점의 신규 진입 후보 수집 (거래대금 순으로 정렬)
            entry_candidates = []
            for ticker, row_idx, row in same_time_events:
                signals = ticker_signals[ticker]
                direction = signals.get('direction', np.zeros(len(ticker_dfs[ticker])))
                
                if direction[row_idx] != 0:
                    # 거래대금 계산
                    tx_amount = row['volume'] * row['close']
                    entry_candidates.append((ticker, row_idx, row, tx_amount, direction[row_idx]))
            
            # 거래대금 순으로 정렬 (내림차순)
            entry_candidates.sort(key=lambda x: x[3], reverse=True)
            
            # 신규 진입 처리 (거래대금 순으로)
            for ticker, row_idx, row, tx_amount, signal_dir in entry_candidates:
                if len(active_positions) >= self.max_positions:
                    break
                
                entry_amount = total_asset * self.entry_capital_ratio
                if cash >= entry_amount and entry_amount > 0:
                    signals = ticker_signals[ticker]
                    position = buy_strategy.create_position(
                        ticker_dfs[ticker], row_idx, int(signal_dir), signals,
                        entry_amount, total_asset, ticker
                    )
                    if position is not None:
                        active_positions.append(position)
                        cash -= position.invested_amount
        
        return PerformanceMetrics.from_trades(
            trades=all_trades,
            initial_capital=self.initial_capital,
            buy_strategy_name=buy_strategy.name,
            buy_params=buy_strategy.config,
            sell_strategy_name=sell_strategy.name,
            sell_params=sell_strategy.config,
            mdd=mdd
        )
    
    def _run_ticker_backtest(
        self,
        ticker: str,
        df: pd.DataFrame,
        buy_strategy: BaseBuyStrategy,
        sell_strategy: BaseSellStrategy,
        use_reverse_signal: bool
    ) -> Tuple[List[TradeRecord], float]:
        """
        단일 종목 백테스트 메인 루프 (다중 포지션 지원)
        
        시뮬레이션 과정:
        1. 매수 전략을 호출하여 전체 데이터에 대한 지표 및 매수 신호를 미리 계산합니다.
           (매수 신호: direction=1(Long), -1(Short), 0(None))
        2. 봉(bar)을 하나씩 순회하며 각 포지션에 대해 독립적으로 청산 조건을 확인합니다.
        3. 최대 포지션 수(max_positions) 미만일 경우에만 새 진입을 허용합니다.
        
        Returns:
            (trades, ticker_mdd) 튜플
        """
        trades: List[TradeRecord] = []
        
        # 1. 매수 신호 계산 (벡터 연산으로 빠르게 수행)
        signals = buy_strategy.calculate_signals(df)
        direction = signals.get('direction', np.zeros(len(df)))
        # VBT(변동성 돌파) 전략 등을 위한 리버스 시그널 (Long->Short 스위칭 등)
        reverse_to_short = signals.get('reverse_to_short', np.zeros(len(df), dtype=bool))
        reverse_to_long = signals.get('reverse_to_long', np.zeros(len(df), dtype=bool))
        
        # 2. 백테스트 시뮬레이션 (다중 포지션 지원)
        active_positions: List[PositionInfo] = []  # 현재 보유 중인 포지션 리스트
        cash = self.initial_capital
        total_asset = self.initial_capital  # 전체 자산 (현금 + 평가금액)
        max_total_asset = self.initial_capital  # MDD 계산용 최고 자산
        mdd = 1.0  # MDD (1.0 = 낙폭 없음)
        
        # 첫 번째 봉은 초기화용으로 건너뛰고 1번 인덱스부터 시작
        for i in range(1, len(df)):
            row = df.iloc[i]
            # 현재 봉의 핵심 데이터를 딕셔너리로 캡슐화 (Sell 전략에 전달용)
            current_bar = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'date': str(row['date']),
                'is_last': i == len(df) - 1
            }
            
            # Buy 전략에서 계산한 지표들을 Sell 전략에서도 쓸 수 있게 전달합니다.
            # (예: ATR 기반 트레일링 스탑을 할 때, Buy 단계에서 계산된 ATR을 사용)
            for indicator_key in ['atr', 'rsi', 'adx', 'ema', 'stoch_k', 'bb_lower', 'bb_upper']:
                if signals.get(indicator_key) is not None:
                    indicator = signals[indicator_key]
                    if hasattr(indicator, '__len__') and len(indicator) > i:
                        val = indicator[i]
                        current_bar[indicator_key] = float(val) if not np.isnan(val) else None
                    else:
                        current_bar[indicator_key] = None
            
            # === 각 활성 포지션에 대해 독립적으로 청산 로직 실행 ===
            positions_to_remove = []  # 청산할 포지션 인덱스 목록
            
            for pos_idx, position in enumerate(active_positions):
                # A. 리버스 시그널 확인 (포지션을 즉시 반대로 스위칭해야 하는 경우)
                if use_reverse_signal and hasattr(sell_strategy, 'check_reverse_signal'):
                    should_reverse, rev_reason, rev_price, new_dir = sell_strategy.check_reverse_signal(
                        position, reverse_to_short[i], reverse_to_long[i], current_bar
                    )
                    if should_reverse:
                        # 기존 포지션 청산 기록 생성
                        trade = sell_strategy.create_trade_record(
                            position, rev_price, i, current_bar['date'], rev_reason,
                            self.commission_fee, self.slippage_fee
                        )
                        trades.append(trade)
                        positions_to_remove.append(pos_idx)
                        
                        # 실현 손익을 현금에 반영
                        cash += position.invested_amount * (1 + trade.pnl)
                        
                        # 반대 방향 포지션으로 즉시 진입 (자금 여유가 있다면)
                        entry_amount = total_asset * self.entry_capital_ratio
                        if cash >= entry_amount:
                            new_position = buy_strategy.create_position(
                                df, i, new_dir, signals, entry_amount, total_asset, ticker
                            )
                            if new_position is not None:
                                active_positions.append(new_position)
                                cash -= new_position.invested_amount
                        continue
                
                # B. 일반 청산 조건 확인 (익절, 손절, 타임아웃 등)
                should_exit, exit_reason, exit_price = sell_strategy.should_exit(
                    position, current_bar, i
                )
                
                if should_exit:
                    # 실제 체결가 계산 및 수수료/슬리피지 반영
                    trade = sell_strategy.create_trade_record(
                        position, exit_price, i, current_bar['date'], exit_reason,
                        self.commission_fee, self.slippage_fee
                    )
                    trades.append(trade)
                    positions_to_remove.append(pos_idx)
                    
                    # 실현 손익을 현금에 반영
                    cash += position.invested_amount * (1 + trade.pnl)
            
            # 청산된 포지션 제거 (인덱스 역순으로 제거해야 순서가 안 깨짐)
            for pos_idx in sorted(positions_to_remove, reverse=True):
                active_positions.pop(pos_idx)
            
            # 전체 자산 재계산 (현금 + 보유 포지션 평가금액)
            position_value = sum(pos.invested_amount * (current_bar['close'] / pos.entry_price) for pos in active_positions)
            total_asset = cash + position_value
            
            # MDD 계산
            if total_asset > max_total_asset:
                max_total_asset = total_asset
            current_dd = total_asset / max_total_asset if max_total_asset > 0 else 1.0
            mdd = min(mdd, current_dd)
            
            # === 신규 진입 로직 (포지션 개수가 max_positions 미만일 때만) ===
            if len(active_positions) < self.max_positions:
                # 매수 신호가 발생했고 진입 가능한 자금이 있는 경우
                if direction[i] != 0:
                    entry_amount = total_asset * self.entry_capital_ratio
                    if cash >= entry_amount and entry_amount > 0:
                        position = buy_strategy.create_position(
                            df, i, int(direction[i]), signals, entry_amount, total_asset, ticker
                        )
                        if position is not None:
                            active_positions.append(position)
                            cash -= position.invested_amount
        
        return trades, mdd
    
    def run_cross_combination_test(
        self,
        data: Dict[str, pd.DataFrame],
        buy_strategies: List[BaseBuyStrategy],
        sell_strategies: List[BaseSellStrategy],
        use_reverse_signal: bool = True,
        is_timeseries_backtest: bool = False
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
            is_timeseries_backtest: 타임시리즈 백테스트 모드 사용 여부
            
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
                    data, buy_strat, sell_strat, use_reverse_signal, is_timeseries_backtest
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
        is_timeseries_backtest: bool = False,
        n_workers: int = None,
        checkpoint_interval: int = 100,
        checkpoint_file: str = None
    ) -> List[PerformanceMetrics]:
        """
        모든 Buy/Sell 전략 조합 테스트 (멀티프로세스)
        
        Args:
            data: {ticker: ohlcv_df} 딕셔너리
            buy_strategies: 매수 전략 리스트
            sell_strategies: 청산 전략 리스트
            use_reverse_signal: 리버스 시그널 사용 여부
            is_timeseries_backtest: 타임시리즈 백테스트 모드 사용 여부
            n_workers: 워커 수 (None이면 CPU 코어 수)
            checkpoint_interval: 중간 저장 주기 (기본 100)
            checkpoint_file: 중간 결과 저장 파일 경로 (None이면 저장 안함)
            
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
        if checkpoint_file:
            print(f"Checkpoint file: {checkpoint_file} (saving every {checkpoint_interval} results)")
        
        # 워커 함수에서 사용할 데이터와 설정 준비 (다중 포지션 설정 포함)
        worker_kwargs = {
            'data': data,
            'use_reverse_signal': use_reverse_signal,
            'is_timeseries_backtest': is_timeseries_backtest,
            'commission_fee': self.commission_fee,
            'slippage_fee': self.slippage_fee,
            'initial_capital': self.initial_capital,
            'max_positions': self.max_positions
        }
        
        # 체크포인트 파일 초기화 (헤더 작성)
        header_written = False
        if checkpoint_file:
            # 파일이 있으면 삭제하고 새로 시작
            import os
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        
        # 멀티프로세스 실행
        results = []
        pending_results = []  # 저장 대기 중인 결과
        saved_count = 0  # 저장된 결과 수 추적
        
        # 워커에 전달할 인자들을 생성 (제너레이터로 메모리 절약)
        def generate_worker_args():
            for buy_strat, sell_strat in combinations:
                yield (buy_strat.config, sell_strat.config,
                       buy_strat.name, sell_strat.name, worker_kwargs)
        
        with Pool(processes=n_workers) as pool:
            # imap_unordered: 결과가 준비되는 대로 즉시 처리 (메모리 효율적)
            for i, result in enumerate(pool.imap_unordered(
                _run_single_backtest_worker_wrapper,
                generate_worker_args(),
                chunksize=10  # 한 번에 워커에 전달할 작업 수
            )):
                if checkpoint_file:
                    pending_results.append(result)
                else:
                    results.append(result)
                
                # 체크포인트 저장 (N개마다 또는 마지막)
                if checkpoint_file and pending_results:
                    if len(pending_results) >= checkpoint_interval or (i + 1) == total:
                        df = self.results_to_dataframe(pending_results)
                        # 첫 저장 시 헤더 포함, 이후는 append
                        if not header_written:
                            df.to_csv(checkpoint_file, index=False, mode='w')
                            header_written = True
                        else:
                            df.to_csv(checkpoint_file, index=False, mode='a', header=False)
                        
                        saved_count += len(pending_results)
                        pending_results = []  # 저장 완료 후 비우기 (메모리 해제)
                
                # 진행 상황 출력
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(f"Progress: {i + 1}/{total} combinations completed")
        
        # checkpoint 모드에서는 파일에 저장된 개수 정보만 반환
        if checkpoint_file:
            print(f"Total {saved_count} results saved to: {checkpoint_file}")
            return [None] * saved_count  # 개수 정보만 전달
        
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
