"""
성과 지표 데이터 클래스

백테스트 결과를 담는 데이터 클래스들입니다.
거래 기록(TradeRecord)과 전체 성과 요약(PerformanceMetrics)을 제공합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TradeRecord:
    """
    개별 거래 기록
    
    하나의 진입-청산 사이클에 대한 모든 정보를 담습니다.
    """
    # 종목 정보
    ticker: str                         # 종목 코드
    direction: int                      # 1: Long, -1: Short
    
    # 가격 정보
    entry_price: float                  # 진입 가격
    exit_price: float                   # 청산 가격
    
    # 시간 정보
    entry_date: str                     # 진입 일자
    exit_date: str                      # 청산 일자
    entry_idx: int                      # 진입 봉 인덱스
    exit_idx: int                       # 청산 봉 인덱스
    holding_bars: int                   # 보유 기간 (봉 개수)
    
    # 거래 사유
    entry_reason: str                   # 진입 사유 (예: 'vbt_long_breakout')
    exit_reason: str                    # 청산 사유 (예: 'stop_loss', 'take_profit', 'timeout')
    
    # 손익 정보
    pnl: float                          # 실현 손익률
    gross_pnl: float = 0.0              # 수수료 제외 손익률
    commission_paid: float = 0.0        # 지불한 수수료
    
    # 추가 정보
    entry_conditions: Dict[str, Any] = field(default_factory=dict)  # 진입 시점 조건
    metadata: Dict[str, Any] = field(default_factory=dict)          # 추가 메타데이터
    
    def is_win(self) -> bool:
        """수익 거래인지 확인"""
        return self.pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'ticker': self.ticker,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_idx': self.entry_idx,
            'exit_idx': self.exit_idx,
            'holding_bars': self.holding_bars,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'gross_pnl': self.gross_pnl,
            'commission_paid': self.commission_paid,
            'entry_conditions': self.entry_conditions,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        direction_str = 'Long' if self.direction == 1 else 'Short'
        result_str = 'WIN' if self.is_win() else 'LOSS'
        return (
            f"TradeRecord({self.ticker}, {direction_str}, "
            f"pnl={self.pnl:.2%}, {result_str}, {self.exit_reason})"
        )


@dataclass
class PerformanceMetrics:
    """
    백테스트 성과 요약
    
    전체 백테스트 결과에 대한 통계 및 성과 지표를 담습니다.
    """
    # === 손익 지표 ===
    total_pnl: float                    # 총 손익률 (최종 자산 / 초기 자산 - 1)
    cumulative_return: float            # 누적 수익률 (최종 자산값)
    
    # === 승패 통계 ===
    trade_count: int                    # 총 거래 횟수
    win_count: int                      # 승리 횟수
    lose_count: int                     # 패배 횟수
    win_ratio: float                    # 승률 (0~1)
    
    # === 리스크 지표 ===
    mdd: float                          # 최대 낙폭 (0~1, 1이면 낙폭 없음)
    max_drawdown_pct: float = 0.0       # 최대 낙폭 퍼센트 (0~100)
    
    # === 추가 성과 지표 ===
    avg_profit_per_trade: float = 0.0   # 거래당 평균 수익률
    avg_win: float = 0.0                # 평균 수익 거래의 수익률
    avg_loss: float = 0.0               # 평균 손실 거래의 손실률
    profit_factor: float = 0.0          # 총 이익 / 총 손실
    avg_holding_bars: float = 0.0       # 평균 보유 기간
    sharpe_ratio: float = 0.0           # 샤프 비율
    
    # === 거래 기록 ===
    trade_history: List[TradeRecord] = field(default_factory=list)
    
    # === 전략 정보 ===
    buy_strategy_name: str = ""
    buy_params: Dict[str, Any] = field(default_factory=dict)
    sell_strategy_name: str = ""
    sell_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_trades(
        cls,
        trades: List[TradeRecord],
        initial_capital: float = 1.0,
        buy_strategy_name: str = "",
        buy_params: Optional[Dict[str, Any]] = None,
        sell_strategy_name: str = "",
        sell_params: Optional[Dict[str, Any]] = None,
        mdd: float = 1.0
    ) -> 'PerformanceMetrics':
        """
        거래 기록에서 성과 지표 계산
        
        Args:
            trades: 거래 기록 리스트
            initial_capital: 초기 자본
            buy_strategy_name: 매수 전략 이름
            buy_params: 매수 전략 파라미터
            sell_strategy_name: 청산 전략 이름
            sell_params: 청산 전략 파라미터
            
        Returns:
            PerformanceMetrics 인스턴스
        """
        if not trades:
            return cls(
                total_pnl=0.0,
                cumulative_return=initial_capital,
                trade_count=0,
                win_count=0,
                lose_count=0,
                win_ratio=0.0,
                mdd=1.0,
                trade_history=[],
                buy_strategy_name=buy_strategy_name,
                buy_params=buy_params or {},
                sell_strategy_name=sell_strategy_name,
                sell_params=sell_params or {}
            )
        
        # 기본 통계
        trade_count = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_count = len(wins)
        lose_count = len(losses)
        win_ratio = win_count / trade_count if trade_count > 0 else 0.0
        
        # 누적 수익률 및 MDD 계산
        cum_capital = initial_capital
        max_capital = initial_capital
        
        pnl_list = [t.pnl for t in trades]
        
        
        # total_pnl 계산 (ticker_mdds를 사용해도 누적 수익률은 거래 기록 기반으로 계산)
        for pnl in pnl_list:
            cum_capital *= (1 + pnl)
        
        total_pnl = (cum_capital / initial_capital) - 1.0
        max_drawdown_pct = (1.0 - mdd) * 100
        
        # 추가 지표 계산
        avg_profit_per_trade = np.mean(pnl_list) if pnl_list else 0.0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        
        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_holding_bars = np.mean([t.holding_bars for t in trades]) if trades else 0.0
        
        # 샤프 비율 (거래당 수익률의 평균 / 표준편차)
        if len(pnl_list) > 1:
            std_pnl = np.std(pnl_list)
            sharpe_ratio = (np.mean(pnl_list) / std_pnl) if std_pnl > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return cls(
            total_pnl=total_pnl,
            cumulative_return=cum_capital,
            trade_count=trade_count,
            win_count=win_count,
            lose_count=lose_count,
            win_ratio=win_ratio,
            mdd=mdd,
            max_drawdown_pct=max_drawdown_pct,
            avg_profit_per_trade=avg_profit_per_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_bars=avg_holding_bars,
            sharpe_ratio=sharpe_ratio,
            trade_history=trades,
            buy_strategy_name=buy_strategy_name,
            buy_params=buy_params or {},
            sell_strategy_name=sell_strategy_name,
            sell_params=sell_params or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (결과 저장/정렬용)"""
        return {
            'total_pnl': self.total_pnl,
            'cumulative_return': self.cumulative_return,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'lose_count': self.lose_count,
            'win_ratio': self.win_ratio,
            'mdd': self.mdd,
            'max_drawdown_pct': self.max_drawdown_pct,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_holding_bars': self.avg_holding_bars,
            'sharpe_ratio': self.sharpe_ratio,
            'buy_strategy_name': self.buy_strategy_name,
            'buy_params': self.buy_params,
            'sell_strategy_name': self.sell_strategy_name,
            'sell_params': self.sell_params
        }
    
    def summary_string(self) -> str:
        """요약 문자열 반환"""
        return (
            f"PnL: {self.total_pnl:.4f} | MDD: {self.mdd:.4f} | "
            f"Win: {self.win_ratio:.2%} | Trades: {self.trade_count} "
            f"(W:{self.win_count} L:{self.lose_count}) | "
            f"Buy: {self.buy_strategy_name} | Sell: {self.sell_strategy_name}"
        )
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(pnl={self.total_pnl:.2%}, "
            f"win_ratio={self.win_ratio:.2%}, mdd={self.mdd:.4f}, "
            f"trades={self.trade_count})"
        )
