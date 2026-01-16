"""
Position 정보 데이터 클래스

매수 전략에서 생성된 포지션 정보를 담는 데이터 클래스입니다.
Sell Strategy로 전달되어 청산 로직에서 사용됩니다.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class PositionInfo:
    """
    포지션 정보를 담는 데이터 클래스
    
    Buy Strategy에서 생성되어 Sell Strategy로 전달됩니다.
    진입 가격, 시점, 이유 등 모든 포지션 관련 정보를 추적합니다.
    """
    
    # === 기본 정보 ===
    ticker: str                         # 종목 코드 (예: 'KRW-BTC')
    direction: int                      # 포지션 방향: 1 (Long), -1 (Short), 0 (None)
    
    # === 진입 정보 ===
    entry_price: float                  # 진입 평균가
    entry_idx: int                      # 진입 시점 인덱스 (봉 번호)
    entry_date: str                     # 진입 일자 (문자열)
    entry_reason: str                   # 진입 사유 (예: 'vbt_long_breakout', 'bb_oversold_rebound')
    entry_conditions: Dict[str, Any]    # 진입 시점의 조건 상세 (예: {'rsi': 28, 'ema_cross': True, 'k': 0.5})
    
    # === 수량 및 자금 관리 ===
    quantity: float                     # 보유 수량
    invested_amount: float              # 투입 금액
    max_investment_ratio: float         # 최대 투자 비율 (전체 자산 대비, 예: 0.1 = 10%)
    current_allocation_ratio: float     # 현재 할당 비율 (전체 자산 대비)
    
    # === 현재 상태 (업데이트 가능) ===
    current_price: float = 0.0          # 현재가
    unrealized_pnl: float = 0.0         # 미실현 손익률
    max_profit: float = 0.0             # 보유 중 최고 수익률 (트레일링 스탑용)
    max_drawdown: float = 0.0           # 보유 중 최대 하락률
    bars_held: int = 0                  # 보유 기간 (봉 개수)
    
    # === 메타데이터 ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # 전략별 추가 정보
    
    def update_current_state(self, current_price: float, current_idx: int) -> None:
        """
        현재가 업데이트 및 미실현 손익 계산
        
        Args:
            current_price: 현재 가격
            current_idx: 현재 봉 인덱스
        """
        self.current_price = current_price
        self.bars_held = current_idx - self.entry_idx
        
        # 미실현 손익 계산
        if self.direction == 1:  # Long
            self.unrealized_pnl = (current_price / self.entry_price) - 1.0
        elif self.direction == -1:  # Short
            self.unrealized_pnl = 1.0 - (current_price / self.entry_price)
        else:
            self.unrealized_pnl = 0.0
        
        # 최고 수익 / 최대 낙폭 추적
        self.max_profit = max(self.max_profit, self.unrealized_pnl)
        self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
    
    def get_position_value(self) -> float:
        """현재 포지션 가치 계산"""
        return self.quantity * self.current_price if self.current_price > 0 else self.invested_amount
    
    def is_profitable(self) -> bool:
        """수익 상태인지 확인"""
        return self.unrealized_pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화용)"""
        return {
            'ticker': self.ticker,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_idx': self.entry_idx,
            'entry_date': self.entry_date,
            'entry_reason': self.entry_reason,
            'entry_conditions': self.entry_conditions,
            'quantity': self.quantity,
            'invested_amount': self.invested_amount,
            'max_investment_ratio': self.max_investment_ratio,
            'current_allocation_ratio': self.current_allocation_ratio,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'max_profit': self.max_profit,
            'max_drawdown': self.max_drawdown,
            'bars_held': self.bars_held,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionInfo':
        """딕셔너리에서 PositionInfo 생성"""
        return cls(
            ticker=data['ticker'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            entry_idx=data['entry_idx'],
            entry_date=data['entry_date'],
            entry_reason=data['entry_reason'],
            entry_conditions=data.get('entry_conditions', {}),
            quantity=data['quantity'],
            invested_amount=data['invested_amount'],
            max_investment_ratio=data['max_investment_ratio'],
            current_allocation_ratio=data['current_allocation_ratio'],
            current_price=data.get('current_price', 0.0),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            max_profit=data.get('max_profit', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            bars_held=data.get('bars_held', 0),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        direction_str = 'Long' if self.direction == 1 else ('Short' if self.direction == -1 else 'None')
        return (
            f"PositionInfo({self.ticker}, {direction_str}, "
            f"entry={self.entry_price:.4f}, pnl={self.unrealized_pnl:.2%}, "
            f"reason={self.entry_reason})"
        )
