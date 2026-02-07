"""
Sell Strategy 레지스트리

전략 이름으로 전략 클래스를 찾아 인스턴스를 생성합니다.
새 전략을 추가할 때 여기에 등록해야 합니다.
"""

from typing import Dict, Type, List, Any
from .base import BaseSellStrategy


def _get_strategy_registry() -> Dict[str, Type[BaseSellStrategy]]:
    """
    전략 레지스트리 반환 (순환 임포트 방지를 위해 함수 내부에서 임포트)
    
    Note: _quick 버전들은 같은 클래스를 사용하며 파라미터만 다릅니다.
    Config의 strategy_name과 동일한 키가 Registry에 있어야 합니다.
    """
    from .vbt_sell_strategy import VBTSellStrategy
    from .vbt_enhanced_sell_strategy import VBTEnhancedSellStrategy
    from .timeseries_sell_strategy import TimeseriesSellStrategy
    from .simple_ratio_sell_strategy import SimpleRatioSellStrategy
    
    # 5개 신규 청산 전략 임포트
    from .adx_exit_strategy import ADXExitStrategy
    from .bb_squeeze_exit_strategy import BBSqueezeExitStrategy
    from .macd_exit_strategy import MACDExitStrategy
    from .chandelier_exit_strategy import ChandelierExitStrategy
    from .vwap_exit_strategy import VWAPExitStrategy
    
    return {
        # VBT 청산 전략
        'bailout_sell': VBTSellStrategy,
        'bailout_sell_quick': VBTSellStrategy,
        'vbt_sell': VBTSellStrategy,
        
        # VBT Enhanced 청산 전략 (ATR 트레일링 스탑)
        'bailout_sell_enhanced': VBTEnhancedSellStrategy,
        'bailout_sell_enhanced_quick': VBTEnhancedSellStrategy,
        
        # 분할 청산 전략
        'timeseries_sell': TimeseriesSellStrategy,
        'timeseries_sell_quick': TimeseriesSellStrategy,
        'fraction_sell': TimeseriesSellStrategy,
        
        # 단순 비율 청산 전략
        'simple_ratio_sell': SimpleRatioSellStrategy,
        'simple_ratio_sell_quick': SimpleRatioSellStrategy,
        'ratio_sell': SimpleRatioSellStrategy,
        
        # === 신규 청산 전략 (2026-02) ===
        # ADX 기반 청산 전략
        'adx_exit': ADXExitStrategy,
        
        # BB Squeeze 청산 전략
        'bb_squeeze_exit': BBSqueezeExitStrategy,
        
        # MACD 크로스 청산 전략
        'macd_exit': MACDExitStrategy,
        
        # Chandelier Exit 전략
        'chandelier_exit': ChandelierExitStrategy,
        
        # VWAP 목표가 청산 전략
        'vwap_exit': VWAPExitStrategy,
    }


def get_sell_strategy(name: str, config: Dict[str, Any]) -> BaseSellStrategy:
    """
    전략 이름과 설정으로 전략 인스턴스 생성
    
    Args:
        name: 전략 이름 (예: 'bailout_sell', 'timeseries_sell')
        config: 전략 설정 딕셔너리
        
    Returns:
        BaseSellStrategy 인스턴스
        
    Raises:
        ValueError: 알 수 없는 전략 이름
    """
    registry = _get_strategy_registry()
    
    if name not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown sell strategy: {name}. Available: {available}")
    
    strategy_class = registry[name]
    
    if 'strategy_name' not in config:
        config = config.copy()
        config['strategy_name'] = name
    
    return strategy_class(config)


def get_available_strategies() -> List[str]:
    """
    사용 가능한 전략 이름 목록 반환
    """
    return list(_get_strategy_registry().keys())


def get_all_sell_param_combinations(
    strategy_name: str, 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    전략의 모든 파라미터 조합 생성
    
    Args:
        strategy_name: 전략 이름
        config: _list suffix를 포함한 설정
        
    Returns:
        파라미터 조합 리스트
    """
    registry = _get_strategy_registry()
    
    if strategy_name not in registry:
        raise ValueError(f"Unknown sell strategy: {strategy_name}")
    
    strategy_class = registry[strategy_name]
    return strategy_class.get_param_combinations(config)


def create_strategies_from_config(config: Dict[str, Any], strategy_name: str = None) -> List[BaseSellStrategy]:
    """
    설정 파일에서 모든 파라미터 조합의 전략 인스턴스 생성
    
    Args:
        config: 전략 설정 (sell_signal_config)
        strategy_name: 전략 이름 (config에 없는 경우 사용)
        
    Returns:
        모든 파라미터 조합의 전략 인스턴스 리스트
    """
    name = strategy_name or config.get('strategy_name')
    if not name:
        raise ValueError("strategy_name must be provided")
    
    combinations = get_all_sell_param_combinations(name, config)
    
    strategies = []
    for params in combinations:
        params['strategy_name'] = name
        strategy = get_sell_strategy(name, params)
        strategies.append(strategy)
    
    return strategies
