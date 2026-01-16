"""
Buy Strategy 레지스트리

전략 이름으로 전략 클래스를 찾아 인스턴스를 생성합니다.
새 전략을 추가할 때 여기에 등록해야 합니다.
"""

from typing import Dict, Type, List, Any
from .base import BaseBuyStrategy


# 전략 레지스트리 (지연 로딩을 위해 함수로 구현)
def _get_strategy_registry() -> Dict[str, Type[BaseBuyStrategy]]:
    """
    전략 레지스트리 반환 (순환 임포트 방지를 위해 함수 내부에서 임포트)
    
    Note: _quick 버전들은 같은 클래스를 사용하며 파라미터만 다릅니다.
    Config의 strategy_name과 동일한 키가 Registry에 있어야 합니다.
    """
    from .vbt_strategy import VBTBuyStrategy
    from .vbt_enhanced_strategy import VBTEnhancedBuyStrategy
    from .low_bb_dru_strategy import LowBBDRUBuyStrategy
    
    return {
        # VBT 전략
        'vbt_with_filters': VBTBuyStrategy,
        'vbt_with_filters_quick': VBTBuyStrategy,
        
        # VBT Enhanced 전략
        'vbt_with_filters_enhanced': VBTEnhancedBuyStrategy,
        'vbt_with_filters_enhanced_quick': VBTEnhancedBuyStrategy,
        
        # Bollinger Band 전략
        'low_bb_dru': LowBBDRUBuyStrategy,
        'low_bb_dru_quick': LowBBDRUBuyStrategy,
    }


def get_buy_strategy(name: str, config: Dict[str, Any]) -> BaseBuyStrategy:
    """
    전략 이름과 설정으로 전략 인스턴스 생성
    
    Args:
        name: 전략 이름 (예: 'vbt_with_filters')
        config: 전략 설정 딕셔너리
        
    Returns:
        BaseBuyStrategy 인스턴스
        
    Raises:
        ValueError: 알 수 없는 전략 이름
    """
    registry = _get_strategy_registry()
    
    if name not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown buy strategy: {name}. Available: {available}")
    
    strategy_class = registry[name]
    
    # config에 strategy_name이 없으면 추가
    if 'strategy_name' not in config:
        config = config.copy()
        config['strategy_name'] = name
    
    return strategy_class(config)


def get_available_strategies() -> List[str]:
    """
    사용 가능한 전략 이름 목록 반환
    """
    return list(_get_strategy_registry().keys())


def get_all_buy_param_combinations(
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
        raise ValueError(f"Unknown buy strategy: {strategy_name}")
    
    strategy_class = registry[strategy_name]
    return strategy_class.get_param_combinations(config)


def create_strategies_from_config(config: Dict[str, Any]) -> List[BaseBuyStrategy]:
    """
    설정 파일에서 모든 파라미터 조합의 전략 인스턴스 생성
    
    Args:
        config: 전략 설정 (strategy_name + buy_signal_config 포함)
        
    Returns:
        모든 파라미터 조합의 전략 인스턴스 리스트
    """
    strategy_name = config.get('strategy_name')
    if not strategy_name:
        raise ValueError("config must contain 'strategy_name'")
    
    buy_config = config.get('buy_signal_config', config)
    combinations = get_all_buy_param_combinations(strategy_name, buy_config)
    
    strategies = []
    for params in combinations:
        params['strategy_name'] = strategy_name
        strategy = get_buy_strategy(strategy_name, params)
        strategies.append(strategy)
    
    return strategies
