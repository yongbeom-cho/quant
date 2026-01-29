"""
Buy Strategy 레지스트리

전략 이름으로 전략 클래스를 찾아 인스턴스를 생성합니다.
새 전략을 추가할 때 여기에 등록해야 합니다.
"""

from typing import Dict, Type, List, Any
import itertools
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
    
    # Explode Volume 전략들
    from .explode_volume_breakout_strategy import ExplodeVolumeBreakoutStrategy
    from .explode_volume_volatility_breakout_strategy import ExplodeVolumeVolatilityBreakoutStrategy
    from .explode_volume_breakout_2_strategy import ExplodeVolumeBreakout2Strategy
    from .explode_volume_volatility_breakout_2_strategy import ExplodeVolumeVolatilityBreakout2Strategy
    
    # Low BB DU 전략들
    from .low_bb_du_strategy import LowBBDUStrategy
    from .low_bb_du_2_strategy import LowBBDU2Strategy
    from .low_bb_du_3_strategy import LowBBDU3Strategy
    from .low_bb_du_4_strategy import LowBBDU4Strategy
    
    # VBT Prev Candle 전략
    from .vbt_prev_candle_strategy import VBTPrevCandleStrategy
    
    # New strategies
    from .adaptive_vbt_strategy import AdaptiveVBTStrategy
    from .volume_weighted_vbt_strategy import VolumeWeightedVBTStrategy
    from .mean_reversion_momentum_strategy import MeanReversionMomentumStrategy
    from .breakout_confirmation_strategy import BreakoutConfirmationStrategy
    
    return {
        # VBT 전략
        'vbt_with_filters': VBTBuyStrategy,
        
        # VBT Enhanced 전략
        'vbt_with_filters_enhanced': VBTEnhancedBuyStrategy,
        
        # Bollinger Band DRU 전략
        'low_bb_dru': LowBBDRUBuyStrategy,
        
        # Explode Volume 전략들
        'explode_volume_breakout': ExplodeVolumeBreakoutStrategy,
        'explode_volume_volatility_breakout': ExplodeVolumeVolatilityBreakoutStrategy,
        'explode_volume_breakout_2': ExplodeVolumeBreakout2Strategy,
        'explode_volume_volatility_breakout_2': ExplodeVolumeVolatilityBreakout2Strategy,
        
        # Low BB DU 전략들
        'low_bb_du': LowBBDUStrategy,
        'low_bb_du_2': LowBBDU2Strategy,
        'low_bb_du_3': LowBBDU3Strategy,
        'low_bb_du_4': LowBBDU4Strategy,
        
        # VBT Prev Candle 전략
        'volatility_breakout_prev_candle': VBTPrevCandleStrategy,
        
        # New Strategies
        'adaptive_vbt': AdaptiveVBTStrategy,
        'volume_weighted_vbt': VolumeWeightedVBTStrategy,
        'mean_reversion_momentum': MeanReversionMomentumStrategy,
        'breakout_confirmation': BreakoutConfirmationStrategy,
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
    설정 파일(JSON)에 있는 리스트형 파라미터들로부터 모든 가능한 조합을 생성합니다.
    
    예를 들어:
    config = {'window_list': [1, 2], 'k_list': [0.5, 0.6]} 이라면,
    (1, 0.5), (1, 0.6), (2, 0.5), (2, 0.6) 총 4개의 파라미터 셋을 만들어 리턴합니다.
    
    Args:
        strategy_name: 전략 이름 (현재 이 함수에서는 사용되지 않지만 인터페이스 일관성을 위해 유지)
        config: _list suffix를 포함한 설정
        
    Returns:
        파라미터 조합 리스트
    """
    # '_list'로 끝나는 모든 키를 찾아 파라미터 후보군을 추출
    param_keys = [k for k in config.keys() if k.endswith('_list')]
    param_values = [config[k] for k in param_keys]
    
    combinations = []
    # itertools.product를 사용하여 모든 조합(Cartesian Product) 생성
    for values in itertools.product(*param_values):
        params = {}
        for key, value in zip(param_keys, values):
            # '_list' 접미사를 제거하고 실제 파라미터 이름으로 저장
            clean_key = key.replace('_list', '')
            params[clean_key] = value
        combinations.append(params)
        
    return combinations


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
