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
    from .xgb_buy_strategy import XGBBuyStrategy
    
    # VBT Prev Candle 전략
    from .vbt_prev_candle_strategy import VBTPrevCandleStrategy
    
    # ADX 모멘텀 필터 전략
    from .adx_momentum_strategy import ADXMomentumStrategy
    # 볼린저밴드 압축 돌파 전략
    from .bb_squeeze_strategy import BBSqueezeStrategy
    # MACD 다이버전스 전략
    from .macd_divergence_strategy import MACDDivergenceStrategy
    # ATR 채널 돌파 전략
    from .atr_channel_strategy import ATRChannelStrategy
    # VWAP 회귀 전략
    from .vwap_reversion_strategy import VWAPReversionStrategy
    
    # (2026-02-09 추가)
    ###
    # 선형 회귀 기울기 전략
    from .linear_regression_slope_strategy import LinearRegressionSlopeBuyStrategy
    # IBS 마이크로 전략
    from .ibs_micro_strategy import IBSBuyStrategy
    # 동적 ATR 전략
    from .dynamic_atr_strategy import DynamicATRBuyStrategy
    ###
    
    # (2026-02-09 추가 - 새 눌림목 전략)
    ###
    # 스토캐스틱 눌림목 전략
    from .stochastic_pullback_strategy import StochasticPullbackBuyStrategy
    # OBV 발산 전략
    from .obv_divergence_strategy import OBVDivergenceBuyStrategy
    # Williams %R + MFI 전략
    from .williams_mfi_strategy import WilliamsMFIBuyStrategy
    ###

    return {
        # VBT 전략
        'vbt_with_filters': VBTBuyStrategy,
        
        # VBT Enhanced 전략
        'vbt_with_filters_enhanced': VBTEnhancedBuyStrategy,
        
        # VBT Prev Candle 전략
        'volatility_breakout_prev_candle': VBTPrevCandleStrategy,
        
        # XGBoost 모델 기반 전략
        'xgb_buy': XGBBuyStrategy,
        
        # === 신규 전략 (2026-02) ===
        # ADX 모멘텀 필터 전략
        'adx_momentum': ADXMomentumStrategy,
        
        # 볼린저밴드 압축 돌파 전략
        'bb_squeeze': BBSqueezeStrategy,
        
        # MACD 다이버전스 전략
        'macd_divergence': MACDDivergenceStrategy,
        
        # ATR 채널 돌파 전략
        'atr_channel': ATRChannelStrategy,
        
        # VWAP 회귀 전략
        'vwap_reversion': VWAPReversionStrategy,
        
        # === 신규 전략 (2026-02-09) ===
        'linear_regression_slope': LinearRegressionSlopeBuyStrategy,
        'ibs_micro': IBSBuyStrategy,
        'dynamic_atr': DynamicATRBuyStrategy,
        
        # === 새 눌림목 전략 (2026-02-09) ===
        'stochastic_pullback': StochasticPullbackBuyStrategy,
        'obv_divergence': OBVDivergenceBuyStrategy,
        'williams_mfi': WilliamsMFIBuyStrategy,
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
