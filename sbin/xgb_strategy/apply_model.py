"""
Apply XGBoost Model Module

학습된 XGBoost 모델을 사용하여 매수 신호를 생성합니다.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from typing import Optional
import os

from .feature import get_features
from .strategy_feature import get_strategy_feature_from_buy_strategy


def apply_strategy_model(
    df: pd.DataFrame,
    model_name: str,
    model_dir: str,
    interval: str,
    buy_strategy_name: Optional[str] = None,
    buy_params: Optional[dict] = None
) -> pd.DataFrame:
    """
    학습된 XGBoost 모델을 사용하여 매수 신호 생성
    
    Args:
        df: OHLCV DataFrame
        model_name: 모델 파일 이름 (예: "xgb-coin-day-pb_du-pb_sma_period=20^pb_sma_up=0.7^pb_rebound_line=0.7-label1-0.545-0.52278996-f5f7f32f44f40f41f37f28f17f33f2f46")
        model_dir: 모델 디렉토리 경로
        interval: 시간 간격
        buy_strategy_name: buy 전략 이름 (model_name에서 추출 불가능한 경우)
        buy_params: buy 전략 파라미터 (model_name에서 추출 불가능한 경우)
    
    Returns:
        'signal' 컬럼이 추가된 DataFrame (1: 매수, 0: 매수 안함)
    """
    # 1. model_name 파싱
    # 형식: xgb-{market}-{interval}-{strategy_name}-{strategy_feature_params}-{label_name}-{min_precision}-{threshold}-{feat_string}
    parts = model_name.split('-')

    if len(parts) < 4:
        raise ValueError(f"Invalid model_name format: {model_name}")
    
    market = parts[1]
    interval_from_name = parts[2]
    strategy_name = parts[3]
    
    # strategy_feature_params, label_name, min_precision, threshold, feat_string 추출
    strategy_feature_params = parts[4]
    label_name = parts[5]
    min_precision = parts[6]
    threshold = float(parts[7])
    feat_string = parts[8]
    
    
    # 2. strategy_feature 생성 (buy_strategy 사용)
    if buy_strategy_name and buy_params:
        df = get_strategy_feature_from_buy_strategy(df, buy_strategy_name, buy_params, interval)
    else:
        # model_name에서 추출한 strategy_name 사용 (기본 파라미터 필요)
        # 이 경우는 기본 파라미터를 사용하거나 에러 발생
        raise ValueError("buy_strategy_name and buy_params are required")
    
    # 3. feature 계산
    df = get_features(df, interval, strategy_name)
    
    # 4. 사용할 feature 추출 (feat_string에서)
    feat_cols = ['feat'+i for i in feat_string.split('f')[1:]]
    
    # 5. 모델 로드
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = XGBClassifier()
    model.load_model(model_path)
    
    # 6. 신호 생성
    df['signal'] = 0
    
    # strategy_feature가 True이고 feature가 유효한 행만 선택
    mask_strategy = df['strategy_feature'] == True
    mask_notna = df[feat_cols].notna().all(axis=1)
    mask_notinf = ~df[feat_cols].isin([np.inf, -np.inf]).any(axis=1)
    mask = mask_strategy & mask_notna & mask_notinf
    
    X = df.loc[mask, feat_cols]
    if len(X) > 0:
        proba = model.predict_proba(X)[:, 1]
        df.loc[mask, 'signal'] = (proba >= threshold).astype(int)
    
    return df

