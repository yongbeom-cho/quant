import talib
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.xgb_strategy import apply_strategy_xgb_feats

def get_feats_and_labels_num(strategy_feature_name):
    if strategy_feature_name == 'low_bb_du':
        return 44, 3
    return 0, 0

def get_strategy_feature_filtered_feature_and_labels(df, strategy_feature_name, interval='minute60'):
    if strategy_feature_name == 'low_bb_du':
        return low_bb_du(df, interval)
    return None, 0, 0

def label_df(df, label_name, upper, lower):
    labels = np.full(len(df), np.nan)

    buy_indices = df.index[df['strategy_feature'] == True]

    for idx in buy_indices:
        buy_close = df.at[idx, 'close']

        future = df.loc[idx+1:]

        # 손절 먼저 체크
        stop_loss = future[future['low'] < buy_close * lower]
        take_profit = future[future['high'] > buy_close * upper]

        if not stop_loss.empty and not take_profit.empty:
            # 둘 다 발생하면 더 먼저 발생한 것
            if stop_loss.index[0] <= take_profit.index[0]:
                labels[idx] = 0
            else:
                labels[idx] = 1

        elif not stop_loss.empty:
            labels[idx] = 0

        elif not take_profit.empty:
            labels[idx] = 1
        # else:
        #     # 끝까지 갔을 때
        #     last_close = future.iloc[-1]['close']
        #     labels[idx] = 1 if last_close > buy_close else 0

    df[label_name] = labels
    return df

def low_bb_du(df, interval):
    df = apply_strategy_xgb_feats(df, interval, 'low_bb_du')

    df = label_df(df, 'label0', 1.12, 0.9)
    df = label_df(df, 'label1', 1.15, 0.875)
    df = label_df(df, 'label2', 1.18, 0.85)
    
    feat_num, label_num = get_feats_and_labels_num("low_bb_du")

    feat_cols = [f"feat{i}" for i in range(feat_num)]
    label_cols = [f"label{i}" for i in range(label_num)]

    assert len(feat_cols) == feat_num, "feat_cols 개수가 feat_num과 다름"
    assert len(label_cols) == label_num, "label_cols 개수가 label_num 다름"
    
    need_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']

    df = df[df['strategy_feature'] == True]
    df = (
        df[feat_cols + label_cols + need_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df