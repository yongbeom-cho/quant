"""
Train Strategy Model

backtest 결과를 바탕으로 XGBoost 모델을 학습합니다.
"""
import os
import sys
import pandas as pd
import argparse
import json
import time
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, Any
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import itertools
import copy
import random
import tempfile

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from xgb_strategy.backtest_analyzer import get_strategy_config_from_backtest
from xgb_strategy.strategy_feature import get_strategy_feature_from_buy_strategy
from xgb_strategy.label import get_labels_from_sell_strategy
from xgb_strategy.feature import get_features


def get_tickers(db_path, table_name):
    """DB에서 ticker 목록 가져오기"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT ticker FROM {table_name}")
    tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def load_ohlcv(db_path, table_name, ticker):
    """DB에서 OHLCV 데이터 로드"""
    max_retries = 3
    retry_delay = 10
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE ticker = ?
        ORDER BY date ASC
    """

    for attempt in range(1, max_retries + 1):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()
            return df
        except Exception as e:
            print(f"[ERROR] DB Load 실패 (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)

    raise RuntimeError("DB load retry failed unexpectedly.")


def save_to_db(db_path, table_base, dfs):
    """DataFrame들을 DB에 저장"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for suffix, df in dfs.items():
        table_name = f"{table_base}_{suffix}"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False
        )

    conn.close()


def load_from_db(db_path, table_base):
    """DB에서 train/val/test 데이터 로드"""
    conn = sqlite3.connect(db_path)

    train_df = pd.read_sql(f"SELECT * FROM {table_base}_train", conn)
    val_df = pd.read_sql(f"SELECT * FROM {table_base}_val", conn)
    test_df = pd.read_sql(f"SELECT * FROM {table_base}_test", conn)

    conn.close()
    return train_df, val_df, test_df


def copy_xgb_model(model):
    """XGBoost 모델 복사"""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    model.save_model(tmp.name)

    new_model = type(model)()
    new_model.load_model(tmp.name)
    return new_model


def get_feat_num(df):
    """df의 column들 중 'feat'로 시작하는 것의 개수 반환"""
    return len([col for col in df.columns if col.startswith('feat')])

def get_label_num(df):
    """df의 column들 중 'label'로 시작하는 것의 개수 반환"""
    return len([col for col in df.columns if col.startswith('label')])
    

def feature_all(feat_num):
    """모든 feature 반환"""
    return [f"feat{i}" for i in range(feat_num)]


def feature_top_k(classifier, best_score_model, k, feat_num):
    """상위 K개 feature 반환"""
    if classifier and best_score_model:
        classifier.model = best_score_model
        importance = classifier.get_importance()
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_feats[:k]]
    lst = feature_all(feat_num)
    random.shuffle(lst)
    return lst[:k]


def prepare_df(df, feat_cols, label_cols, random_state=42):
    """DataFrame 전처리"""
    use_cols = feat_cols + label_cols
    df = df[use_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)
    df = df.dropna(subset=label_cols, how="all")
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df[feat_cols + label_cols]


def prepare_xy(df, feat_cols, label_col):
    """X, Y 분리"""
    use_cols = feat_cols + [label_col]
    xy = df[use_cols]
    xy = xy.dropna(subset=use_cols)
    X = xy[feat_cols]
    Y = xy[label_col]
    return X, Y


def prepare_train_val_test_xy(a_df, b_df, c_df, feat_cols, label_col):
    """train/val/test 데이터 준비"""
    datasets = []

    for name, df in zip(["a", "b", "c"], [a_df, b_df, c_df]):
        X, y = prepare_xy(df, feat_cols, label_col)
        pos_ratio = (y == 1).mean() if len(y) > 0 else 0.0
        datasets.append({
            "name": name,
            "X": X,
            "y": y,
            "pos_ratio": pos_ratio
        })

    datasets = sorted(datasets, key=lambda x: x["pos_ratio"])
    train = datasets[0]
    val = datasets[1]
    test = datasets[2]

    return (
        train["X"], train["y"],
        val["X"], val["y"],
        test["X"], test["y"],
    )


class Classifier:
    """XGBoost Classifier 래퍼"""
    
    def __init__(self, model_config):
        self.model = XGBClassifier(**model_config)
        self.best_threshold = 0.5

    def infer(self, x, y, num_class):
        """추론"""
        y_pred = self.predict(x)
        predictions = [round(value) for value in y_pred]
        test_confusion = confusion_matrix(y, predictions)
        test_accuracy = accuracy_score(y, predictions)
        test_precision = precision_score(y, predictions, average='binary')
        test_recall = recall_score(y, predictions, average='binary')
        test_f1 = f1_score(y, predictions, average='binary')
        test_auc = test_accuracy
        
        print("오차 행렬")
        print(test_confusion)
        print("정답 ratio : %.2f" % (y.mean()))
        print(f"정확도: {test_accuracy:.4f}, 정밀도: {test_precision:.4f}, 재현율: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        return self.model_score(test_confusion, test_precision), test_confusion, test_precision

    def model_score(self, confusion, precision):
        """모델 점수 계산"""
        TN, FP, FN, TP = confusion.ravel()
        base_score = (1.25 ** precision) * (0.8 ** (1 - precision))
        return np.log(base_score) * (TP + FP)

    def get_best_threshold(self, x_train, y_train, x_val, y_val, average='binary', min_precision=0.55, default_threshold=0.5):
        """최적 threshold 찾기"""
        train_proba = self.model.predict_proba(x_train)[:, 1]
        val_proba = self.model.predict_proba(x_val)[:, 1]
        t_prec, t_rec, t_thr = precision_recall_curve(y_train, train_proba)
        v_prec, v_rec, v_thr = precision_recall_curve(y_val, val_proba)

        t_prec, t_rec = t_prec[:-1], t_rec[:-1]
        v_prec, v_rec = v_prec[:-1], v_rec[:-1]

        v_mask = v_prec >= min_precision
        if not v_mask.any():
            return default_threshold

        best_threshold = default_threshold
        best_score = -1

        vals = v_thr[v_mask]
        n = len(vals)
        step = max(1, n // 100)
        for thr in vals[::step]:
            if thr < 0.5:
                continue
            train_pred = (train_proba >= thr).astype(int)

            tp = ((train_pred == 1) & (y_train == 1)).sum()
            fp = ((train_pred == 1) & (y_train == 0)).sum()

            train_precision = tp / (tp + fp + 1e-9)
            if train_precision >= min_precision:
                train_confusion = confusion_matrix(y_train, train_pred)
                train_score = self.model_score(train_confusion, train_precision)

                val_pred = (val_proba >= thr).astype(int)
                tp_v = ((val_pred == 1) & (y_val == 1)).sum()
                fp_v = ((val_pred == 1) & (y_val == 0)).sum()
                fn_v = ((val_pred == 0) & (y_val == 1)).sum()
                val_recall = tp_v / (tp_v + fn_v + 1e-9)
                val_precision = tp_v / (tp_v + fp_v + 1e-9)
                val_confusion = confusion_matrix(y_val, val_pred)
                val_score = self.model_score(val_confusion, val_precision)
                total_score = train_score + val_score
                if (tp_v + fp_v) > 50 and (tp + fp) > 50:
                    if train_score + val_score > best_score:
                        best_score = train_score + val_score
                        best_threshold = thr

        return best_threshold

    def predict(self, x):
        """예측"""
        proba = self.model.predict_proba(x)[:, 1]
        return (proba >= self.best_threshold).astype(int)

    def train(self, x_train, y_train, x_val, y_val, num_class=2, average='binary', min_precision=0.55):
        """학습"""
        self.columns = x_train.columns
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        self.best_threshold = self.get_best_threshold(x_train, y_train, x_val, y_val, average=average, min_precision=min_precision)

        y_pred = self.predict(x_train)
        predictions = [round(value) for value in y_pred]

        train_confusion = confusion_matrix(y_train, predictions)
        train_accuracy = accuracy_score(y_train, predictions)
        train_precision = precision_score(y_train, predictions, average=average)
        train_recall = recall_score(y_train, predictions, average=average)
        train_f1 = f1_score(y_train, predictions, average=average)
        train_auc = train_accuracy

        train_score = self.model_score(train_confusion, train_precision)

        print("#######  train result START ########")
        print("오차 행렬")
        print(train_confusion)
        print("정답 ratio : %.2f" % (y_train.mean()))
        print(f"정확도: {train_accuracy:.4f}, 정밀도: {train_precision:.4f}, 재현율: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print("#######  val result START ########")
        val_score, val_confusion, val_precision = self.infer(x_val, y_val, num_class)

        return self.model, train_score, val_score, train_confusion, val_confusion, train_precision, val_precision

    def get_importance(self):
        """Feature importance 반환"""
        feature_importance = self.model.feature_importances_
        dic = Counter()
        for col, score in zip(self.columns, feature_importance):
            dic[col] = score
        return dic


def format_buy_strategy_params(buy_strategy_params: Dict[str, Any]) -> str:
    """
    buy_strategy params를 파일명에 사용할 수 있는 형식으로 변환
    
    max_investment_ratio와 strategy_name을 제외한 key-value를
    key로 정렬하여 key=value^key=value^... 형식으로 반환
    """
    filtered_params = {
        k: v for k, v in buy_strategy_params.items()
        if k not in ['max_investment_ratio', 'strategy_name']
    }
    
    # key로 정렬
    sorted_items = sorted(filtered_params.items())
    
    # key=value 형식으로 조인 (^로 구분)
    param_parts = [f"{k}={v}" for k, v in sorted_items]
    return "^".join(param_parts)


def prepare_data_from_backtest(
    root_dir: str,
    market: str,
    interval: str,
    backtest_csv_path: str,
    output_dir: str
):
    """
    backtest 결과를 바탕으로 학습 데이터 준비
    
    1. backtest CSV에서 최적 전략 선택
    2. 최적의 strategy_feature, feature, label 생성
    3. train/val/test로 분할하여 DB에 저장
    """
    print(f"=== Preparing data from backtest: {backtest_csv_path} ===")
    
    # 1. backtest 결과에서 최적 전략 선택
    strategy_config = get_strategy_config_from_backtest(backtest_csv_path, interval, top_n=3)
    buy_strategy_config = strategy_config['buy_strategy']
    sell_strategies = strategy_config['sell_strategies']
    label_count = strategy_config['label_count']
    
    print(f"Buy Strategy: {buy_strategy_config['strategy_name']}")
    print(f"Sell Strategies: {[s['strategy_name'] for s in sell_strategies]}")
    print(f"Label Count: {label_count}")
    
    # 2. 데이터 준비
    train_cut = "202107010900"
    val_cut = "202407010900"
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    table_name = f'{market}_ohlcv_{interval}'
    db_path = os.path.join(root_dir, f'var/data/{table_name}.db')
    xgb_dir = output_dir
    os.makedirs(xgb_dir, exist_ok=True)
    
    tickers = get_tickers(db_path, table_name)
    print(f"Processing {len(tickers)} tickers...")
    
    for ticker in tickers:
        df = load_ohlcv(db_path, table_name, ticker)
        if len(df) <= 50:
            continue
        
        # 3. strategy_feature 생성 (buy_strategy 사용)
        df = get_strategy_feature_from_buy_strategy(
            df, buy_strategy_config['strategy_name'], buy_strategy_config['params'], interval
        )
        
        # 4. feature 생성
        df = get_features(df, interval, buy_strategy_config['strategy_name'])
        
        # 5. label 생성 (sell_strategy 사용)
        df = get_labels_from_sell_strategy(df, sell_strategies)
        
        # strategy_feature가 True인 행만 필터링
        df = df[df['strategy_feature'] == True]
        
        feat_num = get_feat_num(df)
        label_num = get_label_num(df)
        
        # 필요한 컬럼만 선택
        feat_cols = [f"feat{i}" for i in range(feat_num)]
        label_cols = [f"label{i}" for i in range(label_num)]
        need_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        use_cols = feat_cols + need_cols + label_cols
        
        df = df[use_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feat_cols + need_cols)
        df = df.dropna(subset=label_cols, how="all")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # train/val/test 분할
        tmp_df = df[df['date'] < train_cut]
        if not tmp_df.empty:
            train_dfs.append(tmp_df)
        
        tmp_df = df[(df['date'] >= train_cut) & (df['date'] < val_cut)]
        if not tmp_df.empty:
            val_dfs.append(tmp_df)
        
        tmp_df = df[df['date'] >= val_cut]
        if not tmp_df.empty:
            test_dfs.append(tmp_df)
    
    # 6. 데이터 합치기 및 저장
    train_df = pd.concat(train_dfs, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_dfs, axis=0).reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 전략 이름으로 table_base 생성
    strategy_name = buy_strategy_config['strategy_name']
    table_base = f"xgb_{market}_{interval}_{strategy_name}"
    xgb_db_path = os.path.join(xgb_dir, f"{table_base}.db")
    
    save_to_db(
        xgb_db_path,
        table_base,
        {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    )
    
    return xgb_db_path, table_base, strategy_name, feat_num, label_num, buy_strategy_config


def train_models(
    root_dir: str,
    market: str,
    interval: str,
    strategy_name: str,
    input_data_dir: str,
    model_output_dir: str,
    best_buy_strategy_config: Dict[str, Any] = None
):
    """XGBoost 모델 학습"""
    print(f"=== Training models for {strategy_name} ===")
    
    table_base = f"xgb_{market}_{interval}_{strategy_name}"
    model_name_base = f"xgb-{market}-{interval}-{strategy_name}"
    xgb_db_path = os.path.join(input_data_dir, f"{table_base}.db")
    
    # buy_strategy_params 문자열 생성
    str_buy_strategy_params = ""
    if best_buy_strategy_config and 'params' in best_buy_strategy_config:
        str_buy_strategy_params = format_buy_strategy_params(best_buy_strategy_config['params'])
    
    a_df, b_df, c_df = load_from_db(xgb_db_path, table_base)
    
    feat_num = get_feat_num(a_df)
    label_num = get_label_num(a_df)
    FEATURE_COLS = [f"feat{i}" for i in range(feat_num)]
    LABEL_COLS = [f"label{i}" for i in range(label_num)]
    
    a_df = prepare_df(a_df, FEATURE_COLS, LABEL_COLS)
    b_df = prepare_df(b_df, FEATURE_COLS, LABEL_COLS)
    c_df = prepare_df(c_df, FEATURE_COLS, LABEL_COLS)
    
    # 모델 설정 로드
    xgb_train_config_path = os.path.join(root_dir, 'sbin/train_xgb/binary_classifier_config.json')
    with open(xgb_train_config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    
    class_imblance_strategy = model_config.get('strategy', 'scale_weight')
    del model_config['strategy']
    
    PARAM_GRID = {
        "learning_rate": [0.05],
        "max_depth": [3, 5],
        "min_child_weight": [5],
        "gamma": [1.0, 5.0],
        "n_estimators": [500, 2000],
        "early_stopping_rounds": [80],
    }
    min_precisions = [0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555]
    if len(a_df) > 10000:
        min_precisions = [0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58]
    
    param_keys = PARAM_GRID.keys()
    param_combinations = list(itertools.product(*PARAM_GRID.values()))
    
    for label_col in LABEL_COLS:
        print(f"######################## {label_col} ########################")
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_train_val_test_xy(
            a_df, b_df, c_df, FEATURE_COLS, label_col
        )
        
        print(y_train.shape, x_train.shape, y_val.shape, x_val.shape)
        
        num_class = 2
        if 'scale_weight' in class_imblance_strategy:
            max_cnt = 0
            min_cnt = 999999999
            for i in range(num_class):
                i_cnt = len(y_train.loc[y_train == i])
                max_cnt = max(max_cnt, i_cnt)
                min_cnt = min(min_cnt, i_cnt)
            model_config['scale_pos_weight'] = max_cnt / min_cnt
            print("scale_pos_weight : ", model_config['scale_pos_weight'])
        elif 'smote' in class_imblance_strategy:
            smote = SMOTE()
            x_train, y_train = smote.fit_resample(x_train, y_train)
            print('after resample : ', len(x_train), len(y_train))
        
        print("TRAIN START")
        
        for min_precision in min_precisions:
            best_score = 0.0
            best_score_model = None
            best_param = ""
            best_feats = ""
            best_threshold = 0.5
            
            for idx, param_values in enumerate(param_combinations):
                new_cfg = dict(zip(param_keys, param_values))
                for k, v in new_cfg.items():
                    model_config[k] = v
                
                classifier = None
                for feat_num_iter in range(len(FEATURE_COLS), 5, -5):
                    feat_cols = feature_top_k(classifier, best_score_model, feat_num_iter, feat_num)
                    config_string = json.dumps(model_config)
                    classifier = Classifier(model_config)
                    average = 'binary'
                    xtr = x_train[feat_cols]
                    xva = x_val[feat_cols]
                    xte = x_test[feat_cols]
                    
                    print("Classifier TRAIN START")
                    model, train_score, val_score, train_confusion, val_confusion, train_precision, val_precision = classifier.train(
                        xtr, y_train, xva, y_val, num_class=num_class, average=average, min_precision=min_precision
                    )
                    print("Classifier TRAIN END")
                    print(feat_cols)
                    print(f"#######  test result START, {model_name_base}-{str_buy_strategy_params}-{label_col}-{min_precision} ########")
                    test_score, test_confusion, test_precision = classifier.infer(xte, y_test, num_class)
                    print("##########################################################################################")
                    
                    has_negative = any(s < 0 for s in (train_score, val_score, test_score))
                    total_score = 0.0
                    if not has_negative:
                        if val_precision > min_precision:
                            total_score = min(min(train_score, val_score), test_score)
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_score_model = copy_xgb_model(model)
                        best_param = config_string
                        best_feats = feat_cols
                        best_confusion = [("train", train_confusion), ("val", val_confusion), ("test", test_confusion)]
                        best_threshold = classifier.best_threshold
                        target = model_name_base + '-' + str_buy_strategy_params + '-' + label_col + '-' + str(best_threshold)
                        print("!!!!!!!!!!!!!! BEST PARAM CANDIDATE !!!!!!!!!!!!!!")
                        print(target, best_param, best_feats, best_score)
                        print(str(min_precision))
                        for key, confusion in best_confusion:
                            print(f"#### {key} ####")
                            print(confusion)
                        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            
            model_out_dir = os.path.join(root_dir, model_output_dir)
            os.makedirs(model_out_dir, exist_ok=True)
            str_feats = "".join(f.replace("feat", "f") for f in best_feats)
            model_output_path = os.path.join(
                model_out_dir,
                model_name_base + '-' + str_buy_strategy_params + '-' + label_col + '-' + str(min_precision) + '-' + str(best_threshold) + '-' + str_feats
            )
            if best_score_model:
                best_score_model.save_model(model_output_path)
                print("@@@@@@@@@@@@@ BEST SCORE START @@@@@@@@@@@@@@@")
                print(model_name_base + '-' + str_buy_strategy_params + '-' + label_col)
                print(min_precision)
                print(best_param)
                print(best_feats)
                print(best_score)
                print(best_threshold)
                for key, confusion in best_confusion:
                    print(f"#### {key} ####")
                    print(confusion)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


parser = argparse.ArgumentParser(description='Train Strategy Model')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute60")
parser.add_argument('--strategy_name', type=str, default="pb_rebound")
parser.add_argument('--backtest_csv', type=str, default=None, help='Backtest result CSV path')
parser.add_argument('--output_data_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant/var/xgb_data")
parser.add_argument('--model_output_dir', type=str, default="var/xgb_model")
parser.add_argument('--skip_data_prep', action='store_true', help='Skip data preparation step')
parser.add_argument('--skip_training', action='store_true', help='Skip training step')

args = parser.parse_args()

if __name__ == "__main__":
    best_buy_strategy_config = None
    if not args.skip_data_prep:
        if args.backtest_csv is None:
            raise ValueError("--backtest_csv is required when --skip_data_prep is False")
        
        xgb_db_path, table_base, strategy_name, feat_num, label_count, best_buy_strategy_config = prepare_data_from_backtest(
            args.root_dir,
            args.market,
            args.interval,
            args.backtest_csv,
            args.output_data_dir
        )
    else:
        strategy_name = args.strategy_name
        
    
    if not args.skip_training:
        train_models(
            args.root_dir,
            args.market,
            args.interval,
            strategy_name,
            args.output_data_dir,
            args.model_output_dir,
            best_buy_strategy_config
        )

