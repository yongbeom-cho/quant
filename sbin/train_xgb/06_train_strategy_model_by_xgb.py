import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import math
import numpy as np
import itertools
import copy
import random
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_xgb.strategy_feature import get_feats_and_labels_num


def copy_xgb_model(model):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    model.save_model(tmp.name)

    new_model = type(model)()   # XGBClassifier or XGBRegressor
    new_model.load_model(tmp.name)
    return new_model


def feature_all(strategy_feature_name):
    feat_num = get_feats_and_labels_num(strategy_feature_name)[0]
    FEATURE_COLS = [f"feat{i}" for i in range(feat_num)]
    return FEATURE_COLS

def feature_top_k(classifier, best_score_model, k, strategy_feature_name):
    if classifier and best_score_model:
        classifier.model = best_score_model
        importance = classifier.get_importance()  
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_feats[:k]]
    lst = feature_all(strategy_feature_name)
    random.shuffle(lst)
    return lst[:k]

def load_from_db(db_path, table_base):
    conn = sqlite3.connect(db_path)

    train_df = pd.read_sql(
        f"SELECT * FROM {table_base}_train",
        conn
    )
    val_df = pd.read_sql(
        f"SELECT * FROM {table_base}_val",
        conn
    )
    test_df = pd.read_sql(
        f"SELECT * FROM {table_base}_test",
        conn
    )

    conn.close()
    return train_df, val_df, test_df

def prepare_df(df, feat_cols, label_cols, random_state=42):
    use_cols = feat_cols + label_cols
    df = df[use_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)
    df = df.dropna(subset=label_cols, how="all")

    # 랜덤 셔플
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df[feat_cols + label_cols]
    # X = df[feat_cols]
    # Y = df[label_cols]

    # return X, Y

def prepare_xy(df, feat_cols, label_col):
    use_cols = feat_cols + [label_col]
    xy = df[use_cols]
    xy = xy.dropna(subset=use_cols)
    X = xy[feat_cols]
    Y = xy[label_col]
    return X, Y

def prepare_train_val_test_xy(a_df, b_df, c_df, feat_cols, label_col):
    datasets = []

    for name, df in zip(["a", "b", "c"], [a_df, b_df, c_df]):
        X, y = prepare_xy(df, feat_cols, label_col)

        # 1 비율 계산
        pos_ratio = (y == 1).mean() if len(y) > 0 else 0.0

        datasets.append({
            "name": name,
            "X": X,
            "y": y,
            "pos_ratio": pos_ratio
        })

    # 1 비율 기준 정렬 (오름차순)
    datasets = sorted(datasets, key=lambda x: x["pos_ratio"])

    train = datasets[0]  # 1 비율 가장 적음
    val   = datasets[1]
    test  = datasets[2]  # 1 비율 가장 많음

    return (
        train["X"], train["y"],
        val["X"],   val["y"],
        test["X"],  test["y"],
    )

class Classifier:
    def __init__(self, model_config):
        self.model = XGBClassifier(**model_config)
        self.best_threshold = 0.5

    def infer(self, x, y, num_class):
        y_pred = self.predict(x)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        test_confusion = confusion_matrix(y, predictions)
        test_accuracy = accuracy_score(y, predictions)
        test_precision = precision_score(y, predictions, average=average)
        test_recall = recall_score(y, predictions, average=average)
        test_f1 = f1_score(y, predictions, average=average)
        #test_auc = roc_auc_score(y, predictions, multi_class='ovo', average=average)
        test_auc = test_accuracy
        
        print("오차 행렬")
        print(test_confusion)
        print("정답 ratio : %.2f" %(y.mean()))
        if average == 'binary':
            print(f"정확도: {test_accuracy:.4f}, 정밀도: {test_precision:.4f}, 재현율: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        else:
            print("정확도: ", test_accuracy, ", 정밀도: ", test_precision, "재현율: ", test_recall, "F1: ", test_f1, "AUC: ", test_auc)

        return self.model_score(test_confusion, test_precision), test_confusion, test_precision

    def model_score(self, confusion, precision):
        TN, FP, FN, TP = confusion.ravel()
        base_score = (1.25 ** precision) * (0.8 ** (1 - precision))
        return np.log(base_score) * (TP + FP)


    def get_best_threshold(self, x_train, y_train, x_val, y_val, average='binary', min_precision=0.55, default_threshold=0.5):
        train_proba = self.model.predict_proba(x_train)[:, 1] 
        val_proba = self.model.predict_proba(x_val)[:, 1]
        t_prec, t_rec, t_thr = precision_recall_curve(y_train, train_proba)
        v_prec, v_rec, v_thr = precision_recall_curve(y_val, val_proba)

        t_prec, t_rec = t_prec[:-1], t_rec[:-1]
        v_prec, v_rec = v_prec[:-1], v_rec[:-1]

        # val 기준 후보
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
            # train precision 확인
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
        proba = self.model.predict_proba(x)[:, 1]
        return (proba >= self.best_threshold).astype(int)

    def train(self, x_train, y_train, x_val, y_val, num_class=2, average='binary', min_precision=0.55):
        self.columns = x_train.columns
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        self.best_threshold = self.get_best_threshold(x_train, y_train, x_val, y_val, average=average, min_precision=min_precision)

        y_pred = self.predict(x_train)
        
        predictions = [ round(value) for value in y_pred]
        # evaluate predictions
            
        train_confusion = confusion_matrix(y_train, predictions)
        train_accuracy = accuracy_score(y_train, predictions)
        train_precision = precision_score(y_train, predictions, average=average)
        train_recall = recall_score(y_train, predictions, average=average)
        train_f1 = f1_score(y_train, predictions, average=average)
        #train_auc = roc_auc_score(y_train, predictions, multi_class='ovo', average=average)
        train_auc = train_accuracy

        train_score = self.model_score(train_confusion, train_precision)
        
        print ("#######  train result START ########")
        
        print("오차 행렬")
        print(train_confusion)
        print("정답 ratio : %.2f" %(y_train.mean()))
        if average == 'binary':
            print(f"정확도: {train_accuracy:.4f}, 정밀도: {train_precision:.4f}, 재현율: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        else:
            print("정확도: ", train_accuracy, ", 정밀도: ", train_precision, "재현율: ", train_recall, "F1: ", train_f1, "AUC: ", train_auc)
        # for i, y_ in enumerate(y_train):
        #     print(y_, y_pred[i])
        
        print ("#######  val result START ########")
        # make predictions for test data
        val_score, val_confusion, val_precision = self.infer(x_val, y_val, num_class)
        
        return self.model, train_score, val_score, train_confusion, val_confusion, train_precision, val_precision
    
    def get_importance(self):
        feature_importance = self.model.feature_importances_
        dic = Counter()
        for col, score in zip(self.columns, feature_importance):
            dic[col] = score
        return dic

    def print_importance(self):
        feature_importance = self.model.feature_importances_
        print("feature count : ", len(feature_importance))
        print(feature_importance)

        plot_importance(model)
        dic = Counter()
        for col, score in zip(self.columns, feature_importance):
            dic[col] = score
        for col, score in dic.most_common():
            print(col, score)

parser = argparse.ArgumentParser(description='06_train_strategy_model_by_xgb')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--input_data_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant/var/xgb_data")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute60")
parser.add_argument('--target_strategy_feature', type=str, default="low_bb_du")
parser.add_argument('--model_output_dir', type=str, default="var/xgb_model")

args = parser.parse_args()


if __name__ == "__main__":
    strategy_feature_name = args.target_strategy_feature
    xgb_dir = args.input_data_dir
    table_base = f"xgb_{args.market}_{args.interval}_{args.target_strategy_feature}"
    model_name_base = f"xgb-{args.market}-{args.interval}-{args.target_strategy_feature}"
    print("######################## ", table_base, " ########################")
    xgb_db_path = os.path.join(xgb_dir, f"{table_base}.db")
    a_df, b_df, c_df = load_from_db(xgb_db_path, table_base)
    feat_num, label_num = get_feats_and_labels_num(args.target_strategy_feature)

    FEATURE_COLS = [f"feat{i}" for i in range(feat_num)]
    LABEL_COLS = [f"label{i}" for i in range(label_num)]

    a_df = prepare_df(a_df, FEATURE_COLS, LABEL_COLS)
    b_df = prepare_df(b_df, FEATURE_COLS, LABEL_COLS)
    c_df = prepare_df(c_df, FEATURE_COLS, LABEL_COLS)

    for label_col in LABEL_COLS:
        print("######################## ", label_col, " ########################")
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_train_val_test_xy(a_df, b_df, c_df, FEATURE_COLS, label_col)        
        
        print(y_train.shape, x_train.shape, y_val.shape, x_val.shape)

        model_config = None

        num_class = 2
        xgb_train_config_path = os.path.join(args.root_dir, 'sbin/train_xgb/binary_classifier_config.json')
        with open(xgb_train_config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            print('binary config loaded')
        
        # 'smote', 'scale_weight', 'baseline
        strategy = 'scale_weight'
        if 'strategy' in model_config:
            strategy = model_config['strategy']
        del model_config['strategy']

        if 'scale_weight' in strategy:
            max_cnt = 0
            min_cnt = 999999999
            for i in range(num_class):
                i_cnt = len(y_train.loc[y_train == i])
                max_cnt = max(max_cnt, i_cnt)
                min_cnt = min(min_cnt, i_cnt)
            model_config['scale_pos_weight'] = max_cnt/min_cnt
            print("scale_pos_weight : ", model_config['scale_pos_weight'])
        elif 'smote' in strategy:
            smote = SMOTE()
            x_train, y_train = smote.fit_resample(x_train, y_train)
            print('after resample : ', len(x_train), len(y_train))

        PARAM_GRID = {
            "learning_rate": [0.05],
            "max_depth": [3, 5],
            "min_child_weight": [5],
            "gamma": [1.0, 5.0],
            "n_estimators": [500, 2000],
            "early_stopping_rounds": [80],
        }
        min_precisions = [0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555]
        if len(x_train) > 10000:
            min_precisions = [0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58]
        
        results = []

        param_keys = PARAM_GRID.keys()
        param_combinations = list(itertools.product(*PARAM_GRID.values()))

        
        
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
                    for feat_num in range(len(FEATURE_COLS), 5, -5):
                        feat_cols = feature_top_k(classifier, best_score_model, feat_num, strategy_feature_name)
                        config_string = json.dumps(model_config)
                        classifier = Classifier(model_config)
                        average = 'binary'
                        xtr = x_train[feat_cols]
                        xva = x_val[feat_cols]
                        xte = x_test[feat_cols]

                        print("Classifier TRAIN START")
                        model, train_score, val_score, train_confusion, val_confusion, train_precision, val_precision = classifier.train(xtr, y_train, xva, y_val, num_class=num_class, average=average, min_precision=min_precision)
                        print("Classifier TRAIN END")
                        print(feat_cols)
                        print ("#######  test result START,", model_name_base + '-' + label_col + '-' + str(min_precision), " ########")
                        test_score, test_confusion, test_precision = classifier.infer(xte, y_test, num_class)
                        print ("##########################################################################################")
                        
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
                            target = model_name_base + '-' + label_col + '-' + str(best_threshold)
                            print("!!!!!!!!!!!!!! BEST PARAM CANDIDATE !!!!!!!!!!!!!!")
                            print(target, best_param, best_feats, best_score)
                            print(str(min_precision))
                            for key, confusion in best_confusion:
                                print("#### %s ####" %(key))
                                print(confusion)
                            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        
                        
            model_out_dir = os.path.join(args.root_dir, args.model_output_dir)
            os.makedirs(model_out_dir, exist_ok=True)
            str_feats = "".join(f.replace("feat", "f") for f in best_feats)
            model_output_path = os.path.join(model_out_dir,  model_name_base + '-' + label_col + '-' + str(min_precision) + '-' + str(best_threshold) + '-' + str_feats)
            if best_score_model:
                best_score_model.save_model(model_output_path)
                print("@@@@@@@@@@@@@ BEST SCORE START @@@@@@@@@@@@@@@")
                print(model_name_base + '-' + label_col)
                print(min_precision)
                print(best_param)
                print(best_feats)
                print(best_score)
                print(best_threshold)
                for key, confusion in best_confusion:
                    print("#### %s ####" %(key))
                    print(confusion)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")