import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import math
import numpy as np
import itertools
import copy
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_xgb.strategy_feature import get_feats_and_labels_num

def copy_xgb_model(model):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    model.save_model(tmp.name)

    new_model = type(model)()   # XGBClassifier or XGBRegressor
    new_model.load_model(tmp.name)
    return new_model


def feature_all():
    FEATURE_COLS = [f"f{i}" for i in range(25)]
    return FEATURE_COLS

def feature_top_k(classifier, k):
    if classifier:
        importance = classifier.get_importance()  
        # {"f3": 123, "f7": 98, ...} 형태 가정
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_feats[:k]]
    return feature_all()

def feature_random(k=10, seed=None):
    rng = np.random.default_rng(seed)
    return list(rng.choice(feature_all(), size=k, replace=False))

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

def prepare_xy(df, feat_cols, label_cols, random_state=42):
    # X, Y 중 하나라도 NaN 있으면 제거
    df = (
        df[feat_cols + label_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 랜덤 셔플
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X = df[feat_cols]
    Y = df[label_cols]

    return X, Y

class Classifier:
    def __init__(self, model_input_path):
        self.model = XGBClassifier()
        self.model.load_model(model_input_path)

    def infer(self, x, y, num_class, average='binary'):
        y_pred = self.model.predict(x)
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

        return self.model_score(test_confusion, test_precision), test_confusion

    def model_score(self, confusion, precision):
        TN, FP, FN, TP = confusion.ravel()
        base_score = (1.25 ** precision) * (0.8 ** (1 - precision))
        return (TP + FP) * np.log(base_score)
         
    def train(self, x_train, y_train, x_val, y_val, num_class=2, average='binary'):
        self.columns = x_train.columns
        # self.model.fit(x_train, y_train, verbose=True,
        #                eval_set=[(x_val, y_val)])
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        y_pred = self.model.predict(x_train)
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
        
        
        print("오차 행렬")
        print(train_confusion)
        print("정답 ratio : %.2f" %(y_train.mean()))
        if average == 'binary':
            print(f"정확도: {train_accuracy:.4f}, 정밀도: {train_precision:.4f}, 재현율: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        else:
            print("정확도: ", train_accuracy, ", 정밀도: ", train_precision, "재현율: ", train_recall, "F1: ", train_f1, "AUC: ", train_auc)
        # for i, y_ in enumerate(y_train):
        #     print(y_, y_pred[i])
        
        print ("#######  val result  ########")
        # make predictions for test data
        val_score, val_confusion = self.infer(x_val, y_val, num_class)

        # for i, y_ in enumerate(y_val):
        #     print(y_, y_pred[i])
        
        
        
        return self.model, train_score, val_score, train_confusion, val_confusion
    
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
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--target_strategy_feature', type=str, default="low_bb_du")
parser.add_argument('--model_input_dir', type=str, default="var/xgb_model")
parser.add_argument('--num_class', type=int, default=2)

args = parser.parse_args()


if __name__ == "__main__":
    num_class = args.num_class
    
    feat_num, label_num = get_feats_and_labels_num(args.target_strategy_feature)
    FEATURE_COLS = [f"feat{i}" for i in range(feat_num)]
    LABEL_COLS = [f"label{i}" for i in range(label_num)]

    interval_feat_labels = {
        "minute60": {
            "label_cols": ['label1', 'label2', 'label3'], 
            "feat_cols_list": [
                ['f15', 'f18', 'f16', 'f21', 'f24', 'f5', 'f20', 'f14', 'f17', 'f19', 'f7', 'f22', 'f2', 'f12', 'f1', 'f3', 'f4', 'f23', 'f9', 'f13'], 
                ['f15', 'f18', 'f21', 'f16', 'f14', 'f19', 'f20', 'f17', 'f24', 'f22', 'f23', 'f1', 'f2', 'f7', 'f5', 'f12', 'f6', 'f3', 'f9', 'f4', 'f13', 'f0', 'f8', 'f10'],
                ['f15', 'f18', 'f14', 'f21', 'f22', 'f6', 'f16', 'f20', 'f24', 'f11', 'f12', 'f17', 'f2', 'f1', 'f19', 'f7', 'f23', 'f9', 'f3', 'f13', 'f5', 'f0']
            ]
        }, 
        "minute240": {
            "label_cols": ['label1', 'label2', 'label3', 'label4'], 
            "feat_cols_list": [
                ['f15', 'f13', 'f4', 'f21', 'f16', 'f19', 'f2', 'f3', 'f24', 'f17', 'f20', 'f8', 'f18', 'f0', 'f23', 'f1'],
                ['f15', 'f4', 'f21', 'f19', 'f23', 'f17', 'f13', 'f20', 'f1', 'f12'],
                ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24'],
                ['f15', 'f21', 'f17', 'f19', 'f20', 'f12', 'f13', 'f23', 'f16', 'f2']
            ]
        },
        "day": {
            "label_cols": ['label1', 'label2', 'label3'], 
            "feat_cols_list": [
                ['f4', 'f14', 'f2', 'f21', 'f16', 'f18', 'f15', 'f3', 'f1', 'f20', 'f9', 'f12', 'f24', 'f0', 'f22'],
                ['f4', 'f16', 'f14', 'f19', 'f8', 'f15', 'f10', 'f13', 'f3', 'f2', 'f1', 'f21', 'f18', 'f23', 'f24', 'f22', 'f12', 'f0'],
                ['f4', 'f14', 'f8', 'f19', 'f5', 'f13', 'f10', 'f16', 'f20', 'f15', 'f3', 'f24', 'f23', 'f0', 'f21', 'f7', 'f22', 'f18', 'f1', 'f9', 'f2', 'f6', 'f17', 'f12']
            ]
        }
    }
    
    for interval, feat_label_dic in interval_feat_labels.items():
        xgb_dir = os.path.join(args.root_dir, "var/xgb_data")
        table_base = f"xgb_{args.market}_{interval}_{args.target_strategy_feature}"
        xgb_db_path = os.path.join(xgb_dir, f"{table_base}.db")
        train_df, val_df, test_df = load_from_db(xgb_db_path, table_base)

        label_cols = feat_label_dic["label_cols"]
        feat_cols_list = feat_label_dic["feat_cols_list"]
        x_train, ys_train = prepare_xy(test_df, FEATURE_COLS, LABEL_COLS)
        x_val, ys_val = prepare_xy(train_df, FEATURE_COLS, LABEL_COLS)
        x_test, ys_test = prepare_xy(val_df, FEATURE_COLS, LABEL_COLS)

        for idx in range(len(label_cols)):
            label_col = label_cols[idx]
            feat_cols = feat_cols_list[idx]

            print("######################## ", label_col, " ########################")
            y_train = ys_train[label_col]
            y_val = ys_val[label_col]
            y_test = ys_test[label_col]
            xtr = x_train[feat_cols]
            xva = x_val[feat_cols]
            xte = x_test[feat_cols]
            
            print(y_train.shape, xtr.shape, y_val.shape, xva.shape, y_test.shape, xte.shape)

            #model load            
            model_in_dir = os.path.join(args.root_dir, args.model_input_dir)
            model_input_path = os.path.join(model_in_dir,  table_base + '_' + label_col + '.classifier')
            
            
            
            classifier = Classifier(model_input_path)
            train_score, train_confusion = classifier.infer(xtr, y_train, num_class)
            val_score, val_confusion = classifier.infer(xva, y_val, num_class)
            test_score, test_confusion = classifier.infer(xte, y_test, num_class)
            print("SCORE")
            print(train_score, val_score, test_score)

            print("CONFUSION")
            print(train_confusion)
            print(val_confusion)
            print(test_confusion)
            print("#############")
            
            