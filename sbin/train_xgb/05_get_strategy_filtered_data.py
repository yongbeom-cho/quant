import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_xgb.strategy_feature import get_strategy_feature_filtered_feature_and_labels, get_feats_and_labels_num


def get_tickers(db_path, table_name):
    column = "ticker"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {column} FROM {table_name}")
    tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers

def load_ohlcv(db_path, table_name, ticker):
    df = None
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

            # 마지막 시도 실패면 예외 던짐
            if attempt == max_retries:
                raise

            # 대기 후 재시도
            time.sleep(retry_delay)

    # 논리적으로 여기까지 오면 안 오지만 안정성 위해
    raise RuntimeError("DB load retry failed unexpectedly.")

    return df

def save_to_db(db_path, table_base, dfs):
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

parser = argparse.ArgumentParser(description='05_get_strategy_filtered_data')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute60")
parser.add_argument('--target_strategy_feature', type=str, default="low_bb_du")

args = parser.parse_args()

if __name__ == "__main__":
    train_cut = "202301010900"
    val_cut   = "202501010900"
    train_dfs = []
    val_dfs = []
    test_dfs = []
    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    xgb_dir = os.path.join(args.root_dir, "var/xgb_data")
    os.makedirs(xgb_dir, exist_ok=True)

    tickers = get_tickers(db_path, table_name)
    for ticker in tickers:
        df = load_ohlcv(db_path, table_name, ticker)
        if len(df) <= 50:
            continue

        df = get_strategy_feature_filtered_feature_and_labels(df, args.target_strategy_feature, args.interval)
        feat_num, label_num = get_feats_and_labels_num(args.target_strategy_feature)
        tmp_df = df[df['date'] < train_cut]
        if not tmp_df.empty:
            train_dfs.append(tmp_df)
        
        tmp_df = df[(df['date'] >= train_cut) & (df['date'] < val_cut)]
        if not tmp_df.empty:
            val_dfs.append(tmp_df)
        
        tmp_df = df[df['date'] >= val_cut]
        if not tmp_df.empty:
            test_dfs.append(tmp_df)

    train_df = (pd.concat(train_dfs, axis=0).sample(frac=1, random_state=42).reset_index(drop=True))
    val_df = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_dfs, axis=0).reset_index(drop=True)
    print(len(train_df), len(val_df), len(test_df))
    
    table_base = f"xgb_{args.market}_{args.interval}_{args.target_strategy_feature}"
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

