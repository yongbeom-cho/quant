import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
import time
import os
import re
from collections import Counter

parser = argparse.ArgumentParser(description='08_get_best_model_by_ts_backtest')
parser.add_argument('--log_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant/var/log/strategy_xgb_timeseries_backtest")
parser.add_argument('--interval', type=str, default="day")
parser.add_argument('--strategy_name', type=str, default="low_bb_du")

args = parser.parse_args()


log_dir = args.log_dir
counter = Counter()

# line에서 key:value 형태 추출용 regex
kv_pattern = re.compile(r'(\w+):([^\s]+)')

for fname in os.listdir(log_dir):
    if not fname.endswith(".txt"):
        continue
    if args.strategy_name not in fname:
        continue

    # 파일명 파싱
    # interval-target_strategy-label-min_precision-threshold.txt
    name = fname.replace(".txt", "")
    parts = name.split("-")

    if len(parts) < 5:
        continue

    interval = parts[0]
    target_strategy = parts[1]
    label = parts[2]
    min_precision = parts[3]
    threshold = parts[4]

    fpath = os.path.join(log_dir, fname)

    with open(fpath, "r") as f:
        for line in f:
            if not line.startswith("############"):
                continue

            kvs = dict(kv_pattern.findall(line))
            # 필요한 값들
            try:
                uppers = kvs["uppers"]
                lower = kvs["lower"]
                close_sell = kvs["close_sell"]
                tx_cnt = kvs["tx_cnt"]
                cur_asset = float(kvs["cur_asset"])
                mdd = float(kvs["mdd"])
            except KeyError:
                continue

            score = cur_asset * mdd

            key = (
                f"{label}-{min_precision}-{threshold}-"
                f"{uppers}-{lower}-{close_sell}-"
                f"{tx_cnt}-{cur_asset}-{mdd}"
            )

            counter[key] += score

# 🔥 상위 20개 출력
print("===== TOP 20 =====")
for k, v in counter.most_common(20):
    print(f"{k} -> score: {v:.6f}")