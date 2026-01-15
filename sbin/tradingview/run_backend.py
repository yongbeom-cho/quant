import sqlite3
import pandas as pd
import numpy as np
import os
from typing import List
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 사용자 설정 경로
DB_DIR = "/Users/yongbeom/cyb/project/2025/quant/var/data"

def clean_val(v):
    if pd.isna(v) or np.isinf(v): return None
    return float(v)

def calculate_wma(series, period):
    if len(series) < period:
        return pd.Series([np.nan] * len(series))
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

@app.get("/api/tickers")
def get_tickers():
    table_name = "coin_ohlcv_month"
    DB_PATH = os.path.join(DB_DIR, f"{table_name}.db")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT ticker FROM {table_name}")
        tickers = [row[0] for row in cur.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error loading tickers from {table_name}: {e}")
        tickers = []
    return sorted(tickers)

@app.get("/api/ohlcv")
def get_ohlcv(
    ticker: str = "KRW-BTC", 
    interval: str = "minute1",
    configs: List[str] = Query([]),
    limit: int = 500,
    offset: int = 0
):
    table_name = f"coin_ohlcv_{interval}"
    DB_PATH = os.path.join(DB_DIR, f"{table_name}.db")
    
    # 지표 계산을 위해 충분한 과거 데이터를 가져옴
    fetch_limit = limit + 200
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM {table_name} WHERE ticker='{ticker}' ORDER BY date DESC LIMIT {fetch_limit} OFFSET {offset}"
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        print(f"Query error on {table_name}: {e}")
        return []

    if df.empty: return []
    df = df.sort_values('date').reset_index(drop=True)

    mas, bbs, ichis = {}, {}, {}
    for config_str in configs:
        try:
            type_name, param = config_str.split('_')
            if type_name in ['sma', 'ema', 'wma']:
                p = int(param)
                key = f"{type_name.upper()}{p}"
                if type_name == 'sma': df[key] = df['close'].rolling(p).mean()
                elif type_name == 'ema': df[key] = df['close'].ewm(span=p, adjust=False).mean()
                elif type_name == 'wma': df[key] = calculate_wma(df['close'], p)
                mas[key] = df[key]
            
            elif type_name == 'bollinger':
                p = int(param)
                mid = df['close'].rolling(p).mean()
                std = df['close'].rolling(p).std()
                bbs[f"BB{p}"] = {"upper": mid + (std * 2), "lower": mid - (std * 2)}
            
            elif type_name == 'ichimoku':
                p1, p2, p3 = map(int, param.split(','))
                def get_hl(p): return (df['high'].rolling(p).max() + df['low'].rolling(p).min()) / 2
                tenkan = get_hl(p1)
                kijun = get_hl(p2)
                span_a = ((tenkan + kijun) / 2).shift(p2)
                span_b = get_hl(p3).shift(p2)
                ichis[f"ICHI{param.replace(',','_')}"] = {"sa": span_a, "sb": span_b}
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            continue

    result = []
    # 최신 데이터부터 limit 개수만큼만 결과에 포함
    start_idx = max(0, len(df) - limit)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        result.append({
            "time": int(pd.to_datetime(row['date']).timestamp()),
            "open": clean_val(row['open']),
            "high": clean_val(row['high']),
            "low": clean_val(row['low']),
            "close": clean_val(row['close']),
            "mas": {k: clean_val(v.iloc[i]) for k, v in mas.items() if not pd.isna(v.iloc[i])},
            "bbs": {k: {"up": clean_val(v["upper"].iloc[i]), "dn": clean_val(v["lower"].iloc[i])} for k, v in bbs.items() if not pd.isna(v["upper"].iloc[i])},
            "ichis": {k: {"sa": clean_val(v["sa"].iloc[i]), "sb": clean_val(v["sb"].iloc[i])} for k, v in ichis.items() if not pd.isna(v["sa"].iloc[i])}
        })
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)