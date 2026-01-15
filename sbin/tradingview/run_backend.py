import sqlite3
import pandas as pd
import numpy as np
import os
import pandas_ta as ta
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

# 데이터 경로 (본인 환경에 맞게 유지)
DB_DIR = "/Users/yongbeom/cyb/project/2025/quant/var/data"

def clean_val(v):
    if pd.isna(v) or np.isinf(v): return None
    return float(v)

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
        print(f"Error loading tickers: {e}")
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
    
    fetch_limit = limit + 500
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM {table_name} WHERE ticker='{ticker}' ORDER BY date DESC LIMIT {fetch_limit} OFFSET {offset}"
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        print(f"Query error: {e}")
        return []

    if df.empty: return []
    
    df = df.sort_values('date').reset_index(drop=True)

    # --- 지표 계산 ---
    calc_inds = {}

    for config_str in configs:
        try:
            parts = config_str.split('_')
            type_name = parts[0]
            param = parts[1] if len(parts) > 1 else "0"
            
            # 1. 이동평균선
            if type_name == 'sma':
                p = int(param)
                df[f"SMA{p}"] = ta.sma(df['close'], length=p)
                calc_inds[f"SMA{p}"] = df[f"SMA{p}"]
            elif type_name == 'ema':
                p = int(param)
                df[f"EMA{p}"] = ta.ema(df['close'], length=p)
                calc_inds[f"EMA{p}"] = df[f"EMA{p}"]
            elif type_name == 'wma':
                p = int(param)
                df[f"WMA{p}"] = ta.wma(df['close'], length=p)
                calc_inds[f"WMA{p}"] = df[f"WMA{p}"]
            
            # 2. 볼린저 밴드 (문제 해결된 부분)
            elif type_name == 'bollinger':
                p = int(param)
                bb = ta.bbands(df['close'], length=p)
                
                if bb is not None:
                    # 컬럼명을 추측하지 않고 'BBU', 'BBL', 'BBP'가 포함된 컬럼을 찾아서 매핑
                    # bbands 결과 컬럼 예시: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0 ...
                    col_u = next((c for c in bb.columns if c.startswith('BBU')), None)
                    col_l = next((c for c in bb.columns if c.startswith('BBL')), None)
                    col_p = next((c for c in bb.columns if c.startswith('BBP')), None) # %B

                    if col_u and col_l:
                        calc_inds[f"BB{p}"] = { "up": bb[col_u], "lo": bb[col_l] }
                    if col_p:
                        calc_inds[f"BBPB{p}"] = bb[col_p]

            # 3. 오실레이터
            elif type_name == 'rsi':
                p = int(param)
                calc_inds[f"RSI{p}"] = ta.rsi(df['close'], length=p)
            elif type_name == 'cci':
                p = int(param)
                calc_inds[f"CCI{p}"] = ta.cci(df['high'], df['low'], df['close'], length=p)
            elif type_name == 'mfi':
                p = int(param)
                calc_inds[f"MFI{p}"] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=p)
            elif type_name == 'atr':
                p = int(param)
                calc_inds[f"ATR{p}"] = ta.atr(df['high'], df['low'], df['close'], length=p)

            # 4. 채널 및 기타
            elif type_name == 'donchian':
                p = int(param)
                dc = ta.donchian(df['high'], df['low'], lower_length=p, upper_length=p)
                if dc is not None:
                    # Donchian도 안전하게 찾기
                    col_u = next((c for c in dc.columns if c.startswith('DCU')), None)
                    col_l = next((c for c in dc.columns if c.startswith('DCL')), None)
                    if col_u and col_l:
                        calc_inds[f"DC{p}"] = {"up": dc[col_u], "lo": dc[col_l]}
            
            elif type_name == 'psar':
                psar = ta.psar(df['high'], df['low'], df['close'])
                if psar is not None:
                    # Long/Short 컬럼 합치기
                    calc_inds["PSAR"] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])

            elif type_name == 'adx':
                p = int(param)
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=p)
                if adx_df is not None:
                    calc_inds[f"ADX{p}"] = {
                        "adx": adx_df[f"ADX_{p}"], 
                        "plus": adx_df[f"DMP_{p}"], 
                        "minus": adx_df[f"DMN_{p}"]
                    }
            
            elif type_name == 'volma':
                p = int(param)
                calc_inds[f"VOLMA{p}"] = ta.sma(df['volume'], length=p)

            elif type_name == 'ichimoku':
                p1, p2, p3 = map(int, param.split(','))
                ichi_df, span_df = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=p1, kijun=p2, senkou=p3)
                if ichi_df is not None:
                    safe_param = param.replace(',', '_')
                    calc_inds[f"ICHI{safe_param}"] = {
                        "sa": ichi_df[f"ISA_{p1}"], 
                        "sb": ichi_df[f"ISB_{p1}"] # 보통 ISB는 선행스팬B이므로 두번째 기간 또는 52 사용됨. ta 라이브러리 리턴 확인 필요시 컬럼매핑이 가장 안전
                    }

        except Exception as e:
            # 서버 로그에 에러를 출력하여 문제 파악 용이하게 함
            print(f"!!! Indicator calc error ({config_str}): {e}")

    # --- 결과 JSON 구성 ---
    result = []
    start_idx = max(0, len(df) - limit)
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        inds_payload = {}
        for k, v in calc_inds.items():
            if isinstance(v, pd.Series):
                inds_payload[k] = clean_val(v.iloc[i])
            elif isinstance(v, dict):
                inds_payload[k] = {sk: clean_val(sv.iloc[i]) for sk, sv in v.items()}
        
        result.append({
            "time": int(pd.to_datetime(row['date']).timestamp()),
            "open": clean_val(row['open']),
            "high": clean_val(row['high']),
            "low": clean_val(row['low']),
            "close": clean_val(row['close']),
            "volume": clean_val(row['volume']),
            "inds": inds_payload
        })

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)