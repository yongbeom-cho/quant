import sqlite3
import pandas as pd
import numpy as np
import os
import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수에서 경로 읽기 (기본값: 스크립트 기준 상대 경로)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_DIR = os.environ.get("QUANT_DB_DIR", os.path.join(PROJECT_ROOT, "var/data"))
TRADES_DIR = os.environ.get("QUANT_TRADES_DIR", os.path.join(PROJECT_ROOT, "var/trades"))

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

    mas, bbs, ichis, oscillators, volumes, atrs, psars, adxs, donchians = {}, {}, {}, {}, {}, {}, {}, {}, {}
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
            
            elif type_name == 'rsi':
                p = int(param)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                oscillators[f"RSI{p}"] = rsi
            
            elif type_name == 'mfi':
                p = int(param)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                raw_money_flow = typical_price * df['volume']
                positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(p).sum()
                negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(p).sum()
                mfi = 100 - (100 / (1 + positive_flow / negative_flow))
                oscillators[f"MFI{p}"] = mfi
            
            elif type_name == 'cci':
                p = int(param)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(p).mean()
                mad = typical_price.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                cci = (typical_price - sma_tp) / (0.015 * mad)
                oscillators[f"CCI{p}"] = cci
            
            elif type_name == 'vol_sma' or type_name == 'volma':
                p = int(param)
                volumes[f"VOLMA{p}"] = df['volume'].rolling(p).mean()
            
            elif type_name == 'atr':
                p = int(param)
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atrs[f"ATR{p}"] = tr.rolling(p).mean()
            
            elif type_name == 'psar':
                p = float(param) if param != '0' else 0.02
                af = p if p > 0 else 0.02
                psar_vals = []
                ep = df['high'].iloc[0]
                psar = df['low'].iloc[0]
                trend = 1
                for i in range(len(df)):
                    if i == 0:
                        psar_vals.append(psar)
                        continue
                    if trend == 1:
                        psar = psar + af * (ep - psar)
                        if df['low'].iloc[i] < psar:
                            trend = -1
                            psar = ep
                            ep = df['low'].iloc[i]
                            af = 0.02
                        else:
                            if df['high'].iloc[i] > ep:
                                ep = df['high'].iloc[i]
                                af = min(af + 0.02, 0.2)
                    else:
                        psar = psar + af * (ep - psar)
                        if df['high'].iloc[i] > psar:
                            trend = 1
                            psar = ep
                            ep = df['high'].iloc[i]
                            af = 0.02
                        else:
                            if df['low'].iloc[i] < ep:
                                ep = df['low'].iloc[i]
                                af = min(af + 0.02, 0.2)
                    psar_vals.append(psar)
                psars["PSAR"] = pd.Series(psar_vals, index=df.index)
            
            elif type_name == 'adx':
                p = int(param)
                high_diff = df['high'].diff()
                low_diff = -df['low'].diff()
                plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                tr = pd.concat([df['high'] - df['low'], 
                               np.abs(df['high'] - df['close'].shift()),
                               np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
                atr = tr.rolling(p).mean()
                plus_di = 100 * (plus_dm.rolling(p).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(p).mean() / atr)
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(p).mean()
                adxs[f"ADX{p}"] = {"adx": adx, "plus": plus_di, "minus": minus_di}
            
            elif type_name == 'donchian':
                p = int(param)
                donchians[f"DC{p}"] = {"up": df['high'].rolling(p).max(), "lo": df['low'].rolling(p).min()}
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
            "volume": clean_val(row.get('volume', 0)),
            "mas": {k: clean_val(v.iloc[i]) for k, v in mas.items() if not pd.isna(v.iloc[i])},
            "bbs": {k: {"up": clean_val(v["upper"].iloc[i]), "dn": clean_val(v["lower"].iloc[i])} for k, v in bbs.items() if not pd.isna(v["upper"].iloc[i])},
            "ichis": {k: {"sa": clean_val(v["sa"].iloc[i]), "sb": clean_val(v["sb"].iloc[i])} for k, v in ichis.items() if not pd.isna(v["sa"].iloc[i])},
            "oscillators": {k: clean_val(v.iloc[i]) for k, v in oscillators.items() if not pd.isna(v.iloc[i])},
            "volumes": {k: clean_val(v.iloc[i]) for k, v in volumes.items() if not pd.isna(v.iloc[i])},
            "atrs": {k: clean_val(v.iloc[i]) for k, v in atrs.items() if not pd.isna(v.iloc[i])},
            "psars": {k: clean_val(v.iloc[i]) for k, v in psars.items() if not pd.isna(v.iloc[i])},
            "adxs": {k: {"adx": clean_val(v["adx"].iloc[i]), "plus": clean_val(v["plus"].iloc[i]), "minus": clean_val(v["minus"].iloc[i])} for k, v in adxs.items() if not pd.isna(v["adx"].iloc[i])},
            "donchians": {k: {"up": clean_val(v["up"].iloc[i]), "lo": clean_val(v["lo"].iloc[i])} for k, v in donchians.items() if not pd.isna(v["up"].iloc[i])}
        })
    return result


@app.get("/api/trades")
def get_trades_index():
    """
    거래 내역 인덱스 목록 조회
    trades_index.json에서 모든 거래 기록의 메타데이터를 반환합니다.
    """
    index_path = os.path.join(TRADES_DIR, "trades_index.json")
    
    if not os.path.exists(index_path):
        return {"trades": [], "message": "No trade data found"}
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            trades_index = json.load(f)
        
        # 인덱스를 리스트 형태로 변환 (프론트엔드 사용 편의)
        trades_list = [
            {"id": trade_id, **metadata}
            for trade_id, metadata in trades_index.items()
        ]
        
        # total_pnl 기준 내림차순 정렬
        trades_list.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)
        
        return {"trades": trades_list, "count": len(trades_list)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trades index: {str(e)}")


@app.get("/api/trades/{trade_id}")
def get_trade_detail(trade_id: str):
    """
    개별 거래 내역 조회
    차트에 마커를 표시하기 위한 상세 거래 데이터를 반환합니다.
    """
    # 인덱스에서 메타데이터 조회
    index_path = os.path.join(TRADES_DIR, "trades_index.json")
    trade_path = os.path.join(TRADES_DIR, f"{trade_id}.json")
    
    if not os.path.exists(trade_path):
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    
    try:
        # 메타데이터 로드
        metadata = {}
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                trades_index = json.load(f)
                metadata = trades_index.get(trade_id, {})
        
        # 거래 내역 로드
        with open(trade_path, 'r', encoding='utf-8') as f:
            trade_data = json.load(f)
        
        # 차트 마커용 데이터 변환 (timestamp 추가)
        markers = []
        for trade in trade_data.get("trades", []):
            # 진입 마커
            entry_date = trade.get("entry_date", "")
            if entry_date:
                # YYYYMMDD 형식을 timestamp로 변환
                try:
                    entry_ts = int(pd.to_datetime(entry_date).timestamp())
                    markers.append({
                        "time": entry_ts,
                        "position": "belowBar" if trade.get("direction", 1) == 1 else "aboveBar",
                        "color": "#26a69a" if trade.get("direction", 1) == 1 else "#ef5350",
                        "shape": "arrowUp" if trade.get("direction", 1) == 1 else "arrowDown",
                        "text": f"Entry: {trade.get('entry_reason', '')}",
                        "price": trade.get("entry_price"),
                        "details": {
                            "type": "entry",
                            "price": trade.get("entry_price"),
                            "reason": trade.get("entry_reason"),
                            "date": trade.get("entry_date")
                        }
                    })
                except:
                    pass
            
            # 청산 마커
            exit_date = trade.get("exit_date", "")
            if exit_date:
                try:
                    exit_ts = int(pd.to_datetime(exit_date).timestamp())
                    pnl = trade.get("pnl", 0)
                    markers.append({
                        "time": exit_ts,
                        "position": "aboveBar" if trade.get("direction", 1) == 1 else "belowBar",
                        "color": "#26a69a" if pnl > 0 else "#ef5350",
                        "shape": "circle",
                        "text": f"Exit: {trade.get('exit_reason', '')} ({pnl:.2%})",
                        "price": trade.get("exit_price"),
                        "details": {
                            "type": "exit",
                            "price": trade.get("exit_price"),
                            "reason": trade.get("exit_reason"),
                            "pnl": pnl,
                            "pnlAmount": trade.get("pnl_amount"),
                            "holdingBars": trade.get("holding_bars"),
                            "date": trade.get("exit_date")
                        }
                    })
                except:
                    pass
        
        return {
            "id": trade_id,
            "metadata": metadata,
            "trades": trade_data.get("trades", []),
            "markers": markers
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trade data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)